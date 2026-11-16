#!/usr/bin/env python3
"""SHURA (VRAMLOGIC4) engine test, CPU only: the state after a pause on a small hybrid model with a chat template (Qwen3.5-2B):
JSON schema, tool_choice required / auto, thinking with and without a reasoning budget, a stop string cut by the pause,
repetition penalties. Each scenario is compared with the answer without a pause.

    python3 tools/server/tests/shura/pause_state_cpu.py build/bin/llama-server Qwen3.5-2B-Q4_K_M.gguf /tmp/out [--old] [scenario ...]
"""
import json, os, subprocess, sys, time, urllib.request

BIN, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
OLD = "--old" in sys.argv
ONLY = [a for a in sys.argv[4:] if not a.startswith("--")]
P = 18398
SLOTS = os.path.join(OUT, "slots")
os.makedirs(SLOTS, exist_ok=True)


def http(m, path, body=None, timeout=300):
    r = urllib.request.Request(f"http://127.0.0.1:{P}{path}", json.dumps(body).encode() if body is not None else None,
                               {"Content-Type": "application/json"}, method=m)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, json.loads(f.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    except Exception as e:
        return None, {"exception": repr(e)}


def start(tag):
    assert not subprocess.run(["ss", "-ltnH", f"sport = :{P}"], capture_output=True, text=True).stdout.strip(), "port taken"
    srv = subprocess.Popen([BIN, "-m", MODEL, "--port", str(P), "-c", "4096", "-np", "1", "--load-mode", "none", "-ngl", "0", "-t", "4",
                            "--jinja", "--slot-save-path", SLOTS + "/", "--vram-test-hooks", "--cache-ram", "0"],
                           stdout=open(os.path.join(OUT, f"{tag}.log"), "wb"), stderr=subprocess.STDOUT)
    for _ in range(600):
        if srv.poll() is not None:
            raise SystemExit(f"server {tag} exited")
        try:
            if http("GET", "/health", timeout=2)[1].get("status") == "ok":
                return srv
        except Exception:
            pass
        time.sleep(0.1)
    raise SystemExit("no health")


def stop(srv):
    if srv.poll() is None:
        srv.terminate()
        srv.wait(30)


TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["c", "f"]}},
                         "required": ["city", "unit"]}}}]
SCHEMA = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}, "tags": {"type": "array", "items": {"type": "string"}}},
          "required": ["name", "age", "tags"]}
BASE = {"temperature": 0, "top_k": 1, "seed": 1, "id_slot": 0, "cache_prompt": True}
NOTHINK = {"chat_template_kwargs": {"enable_thinking": False}}

SC = {
    "json": ("/v1/chat/completions", dict(BASE, **NOTHINK, max_tokens=120, messages=[{"role": "user", "content": "Describe a fictional person as JSON."}],
              response_format={"type": "json_schema", "json_schema": {"name": "p", "schema": SCHEMA}}), 12),
    "tool_required": ("/v1/chat/completions", dict(BASE, **NOTHINK, max_tokens=120, messages=[{"role": "user", "content": "What is the weather in Almaty in celsius?"}],
                       tools=TOOLS, tool_choice="required"), 9),
    "tool_auto": ("/v1/chat/completions", dict(BASE, **NOTHINK, max_tokens=120, messages=[{"role": "user", "content": "Call the weather tool for Paris in fahrenheit."}],
                   tools=TOOLS, tool_choice="auto"), 9),
    "thinking": ("/v1/chat/completions", dict(BASE, max_tokens=400, messages=[{"role": "user", "content": "Is 91 prime? Answer briefly."}]), 10),
    "thinking_budget": ("/v1/chat/completions", dict(BASE, max_tokens=400, reasoning_budget_tokens=24,
                         messages=[{"role": "user", "content": "Is 91 prime? Answer briefly."}]), 10),
    "stop_split": ("/completion", dict(BASE, n_predict=80, ignore_eos=True, stop=["STOPMARK"],
                    prompt="Repeat exactly: one two three STOPMARK four five. Answer: one two three"), 0),
    "json_double": ("/v1/chat/completions", dict(BASE, **NOTHINK, max_tokens=120, messages=[{"role": "user", "content": "Describe a fictional person as JSON."}],
                     response_format={"type": "json_schema", "json_schema": {"name": "p", "schema": SCHEMA}}), [12, 7]),
    "stop_split_stream": ("/completion", dict(BASE, n_predict=80, ignore_eos=True, stop=["STOPMARK"],
                           prompt="Repeat exactly: one two three STOPMARK four five. Answer: one two three"), 0),
    "penalties": ("/completion", dict(BASE, n_predict=80, ignore_eos=True, repeat_penalty=1.3, presence_penalty=0.6, frequency_penalty=0.4,
                   prompt="List of fruits: apple, banana, apple, banana, apple,"), 25),
}


def stream_text(path, body):
    """the text a streaming client gets (raw completion): the concatenated content of the SSE events"""
    r = urllib.request.Request(f"http://127.0.0.1:{P}{path}", json.dumps(dict(body, stream=True)).encode(), {"Content-Type": "application/json"})
    text, last = "", {}
    try:
        with urllib.request.urlopen(r, timeout=300) as f:
            for line in f:
                line = line.decode(errors="replace").strip()
                if line.startswith("data: ") and line[6:] != "[DONE]":
                    ev = json.loads(line[6:])
                    text += ev.get("content", "")
                    last = ev
    except Exception as e:
        last = {"exception": repr(e)}
    return text, last


def run(path, body, k, tag, stream=False):
    """returns (code, answer, info) of a run with pauses after k tokens (k: int or a list, one pause per part; 0: no pause)"""
    ks = k if isinstance(k, list) else ([k] if k else [])
    for f in os.listdir(SLOTS):
        os.unlink(os.path.join(SLOTS, f))
    info = {"parts": []}
    seen = ""
    srv = start(tag + "_a")
    cur_body = body
    for j in range(len(ks) + 1):
        if j < len(ks):
            http("POST", "/test/pause", {"after_tokens": ks[j]})
        if stream:
            t, last = stream_text(path, cur_body)
            seen += t
            c, r = (200, {"content": t, "stop_type": last.get("stop_type"), "tokens_predicted": last.get("tokens_predicted")}) if j == len(ks) else (None, {})
        else:
            c, r = http("POST", path, cur_body)
        if j == len(ks):
            stop(srv)
            info["client_text"] = seen
            return c, r, info
        for _ in range(100):
            if srv.poll() is not None:
                break
            time.sleep(0.05)
        info["parts"].append({"exit75": srv.poll() == 75})
        stop(srv)
        tf = next((f for f in os.listdir(SLOTS) if f.endswith(".tokens.json")), None)
        if not tf:
            return None, {}, dict(info, error="no pause file")
        side_path = os.path.join(SLOTS, tf)
        side = json.load(open(side_path))
        info["parts"][-1]["side"] = {x: side.get(x) for x in ("n_past", "generated_tokens", "generated_total", "pending_token", "unsent_bytes", "n_predict_left")}
        if OLD:
            for x in ("pending_token", "generated_total", "unsent_bytes"):
                side.pop(x, None)
            json.dump(side, open(side_path, "w"))
        srv = start(f"{tag}_r{j}")
        info["parts"][-1]["restore"] = http("POST", "/slots/0?action=restore", {"filename": tf[: -len(".tokens.json")]})[0]
        for f in os.listdir(SLOTS):   # the next pause writes its own files
            if f != tf and not f.startswith(tf[: -len(".tokens.json")]):
                os.unlink(os.path.join(SLOTS, f))
        cur_body = dict(body, resume={"tokens_file": tf})
        # a second pause in the same answer renames nothing: keep the restored files apart
        os.rename(side_path, side_path + f".{j}")
        cur_body = dict(body, resume={"tokens_file": tf + f".{j}"})


def answer(path, r):
    if path == "/completion":
        return {"content": r.get("content"), "n": r.get("tokens_predicted"), "timings_n": (r.get("timings") or {}).get("predicted_n"),
                "stop": r.get("stop_type")}
    ch = (r.get("choices") or [{}])[0]
    m = ch.get("message") or {}
    return {"content": m.get("content"), "reasoning": m.get("reasoning_content"),
            "tool_calls": [{"name": t["function"]["name"], "arguments": t["function"]["arguments"]} for t in (m.get("tool_calls") or [])],
            "finish": ch.get("finish_reason"), "usage": r.get("usage"), "timings_n": (r.get("timings") or {}).get("predicted_n")}


def valid(name, a):
    try:
        if name == "json":
            o = json.loads(a["content"])
            return isinstance(o.get("name"), str) and isinstance(o.get("age"), int) and isinstance(o.get("tags"), list)
        if name.startswith("tool"):
            if not a["tool_calls"]:
                return False
            args = json.loads(a["tool_calls"][0]["arguments"])
            return a["tool_calls"][0]["name"] == "get_weather" and isinstance(args.get("city"), str) and args.get("unit") in ("c", "f")
        if name.startswith("thinking"):
            return bool(a["content"]) and "<think>" not in (a["content"] or "") and "</think>" not in (a["content"] or "")
    except Exception:
        return False
    return True


res = {"old": OLD}
for name, (path, body, k) in SC.items():
    if ONLY and name not in ONLY:
        continue
    c0, r0, _ = run(path, body, 0, name + "_ref")
    ref = answer(path, r0)
    if name.startswith("stop_split"):   # the pause between the last two tokens of the stop string: the answer must stop at the same place
        k = max(1, (ref.get("n") or 2) - 1)
    c1, r1, info = run(path, body, k, name + "_pause", stream=name.endswith("_stream"))
    got = answer(path, r1) if c1 == 200 else {"error": r1}
    if name.endswith("_stream") and c1 == 200:
        # what the streaming client saw over both parts equals the answer without a pause
        same = {"client_text": info.get("client_text") == ref.get("content"), "stop": r1.get("stop_type") == ref.get("stop")}
    elif path == "/completion" and c1 == 200:
        # a raw completion returns only the text after the pause (the client joins): the reference ends with it, the counts add up
        same = {"content": got.get("content") is not None and (ref.get("content") or "").endswith(got["content"]),
                "n": (got.get("n") or 0) + k == ref.get("n"), "stop": got.get("stop") == ref.get("stop")}
    else:
        same = {x: got.get(x) == ref.get(x) for x in ref if x not in ("usage", "timings_n")}
    out = {"k": k, "ref": ref, "paused": got, "info": info, "same_fields": same, "all_same": all(same.values()),
           "ref_valid": valid(name, ref), "paused_valid": valid(name, got) if c1 == 200 else False}
    res[name] = out
    print(name, "all_same", out["all_same"], "valid", out["ref_valid"], out["paused_valid"], {x: v for x, v in same.items() if not v}, flush=True)
json.dump(res, open(os.path.join(OUT, "vl4_state" + ("_old" if OLD else "") + ".json"), "w"), indent=1, ensure_ascii=False)
sys.exit(0 if OLD or all(v.get("all_same") for k, v in res.items() if isinstance(v, dict)) else 1)
