#!/usr/bin/env python3
"""SHURA (VRAMLOGIC4) engine test, CPU only: the answer with a pause (the server's own pause task, POST /test/pause with
after_tokens, --vram-test-hooks) equals the answer without one, token for token and log-probability for log-probability, at 5
positions, on a small hybrid model:

    test-llama-archs -a '^qwen35moe$' -o M
    python3 tools/server/tests/shura/pause_resume_cpu.py build/bin/llama-server M/qwen35moe-moe.gguf /tmp/out [--old]

--old drops pending_token from the pause sidecar (the behaviour before VRAMLOGIC4: the last token repeats at the seam).
"""
import json, os, subprocess, sys, time, urllib.request

BIN, MODEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
OLD = "--old" in sys.argv
P = 18398
SLOTS = os.path.join(OUT, "slots")
os.makedirs(SLOTS, exist_ok=True)
N = 60
PROMPT = [(i * 7919 + 13) % 100 for i in range(40)]


def http(m, path, body=None, timeout=60):
    r = urllib.request.Request(f"http://127.0.0.1:{P}{path}", json.dumps(body).encode() if body is not None else None,
                               {"Content-Type": "application/json"}, method=m)
    try:
        with urllib.request.urlopen(r, timeout=timeout) as f:
            return f.status, json.loads(f.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")
    except Exception as e:  # the server exits during a pause
        return None, {"exception": repr(e)}


def start(tag, extra=()):
    assert not subprocess.run(["ss", "-ltnH", f"sport = :{P}"], capture_output=True, text=True).stdout.strip(), "port taken"
    srv = subprocess.Popen([BIN, "-m", MODEL, "--port", str(P), "-c", "2048", "-np", "1", "--load-mode", "none", "-ngl", "0", "-t", "2",
                            "--slot-save-path", SLOTS + "/", "--vram-test-hooks", "--cache-ram", "0", *extra],
                           stdout=open(os.path.join(OUT, f"{tag}.log"), "wb"), stderr=subprocess.STDOUT)
    for _ in range(200):
        if srv.poll() is not None:
            raise SystemExit(f"server {tag} exited")
        try:
            if http("GET", "/health", timeout=2)[1].get("status") == "ok":
                return srv
        except Exception:
            pass
        time.sleep(0.05)
    raise SystemExit("no health")


def stop(srv):
    if srv.poll() is None:
        srv.terminate()
        srv.wait(20)


def gen(extra=None, n=N):
    body = {"prompt": PROMPT, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1, "ignore_eos": True, "cache_prompt": True,
            "n_probs": 5, "id_slot": 0}
    body.update(extra or {})
    return http("POST", "/completion", body)


def toks(r):
    cp = r.get("completion_probabilities") or []
    return [t["id"] for t in cp], [[c["logprob"] for c in t["top_logprobs"]] for t in cp]


res = {"old": OLD}
srv = start("ref")
c, r = gen()
stop(srv)
ref_ids, ref_lp = toks(r)
res["ref_n"] = len(ref_ids)
assert len(ref_ids) == N, ("the reference failed", r)
srv = start("ref2")
c, r2 = gen()
stop(srv)
res["ref_repeat_same"] = toks(r2) == (ref_ids, ref_lp)

cases = []
for k in (1, 2, 10, 30, N - 1):
    for f in os.listdir(SLOTS):
        os.unlink(os.path.join(SLOTS, f))
    srv = start(f"pause{k}")
    http("POST", "/test/pause", {"after_tokens": k})
    c, rp = gen()
    for _ in range(100):
        if srv.poll() is not None:
            break
        time.sleep(0.05)
    exit_code = srv.poll()
    stop(srv)
    tf = next((f for f in os.listdir(SLOTS) if f.endswith(".tokens.json")), None)
    case = {"k": k, "exit": exit_code, "pause_answer_code": c, "tokens_file": tf}
    if tf:
        side_path = os.path.join(SLOTS, tf)
        side = json.load(open(side_path))
        case["side"] = {x: side.get(x) for x in ("n_past", "generated_tokens", "pending_token", "n_predict_left", "prompt_done")}
        if OLD and "pending_token" in side:
            side.pop("pending_token")
            json.dump(side, open(side_path, "w"))
        srv = start(f"resume{k}")
        case["restore"] = http("POST", "/slots/0?action=restore", {"filename": tf[: -len(".tokens.json")]})[0]
        c, rr = gen({"prompt": [1], "resume": {"tokens_file": tf}})
        log = open(os.path.join(OUT, f"resume{k}.log"), errors="replace").read()
        stop(srv)
        ids, lp = toks(rr) if c == 200 else ([], [])
        full_ids = ref_ids[:k] + ids
        full_lp = ref_lp[:k] + lp
        first = next((i for i, (a, b) in enumerate(zip(full_ids, ref_ids)) if a != b), None)
        md = max((abs(a - b) for i in range(min(len(full_lp), len(ref_lp)) if first is None else first)
                  for a, b in zip(full_lp[i], ref_lp[i])), default=0.0)
        case["resumed_head"] = ids[:4]; case["ref_around"] = ref_ids[max(0, k - 2): k + 3]
        case.update({"resume_code": c, "resumed_n": len(ids), "total_n": len(full_ids), "same_tokens": full_ids == ref_ids,
                     "first_diff": first, "max_logprob_diff_before_split": md, "same_logprobs": full_lp == ref_lp,
                     "checkpoint_reread": "restored" in log and "context checkpoint" in log,
                     "log": [l[l.find("resume:"):][:160] for l in log.splitlines() if "resume:" in l or "erased invalidated" in l or "restored" in l][:6]})
    cases.append(case)
    print(json.dumps({x: case.get(x) for x in ("k", "exit", "resumed_n", "total_n", "same_tokens", "first_diff", "same_logprobs", "max_logprob_diff_before_split")}), flush=True)
res["cases"] = cases
json.dump(res, open(os.path.join(OUT, "vl4_cpu" + ("_old" if OLD else "") + ".json"), "w"), indent=1)
print("ref_repeat_same", res["ref_repeat_same"])
sys.exit(0 if OLD or all(c.get("same_tokens") and c.get("same_logprobs") for c in cases) else 1)
