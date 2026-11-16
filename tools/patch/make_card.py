#!/usr/bin/env python3
"""SHURA (PATCH1): build an expansion card (a patch) for shura.cpp — a GGUF file with slices of the base model's tensors in the base's
own format, its manifest and the hashes the engine checks before writing anything (src/llama-patch.cpp).

  make_card.py --base BASE.gguf --out CARD.gguf --name NAME [--donor DONOR.gguf --pairs pairs.json [--kinds gate,up,down]]
               [--identity N] [--whole TENSOR[,TENSOR...]] [--recipe TEXT] [--switch always|JSON] [--privacy local]
               [--previous HASH] [--full-base-hash]

Slices:
  --donor + --pairs   experts copied byte for byte from a donor of the same type and shape (pairs.json: {"pairs": [[layer, expert], ...]}),
                      e.g. RUMIX0's recipe hy_qwen_5 (Qwen3.6 APEX experts over Tiel);
  --identity N        N expert slices with the base's own bytes (a card that changes nothing: the identity check);
  --whole T           a whole base tensor with the base's own bytes (tests).
The base file is only read. Card format (all little endian, GGUF v3):
  shura.patch.version          u32  1
  shura.patch.manifest         str  JSON (data only): name, content_sha256, base {file, file_sha256, layout_sha256}, recipe, switch,
                                    slice budget, tests, impact, previous version, privacy class
  shura.patch.base_layout_sha256 str sha256 over "name\\ttype\\tnbytes\\n" of every base tensor in file order
  shura.patch.targets          [str]  base tensor of slice i
  shura.patch.experts          [i32]  expert index of slice i (-1: the whole tensor)
  shura.patch.base_sha256      [str]  sha256 of the base bytes slice i replaces
  tensors slice.0 .. slice.N-1        the bytes, same type as the target, shape one expert (ne0, ne1, 1) or the whole tensor
content_sha256 = sha256 over "target\\texpert\\tsha256(slice bytes)\\n" for every slice in order.
"""
import argparse, hashlib, json, mmap, os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import numpy as np  # noqa: E402
from gguf import GGUFReader, GGUFWriter  # noqa: E402

MAX_SLICES = 1 << 20


def layout_sha256(r):
    h = hashlib.sha256()
    for t in r.tensors:
        h.update(f"{t.name}\t{int(t.tensor_type)}\t{int(t.n_bytes)}\n".encode())
    return h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(64 << 20)
            if not b:
                return h.hexdigest()
            h.update(b)


def content_sha256(slices):
    h = hashlib.sha256()
    for n, e, data, ty, ne, base_h in slices:
        h.update(f"{n}\t{e}\t{hashlib.sha256(data).hexdigest()}\n".encode())
    return h.hexdigest()


def write_card(out, man, slices, layout=None):
    """slices: (target, expert, bytes, ggml type, ggml shape ne, sha256 of the base bytes). Tests call it with broken input."""
    tmp = out + ".tmp"
    w = GGUFWriter(tmp, arch="shura-card", use_temp_file=False)
    w.add_uint32("shura.patch.version", 1)
    w.add_string("shura.patch.manifest", json.dumps(man, ensure_ascii=False, separators=(",", ":")))
    w.add_string("shura.patch.base_layout_sha256", layout or man["base"]["layout_sha256"])
    w.add_array("shura.patch.targets", [s[0] for s in slices])
    w.add_array("shura.patch.experts", [int(s[1]) for s in slices])
    w.add_array("shura.patch.base_sha256", [s[5] for s in slices])
    for i, (n, e, data, ty, ne, _) in enumerate(slices):
        rows_n = int(np.prod(ne[1:]))
        arr = np.frombuffer(data, dtype=np.uint8).reshape(list(reversed(ne[1:])) + [len(data) // rows_n])
        w.add_tensor(f"slice.{i}", arr, raw_dtype=ty)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    os.replace(tmp, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--donor")
    ap.add_argument("--pairs")
    ap.add_argument("--kinds", default="gate,up,down")
    ap.add_argument("--identity", type=int, default=0)
    ap.add_argument("--identity-seed", type=int, default=1)
    ap.add_argument("--whole", default="")
    ap.add_argument("--recipe", default="")
    ap.add_argument("--switch", default="always")
    ap.add_argument("--privacy", default="local")
    ap.add_argument("--previous", default="")
    ap.add_argument("--full-base-hash", action="store_true")
    a = ap.parse_args()

    rb = GGUFReader(a.base)
    tb = {t.name: t for t in rb.tensors}
    bm = mmap.mmap(os.open(a.base, os.O_RDONLY), 0, access=mmap.ACCESS_READ)
    plan = []   # (target, expert, bytes, type, ggml shape ne)

    def expert_slice(t, e):
        n_exp = int(t.shape[-1])
        per = int(t.n_bytes) // n_exp
        assert per * n_exp == int(t.n_bytes)
        return per, int(t.data_offset) + e * per

    if a.donor:
        rd = GGUFReader(a.donor)
        td = {t.name: t for t in rd.tensors}
        dm = mmap.mmap(os.open(a.donor, os.O_RDONLY), 0, access=mmap.ACCESS_READ)
        pairs = sorted({(int(L), int(e)) for L, e in json.load(open(a.pairs))["pairs"]})
        for L, e in pairs:
            for k in a.kinds.split(","):
                n = f"blk.{L}.ffn_{k}_exps.weight"
                b, d = tb[n], td[n]
                if list(b.shape) != list(d.shape) or b.tensor_type != d.tensor_type or b.n_bytes != d.n_bytes:
                    sys.exit(f"{n}: the donor is not byte-compatible")
                per, ob = expert_slice(b, e)
                _, od = expert_slice(d, e)
                plan.append((n, e, dm[od:od + per], b.tensor_type, [int(b.shape[0]), int(b.shape[1]), 1]))
    if a.identity:
        import random
        rnd = random.Random(a.identity_seed)
        exps = [t for t in rb.tensors if "_exps" in t.name and t.name.startswith("blk.")]
        seen = {(n, e) for n, e, *_ in plan}
        added = 0
        while added < a.identity:
            t = rnd.choice(exps)
            e = rnd.randrange(int(t.shape[-1]))
            if (t.name, e) in seen:
                continue
            seen.add((t.name, e))
            per, ob = expert_slice(t, e)
            plan.append((t.name, e, bm[ob:ob + per], t.tensor_type, [int(t.shape[0]), int(t.shape[1]), 1]))
            added += 1
    for n in [x for x in a.whole.split(",") if x]:
        t = tb[n]
        ob = int(t.data_offset)
        plan.append((n, -1, bm[ob:ob + int(t.n_bytes)], t.tensor_type, [int(x) for x in t.shape]))
    if not plan or len(plan) > MAX_SLICES:
        sys.exit("no slices (or too many)")

    # hashes: the replaced base bytes, the slice bytes, the content
    rows, ch = [], hashlib.sha256()
    for n, e, data, ty, ne in plan:
        t = tb[n]
        if e >= 0:
            per, ob = expert_slice(t, e)
        else:
            per, ob = int(t.n_bytes), int(t.data_offset)
        base_h = hashlib.sha256(bm[ob:ob + per]).hexdigest()
        sh = hashlib.sha256(data).hexdigest()
        ch.update(f"{n}\t{e}\t{sh}\n".encode())
        rows.append((n, e, base_h, sh))
    content = ch.hexdigest()
    n_changed = sum(1 for n, e, bh, sh in rows if bh != sh)

    man = {
        "format": "shura.card/1",
        "name": a.name,
        "content_sha256": content,
        "base": {"file": os.path.basename(a.base), "file_sha256": file_sha256(a.base) if a.full_base_hash else None,
                 "layout_sha256": layout_sha256(rb)},
        "slices": len(plan), "slices_changed": n_changed, "bytes": sum(len(p[2]) for p in plan),
        "recipe": a.recipe or ("donor " + os.path.basename(a.donor) + " pairs " + os.path.basename(a.pairs) if a.donor else "identity"),
        "data_refs": [], "tests_passed": [], "impact": {"speed": None, "ram_mib": 0, "vram_mib": 0},
        "previous_version": a.previous or None, "privacy": a.privacy,
        "switch": a.switch if a.switch == "always" else json.loads(a.switch),
        "slice_budget": None,
        "built": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    write_card(a.out, man, [(n, e, data, ty, ne, r[2]) for (n, e, data, ty, ne), r in zip(plan, rows)])
    print(json.dumps({"out": a.out, "slices": len(plan), "changed": n_changed, "mib": round(man["bytes"] / 2**20, 1),
                      "content_sha256": content, "layout": man["base"]["layout_sha256"][:12]}))


if __name__ == "__main__":
    main()
