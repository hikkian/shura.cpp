#!/usr/bin/env python3
"""SHURA (PATCH1): the cards test-patch uses, built over a small MoE model (test-llama-archs -a deepseek2 -o DIR) and a donor of the
same layout (the same generator with another seed).

  test_cards.py BASE.gguf DONOR.gguf OUTDIR

Good cards: id (base bytes, identity), donor (8 experts of the donor, all expert tensors), donor2 (8 other experts), whole (one whole
tensor from the donor), conflict (shares one expert with donor).
Cards the loader must reject before writing anything: bad_truncated, bad_otherbase, bad_layout, bad_overlap, bad_range, bad_type,
bad_shape, bad_tamper, bad_notensor, bad_mtp, bad_nomanifest.
"""
import copy, hashlib, json, mmap, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "gguf-py"))
import make_card as mc  # noqa: E402
from gguf import GGUFReader, GGMLQuantizationType  # noqa: E402

base, donor, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
rb, rd = GGUFReader(base), GGUFReader(donor)
tb, td = {t.name: t for t in rb.tensors}, {t.name: t for t in rd.tensors}
bm = mmap.mmap(os.open(base, os.O_RDONLY), 0, access=mmap.ACCESS_READ)
dm = mmap.mmap(os.open(donor, os.O_RDONLY), 0, access=mmap.ACCESS_READ)
layout = mc.layout_sha256(rb)
exps = sorted(n for n, t in tb.items() if n.endswith("_exps.weight") and len(t.shape) == 3)
assert exps, "no stacked expert tensors"


def sl(n, e, src_map, src_t):
    t, s = tb[n], src_t[n]
    if e >= 0:
        per = int(t.n_bytes) // int(t.shape[-1])
        ob, os_ = int(t.data_offset) + e * per, int(s.data_offset) + e * per
        ne = [int(t.shape[0]), int(t.shape[1]), 1]
    else:
        per, ob, os_ = int(t.n_bytes), int(t.data_offset), int(s.data_offset)
        ne = [int(x) for x in t.shape]
    return [n, e, bytes(src_map[os_:os_ + per]), t.tensor_type, ne, hashlib.sha256(bm[ob:ob + per]).hexdigest()]


def card(name, slices, layout_=None, man_patch=None):
    man = {"format": "shura.card/1", "name": name, "content_sha256": mc.content_sha256(slices),
           "base": {"file": os.path.basename(base), "layout_sha256": layout}, "slices": len(slices), "switch": "always", "privacy": "local"}
    if man_patch:
        man_patch(man)
    p = os.path.join(out, name + ".gguf")
    mc.write_card(p, man, slices, layout_)
    return p


n_exp = int(tb[exps[0]].shape[-1])
card("id", [sl(n, e, bm, tb) for n in exps for e in (1, 5, 9, 62)])   # the random router of the test model picks 62 and 63
card("donor", [sl(n, e, dm, td) for n in exps for e in list(range(0, 16, 2)) + [62]])
card("donor2", [sl(n, e, dm, td) for n in exps for e in list(range(1, 17, 2)) + [63]])
card("whole", [sl(exps[0], -1, dm, td)])
card("conflict", [sl(exps[-1], 4, dm, td)])

# rejected cards
p = card("bad_truncated", [sl(n, e, dm, td) for n in exps for e in range(0, 16, 2)])
sz = os.path.getsize(p)
with open(p, "r+b") as f:
    f.truncate(sz * 7 // 10)
# a card made for another base with the same layout: the base hashes it records are the donor's
rb_save, bm_save = None, None
s = [sl(n, e, dm, td) for n in exps for e in (2, 3)]
for x in s:
    x[5] = hashlib.sha256(x[2]).hexdigest()   # "its" base bytes = the donor's
card("bad_otherbase", s)
card("bad_layout", [sl(exps[0], 1, dm, td)], layout_="0" * 64)
card("bad_overlap", [sl(exps[0], 1, dm, td), sl(exps[0], 1, dm, td)])
x = sl(exps[0], 1, dm, td); x[1] = n_exp
card("bad_range", [x])
x = sl(exps[0], 1, dm, td); x[3] = GGMLQuantizationType.F16; x[2] = x[2][:len(x[2]) // 2]
card("bad_type", [x])
x = sl(exps[0], 1, dm, td); x[4] = [x[4][0] // 2, x[4][1] * 2, 1]
card("bad_shape", [x])
p = card("bad_tamper", [sl(exps[0], 1, dm, td)])
with open(p, "r+b") as f:   # one byte of the slice data (the end of the file) changes after the manifest was written
    f.seek(-3, 2)
    b = f.read(1)
    f.seek(-3, 2)
    f.write(bytes([b[0] ^ 0x5A]))
x = sl(exps[0], 1, dm, td); x[0] = "blk.1.ffn_up_exps.weight.missing"
card("bad_notensor", [x])
x = sl(exps[0], 1, dm, td); x[0] = exps[0].replace("ffn_", "nextn.ffn_")
card("bad_mtp", [x])
card("bad_nomanifest", [sl(exps[0], 1, dm, td)], man_patch=lambda m: m.pop("content_sha256"))
print(json.dumps({"layout": layout[:12], "experts": n_exp, "tensors": exps, "out": out}))
