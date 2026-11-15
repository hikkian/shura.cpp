#!/usr/bin/env python3
"""release-manifest.json for shura.cpp releases: build one entry, merge entries, check a manifest against the contract.

  manifest.py entry  --family F --version V --arch A --archive PATH --built-with JSON --commit SHA --url-base URL [--glibc-min X] [--cpu-isa a,b] --out FRAG
  manifest.py merge  FRAG... --out release-manifest.json
  manifest.py check  release-manifest.json [--dir DIR_WITH_ARCHIVES]    (exit 1 and a reason per problem)

Contract: shura-integration/docs/RELEASE-CONTRACT.md (schema 1). No network. `check` with --dir also verifies sha256 and size of every
archive against the files and that the archive is inside the size window; it does not need the installer code."""
import argparse
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from families import FAMILIES, MAX_ARCHIVE, MIN_ARCHIVE  # noqa: E402

NAME = re.compile(r"^shura-cpp-(?P<version>\d[0-9A-Za-z.+~]*)-(?P<family>hip|cuda|vulkan|sycl|cpu)-(?P<os>linux)-(?P<arch>x86_64|arm64)\.tar\.gz$")
GLIBC_CEILING = (2, 35)
REPO = "hikkian/shura.cpp"
RELEASE_BASE = "https://github.com/hikkian/shura.cpp/releases/download/shura-v{version}"
SHA = re.compile(r"^[0-9a-f]{64}$")
MANIFEST_FAMILIES = {"HIP": "hip", "CUDA": "cuda", "Vulkan": "vulkan", "SYCL": "sycl", "CPU": "cpu"}
RUNTIME_BASE = {"HIP": "rocm", "CUDA": "nvidia_driver", "Vulkan": "vulkan", "SYCL": "oneapi", "CPU": None}
REQUIRED = ("id", "engine", "version", "family", "gpu_targets", "runtime", "glibc_min", "parts", "built_with", "assets")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def archive_name(version, family, arch):
    return f"shura-cpp-{version}-{family}-linux-{arch}.tar.gz"


def make_entry(*, family, version, arch, archive, built_with, commit, url_base, glibc_min, cpu_isa=None):
    f = FAMILIES[family]
    name = archive_name(version, family, arch)
    if os.path.basename(archive) != name:
        raise SystemExit(f"archive file is {os.path.basename(archive)}, the contract name is {name}")
    runtime = dict(f["runtime"])
    if f["manifest_family"] == "HIP" and built_with.get("runtime", {}).get("rocm"):
        pass  # declared range stays the one in families.py; the version used is in built_with.runtime
    entry = {
        "id": f"shura-{version}-{family}-linux-{arch}", "engine": "shura.cpp", "version": version,
        "family": f["manifest_family"], "arch": arch, "gpu_targets": list(f["gpu_targets"]), "runtime": runtime,
        "glibc_min": glibc_min, "parts": list(f["parts"]),
        "built_with": {"commit": commit, "cmake_flags": built_with["cmake_flags"], "compiler": built_with.get("compiler", ""),
                       "cmake": built_with.get("cmake", ""), "runtime": built_with.get("runtime", {})},
        "assets": [{"name": name, "url": f"{url_base.rstrip('/')}/{name}", "sha256": sha256_file(archive),
                    "size": os.path.getsize(archive)}],
    }
    if cpu_isa:
        entry["cpu_isa"] = list(cpu_isa)
    return entry


def vkey(text):
    m = re.match(r"^\s*(\d+(?:\.\d+)*)", str(text) if text is not None else "")
    return tuple(int(x) for x in m[1].split(".")) if m else None


def check(manifest, directory=None):
    """List of problems (strings); empty list = the manifest follows the contract."""
    bad = []
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        return ["schema must be 1"]
    rels = manifest.get("releases")
    if not isinstance(rels, list) or not rels:
        return ["'releases' must be a non-empty list"]
    seen = set()
    for i, r in enumerate(rels):
        w = f"release #{i + 1}" + (f" ({r.get('id')})" if isinstance(r, dict) else "")
        if not isinstance(r, dict):
            bad.append(f"{w}: not an object"); continue
        miss = [k for k in REQUIRED if k not in r]
        if miss:
            bad.append(f"{w}: missing fields {miss}"); continue
        if r["engine"] != "shura.cpp":
            bad.append(f"{w}: engine must be shura.cpp")
        fam = r["family"]
        key = MANIFEST_FAMILIES.get(fam)
        if key is None:
            bad.append(f"{w}: unknown family {fam!r}"); continue
        arch = r.get("arch", "x86_64")
        if arch not in ("x86_64", "arm64"):
            bad.append(f"{w}: bad arch {arch!r}")
        if r["id"] in seen:
            bad.append(f"{w}: duplicate id")
        seen.add(r["id"])
        if r["id"] != f"shura-{r['version']}-{key}-linux-{arch}":
            bad.append(f"{w}: id should be shura-{r['version']}-{key}-linux-{arch}")
        tg = r["gpu_targets"]
        if not isinstance(tg, list) or not all(isinstance(t, str) for t in tg):
            bad.append(f"{w}: gpu_targets must be a list of strings")
        elif fam in ("HIP", "CUDA") and not tg:
            bad.append(f"{w}: gpu_targets is empty for {fam}")
        elif fam not in ("HIP", "CUDA") and tg:
            bad.append(f"{w}: gpu_targets must be [] for {fam}")
        elif fam == "HIP" and not all(re.fullmatch(r"gfx[0-9a-f]+", t) for t in tg):
            bad.append(f"{w}: HIP targets must be gfx codes: {tg}")
        elif fam == "CUDA" and not all(re.fullmatch(r"\d+\.\d+", t) for t in tg):
            bad.append(f"{w}: CUDA targets must be compute capabilities like 8.9: {tg}")
        rt = r["runtime"]
        base = RUNTIME_BASE[fam]
        if not isinstance(rt, dict):
            bad.append(f"{w}: runtime must be an object")
        else:
            for k, v in rt.items():
                if base is None or k not in (base + "_min", base + "_max") or vkey(v) is None:
                    bad.append(f"{w}: bad runtime entry {k}={v!r} for {fam}")
            if fam in ("HIP", "CUDA") and not rt:
                bad.append(f"{w}: {fam} needs at least one runtime min or max")
        if vkey(r["glibc_min"]) is None:
            bad.append(f"{w}: bad glibc_min {r['glibc_min']!r}")
        elif vkey(r["glibc_min"]) > GLIBC_CEILING:
            bad.append(f"{w}: glibc_min {r['glibc_min']} above the ceiling {'.'.join(map(str, GLIBC_CEILING))}")
        if not (isinstance(r["parts"], list) and r["parts"] and all(isinstance(p, str) and p for p in r["parts"])):
            bad.append(f"{w}: parts must be a non-empty list of names")
        else:
            expect = FAMILIES[key]["parts"]
            if sorted(r["parts"]) != sorted(expect):
                bad.append(f"{w}: parts {r['parts']} differ from the table for {fam}: {expect}")
        bw = r["built_with"]
        if not (isinstance(bw, dict) and isinstance(bw.get("commit"), str) and re.fullmatch(r"[0-9a-f]{7,40}", bw["commit"])):
            bad.append(f"{w}: built_with.commit must be a git sha")
        elif not (isinstance(bw.get("cmake_flags"), list) and bw["cmake_flags"]):
            bad.append(f"{w}: built_with.cmake_flags must be a non-empty list")
        else:
            fl = bw["cmake_flags"]
            if "-DGGML_NATIVE=OFF" not in fl:
                bad.append(f"{w}: built_with.cmake_flags lacks -DGGML_NATIVE=OFF (a native build is not portable)")
            if "-DLLAMA_OPENSSL=OFF" not in fl:
                bad.append(f"{w}: built_with.cmake_flags lacks -DLLAMA_OPENSSL=OFF (no TLS library in the build)")
        if "cpu_isa" in r and not (isinstance(r["cpu_isa"], list) and all(isinstance(t, str) for t in r["cpu_isa"])):
            bad.append(f"{w}: cpu_isa must be a list of strings")
        assets = r["assets"]
        if not (isinstance(assets, list) and assets):
            bad.append(f"{w}: no assets"); continue
        for a in assets:
            n = a.get("name") if isinstance(a, dict) else None
            m = NAME.match(n) if isinstance(n, str) else None
            if not m:
                bad.append(f"{w}: asset name {n!r} does not follow the contract"); continue
            if (m["version"], m["family"], m["arch"]) != (r["version"], key, arch):
                bad.append(f"{w}: asset name {n} disagrees with version/family/arch of the release")
            want_url = RELEASE_BASE.format(version=r["version"]) + "/" + n
            if a.get("url") != want_url:
                bad.append(f"{w}: {n}: url must be exactly {want_url}, got {a.get('url')!r}")
            if not (isinstance(a.get("sha256"), str) and SHA.match(a["sha256"])):
                bad.append(f"{w}: {n}: sha256 must be 64 lowercase hex")
            if "size" in a and not (isinstance(a["size"], int) and a["size"] > 0):
                bad.append(f"{w}: {n}: bad size")
            elif isinstance(a.get("size"), int) and not MIN_ARCHIVE <= a["size"] <= MAX_ARCHIVE:
                bad.append(f"{w}: {n}: size {a['size']} outside [{MIN_ARCHIVE}, {MAX_ARCHIVE}]")
            if directory is not None:
                p = os.path.join(directory, n)
                if not os.path.isfile(p):
                    bad.append(f"{w}: {n}: file not found in {directory}")
                else:
                    if os.path.getsize(p) != a.get("size"):
                        bad.append(f"{w}: {n}: size {a.get('size')} != file {os.path.getsize(p)}")
                    if sha256_file(p) != a.get("sha256"):
                        bad.append(f"{w}: {n}: sha256 does not match the file")
                    sf = p + ".sha256"
                    if os.path.isfile(sf) and open(sf).read().split()[:1] != [a.get("sha256")]:
                        bad.append(f"{w}: {n}: {n}.sha256 differs from the manifest")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("entry")
    for k in ("family", "version", "arch", "archive", "built-with", "commit", "url-base", "out"):
        e.add_argument("--" + k, required=True)
    e.add_argument("--glibc-min", default=None)
    e.add_argument("--cpu-isa", default="")
    mg = sub.add_parser("merge")
    mg.add_argument("frags", nargs="+")
    mg.add_argument("--out", required=True)
    c = sub.add_parser("check")
    c.add_argument("manifest")
    c.add_argument("--dir")
    a = ap.parse_args(argv)
    if a.cmd == "entry":
        bw = json.load(open(a.built_with))
        glibc = a.glibc_min or bw.get("build_glibc") or ""
        ent = make_entry(family=a.family, version=a.version, arch=a.arch, archive=a.archive, built_with=bw, commit=a.commit,
                         url_base=a.url_base, glibc_min=glibc, cpu_isa=[x for x in a.cpu_isa.split(",") if x])
        json.dump(ent, open(a.out, "w"), indent=2)
        return 0
    if a.cmd == "merge":
        rels = [json.load(open(p)) for p in a.frags]
        rels.sort(key=lambda r: r["id"])
        json.dump({"schema": 1, "releases": rels}, open(a.out, "w"), indent=2)
        a.manifest, a.dir = a.out, None
    try:
        manifest = json.load(open(a.manifest))
    except (OSError, ValueError) as ex:
        print(f"cannot read manifest: {ex}")
        return 1
    problems = check(manifest, a.dir)
    for p in problems:
        print("PROBLEM", p)
    print(f"{a.manifest}: {len(manifest.get('releases', []))} release(s), {len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
