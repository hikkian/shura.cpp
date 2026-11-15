#!/usr/bin/env python3
"""package.py: pack a built shura.cpp into the release archive and write its manifest fragment.

  package.py --family F --version V --arch A --bin-dir build/bin --src REPO --built-with built_with.json --out-dir DIR
             [--extra-lib PATH ...] [--url-base URL] [--cpu-isa a,b]

Archive (contract name shura-cpp-<version>-<family>-linux-<arch>.tar.gz), one top directory shura-cpp-<version>-<family>/:
  llama-server and all lib*.so* side by side (rpath $ORIGIN needs one directory)
  LICENSE, NOTICE (ours; it carries the upstream credits), LICENSE-upstream (if the tree has one), docs/SHURA.md, licenses/*, BUILD-INFO.json
Also writes <archive>.sha256 ("<hash>  <name>") and <archive>.manifest.json (one manifest entry, merged later by manifest.py merge).
No network. Refuses an archive outside the size window."""
import argparse
import glob
import io
import json
import hashlib
import os
import re
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import manifest as mf  # noqa: E402
from families import FAMILIES, MAX_ARCHIVE, MIN_ARCHIVE  # noqa: E402


CUDA_LIB = re.compile(r"^(libcudart\.so|libcublas\.so|libcublasLt\.so)(\.[0-9]+)*$")
CUDA_NEEDED = ("libcudart", "libcublas", "libcublasLt")
FOREIGN_PREFIXES = ("libcuda", "libcudart", "libcublas", "libcublasLt", "libamdhip64", "libhipblas", "librocblas", "librocsolver",
                    "libhsa-runtime", "libamd_comgr", "libvulkan.so", "libOpenCL", "libssl", "libcrypto")
CUDA_EULA = "NVIDIA-CUDA-EULA-12.8.txt"
CUDA_EULA_SHA256 = "00df22cae0f47779ce9734d360bee6fb3fbb616b4fe4a3aa31098bd8a61ac328"   # the licence text we ship, pinned
OUR_LIB = re.compile(r"^lib(ggml|llama|mtmd)[-A-Za-z0-9_]*\.so(\.[0-9]+)*$")   # our backend and server libraries only
CUDA_NOTICE = ("This product includes NVIDIA CUDA Toolkit 12.8 runtime libraries (libcudart, libcublas, libcublasLt), redistributed "
               "unmodified under the NVIDIA Software License Agreement and CUDA Supplement (version 12.8), "
               "https://docs.nvidia.com/cuda/eula/index.html . The license text is in licenses/NVIDIA-CUDA-EULA-12.8.txt.")


def sha_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def add_bytes(tar, data, arcname, mode=0o644):
    ti = tarfile.TarInfo(arcname)
    ti.size, ti.mode, ti.mtime = len(data), mode, int(os.environ.get("SOURCE_DATE_EPOCH", 0))
    tar.addfile(ti, io.BytesIO(data))


def add_file(tar, src, arcname, mode=None):
    ti = tar.gettarinfo(src, arcname)
    ti.uid = ti.gid = 0
    ti.uname = ti.gname = ""
    ti.mtime = int(os.environ.get("SOURCE_DATE_EPOCH", ti.mtime))
    if mode is not None:
        ti.mode = mode
    with open(src, "rb") as f:
        tar.addfile(ti, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=sorted(FAMILIES))
    ap.add_argument("--version", required=True)
    ap.add_argument("--arch", default="x86_64")
    ap.add_argument("--bin-dir", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--built-with", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--extra-lib", action="append", default=[], help="runtime library to ship next to llama-server (CUDA runtime)")
    ap.add_argument("--url-base", default="https://github.com/hikkian/shura.cpp/releases/download/shura-v{version}")
    ap.add_argument("--cpu-isa", default="")
    a = ap.parse_args()

    name = mf.archive_name(a.version, a.family, a.arch)
    if not mf.NAME.match(name):
        sys.exit(f"version {a.version!r} does not give a contract name")
    server = os.path.join(a.bin_dir, "llama-server")
    if not os.path.isfile(server):
        sys.exit(f"no llama-server in {a.bin_dir}")
    bw = json.load(open(a.built_with))
    top = f"shura-cpp-{a.version}-{a.family}"
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, name)

    here = os.path.dirname(os.path.abspath(__file__))
    cuda_eula = os.path.join(here, "licenses-cuda", CUDA_EULA)
    if a.family != "cuda" and a.extra_lib:
        sys.exit(f"--extra-lib is allowed only for the cuda family, not {a.family}")
    if a.family == "cuda":
        for p in a.extra_lib:
            if not CUDA_LIB.match(os.path.basename(p)):
                sys.exit(f"{os.path.basename(p)}: not in Attachment A of the NVIDIA licence (libcudart, libcublas, libcublasLt only)")
        names = [os.path.basename(p) for p in a.extra_lib]
        for need in CUDA_NEEDED:
            if not any(n.startswith(need + ".so") for n in names):
                sys.exit(f"cuda archive needs --extra-lib {need}.so*; given: {names}")
        if not os.path.isfile(cuda_eula):
            sys.exit(f"missing {cuda_eula}")
        if sha_of(cuda_eula) != CUDA_EULA_SHA256:
            sys.exit("the NVIDIA licence text differs from the pinned copy")
    else:
        foreign = sorted(n for n in os.listdir(a.bin_dir) if n.startswith(FOREIGN_PREFIXES) and not n.startswith("libggml-"))
        if foreign:
            sys.exit(f"foreign GPU or TLS library in {a.bin_dir}: {foreign}")
    if a.family == "hip":
        want = FAMILIES["hip"]["rocm_soname_major"]
        got = (bw.get("runtime") or {}).get("rocm_major")
        if got != want:
            sys.exit(f"built_with runtime.rocm_major is {got!r}, the table says {want}")

    libs = sorted(glob.glob(os.path.join(a.bin_dir, "lib*.so*")))
    alien = [os.path.basename(p) for p in libs if not OUR_LIB.match(os.path.basename(p))]
    if alien:
        sys.exit(f"not our library in {a.bin_dir} (would ship in the archive): {alien}")
    if not libs:
        sys.exit(f"no lib*.so in {a.bin_dir}: a BACKEND_DL build needs its modules")
    docs = [("LICENSE", "LICENSE"), ("NOTICE", "NOTICE"), ("docs/SHURA.md", "docs/SHURA.md")]
    for extra in ("LICENSE-upstream", "AUTHORS"):
        if os.path.isfile(os.path.join(a.src, extra)):
            docs.append((extra, extra))
    for s, _ in docs[:3]:
        if not os.path.isfile(os.path.join(a.src, s)):
            sys.exit(f"missing {s} in the source tree")
    lic = sorted(p for p in glob.glob(os.path.join(a.src, "licenses", "*")) if os.path.isfile(p))
    if not lic:
        sys.exit("licenses/ is empty: vendored licence texts must ship")

    info = dict(bw)
    info.update({"version": a.version, "family": a.family, "arch": a.arch, "parts": FAMILIES[a.family]["parts"],
                 "gpu_targets": FAMILIES[a.family]["gpu_targets"]})
    info_bytes = json.dumps(info, indent=2).encode()

    with tarfile.open(out, "w:gz", compresslevel=9) as tar:
        add_file(tar, server, f"{top}/llama-server", 0o755)
        for p in libs + a.extra_lib:
            if os.path.islink(p):      # ship the real file under the link's name (a tar symlink chain is brittle)
                add_file(tar, os.path.realpath(p), f"{top}/{os.path.basename(p)}", 0o755)
            else:
                add_file(tar, p, f"{top}/{os.path.basename(p)}", 0o755)
        for s, d in docs:
            if a.family == "cuda" and d == "NOTICE":
                notice = open(os.path.join(a.src, s), "rb").read()
                add_bytes(tar, notice + b"\n" + CUDA_NOTICE.encode() + b"\n", f"{top}/{d}")
            else:
                add_file(tar, os.path.join(a.src, s), f"{top}/{d}", 0o644)
        for p in lic:
            add_file(tar, p, f"{top}/licenses/{os.path.basename(p)}", 0o644)
        if a.family == "cuda":
            add_file(tar, cuda_eula, f"{top}/licenses/{CUDA_EULA}", 0o644)
        ti = tarfile.TarInfo(f"{top}/BUILD-INFO.json")
        ti.size, ti.mode, ti.mtime = len(info_bytes), 0o644, int(os.environ.get("SOURCE_DATE_EPOCH", 0))
        tar.addfile(ti, io.BytesIO(info_bytes))

    if a.family == "cuda":   # the shipped bytes of every NVIDIA library must equal the source file
        with tarfile.open(out, "r:gz") as tar:
            for p in a.extra_lib:
                m = tar.extractfile(f"{top}/{os.path.basename(p)}")
                if m is None or hashlib.sha256(m.read()).hexdigest() != sha_of(os.path.realpath(p)):
                    sys.exit(f"{os.path.basename(p)}: archive copy differs from the source file")

    size = os.path.getsize(out)
    if not MIN_ARCHIVE <= size <= MAX_ARCHIVE:
        sys.exit(f"{name}: size {size} outside [{MIN_ARCHIVE}, {MAX_ARCHIVE}]")
    digest = mf.sha256_file(out)
    with open(out + ".sha256", "w") as f:
        f.write(f"{digest}  {name}\n")
    entry = mf.make_entry(family=a.family, version=a.version, arch=a.arch, archive=out, built_with=bw, commit=bw["commit"],
                          url_base=a.url_base.format(version=a.version), glibc_min=bw.get("build_glibc") or "",
                          cpu_isa=[x for x in a.cpu_isa.split(",") if x])
    with open(out + ".manifest.json", "w") as f:
        json.dump(entry, f, indent=2)
    print(f"{name}  {size} bytes  sha256 {digest}")


if __name__ == "__main__":
    main()
