"""Shared helpers for the release-rule tests: a fake source tree, fake bin dir, a package.py runner. No network, no build."""
import json
import os
import sys
import tarfile
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.path.dirname(HERE)
sys.path.insert(0, REL)
import package  # noqa: E402

package.MIN_ARCHIVE = 1   # the real window is 10 MiB; the fakes are tiny
NOTICE_TEXT = "shura.cpp NOTICE\nupstream credits\n"


def make_tree(root):
    src, binp, out = (os.path.join(root, d) for d in ("src", "bin", "out"))
    os.makedirs(os.path.join(src, "docs")); os.makedirs(os.path.join(src, "licenses")); os.makedirs(binp)
    for name, text in (("LICENSE", "lic\n"), ("NOTICE", NOTICE_TEXT), ("docs/SHURA.md", "doc\n"), ("licenses/x.txt", "x\n")):
        open(os.path.join(src, name), "w").write(text)
    open(os.path.join(binp, "llama-server"), "w").write("fake server\n")
    open(os.path.join(binp, "libggml-base.so"), "w").write("base\n")
    bw = os.path.join(root, "built_with.json")
    json.dump({"commit": "abcdef1", "cmake_flags": ["-DGGML_NATIVE=OFF", "-DLLAMA_OPENSSL=OFF"], "build_glibc": "2.35"}, open(bw, "w"))
    return src, binp, out, bw


def fake_libs(root, names):
    d = os.path.join(root, "cudalibs"); os.makedirs(d, exist_ok=True)
    paths = []
    for n in names:
        p = os.path.join(d, n); open(p, "w").write("fake " + n + "\n"); paths.append(p)
    return paths


def run_package(family, src, binp, out, bw, extra=()):
    argv = ["package.py", "--family", family, "--version", "1.0.0", "--bin-dir", binp, "--src", src, "--built-with", bw, "--out-dir", out]
    for p in extra:
        argv += ["--extra-lib", p]
    old = sys.argv
    sys.argv = argv
    try:
        package.main()
    finally:
        sys.argv = old
    return os.path.join(out, f"shura-cpp-1.0.0-{family}-linux-x86_64.tar.gz")
