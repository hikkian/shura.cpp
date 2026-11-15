#!/usr/bin/env bash
# ci-build.sh <family> <version>   - the whole job body of one family in the release workflow: deps, build, tests, package, check.
# Runs on a clean Ubuntu 22.04 runner or in the pinned CUDA container (as root). Env: OUT_DIR (default /dev/shm/RELBIN1/out/<family> if
# /dev/shm exists, else $PWD/out/<family>), SHURA_TINY_MODEL (optional). Uses the network only for apt and, for HIP, pip with --require-hashes.
set -euo pipefail
fam="${1:?family}"; version="${2:?version}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$here/../.." && pwd)"; export SRC
OUT_DIR="${OUT_DIR:-$SRC/out/$fam}"
mkdir -p "$OUT_DIR"
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"

# a value left as PIN_ME means the pin was never filled in: refuse instead of building from an unpinned source
if grep -rn 'PIN_ME' "$here/rocm-requirements.txt" >/dev/null 2>&1 && [ "$fam" = hip ]; then
  echo "rocm-requirements.txt still has PIN_ME: fill the wheel hashes first" >&2; exit 1
fi

export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
common=(build-essential cmake ninja-build git python3 python3-venv ca-certificates curl)
pkgs=("${common[@]}")
[ "$fam" = vulkan ] && pkgs+=(libvulkan-dev glslc spirv-headers)
$SUDO apt-get install -y --no-install-recommends "${pkgs[@]}"
dpkg-query -W -f='${Package}=${Version}\n' "${pkgs[@]}" > "$OUT_DIR/apt-versions.txt" || true

extra_libs=()
if [ "$fam" = hip ]; then
  python3 -m venv "$SRC/.venv-rocm"
  # shellcheck disable=SC1091
  . "$SRC/.venv-rocm/bin/activate"
  # every wheel by hash; the index is AMD's, hashes make it safe
  python -m pip install --require-hashes -r "$here/rocm-requirements.txt" \
    --index-url "${ROCM_INDEX_URL:-https://stable.repo.amd.com/rocm/whl-next/}"
  ROCM_PATH="$(rocm-sdk path --root)"; export ROCM_PATH HIP_PATH="$ROCM_PATH"
  CMAKE_PREFIX_PATH="$(rocm-sdk path --cmake)"; export CMAKE_PREFIX_PATH
  PATH="$(rocm-sdk path --bin):$PATH"; export PATH
  export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
fi
if [ "$fam" = cuda ]; then
  major="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
  for l in libcudart libcublas libcublasLt; do extra_libs+=(--extra-lib "/usr/local/cuda/lib64/$l.so.$major"); done   # the link name: package.py ships the real file under it
fi

BUILD_DIR="$SRC/build-$fam" OUT="$OUT_DIR" "$here/build.sh" "$fam"

if [ "$fam" = cpu ]; then
  # our GPU-free tests are built once, in the CPU job only (the release build itself has LLAMA_BUILD_TESTS=OFF)
  cmake -S "$SRC" -B "$SRC/build-tests" -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_CURL=OFF
  cmake --build "$SRC/build-tests" -j "$(nproc)" --target test-vram-governor test-reasoning-effort
  export TESTS_BIN_DIR="$SRC/build-tests/bin"
fi

# built_with.json: every apt package this script installed, name -> version
python3 -B - "$OUT_DIR/built_with.json" "${pkgs[@]}" <<'PY'
import json, subprocess, sys
path, pkgs = sys.argv[1], sys.argv[2:]
bw = json.load(open(path))
apt = {}
for p in pkgs:
    r = subprocess.run(["dpkg-query", "-W", "-f=${Version}", p], capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout:
        sys.exit(f"cannot read the installed version of {p}")
    apt[p] = r.stdout
bw["apt_packages"] = apt
json.dump(bw, open(path, "w"), indent=2)
PY

if [ "$fam" = hip ]; then
  # the ROCm runtime major is read from the build, not assumed: the soname libggml-hip.so needs
  soname="$(readelf -d "$SRC/build-hip/bin/libggml-hip.so" | sed -n 's/.*NEEDED.*\[\(libamdhip64\.so\.[0-9][0-9]*\)\].*/\1/p' | head -n1)"
  [ -n "$soname" ] || { echo "libggml-hip.so has no NEEDED libamdhip64.so.N" >&2; exit 1; }
  python3 -B - "$OUT_DIR/built_with.json" "$soname" <<'PY'
import json, sys
path, soname = sys.argv[1], sys.argv[2]
bw = json.load(open(path))
rt = bw.get("runtime") if isinstance(bw.get("runtime"), dict) else {}
rt.update({"rocm_soname": soname, "rocm_major": int(soname.rsplit(".", 1)[1])})
bw["runtime"] = rt
json.dump(bw, open(path, "w"), indent=2)
PY
fi

cpu_isa=""
if [ "$fam" = cpu ]; then
  cpu_isa="$(find "$SRC/build-cpu/bin" -name 'libggml-cpu-*.so' -printf '%f\n' | sed 's/^libggml-cpu-//; s/\.so$//' | sort | paste -sd,)"
fi

python3 -B "$here/package.py" --family "$fam" --version "$version" --bin-dir "$SRC/build-$fam/bin" --src "$SRC" \
  --built-with "$OUT_DIR/built_with.json" --out-dir "$OUT_DIR" --cpu-isa "$cpu_isa" "${extra_libs[@]}"

archive="$OUT_DIR/shura-cpp-$version-$fam-linux-x86_64.tar.gz"
"$here/check.sh" "$fam" "$archive" "$OUT_DIR/check-report.txt" || { cat "$OUT_DIR/check-report.txt"; exit 1; }
cat "$OUT_DIR/check-report.txt"
