#!/usr/bin/env bash
# ci-build.sh <family> <version>   - the whole job body of one family in the release workflow: deps, build, tests, package, check.
# Runs on a clean Ubuntu 22.04 runner, or in the pinned CUDA / ROCm container (as root). Env: OUT_DIR (default $SRC/out/<family>),
# SHURA_TINY_MODEL (optional). Uses the network only for apt; the ROCm toolchain comes from the pinned ROCm image (/opt/rocm).
set -euo pipefail
fam="${1:?family}"; version="${2:?version}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$here/../.." && pwd)"; export SRC
OUT_DIR="${OUT_DIR:-$SRC/out/$fam}"
mkdir -p "$OUT_DIR"
SUDO=""; [ "$(id -u)" = 0 ] || SUDO="sudo"

export DEBIAN_FRONTEND=noninteractive
$SUDO apt-get update -qq
common=(build-essential cmake ninja-build git python3 python3-venv ca-certificates curl python3-tomli)
pkgs=("${common[@]}")
[ "$fam" = vulkan ] && pkgs+=(libvulkan-dev spirv-headers)
[ "$fam" = hip ] && pkgs+=(rocblas-dev hipblas-dev)      # the same packages as the fork's shura.yml hip job
$SUDO apt-get install -y --no-install-recommends "${pkgs[@]}"
dpkg-query -W -f='${Package}=${Version}\n' "${pkgs[@]}" > "$OUT_DIR/apt-versions.txt" || true

extra_libs=()
if [ "$fam" = hip ]; then
  # the ROCm toolchain is the one in the pinned image (rocm/dev-ubuntu-22.04:7.2.1); no wheels are downloaded
  [ -x /opt/rocm/bin/hipconfig ] || { echo "no /opt/rocm in this image: the hip job must run in the pinned ROCm container" >&2; exit 1; }
  ROCM_PATH=/opt/rocm; export ROCM_PATH HIP_PATH="$ROCM_PATH" CMAKE_PREFIX_PATH="$ROCM_PATH"
  PATH="$ROCM_PATH/bin:$PATH"; export PATH
  export LD_LIBRARY_PATH="$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}"
  # check the pin before the build: package installs write "7.2.1" or "7.2.1-<build>"
  image_version="$(cat "$ROCM_PATH/.info/version")"
  case "$image_version" in 7.2.1|7.2.1-*) ;; *) echo "ROCm in the image is '$image_version', the pin is 7.2.1" >&2; exit 1 ;; esac
fi
if [ "$fam" = cuda ]; then
  major="$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
  for l in libcudart libcublas libcublasLt; do extra_libs+=(--extra-lib "/usr/local/cuda/lib64/$l.so.$major"); done   # the link name: package.py ships the real file under it
fi

if [ "$fam" = vulkan ]; then
  # glslc is not an apt package on Ubuntu 22.04: take it from the LunarG Vulkan SDK archive pinned by size and SHA-256 (vulkan-sdk.env)
  # shellcheck disable=SC1091
  . "$here/vulkan-sdk.env"
  curl -fsSL --proto '=https' --tlsv1.2 -A "shura-release-ci/1.0" -o "$SRC/vulkan-sdk.tar.xz" "$VULKAN_SDK_URL"
  [ "$(stat -c %s "$SRC/vulkan-sdk.tar.xz")" = "$VULKAN_SDK_BYTES" ] || { echo "Vulkan SDK archive size does not match the pin" >&2; exit 1; }
  [ "$(sha256sum "$SRC/vulkan-sdk.tar.xz" | cut -d' ' -f1)" = "$VULKAN_SDK_SHA256" ] || { echo "Vulkan SDK archive checksum does not match the pin" >&2; exit 1; }
  mkdir -p "$SRC/vulkan-sdk"
  tar -xJf "$SRC/vulkan-sdk.tar.xz" -C "$SRC/vulkan-sdk" --strip-components=3 "${VULKAN_SDK_VERSION}/x86_64/bin/glslc"
  rm -f "$SRC/vulkan-sdk.tar.xz"
  PATH="$SRC/vulkan-sdk:$PATH"; export PATH
  glslc --version >/dev/null || { echo "glslc from the Vulkan SDK does not run" >&2; exit 1; }
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

if [ "$fam" = vulkan ]; then
  # the pinned Vulkan SDK archive glslc came from (vulkan-sdk.env), recorded next to the apt packages
  python3 -B - "$OUT_DIR/built_with.json" "$VULKAN_SDK_VERSION" "$VULKAN_SDK_URL" "$VULKAN_SDK_SHA256" "$VULKAN_SDK_BYTES" <<'PY'
import json, sys
path, version, url, sha256, size = sys.argv[1:6]
bw = json.load(open(path))
bw["vulkan_sdk"] = {"version": version, "url": url, "sha256": sha256, "bytes": int(size)}
json.dump(bw, open(path, "w"), indent=2)
PY
fi

if [ "$fam" = hip ]; then
  # the ROCm runtime major is read from the build, not assumed: the soname libggml-hip.so needs
  soname="$(readelf -d "$SRC/build-hip/bin/libggml-hip.so" | sed -n 's/.*NEEDED.*\[\(libamdhip64\.so\.[0-9][0-9]*\)\].*/\1/p' | head -n1)"
  [ -n "$soname" ] || { echo "libggml-hip.so has no NEEDED libamdhip64.so.N" >&2; exit 1; }
  python3 -B - "$OUT_DIR/built_with.json" "$soname" "$image_version" <<'PY'
import json, sys
path, soname, image_version = sys.argv[1], sys.argv[2], sys.argv[3]
bw = json.load(open(path))
rt = bw.get("runtime") if isinstance(bw.get("runtime"), dict) else {}
rt.update({"rocm_soname": soname, "rocm_major": int(soname.rsplit(".", 1)[1]), "rocm_image_version": image_version})
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

# the CPU job downloads the pinned tiny model (smoke-model.env) and checks it before the self-test in check.sh
if [ "$fam" = cpu ]; then
  # shellcheck disable=SC1091
  . "$here/smoke-model.env"
  mdir="$SRC/smoke-model"; mkdir -p "$mdir"
  curl -fsSL --proto '=https' --tlsv1.2 -A "shura-release-ci/1.0" -o "$mdir/$SMOKE_MODEL_NAME" "$SMOKE_MODEL_URL"
  echo "$SMOKE_MODEL_SHA256  $mdir/$SMOKE_MODEL_NAME" | sha256sum -c --status || { echo "smoke model checksum does not match the pin" >&2; exit 1; }
  export SHURA_TINY_MODEL="$mdir/$SMOKE_MODEL_NAME"
fi

archive="$OUT_DIR/shura-cpp-$version-$fam-linux-x86_64.tar.gz"
"$here/check.sh" "$fam" "$archive" "$OUT_DIR/check-report.txt" || { cat "$OUT_DIR/check-report.txt"; exit 1; }
cat "$OUT_DIR/check-report.txt"
