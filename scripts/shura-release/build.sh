#!/usr/bin/env bash
# build.sh <hip|cuda|vulkan|cpu>  - configure and build llama-server (+ backend modules) for one release family.
# Env: SRC (repo root, default: two levels up), BUILD_DIR (default $SRC/build-<family>), OUT (where built_with.json goes, default BUILD_DIR),
#      JOBS (default nproc). Nothing is downloaded: the UI is not built (LLAMA_BUILD_UI=OFF, no prebuilt fetch) and no submodules are used.
# Writes $OUT/built_with.json: commit, cmake flags, compiler, cmake version, runtime versions (what was really used).
set -euo pipefail

fam="${1:-}"
case "$fam" in hip|cuda|vulkan|cpu) ;; *) echo "usage: $0 hip|cuda|vulkan|cpu" >&2; exit 2 ;; esac

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-$(cd "$here/../.." && pwd)}"
BUILD_DIR="${BUILD_DIR:-$SRC/build-$fam}"
OUT="${OUT:-$BUILD_DIR}"
JOBS="${JOBS:-$(nproc)}"

# flags every family shares (upstream release.yml: BACKEND_DL, NATIVE=OFF, CPU_ALL_VARIANTS, $ORIGIN rpath)
# shellcheck disable=SC2016
flags=(
  -DCMAKE_BUILD_TYPE=Release
  -DGGML_NATIVE=OFF
  -DGGML_BACKEND_DL=ON
  -DGGML_CPU_ALL_VARIANTS=ON
    -DCMAKE_INSTALL_RPATH='$ORIGIN'
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON
  -DLLAMA_BUILD_SERVER=ON
  -DLLAMA_BUILD_TOOLS=ON
  -DLLAMA_BUILD_TESTS=OFF
  -DLLAMA_BUILD_EXAMPLES=OFF
  -DLLAMA_BUILD_UI=OFF            # no npm, no network
  -DLLAMA_USE_PREBUILT_UI=OFF     # no Hugging Face fetch; the server then has an empty UI table (API only)
  -DLLAMA_CURL=OFF
  -DLLAMA_OPENSSL=OFF             # no libssl/libcrypto in the archive: the server speaks plain HTTP on a local address
)

targets="$(python3 -B "$here/families.py" targets "$fam")"
case "$fam" in
  hip)
    flags+=(-DGGML_HIP=ON -DHIP_PLATFORM=amd "-DAMDGPU_TARGETS=$targets" "-DGPU_TARGETS=$targets")
    if command -v hipconfig >/dev/null 2>&1; then flags+=("-DCMAKE_HIP_COMPILER=$(hipconfig -l)/clang"); fi
    ;;
  cuda)
    flags+=(-DGGML_CUDA=ON -DGGML_CUDA_NCCL=OFF "-DCMAKE_CUDA_ARCHITECTURES=$targets")
    ;;
  vulkan)
    flags+=(-DGGML_VULKAN=ON)
    ;;
  cpu)
    : # shared flags are enough; backends: CPU variants only
    ;;
esac

mkdir -p "$BUILD_DIR" "$OUT"
echo "== configure ($fam): ${flags[*]}"
cmake -S "$SRC" -B "$BUILD_DIR" "${flags[@]}"

# BACKEND_DL: the backends are loadable modules and llama-server does not depend on them, so ask for them by name.
mods="$(cmake --build "$BUILD_DIR" --target help | sed -n 's/^\.\.\. \(ggml-[a-z0-9_.-]*\)$/\1/p' | grep -v '^ggml-base$' || true)"
echo "== targets: llama-server ${mods//$'\n'/ }"
# shellcheck disable=SC2086
cmake --build "$BUILD_DIR" --config Release -j "$JOBS" --target llama-server $mods

test -x "$BUILD_DIR/bin/llama-server" || { echo "llama-server was not produced" >&2; exit 1; }
if [ "$fam" = hip ]   && ! ls "$BUILD_DIR"/bin/libggml-hip.so*  >/dev/null 2>&1; then echo "libggml-hip.so was not produced" >&2; exit 1; fi
if [ "$fam" = cuda ]  && ! ls "$BUILD_DIR"/bin/libggml-cuda.so* >/dev/null 2>&1; then echo "libggml-cuda.so was not produced" >&2; exit 1; fi
if [ "$fam" = vulkan ] && ! ls "$BUILD_DIR"/bin/libggml-vulkan.so* >/dev/null 2>&1; then echo "libggml-vulkan.so was not produced" >&2; exit 1; fi

# built_with: what was actually used, written by a tiny python so the JSON is always valid
cc_ver="$( (c++ --version 2>/dev/null || true) | head -n1)"
if [ "$fam" = hip ] && command -v hipconfig >/dev/null 2>&1; then cc_ver="$("$(hipconfig -l)/clang" --version | head -n1)"; fi
if [ "$fam" = cuda ] && command -v nvcc >/dev/null 2>&1; then cc_ver="$cc_ver; $(nvcc --version | tail -n1)"; fi
cm_ver="$(cmake --version | head -n1)"
glibc="$(ldd --version 2>&1 | head -n1 | grep -oE '[0-9]+\.[0-9]+$' || true)"
commit="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)"
rt_rocm=""; rt_cuda=""; rt_vk=""
[ "$fam" = hip ]    && rt_rocm="$(cat "${ROCM_PATH:-/opt/rocm}/.info/version" 2>/dev/null || echo "${ROCM_VERSION:-unknown}")"
[ "$fam" = cuda ]   && rt_cuda="$(nvcc --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p')"
[ "$fam" = vulkan ] && rt_vk="$( (glslc --version 2>/dev/null || true) | head -n1)"
printf '%s\0' "${flags[@]}" | OUT="$OUT" FAM="$fam" COMMIT="$commit" CC_VER="$cc_ver" CM_VER="$cm_ver" GLIBC="$glibc" \
  RT_ROCM="$rt_rocm" RT_CUDA="$rt_cuda" RT_VK="$rt_vk" python3 -B -c '
import json, os, sys
flags = [f for f in sys.stdin.buffer.read().decode().split("\0") if f]
rt = {k: v for k, v in (("rocm", os.environ["RT_ROCM"]), ("cuda", os.environ["RT_CUDA"]), ("glslc", os.environ["RT_VK"])) if v}
json.dump({"commit": os.environ["COMMIT"], "family": os.environ["FAM"], "cmake_flags": flags,
           "compiler": os.environ["CC_VER"], "cmake": os.environ["CM_VER"], "runtime": rt,
           "build_glibc": os.environ["GLIBC"]}, open(os.environ["OUT"] + "/built_with.json", "w"), indent=2)
' 
echo "== built_with: $OUT/built_with.json"
