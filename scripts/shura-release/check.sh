#!/usr/bin/env bash
# check.sh <family> <archive.tar.gz> [report-file]   - checks inside the release job, no GPU needed.
# Env: SRC (repo root, for touchpoints; default two levels up), SHURA_TINY_MODEL (path to a tiny .gguf for the CPU self-test; no download
#      is done here - without it the self-test is skipped and the report says so), TESTS_BIN_DIR (where test-vram-governor etc. are, if built),
#      WORK (scratch dir, default /dev/shm/RELBIN1/check), CHECK_PORT (default 39517).
# Exit 1 if any check FAILed; SKIP and NONE lines do not fail but are written to the report.
set -uo pipefail

fam="${1:-}"; archive="${2:-}"; report="${3:-/dev/stdout}"
case "$fam" in hip|cuda|vulkan|cpu) ;; *) echo "usage: $0 hip|cuda|vulkan|cpu archive [report]" >&2; exit 2 ;; esac
[ -f "$archive" ] || { echo "no archive $archive" >&2; exit 2; }

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SRC:-$(cd "$here/../.." && pwd)}"
WORK="${WORK:-/dev/shm/RELBIN1/check}"
PORT="${CHECK_PORT:-39517}"
MIN_BYTES=$((10 * 1024 * 1024))          # 10 MiB   (same window as families.py)
MAX_BYTES=$((2 * 1024 * 1024 * 1024))    # 2 GiB
fail=0
say() { echo "$*" >>"$report"; }
ok()  { say "OK    $*"; }
bad() { say "FAIL  $*"; fail=1; }

[ "$report" = /dev/stdout ] || : >"$report"
rm -rf "$WORK"; mkdir -p "$WORK"

# 1. archive name, size window, sha256 file
name="$(basename "$archive")"
if [[ "$name" =~ ^shura-cpp-[0-9][0-9A-Za-z.+~]*-${fam}-linux-(x86_64|arm64)\.tar\.gz$ ]]; then ok "name $name"; else bad "name $name does not follow the contract for $fam"; fi
size="$(stat -c %s "$archive")"
if [ "$size" -ge "$MIN_BYTES" ] && [ "$size" -le "$MAX_BYTES" ]; then ok "size $size bytes in [$MIN_BYTES, $MAX_BYTES]"; else bad "size $size outside [$MIN_BYTES, $MAX_BYTES]"; fi
if [ -f "$archive.sha256" ]; then
  want="$(cut -d' ' -f1 "$archive.sha256")"; have="$(sha256sum "$archive" | cut -d' ' -f1)"
  if [ "$want" = "$have" ]; then ok "sha256 $have"; else bad "sha256 file says $want, archive is $have"; fi
else bad "no $name.sha256 next to the archive"; fi

# 2. unpack, contents
if tar -xzf "$archive" -C "$WORK"; then ok "unpacked"; else bad "tar -xzf failed"; say "stopping"; exit 1; fi
dir="$(find "$WORK" -mindepth 1 -maxdepth 1 -type d | head -n1)"
for f in llama-server LICENSE NOTICE docs/SHURA.md BUILD-INFO.json; do
  if [ -e "$dir/$f" ]; then ok "has $f"; else bad "missing $f"; fi
done
if ls "$dir"/licenses/* >/dev/null 2>&1; then ok "has licenses/"; else bad "licenses/ empty"; fi
if ls "$dir"/libggml-base.so* >/dev/null 2>&1; then ok "has libggml-base"; else bad "no libggml-base.so"; fi
case "$fam" in
  hip) pat='libggml-hip.so*' ;; cuda) pat='libggml-cuda.so*' ;; vulkan) pat='libggml-vulkan.so*' ;; cpu) pat='libggml-cpu-*.so*' ;;
esac
# shellcheck disable=SC2086
if ls "$dir"/$pat >/dev/null 2>&1; then ok "has backend module $pat"; else bad "no $pat"; fi
if [ "$fam" != cpu ]; then
  # shellcheck disable=SC2086
  if ls "$dir"/libggml-cpu-*.so* >/dev/null 2>&1; then ok "has CPU variants (fallback and host side of the GPU build)"; else bad "no libggml-cpu-*.so"; fi
fi
if [ "$fam" = cpu ]; then n="$(find "$dir" -maxdepth 1 -name "libggml-cpu-*.so" | wc -l)"; if [ "$n" -ge 3 ]; then ok "CPU instruction-set variants: $n"; else bad "only $n CPU variants (GGML_CPU_ALL_VARIANTS lost?)"; fi; fi
if grep -q 'GGML_NATIVE=OFF' "$dir/BUILD-INFO.json" 2>/dev/null; then ok "BUILD-INFO: GGML_NATIVE=OFF"; else bad "BUILD-INFO lacks GGML_NATIVE=OFF"; fi

# 3. the binary runs: --version and --list-devices (no GPU: must not crash; a non-zero code is reported, a signal is a FAIL)
export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if "$dir/llama-server" --version >"$WORK/version.txt" 2>&1; then ok "--version: $(grep -m1 -i 'version' "$WORK/version.txt" | head -c 120)"; else bad "--version exit $? : $(head -c 300 "$WORK/version.txt")"; fi
timeout 120 "$dir/llama-server" --list-devices >"$WORK/devices.txt" 2>&1; rc=$?
if [ "$rc" -eq 0 ]; then ok "--list-devices exit 0 ($(grep -c . "$WORK/devices.txt") lines)";
elif [ "$rc" -ge 124 ]; then bad "--list-devices crashed or hung (exit $rc)";
else say "NOTE  --list-devices exit $rc without crash: $(tail -n2 "$WORK/devices.txt" | tr '\n' ' ' | head -c 200)"; fi
if ldd "$dir/llama-server" 2>&1 | grep -q 'not found'; then bad "ldd: unresolved: $(ldd "$dir/llama-server" 2>&1 | grep 'not found' | tr -s ' \t\n' ' ')"; else ok "ldd: all libraries of llama-server resolve"; fi

# 3b. release rules: no TLS library, glibc ceiling, no foreign GPU libraries, licence files, ROCm major
elfs=("$dir/llama-server"); while IFS= read -r f; do elfs+=("$f"); done < <(find "$dir" -maxdepth 1 -name 'lib*.so*' | sort)
tls=""
for f in "${elfs[@]}"; do
  hit="$(ldd "$f" 2>&1 | grep -E 'libssl|libcrypto' | tr -s ' \t\n' ' ')"
  [ -z "$hit" ] || tls="$tls $(basename "$f"):$hit"
done
if [ -z "$tls" ]; then ok "ldd: no libssl or libcrypto in any ELF"; else bad "ldd: TLS library linked:$tls"; fi
GLIBC_CEIL="2.35"
if command -v objdump >/dev/null 2>&1; then
  gmax="$(for f in "${elfs[@]}"; do objdump -T "$f" 2>/dev/null; done | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/^GLIBC_//' | sort -V | tail -n1)"
  if [ -z "$gmax" ]; then bad "glibc: no GLIBC_x.y symbol versions found (objdump -T empty)";
  elif [ "$(printf '%s\n%s\n' "$gmax" "$GLIBC_CEIL" | sort -V | tail -n1)" != "$GLIBC_CEIL" ]; then bad "glibc: highest GLIBC_$gmax is above the ceiling $GLIBC_CEIL";
  else ok "glibc: highest GLIBC_$gmax, ceiling $GLIBC_CEIL"; fi
else bad "glibc: objdump not found"; fi
if [ "$fam" != cuda ]; then
  foreign="$(find "$dir" -maxdepth 2 \( -name 'libcudart*' -o -name 'libcublas*' -o -name 'libcuda*' -o -name 'libamdhip64*' -o -name 'librocblas*' -o -name 'libhsa-runtime*' -o -name 'libvulkan.so*' \) | sed "s|^$dir/||" | tr '\n' ' ')"
  if [ -z "$foreign" ]; then ok "no foreign GPU libraries in the archive"; else bad "foreign GPU libraries in the archive: $foreign"; fi
else
  if [ -f "$dir/licenses/NVIDIA-CUDA-EULA-12.8.txt" ]; then ok "has licenses/NVIDIA-CUDA-EULA-12.8.txt"; else bad "missing licenses/NVIDIA-CUDA-EULA-12.8.txt"; fi
  if grep -q 'NVIDIA CUDA Toolkit 12.8' "$dir/NOTICE" 2>/dev/null; then ok "NOTICE names NVIDIA CUDA Toolkit 12.8"; else bad "NOTICE lacks 'NVIDIA CUDA Toolkit 12.8'"; fi
fi
if [ "$fam" = hip ]; then
  rmaj="$(python3 -B -c 'import json,sys; print(json.load(open(sys.argv[1])).get("runtime",{}).get("rocm_major",""))' "$dir/BUILD-INFO.json" 2>/dev/null)"
  want="$(python3 -B -c 'import sys; sys.path.insert(0, sys.argv[1]); from families import FAMILIES; print(FAMILIES["hip"]["rocm_soname_major"])' "$here" 2>/dev/null)"
  if [ -n "$want" ] && [ "$rmaj" = "$want" ]; then ok "BUILD-INFO: runtime.rocm_major $rmaj"; else bad "BUILD-INFO: runtime.rocm_major '$rmaj', expected '$want'"; fi
fi
if [ "$fam" = hip ] || [ "$fam" = cuda ]; then
  say "NOTE  $fam runtime libraries (ROCm/CUDA driver side) are not required on the build runner; backend module load is not tested without a GPU"
fi

# 4. self-test on a tiny model (CPU only; no download here)
if [ "$fam" = cpu ]; then
  if [ -n "${SHURA_TINY_MODEL:-}" ] && [ -f "$SHURA_TINY_MODEL" ]; then
    "$dir/llama-server" -m "$SHURA_TINY_MODEL" --port "$PORT" --host 127.0.0.1 -c 256 -ngl 0 >"$WORK/server.log" 2>&1 &
    spid=$!
    up=0; for _ in $(seq 1 60); do sleep 1; if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then up=1; break; fi; kill -0 "$spid" 2>/dev/null || break; done
    if [ "$up" = 1 ]; then
      reply="$(curl -sf "http://127.0.0.1:$PORT/completion" -d '{"prompt":"Once upon a time","n_predict":8,"temperature":0}' || true)"
      if printf '%s' "$reply" | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("content","").strip() else 1)' 2>/dev/null; then ok "self-test: tiny model answered with text"; else bad "self-test: no text in the completion"; fi
    else bad "self-test: server did not come up: $(tail -n3 "$WORK/server.log" | tr '\n' ' ' | head -c 300)"; fi
    kill "$spid" 2>/dev/null; wait "$spid" 2>/dev/null
  else
    say "SKIP  self-test on a tiny model: SHURA_TINY_MODEL not set (nothing is downloaded in this script)"
  fi
else
  say "SKIP  self-test with a model: needs a GPU, not available in the build job"
fi

# 5. our tests that need no GPU
if [ -f "$SRC/scripts/shura-touchpoints.py" ] && [ -f "$SRC/touchpoints.toml" ]; then
  if python3 -B "$SRC/scripts/shura-touchpoints.py" >"$WORK/touchpoints.txt" 2>&1; then ok "touchpoints: $(tail -n1 "$WORK/touchpoints.txt")"; else bad "touchpoints: $(tail -n3 "$WORK/touchpoints.txt" | tr '\n' ' ')"; fi
else say "NONE  touchpoints: scripts/shura-touchpoints.py or touchpoints.toml not in the tree"; fi
for t in test-vram-governor test-reasoning-effort; do
  src="$SRC/tests/$t.cpp"
  if [ ! -f "$src" ]; then say "NONE  $t: no tests/$t.cpp in the tree"; continue; fi
  if [ -n "${TESTS_BIN_DIR:-}" ] && [ -x "$TESTS_BIN_DIR/$t" ]; then
    if "$TESTS_BIN_DIR/$t" >"$WORK/$t.txt" 2>&1; then ok "$t"; else bad "$t: $(tail -n3 "$WORK/$t.txt" | tr '\n' ' ')"; fi
  else say "SKIP  $t: source exists but the release build has LLAMA_BUILD_TESTS=OFF and TESTS_BIN_DIR has no binary"; fi
done

if [ "$fail" = 0 ]; then say "RESULT all checks passed"; else say "RESULT FAILED"; fi
exit "$fail"
