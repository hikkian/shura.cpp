#!/usr/bin/env bash
# check-pins.sh - the guard job: the CUDA container image of build-cuda must be pinned by digest.
# Reads .github/workflows/shura-release.yml (path from this script's location), prints the digest, exit 1 if the pin is open or malformed.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wf="$here/../../.github/workflows/shura-release.yml"
[ -f "$wf" ] || { echo "no workflow file $wf" >&2; exit 1; }

if grep -E 'image:' "$wf" | grep -q 'PIN_ME'; then
  echo "an image: line in the workflow still has PIN_ME" >&2; exit 1
fi
refs="$(grep -c 'nvidia/cuda' "$wf" || true)"
[ "$refs" = 1 ] || { echo "expected exactly one nvidia/cuda reference in the workflow, found $refs" >&2; exit 1; }
lines="$(grep -E '^[[:space:]]*image:[[:space:]]*nvidia/cuda:' "$wf" || true)"
n="$(printf '%s' "$lines" | grep -c . || true)"
[ "$n" = 1 ] || { echo "expected exactly one 'image: nvidia/cuda:' line in the workflow, found $n" >&2; exit 1; }
img="$(printf '%s' "$lines" | sed -E 's/^[[:space:]]*image:[[:space:]]*//; s/[[:space:]]+#.*$//; s/[[:space:]]+$//')"
if [[ "$img" =~ ^nvidia/cuda:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}$ ]]; then
  digest="${img#*@}"
  pinned="$(tr -d '[:space:]' < "$here/cuda-image.digest" 2>/dev/null || true)"
  [ "$digest" = "$pinned" ] || { echo "CUDA image digest $digest differs from cuda-image.digest ($pinned)" >&2; exit 1; }
  echo "$digest"
else
  echo "CUDA image is not pinned as nvidia/cuda:<tag>@sha256:<64 hex>: '$img'" >&2; exit 1
fi
