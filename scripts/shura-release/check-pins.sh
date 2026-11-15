#!/usr/bin/env bash
# check-pins.sh - the guard job: every toolchain image in the release workflow is pinned by digest, and the digest is the one
# written in the pin file next to this script (cuda-image.digest, rocm-image.digest). Exit 1 otherwise.
# Env (tests only): PINS_WORKFLOW (the workflow file), PINS_DIR (the folder with the pin files).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
wf="${PINS_WORKFLOW:-$here/../../.github/workflows/shura-release.yml}"
pins="${PINS_DIR:-$here}"
[ -f "$wf" ] || { echo "no workflow file $wf" >&2; exit 1; }

if grep -E 'image:' "$wf" | grep -q 'PIN_ME'; then
  echo "an image: line in the workflow still has PIN_ME" >&2; exit 1
fi

check_image() {   # check_image <repository prefix, e.g. nvidia/cuda:> <pin file>
  local prefix="$1" pinfile="$2" refs lines n img digest pinned
  refs="$(grep -c "$prefix" "$wf" || true)"
  [ "$refs" = 1 ] || { echo "expected exactly one $prefix reference in the workflow, found $refs" >&2; exit 1; }
  lines="$(grep -E "^[[:space:]]*image:[[:space:]]*${prefix}" "$wf" || true)"
  n="$(printf '%s' "$lines" | grep -c . || true)"
  [ "$n" = 1 ] || { echo "expected exactly one 'image: $prefix' line, found $n" >&2; exit 1; }
  img="$(printf '%s' "$lines" | sed -E 's/^[[:space:]]*image:[[:space:]]*//; s/[[:space:]]+#.*$//; s/[[:space:]]+$//')"
  if [[ "$img" =~ ^[A-Za-z0-9./_-]+:[A-Za-z0-9._-]+@sha256:[0-9a-f]{64}$ ]]; then
    digest="${img#*@}"
    pinned="$(tr -d '[:space:]' < "$pins/$pinfile" 2>/dev/null || true)"
    [ "$digest" = "$pinned" ] || { echo "$prefix digest $digest differs from $pinfile ($pinned)" >&2; exit 1; }
    echo "$prefix pinned: $digest"
  else
    echo "$prefix image is not pinned as <name>:<tag>@sha256:<64 hex>: '$img'" >&2; exit 1
  fi
}

check_image "nvidia/cuda:" cuda-image.digest
check_image "rocm/dev-ubuntu-22.04:" rocm-image.digest
