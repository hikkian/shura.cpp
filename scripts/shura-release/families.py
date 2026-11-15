#!/usr/bin/env python3
"""One table for the four release families of shura.cpp (card RELBIN1). Used by build.sh, package.py, manifest.py.

Contract: shura-integration/docs/RELEASE-CONTRACT.md. Values marked (config) are our decisions, not facts about hardware:
change them here and nowhere else.  CLI: families.py targets|cmake-targets|manifest-family <family>."""
import sys

FAMILIES = {
    "hip": {
        "manifest_family": "HIP",
        "gpu_targets": ["gfx1030", "gfx1100", "gfx1101", "gfx1102"],
        # (config) now checkable: the major version comes from the soname libamdhip64.so.N that libggml-hip.so was built against;
        # the rule is "the same major version fits". Only a run of the archive from GitHub finally confirms it.
        "runtime": {"rocm_min": "7.0", "rocm_max": "7.99"},
        "rocm_soname_major": 7,
        # our parts that work in this build: VRAM governor and GQA flash-attention kernel are HIP-only
        "parts": ["vram_governor", "gqa_kernel", "mtp", "moe_cache"],
    },
    "cuda": {
        "manifest_family": "CUDA",
        # compute capability, CUDA 12.8: 7.5 Turing, 8.0/8.6/8.9 Ampere/Ada, 12.0 Blackwell consumer
        "gpu_targets": ["7.5", "8.0", "8.6", "8.9", "12.0"],
        "runtime": {"nvidia_driver_min": "570.26"},   # (config) CUDA 12.8 needs 570.26 for sm_120
        # no VRAM governor and no GQA kernel on CUDA (card RELBIN1)
        "parts": ["mtp", "moe_cache"],
    },
    "vulkan": {"manifest_family": "Vulkan", "gpu_targets": [], "runtime": {"vulkan_min": "1.2"},   # (config)
               "parts": ["mtp", "moe_cache"]},
    "cpu": {"manifest_family": "CPU", "gpu_targets": [], "runtime": {}, "parts": ["mtp", "moe_cache"]},
}

# size window of one archive, bytes (checked by check.sh and manifest.py)
MIN_ARCHIVE = 10 * 1024 * 1024
MAX_ARCHIVE = 2 * 1024 * 1024 * 1024


def cuda_archs(f=FAMILIES["cuda"]):
    """'7.5' -> '75-real'; the cmake list for CMAKE_CUDA_ARCHITECTURES."""
    return ";".join(t.replace(".", "") + "-real" for t in f["gpu_targets"])


if __name__ == "__main__":
    cmd, fam = sys.argv[1], sys.argv[2]
    f = FAMILIES[fam]
    if cmd == "targets":
        print(cuda_archs() if fam == "cuda" else ";".join(f["gpu_targets"]))
    elif cmd == "manifest-family":
        print(f["manifest_family"])
    else:
        sys.exit(f"unknown command {cmd}")
