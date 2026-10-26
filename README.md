<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/shura-cpp/wordmark-dark.png">
  <img alt="shura.cpp" src="assets/shura-cpp/wordmark-light.png" width="720">
</picture>

</div>

**shura.cpp is the inference engine of [Shura](https://github.com/hikkian/shura): a fork of llama.cpp that shares one
gaming GPU with the desktop you keep using.** It runs large Mixture-of-Experts models on a single consumer card by keeping
the hot experts in VRAM and the rest in system RAM, and it gives the card back to the desktop the moment the desktop
needs it.

> [!NOTE]
> **This is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).** Everything that makes it run is
> theirs: the file format, the backends, the server, years of careful work by many people. We are grateful for it. Our
> changes sit on top as separate commits on the `shura` branch; the upstream README is kept at
> [docs/UPSTREAM-README.md](docs/UPSTREAM-README.md). Nothing here is endorsed by or affiliated with the llama.cpp or
> ggml projects.

## What this fork adds

| | What | Where to read |
|---|---|---|
| **VRAM governor** | A thread inside `llama-server` polls the card every 200 ms and keeps a configurable amount of VRAM free for the desktop (1.5 GiB by default in Shura). It shrinks the expert cache in a safe point between compute steps and grows it back when the desktop is quiet. Endpoints `GET /vram`, `POST /vram/policy`, `POST /vram/pause`, `GET /vram/journal`. | [docs/SHURA.md](docs/SHURA.md), feature 8 |
| **Elastic expert cache** | Hot experts of every layer live in VRAM in packs, the cold ones are computed from RAM; the two halves are merged exactly, so the output is the same as without the cache. Live resize over HTTP; a resize releases the old bank first and never holds two banks at once. Static, LRU, hybrid and two-channel policies; a routing-trace recorder (`tools/moe-trace`) to choose the hot set. | feature 6 |
| **Pause with a checkpoint, and resume** | When the desktop needs more than the cache can give back, the server saves the slot, tells the client, and exits with code 75. The client (Shura's gateway) restarts it in a lighter layout, restores the slot and **continues the same answer** with `resume` in the chat request. | feature 8 |
| **TurboQuant KV types** | `TURBO2_0`, `TURBO3_0`, `TURBO4_0` for the KV cache (CPU reference, CUDA/HIP kernels, flash-attention paths), so that a 200k window fits next to the weights. | feature 1 |
| **Kernels for one consumer GPU** | GQA-vectorised flash-attention decode, head size 512 for Gemma, a faster `MUL_MAT_ID` prefill with one pass over the ids, expert prefetch in the scheduler during large batches. CUDA and HIP. | feature 7 |
| **Loading that does not evict your browser** | `--load-mode dio` reads the weights without going through the page cache, drops the file pages after pinning the CPU experts, and caps pinned host memory. | feature 4 |
| **Page-fault hunt** | A GPU page fault on long, growing sessions was traced to an out-of-bounds read in the q8_1 padding of the MoE matmul (`MUL_MAT_ID` ids path). Upstream has since rewritten that code; this fork carries upstream's version and keeps the reproducer in its tests. | [docs/SHURA.md](docs/SHURA.md), "Where we use upstream's version" |

Where upstream already does something we had, we take upstream's version and drop ours (slot checkpoints, the `-1` id
skip in `mul_mat_id`, flash attention for RDNA, shared-expert MMVQ). The list is in `docs/SHURA.md`.

## Measured

One machine: AMD RX 7800 XT 16 GB, Ryzen 5 5600, 32 GB DDR4-3200, Fedora 44, KDE on Wayland, ROCm 7.2.4; model
Tiel-Coder-35B-A3B (a Qwen3.5-class MoE, 8 of 256 experts active). Every number has a date. Numbers marked * were taken
on the previous base of this fork and are re-measured on this build before publication (card REBASE2V in the Shura
project).

| What | Result | Date |
|---|---|---|
| Desktop takes +400 MiB of VRAM during decode | free VRAM back above 1.5 GiB after **0.24 s**; longest gap between tokens 51 ms | 2026-10-25 |
| The same during a 100k-token prefill | back above 1.5 GiB after **0.29 s**, prefill continues | 2026-10-25 |
| Desktop takes more than the cache can give back | checkpoint 0.6 s, exit, restart, restore 22 ms, the answer continues | 2026-10-25 |
| 44-minute soak: 171 requests from 0.2k to 28k tokens plus 25 desktop jumps of 100–450 MiB | 15 cache shrinks, 12 regrowths, no pause needed, the buffer pool did not grow after warm-up | 2026-10-25 |
| Decode, Tiel 35B-A3B, short prompt / 28k / 100k filled context | **75 / 71 / 49 tok/s** * | 2026-10-24 |
| 40-minute long-session hammer: 170 requests, prompts up to 180,300 tokens, a cache resize on every request | 0 GPU page faults, 0 errors | 2026-10-24 |
| Host memory of the server under a steady replay | flat on ROCm 7.2.4 (0.01 GiB/h); the same binary on ROCm 7.1.1 grew 4.5 GiB/h, a runtime leak, which is why we build against 7.2.x | 2026-10-24 |

The benchmark scripts, the raw data and the method (alternating pairs, KL gate 0.0385 against the reference layout, a
VRAM watchdog that stops any run leaving less than 1.5 GiB free) are published with Shura 1.0.

## Build

ROCm/HIP (what the numbers above were measured with; ROCm 7.2.x, because 7.1.x leaks host memory from HIP graphs):

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1101 -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF
cmake --build build --target llama-server llama-bench test-backend-ops
```

Replace `gfx1101` (RX 7800 XT / 7700 XT) with your card's target; the HIP build is tested on gfx1101 only. If ROCm is
installed somewhere other than the system paths, point `HIP_PATH` and `CMAKE_PREFIX_PATH` at it.

CUDA:

```bash
cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-bench test-backend-ops
```

The CUDA build is kept compiling but is not measured on this fork's current base. Vulkan, Metal and SYCL build as in
upstream; the expert cache, the governor and the TurboQuant KV types are CUDA/HIP only, the other backends simply do not
see them.

Checks that need no model:

```bash
build/bin/test-backend-ops test -o MUL_MAT_ID     # includes the ids == -1 cases
build/bin/test-backend-ops test -o FLASH_ATTN_EXT
build/bin/test-backend-ops test -o TURBO_WHT
build/bin/test-moe-cache-policy
build/bin/test-vram-governor
```

## Run

The flags Shura uses, as a starting point for a 16 GB card and a 35B-A3B model:

```bash
build/bin/llama-server -m model.gguf -ngl 99 -ncmoe 10 -fa on -c 32768 \
  -ctk q8_0 -ctv turbo4 --load-mode dio \
  --moe-cache-profile profile.csv --moe-cache-slots 20 --moe-cache-policy two-channel \
  --moe-cache-pinned-source --moe-cache-live-resize \
  --vram-governor --vram-reserve-mib 1536 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.2
```

`-ncmoe` is how many layers of experts stay on the CPU; the governor then uses whatever VRAM is left for the hot
experts. `profile.csv` comes from `tools/moe-trace` (record a routing trace on your own prompts). Everything else is
upstream's. `curl localhost:8080/vram` shows what the governor is doing.

## Keeping up with upstream

The `upstream` branch mirrors `ggml-org/llama.cpp` master and is never committed to; `shura` is our commits rebased on
top of it, one feature per commit, so a conflict names the feature it belongs to. The hook points into upstream files
are listed in [touchpoints.toml](touchpoints.toml) and checked by `scripts/shura-touchpoints.py`. The memo for moving
to a newer upstream, with the things that broke on previous rebases, is [docs/SHURA-UPDATE.md](docs/SHURA-UPDATE.md).

## License

MIT, like llama.cpp. The upstream [LICENSE](LICENSE) and [AUTHORS](AUTHORS) are kept unchanged; the changes in this
fork are © 2026 hikkian and are released under the same MIT license. Model weights are not included and are covered by
their own licenses.
