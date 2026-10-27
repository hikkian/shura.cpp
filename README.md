<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/shura-cpp/banner-dark.svg">
  <img alt="shura.cpp: a fork of llama.cpp that shares one gaming GPU with the desktop you keep using" src="assets/shura-cpp/banner-light.svg" width="100%">
</picture>

**shura.cpp runs large Mixture-of-Experts models on one gaming GPU and gives the VRAM back the moment your desktop needs
it.** It is a fork of llama.cpp and the inference engine of [Shura](https://github.com/hikkian/shura).

## What it is

A 35B MoE model does not fit on a 16 GB card next to a desktop. shura.cpp keeps the experts of some layers in system
RAM, caches the ones those layers route to most often in VRAM and computes the rest from RAM. A governor inside
`llama-server` keeps a 1.5 GiB floor and a 256 MiB buffer of VRAM free for the desktop. When a browser, a video call or a
game takes more, the expert cache gives back the whole deficit in one step; when the desktop lets go, the cache grows
again.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/shura-cpp/governor-dark.svg">
  <img alt="The card's memory to scale: model 11.4 GiB, expert cache, the desktop's free room (1.5 GiB floor + 256 MiB buffer), desktop" src="assets/shura-cpp/governor-light.svg" width="100%">
</picture>

One loop, to scale: the desktop grows 1.25 GiB into its free room at once, the expert cache gives back five banks in one
step so the room is whole again, and once the desktop lets go the cache grows back bank by bank.

It is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp): the file format, the backends and the server
are theirs. Our changes are separate commits on the `shura` branch, rebased on upstream. Upstream's README is kept at
[docs/UPSTREAM-README.md](docs/UPSTREAM-README.md). This fork is not affiliated with or endorsed by the llama.cpp or ggml
projects.

Tested on AMD gfx1101 (RX 7800 XT) with ROCm. The CUDA build compiles but is not measured on the current base.

## What this fork adds

**VRAM governor.** Keeps free VRAM above the floor plus the buffer (512 MiB while the desktop is busy, set by the client).
Below that it shrinks the expert cache by the whole deficit in one resize, polling every 50 ms under pressure. A resize
waits until the card shows the old bank released before it allocates the new one, so the two never count at once.
Endpoints: `GET /vram`, `POST /vram/policy`, `POST /vram/pause`, `GET /vram/journal`.

**Stopping a long prefill.** An urgent trim or a pause interrupts the running prefill step. The CUDA/HIP backend checks
the abort callback in large batches (upstream checks it only on the CPU backend), `llama_decode` checks it between
micro-batches, and the server rolls the interrupted part back and runs it again. The check's cost is within the noise.
On models with recurrent or hybrid state (Qwen3.5 / Qwen3-Next style, which includes the reference models) this is
switched off for now: the trim or pause waits for the end of the current slice. See Known issues.

**Pause with a checkpoint, and resume.** When the desktop needs more than the cache can give, the server saves the slot,
tells the client and exits with code 75. The client (Shura's gateway) restarts it in a lighter layout, restores the slot
and continues the same answer with `resume` in the chat request. A pause in the middle of a prompt continues from there.

**Elastic expert cache.** Live resize over HTTP with a rollback-safe release, a bounded LRU policy, hybrid and
two-channel placement, a pinned source for the cold experts.

**Loading that leaves your browser alone.** On top of upstream's `--load-mode dio`, which reads without the page cache,
`--drop-file-cache` drops the file pages after the CPU experts are pinned and `--host-pinned-max-mib` caps pinned host
memory.

**Kernels for one consumer GPU.** GQA-vectorised flash-attention decode, faster `MUL_MAT_ID` prefill, an optional
ADD+RMS_NORM+MUL fusion, a per-op profiler and failure diagnostics, for CUDA and HIP.

Where upstream already does something we had, we use upstream's version: slot checkpoints, the `-1` id skip in
`mul_mat_id`, flash attention for RDNA, shared-expert MMVQ, and the q8_1 padding of the MoE matmul, where we had traced
a GPU page fault on long sessions (its reproducer stays in our tests). Every feature, by commit:
[docs/SHURA.md](docs/SHURA.md).

## Measured

On 2026-10-27 and 2026-10-28, on one machine: AMD RX 7800 XT 16 GB, Ryzen 5 5600, 32 GB DDR4-3200, Fedora 44, KDE on
Wayland, ROCm 7.2.4. Models Tiel-Coder-35B-A3B and Occamy-1.0-35B-A3B (Qwen3.5-class MoE, 8 of 256 experts active). This
build against the previous one (the same fork on an upstream of 2026-10-06), alternating pairs, medians of three.

| What | Result |
|---|---|
| Prefill, 28k-token prompt, short-context layout | Tiel **+16.3 %** (1499 → 1742 tok/s), Occamy **+15.4 %** (1487 → 1716) |
| Prefill, 100k-token prompt, long-context layout | Tiel **+7.9 %** (1281 → 1381 tok/s), Occamy **+7.2 %** (1285 → 1378) |
| Decode | short-context within the noise (−1 % Tiel, +1 % Occamy); long-context layout +17 % Tiel, +9 % Occamy |
| KL divergence against the previous build, 8 × 512 tokens | Tiel 0.0346 (the previous build against itself: 0.0343), Occamy 0.0185; gate 0.0385 |
| The desktop takes **+1.7 GiB at once** while a 28k prefill runs, cache full | one cache trim, the running step interrupted, free VRAM below 1.5 GiB for **0.05–0.2 s** (dense models; hybrid models wait for the slice, see Known issues) |
| The same during decode | one trim, below 1.5 GiB for **0.1 s** |
| The same with an empty cache (nothing left to give back) during a 100k prefill | prefill cut 14 ms after the decision, slot saved in 214 ms, exit 75; below 1.5 GiB for **0.85 s** (measured before the hybrid-model switch-off; hybrid models now finish the slice first) |
| Cache resize, 50 Hz trace of free VRAM, grow and shrink | dips below the lower of before/after: **0–9 MiB** (previous build: 238–1820 MiB) |
| Cost of the interrupt check in a prefill | 2313 vs 2321 tok/s (8k), 2003 vs 2001 (28k): within the noise |
| 40-minute hammer: 384 requests, new 28–34k prompts mixed with small ones, governor on † | 0 failures, 0 GPU errors, free VRAM never below 1595 MiB; server RSS settles (+11 MiB/h at the end) |
| 20 loads in a row with `--load-mode dio` † | ready in 8.0–8.5 s each, page cache +24…60 MiB, no swap, the browser untouched |

† Measured on the commit before the last governor changes (8f177300e); everything else on the current `shura` branch.

The benchmark scripts, the raw data and the method (a VRAM watchdog stops any run that leaves less than 1.5 GiB free for
more than a second) are published with Shura 1.0.

## Build

ROCm/HIP, as measured above. Use ROCm 7.2.x: 7.1.x leaks host memory from HIP graphs.

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
cmake -S . -B build -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1101 -DCMAKE_BUILD_TYPE=Release \
      -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF
cmake --build build --target llama-server llama-bench test-backend-ops
```

Replace `gfx1101` (RX 7800 XT / 7700 XT) with your card's target. If ROCm is not in the system paths, point `HIP_PATH`
and `CMAKE_PREFIX_PATH` at it.

CUDA:

```bash
cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target llama-server llama-bench test-backend-ops
```

Vulkan, Metal and SYCL build as in upstream; they do not get the expert cache, the governor or the TurboQuant KV types.

Checks that need no model:

```bash
build/bin/test-backend-ops test -o MUL_MAT_ID     # includes the ids == -1 cases
build/bin/test-backend-ops test -o FLASH_ATTN_EXT
build/bin/test-backend-ops test -o TURBO_WHT
build/bin/test-moe-cache-policy
build/bin/test-vram-governor
# interrupting llama_decode, with a small model made by test-llama-archs:
build/bin/test-llama-archs -a "^llama$" -o /tmp/models && build/bin/test-vram-abort /tmp/models/llama-dense.gguf
```

## Run

The flags Shura uses on a 16 GB card with a 35B-A3B model:

```bash
build/bin/llama-server -m model.gguf -ngl 99 -ncmoe 10 -fa on -c 32768 \
  -ctk q8_0 -ctv turbo4 --load-mode dio \
  --moe-cache-profile profile.csv --moe-cache-slots 20 --moe-cache-policy two-channel \
  --moe-cache-pinned-source --moe-cache-live-resize \
  --vram-governor --vram-reserve-mib 1536 --vram-pause-after-ms 250 \
  --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.2
```

`-ncmoe` is how many layers keep their experts on the CPU; the governor gives the VRAM that is left to the hot experts.
`profile.csv` comes from `tools/moe-trace`: record a routing trace on your own prompts. `curl localhost:8080/vram` shows
what the governor is doing; `POST /vram/policy {"buffer_mib": 512}` widens the buffer at run time (Shura's gateway does
it while a video plays or the browser is busy).

## Known issues

* **bf16 `MUL_MAT_ID` with skipped ids** (`n_used = 1`, `n = 4`, `skip = 1`) fails in about half of the long
  `test-backend-ops` runs with an error of ~0.5, and never alone or in its own family of cases. It depends on what earlier
  cases left in GPU memory. The quantized types the models use do not show it.
* **Upstream: `MUL_MAT_ID` with an offset ids view** (`iq3_xxs`, `n = 8`) fails in 8 of 10 runs on ROCm on plain upstream
  74bca3950 too. Our graphs never pass offset ids views.
* **Interrupting a prefill is off on models with recurrent or hybrid state.** Rolling back an interrupted slice did
  not restore the recurrent state, and the retried slice failed with HTTP 500 (found on 2026-10-28). Until a proper
  rollback lands, such models finish the current slice before a trim or pause (`GET /vram` → `cache.interrupts`
  is `off_recurrent_state`); on the reference machine that keeps free VRAM below the floor for 0.1–0.6 s in a long prefill.
* **Decode steps run as HIP/CUDA graphs and are not interrupted.** An urgent trim during decode waits for the current
  step, about 13 ms on the reference machine. Backends other than CPU and CUDA/HIP stop only between micro-batches.
* The CUDA build is not measured on this base.

Details in [docs/SHURA.md](docs/SHURA.md#known-issues).

## Built on

shura.cpp started from a snapshot of **[thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp)** (branch `perf`,
2026-09-03), the MoE performance fork by **Anirban Kar**. The ideas and the first implementations of these parts are his:

* the **expert cache**: the most-routed experts of each layer kept in VRAM as packs, the rest computed from RAM and the
  two halves merged exactly;
* **expert prefetch** in the scheduler during large batches and CPU splits on a worker thread;
* **pinning** the memory-mapped CPU experts so uploads go over DMA;
* the routing-trace recorder `tools/moe-trace` and the qwen4exp conversion support.

The **TurboQuant KV cache types** (after arXiv [2504.19874](https://arxiv.org/abs/2504.19874)) with their CUDA/HIP
flash-attention kernels and rotation data came into that base from **[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)**
(branch `feature/turboquant-kv-cache`).

We moved all of it onto a current upstream, fixed what broke and built the rest on top. Thank you, Anirban and Tom.
Every source and its license is listed in [NOTICE](NOTICE).

The `upstream` branch mirrors `ggml-org/llama.cpp` master and is never committed to; `shura` is our commits rebased on
top of it, one feature per commit, so a conflict names the feature it belongs to. The hook points into upstream files are
listed in [touchpoints.toml](touchpoints.toml) and checked by `scripts/shura-touchpoints.py`; the memo for moving to a
newer upstream is [docs/SHURA-UPDATE.md](docs/SHURA-UPDATE.md).

## License

MIT, like llama.cpp. The upstream [LICENSE](LICENSE) and [AUTHORS](AUTHORS) are kept unchanged. The parts that come
from [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp) are © Anirban Kar, the TurboQuant parts from
[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) © their author, the rest of this fork's
changes © 2026 hikkian, all under the same MIT license; see [NOTICE](NOTICE). Model weights are not included and are
covered by their own licenses.
