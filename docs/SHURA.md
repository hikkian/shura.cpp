# shura.cpp 1.0: what it changes in llama.cpp v1.0.0

shura.cpp is the inference engine of [Shura](https://github.com/hikkian/shura): large MoE models on one consumer GPU with the experts
that do not fit kept in host memory, a VRAM governor that gives the card back to the desktop on demand, and pause/resume of a request.

**shura.cpp 1.0 is llama.cpp v1.0.0 plus only what upstream does not have**, added on top of upstream's own mechanisms where there is
one. Earlier versions (the `shura` branch) were a separate fork line based on a snapshot of thecodacus/llama.cpp; 1.0 was rebuilt on
upstream (design: Shura's report SHCPP0; port and measurements: REBASE1). Tested on AMD gfx1101 (RX 7800 XT, ROCm 10.1); CUDA and HIP
are kept compiling by CI; the other backends get upstream's code only.

## Branches

| branch | content |
|---|---|
| `shura-v1` | llama.cpp tag `v1.0.0` + the commits below, one part per commit |
| `shura` | the earlier line (until 1.0 replaces it in Shura) |

The hook points into upstream files are listed in [touchpoints.toml](../touchpoints.toml); `scripts/shura-touchpoints.py` checks that
every one is still in place. Moving to a newer upstream: [SHURA-UPDATE.md](SHURA-UPDATE.md).

## Upstream's parts that replace ours

| ours (the `shura` branch) | upstream v1.0.0, used as is |
|---|---|
| expert cache: equal slots per layer, static/hybrid/two-channel policies, hot/cold packs (`ids == -1`) | `llama_moe_cache` (`src/llama-moe-cache.cpp`): one budget in MiB per group of layers, ClassLRU eviction with frequency memory, a CPU share of the misses, in-device copies for large batches; `--moe-cache-mib`, `--moe-cache-cpu-share` |
| cache bypass during prefill | the same behaviour (no cache update for batches above 32 tokens) |
| TurboQuant KV types (TURBO2/3/4_0) | `q4_0` / `q8_0` KV with upstream's Hadamard rotation of quantized K and V (`LLAMA_ATTN_ROT_DISABLE=1` turns it off). REBASE1: V `q4_0` had a lower KL divergence than V `turbo4` (0.0106 against 0.0135 at 2–4k context), so TurboQuant was not ported |
| MTP export, `mtp-dump` | NextN API and MTP drafting (`--spec-type draft-mtp`) |
| loader: direct I/O | `--load-mode dio` |
| head size 512 attention, MUL_MAT_ID prefill pass, ADD+RMS_NORM+MUL fusion, GDN knobs | upstream's kernels |
| slot checkpoints in slot files | upstream always appends them; `--slot-save-checkpoints` is accepted and ignored |
| LoRA with the expert cache | upstream's (the CPU share is off with LoRA); an MTP draft context that shares the model gets the same adapters |

## Our parts (on top of upstream)

| part | what | where |
|---|---|---|
| **expert cache: live resize** | `llama_moe_cache_resize(ctx, bytes)`: under the micro-batch lock, backends synchronized, banks released, the governor's release wait, new banks allocated and zeroed; an allocation failure restores the previous size, a second one leaves the cache empty (layers read the host experts, the server keeps answering); the slots are warmed from the decayed use counters; the layout generation is a graph parameter (no stale graph reuse); the scheduler is reserved lazily on the decoding thread with the same rollback; `invalidate(il, e)` re-reads an expert whose host copy changed | `src/llama-moe-cache.{h,cpp}`, `src/llama-context.{h,cpp}`, `src/llama-graph.h`, `include/llama.h` |
| expert cache: empty state, observability | `--moe-cache-live-resize`: the cache exists at size 0; geometry and counters as an atomic snapshot (`llama_moe_cache_get_info`); `llama_moe_cache_set_cpu_share`; the cache is reported as a part of `context` in the memory breakdown (`moe_cache`) | same |
| **VRAM governor** | `--vram-governor` and the `--vram-*` flags, `/vram`, `/vram/policy`, `/vram/pause`, `/vram/journal`, pause with exit code 75 and `<checkpoint>.tokens.json`, `resume` in chat requests. Chunk 0 = empty, chunk 1 = the smallest cache that serves the MTP verify batch, then `--vram-chunk-mib` steps; `--moe-cache-cpu-share-table` switches the CPU share by size. With recurrent/hybrid memory (Qwen3.5/3.6 MoE) a ubatch is never interrupted: the recurrent state cannot be rolled back over a slice | `tools/server/server-vram*.{h,cpp}`, hooks in `server-context.cpp`, `server.cpp` |
| CUDA/HIP hooks | the abort callback inside large node-by-node graphs (two marks, the GPU queue never drains) and between ubatches; pool freeze and growth accounting; pinned host memory limit (`--host-pinned-max-mib`); `GGML_CUDA_LOG_ALLOCS`; failure diagnostics (`PFAULT1`) | `ggml/src/ggml-cuda/{ggml-cuda.cu,common.cuh}`, `src/llama-context.cpp` |
| **expert upload overlap** (`GGML_SCHED_PREFETCH_EXPERTS=3`) | for a large-batch MUL_MAT_ID split the whole expert tensor is uploaded through a second backend instance into a staging slot, overlapping the previous split; brings prefill back to the earlier engine's level | `ggml/src/ggml-backend.cpp` |
| **GQA-vectorised decode attention** (`GGML_HIP_FA_GQA_VEC=1`) | HIP, K `q8_0` with V `q8_0`/`q4_0`, head 256, GQA 4 or 8, up to 3 tokens (the MTP verify batch): one block handles all Q heads of a KV head | `ggml/src/ggml-cuda/fattn-vec.cuh` |
| reasoning budget (THINK3b) | soft target, ThinkBrake, a grace window at the ceiling, counters in `timings`/`usage`, shadow mode; off by default, the output without the new fields is upstream's | `common/reasoning-budget.cpp`, `common/sampling.cpp`, server schema |
| SIGABRT backtraces | the stacks of all threads on a glibc abort | `common/common.cpp`, `tools/server/server.cpp` |
| `llama-perplexity` streaming (`SHURA_PPL_STREAM=1`) | KL divergence batch by batch, so the memory does not grow with the context (a 248k vocabulary) | `tools/perplexity/perplexity.cpp` |

## Recommended settings (desktop, RX 7800 XT 16 GB, Tiel / Ornith-1.5 35B-A3B)

From REBASE1's measurements (see the report for the numbers and the decision rule):

| profile | settings |
|---|---|
| live 32k | `-c 32768 -ub 1024 -ncmoe 14 -ctk q8_0 -ctv q4_0 -fa on --load-mode dio`, MTP `--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-p-min 0.25 -ctkd q8_0 -ctvd q8_0`, cache `--moe-cache-live-resize --moe-cache-mib 0 --moe-cache-cpu-share 0.5` with `--vram-governor`; env `GGML_SCHED_PREFETCH_EXPERTS=3 GGML_HIP_FA_GQA_VEC=1 GPU_MAX_HW_QUEUES=2` |
| long, the full context 262 144 with MTP | as above with `-c 262144 -ncmoe 18 -ub 512 -ctk q4_0 -ctv q4_0` (fits with ≥ 1.5 GiB free at ~255k tokens) |

## Not carried over from the `shura` branch

TurboQuant KV types and their kernels (upstream's rotated `q4_0` is more accurate); the static/hybrid/two-channel policies and the
hot/cold packs (upstream's cache; the two-channel rule may come back inside upstream's `plan()` if measurements ask for it); the
CPU split worker (`GGML_SCHED_ASYNC_CPU`); `tools/moe-trace` (upstream's `LLAMA_MOE_CACHE_TRACE` and `scripts/moe-cache-tune.py`
replace it; recording on large batches is still to do); `mtp-dump` (to be rewritten on the NextN API); the op profiler; the
`/moe-cache/slots` endpoint of the old cache (the governor's `/moe-cache/chunks` remains under test hooks).

## Known limits

* `PAD_REFLECT_1D` fails in `test-backend-ops` on HIP gfx1101 in upstream v1.0.0 ("invalid configuration argument"); our models
  do not use it.
* `iq4_nl` KV has no flash-attention kernel on CUDA/HIP in v1.0.0 (the attention would run elsewhere).
* The governor reads amdgpu's sysfs (`mem_info_vram_*`); on NVIDIA it reports itself unavailable and the server runs without it.
* MTP acceptance per drafted token is ~1.7 points lower than the earlier engine's at `--spec-draft-p-min 0.2` (upstream's MTP drafts
  slightly more often; accepted tokens per generated token are the same); `0.25` brings it above the earlier level.
