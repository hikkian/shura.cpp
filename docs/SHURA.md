# shura.cpp: what this fork changes

shura.cpp is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for one job: running large MoE models on a
single consumer GPU (AMD gfx1101 tested; CUDA builds are kept compiling) with a CPU-side expert store, quantized KV cache and a
VRAM governor that gives the card back to the desktop on demand.

## Branches

| branch | content |
|---|---|
| `upstream` | mirror of `ggml-org/llama.cpp` master, never committed to |
| `shura` | our changes as separate commits on top of `upstream` (rebased, not merged) |

Each commit is one feature, so a conflict during a rebase names the feature it belongs to. The memo for moving to a newer upstream is
[SHURA-UPDATE.md](SHURA-UPDATE.md). The list of places where our code hooks into upstream files is [touchpoints.toml](../touchpoints.toml);
`scripts/shura-touchpoints.py` checks that all of them are still present.

## Own files (nothing of upstream in them)

| file | what |
|---|---|
| `ggml/src/ggml-turbo-quant.c`, `ggml/src/ggml-cuda/turbo-*.cu(h)`, `fattn-mma-turbo.cuh`, `src/turbo-rotation-data*.h` | TurboQuant KV types (TURBO2/3/4_0): CPU reference, CUDA/HIP kernels, rotation tables |
| `src/llama-moe-cache-policy.h`, `src/llama-moe-cache-two-channel-policy.h` | expert cache policies (static, hybrid, two-channel) as plain C++ with no ggml dependency; tested by `tests/test-moe-cache-policy.cpp` |
| `tools/server/server-vram-governor.h` | the VRAM decision core (pure functions); tested by `tests/test-vram-governor.cpp` |
| `tools/server/server-vram.{h,cpp}` | governor runtime: sampling, warm-up, pause/resume, endpoints |
| `tools/moe-trace/` | routing trace recorder and cache simulator |
| `scripts/turbo-quality-gate.sh`, `scripts/shura-touchpoints.py` | KL gate for KV types; hook point check |

## Features, in commit order

1. **turbo-kv**: KV cache types TURBO2/3/4_0 (enum 43..45), `GGML_OP_TURBO_WHT`, flash-attention vec and MMA paths, set-rows, dequantize. CUDA/HIP only; the other GPU backends (Vulkan, Metal, SYCL, MUSA) are not supported for these types and use upstream code.
2. **scheduler-prefetch**: `ggml_backend_sched` prefetches full expert tensors through a second backend during large batches; CPU splits can run on a worker (`cpu_async`).
3. **qwen4exp**: converter side only (`mtp_shared_embd`, `nextn_shared_target_tensors`); the architecture itself is upstream's.
4. **llama core**: KV type plumbing, loader options (`--load-mode dio`, page-cache drop, pinned limit), graph and model changes.
5. **moe-trace**, **README**.
6. **moe cache (hot/cold packs)**: separate expert packs on the GPU with `ids == -1` for "not owned by this pack"; LRU, hybrid and two-channel policies; bypass during prefill; pinned source; live resize without a double VRAM peak.
7. **CUDA/HIP kernels**: GQA-vectorised FA decode (`GGML_HIP_FA_GQA_VEC`), head size 512 FA for Gemma, faster MUL_MAT_ID prefill (one pass over the ids), ADD+RMS_NORM+MUL fusion (off by default), GDN tuning knobs, per-op profiler, failure diagnostics.
8. **VRAM governor**: `--vram-governor`, `/vram`, `/vram/policy`, `/vram/pause`, pause exit code 75 with `<checkpoint>.tokens.json`, `resume` in chat requests, RAM guard. VRAMLOGIC2: the reserve is the floor, the working target is reserve + `--vram-buffer-mib` (512): trimming starts below the target by the whole deficit, polling is `--vram-poll-fast-ms` (50) under pressure, and an urgent trim (already below the floor) interrupts the running ubatch through the abort callback; the server rewinds and retries that batch slice (`rewind_slice`, `ret == 2` in `decode()`).

## Where we use upstream's version instead of ours

Recorded for every rebase in `reports/REPORT-REBASE2.md` (in the stack repository). Currently: mmq ids padding (page fault fix),
flash attention for RDNA (#28102) and the follow-up fixes, slot checkpoints in slot files (#26004, #30514, #30574; our
`--slot-save-checkpoints` flag is gone, upstream always appends the checkpoints), upstream's `-1` id skip in the CPU `mul_mat_id`,
shared-expert MMVQ (#29184, replaces our fuse3), the qwen4exp MTP export.

## Hook points in upstream files

Grouped by feature; the exact probe strings are in `touchpoints.toml`.

* **turbo-kv**: `ggml.h` (type and op enums), `ggml.c` (op tables, `GGML_OP_COUNT` assertions), `ggml-cpu/` (type traits, `ops.cpp` WHT), `ggml-common.h`, `ggml-quants.h`, `ggml-cuda/` (`common.cuh`, `convert.cu`, `dequantize.cuh`, `fattn*.cu(h)`, `ggml-cuda.cu`, `set-rows.cu`, `quantize.cu`), cmake lists for the FA instances, `llama-graph.cpp` (query pre-rotation in every `build_attn`), gguf-py constants.
* **scheduler-prefetch**: `ggml-backend.cpp` (inside `ggml_backend_sched_compute_splits`, the host-weight copy loop), `ggml-backend.h`, `llama-context.cpp` (`sched_new()`).
* **moe cache**: `llama-graph.cpp/.h` (`build_moe_ffn` pack path and the `moe_cache` argument), `llama-model.cpp/.h` (`init_moe_expert_cache`, resize, stats; this code lives in `llama-model.cpp` because it needs `llama_model::impl`), `src/models/{qwen35moe,gemma4,laguna,deepseek2}.cpp` (last argument of `build_moe_ffn`), `ggml-cuda/{mmvq,mmid,mmq,mmf,mmvf}.cu`, `llama-model-loader.cpp/.h`, `include/llama.h`, `common/{arg.cpp,common.h}`.
* **governor**: `server.cpp` (`server_prepare_params`, `server_register_routes`), `server-context.cpp` (pause task, resume request), `server-queue.cpp` and `server-task.cpp` (`seed_resume`), `common/arg.cpp`, `llama-mmap.cpp` (`O_DIRECT`).
* **tests and tools**: `tests/test-backend-ops.cpp` (cases for `ids == -1`, FA shapes, the fused op), `tools/llama-bench` (`--sched-async-cpu`).

## Known untidy places

* `llama-model.cpp` carries about 1,100 lines of the expert cache. They cannot move to their own translation unit without moving the private `llama_model::impl` into a header, which would touch far more of upstream than it saves.
* In `build_moe_ffn` the pack path sits next to upstream's slot-cache code inside `if (!use_moe_packs) { ... }` with unchanged indentation, to keep the diff small.
