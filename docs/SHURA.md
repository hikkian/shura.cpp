# shura.cpp: what this fork changes

shura.cpp is a fork of [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) for one job: running large MoE models on a
single consumer GPU (AMD gfx1101 tested; CUDA builds are kept compiling) with a CPU-side expert store, quantized KV cache and a
VRAM governor that gives the card back to the desktop on demand.

## Origins

The fork started from a snapshot of [thecodacus/llama.cpp](https://github.com/thecodacus/llama.cpp) (branch `perf`, 2026-09-03), the
MoE performance fork by **Anirban Kar**: the expert cache (hot expert packs in VRAM), the scheduler's expert prefetch and CPU worker,
pinning of the CPU experts, `tools/moe-trace` and the qwen4exp conversion support come from there (features 2, 3, 5 and the base of 6
below). The TurboQuant KV types, their CUDA/HIP flash-attention kernels and rotation data (feature 1) came into that base from
[TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant) (branch `feature/turboquant-kv-cache`). All of it was moved
onto a current upstream as the commits of this branch, so the history here does not show their authorship; [NOTICE](../NOTICE), this note
and the README do. Everything else (the governor, pause and resume, live resize, the bounded LRU, hybrid and two-channel policies, the
loader options, the kernels of feature 7, the fixes) was written for Shura.

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

1. **turbo-kv** (from TheTom/llama-cpp-turboquant via thecodacus/llama.cpp): KV cache types TURBO2/3/4_0 (enum 43..45), `GGML_OP_TURBO_WHT`, flash-attention vec and MMA paths, set-rows, dequantize. CUDA/HIP only; the other GPU backends (Vulkan, Metal, SYCL, MUSA) are not supported for these types and use upstream code.
2. **scheduler-prefetch** (from thecodacus/llama.cpp): `ggml_backend_sched` prefetches full expert tensors through a second backend during large batches; CPU splits can run on a worker (`cpu_async`).
3. **qwen4exp** (from thecodacus/llama.cpp): converter side only (`mtp_shared_embd`, `nextn_shared_target_tensors`); the architecture itself is upstream's.
4. **llama core**: KV type plumbing, loader options (`--load-mode dio`, page-cache drop, pinned limit), graph and model changes.
5. **moe-trace** (from thecodacus/llama.cpp), **README**.
6. **moe cache (hot/cold packs)** (the packs and static placement from thecodacus/llama.cpp; LRU, hybrid, two-channel, live resize ours): separate expert packs on the GPU with `ids == -1` for "not owned by this pack"; LRU, hybrid and two-channel policies; bypass during prefill; pinned source; live resize without a double VRAM peak.
7. **CUDA/HIP kernels**: GQA-vectorised FA decode (`GGML_HIP_FA_GQA_VEC`), head size 512 FA for Gemma, faster MUL_MAT_ID prefill (one pass over the ids), ADD+RMS_NORM+MUL fusion (off by default), GDN tuning knobs, per-op profiler, failure diagnostics.
8. **VRAM governor**: `--vram-governor`, `/vram`, `/vram/policy`, `/vram/pause`, `/vram/journal`, pause exit code 75 with `<checkpoint>.tokens.json`, `resume` in chat requests, RAM guard, the abort callback in the CUDA/HIP backend and between micro-batches. See "VRAM governor" below.

## Where we use upstream's version instead of ours

Recorded at every rebase in the Shura project's notes. Currently: mmq ids padding (page fault fix),
flash attention for RDNA (#28102) and the follow-up fixes, slot checkpoints in slot files (#26004, #30514, #30574; our
`--slot-save-checkpoints` flag is gone, upstream always appends the checkpoints), upstream's `-1` id skip in the CPU `mul_mat_id`,
shared-expert MMVQ (#29184, replaces our fuse3), the qwen4exp MTP export.

## VRAM governor

A thread inside `llama-server` (`tools/server/server-vram.{h,cpp}`, decisions in `server-vram-governor.h`) owns the expert cache. All
sizes are MiB, times seconds.

* **Floor and target.** `reserve_mib` (default 1536) is the floor free VRAM should never go below; the working target is
  `reserve_mib + buffer_mib` (buffer default 256). Below the target the cache gives the whole deficit back in one resize, rounded up to
  whole chunks (`--moe-cache-chunk-slots` slots each); it grows back only into room above target + `hysteresis_mib` (256), one step per
  `grow_seconds` (30) of calm, never while a request runs. Polling is `poll_ms` (200), and `poll_fast_ms` (50) while free VRAM is below
  target + hysteresis or fell by 128 MiB since the previous poll.
* **Urgent trims.** A trim decided when free VRAM is already below the floor raises the abort callback while the resize waits for the
  cache lock. The CUDA/HIP backend checks it every `GGML_CUDA_ABORT_CHECK_NODES` (64) nodes of a large-batch graph evaluated node by node
  (prefill; decode runs as HIP/CUDA graphs and is not checked), keeping one interval of work queued so the GPU never runs dry; the CPU
  backend checks it per node; `llama_decode` checks it between micro-batches. The server rolls the interrupted batch slice back and
  decodes it again (`cache.ubatch_interrupts`).
* **Release before allocation.** Between freeing the old bank and allocating the new one, a resize waits (at most 250 ms) until the card
  shows most of the old bank free again: on ROCm the release is accounted later than the free call returns, and without the wait both
  banks counted at once. The resize log line reports how long it waited.
* **Pause.** With the cache at its minimum and free VRAM below the floor by more than 48 MiB for `pause_after_ms` (server default
  2000; Shura's gateway sends 250), or below `pause_floor_mib` (768) for two polls, the server saves every slot that holds tokens and
  exits with code 75. A running prompt slice is cut for it (`cache.pause_cuts`); a decode step is not.

Endpoints:

| endpoint | what |
|---|---|
| `GET /vram` | `state` (WARMUP, FULL, TRIM, MIN, GROW, PAUSING, PAUSED), `reason`, `ready`, `card` (`free_mib`, `total_mib`, `min_free_mib_window` since the last call), `fixed_mib` (weights, KV, draft, pool, scheduler; measured after warm-up), `cache` (chunks, slots, MiB, `ubatch_interrupts`, `pause_cuts`, the last release wait), `policy`, `counters`, `host` (RSS parts, pinned memory, system memory and pressure) |
| `GET`/`POST /vram/policy` | the policy above; POST takes any subset, applies it atomically or answers 400 and changes nothing. Fields: `reserve_mib` 256..4096, `buffer_mib` 0..2048, `hysteresis_mib` 0..1024, `grow_seconds` 0..600, `pause_floor_mib` 0..2048, `pause_after_ms` 0..600000 (`pause_after_seconds` is the older name and takes fractions), `min_chunks` 0..64, `max_chunks` null or 0..64 (0 = release everything, a game), `prefill_yield`, `poll_ms` 50..1000, `poll_fast_ms` 20..1000, `trim_abort`, `sleep_level` 0..1 (light sleep: the cache goes to zero, the model stays), `human_need` 0..1 and `note` (both only recorded) |
| `POST /vram/pause` | `{"checkpoint": "<file name in --slot-save-path>", "reason": "..."}`: the same pause, on request (202) |
| `GET /vram/journal?since=<id>&limit=<n>` | the last 512 concessions in memory: trims, grows, pauses, sleeps, policy changes, how long the card stayed short |

**The pause frame and resume.** A client streaming a paused answer gets an error frame of type `vram_pause` whose `message` is a JSON
string: `checkpoint`, `slot`, `reason`, `n_past`, `generated_tokens`, `saved`, `tokens_file`. Next to the slot file the server writes
`<checkpoint>.tokens.json`: the exact token ids of the slot, `generated_tokens`, `n_predict_left`, the request's sampling parameters, and
`prompt_done` (false when the pause came while the prompt was still being read). To continue, restart the server, restore the slot
(`POST /slots/0?action=restore {"filename": "<checkpoint>"}`) and send the same chat request again with
`"resume": {"tokens_file": "<checkpoint>.tokens.json", "tool_call_ids": [...]}`: the server continues from the saved tokens and streams
only what is new; when `prompt_done` is false it reads the prompt from the request again and reuses the restored prefix. Two runs of
the same request are not bit-identical on this build even without a pause (the two-channel cache changes which experts run where), so
a resumed answer matches a continuous one in substance, not byte for byte.

Test hooks (`--vram-test-hooks`): `POST /vram/simulate {"free_delta_mib": -1024, "seconds": 5}` and `POST /moe-cache/chunks {"chunks": N}`.
Diagnostics: `GGML_CUDA_LOG_ALLOCS=1` logs every device allocation and release of the CUDA/HIP buffers and pool.

## Known issues

* **bf16 `MUL_MAT_ID` with skipped ids** (`n_used = 1`, `n = 4`, `skip = 1`): in a long `test-backend-ops` series (226 cases with skipped
  ids) it fails in about half the runs with an error of ~0.5, always after the f16 `n = 33/129` cases; alone, or in its family of 18
  cases, it never failed (0 of 20, 0 of 12). It depends on the state of the GPU memory left by earlier cases, not on the case. The
  quantized types the models use do not show it. Not checkable on upstream, which declares `ids = -1` unsupported on CUDA/ROCm.
* **Upstream: `MUL_MAT_ID` with an offset ids view** (`type_a = iq3_xxs`, `n = 8`, ids passed as a view with a 4-byte offset) fails in
  8 of 10 runs on ROCm on plain upstream 74bca3950 too (the mmvq path). Our graphs never pass offset ids views. Not reported upstream yet.
* The CUDA build is kept compiling but is not measured on this base; Vulkan, Metal and SYCL do not get the expert cache, the governor or
  the TurboQuant types.
* Decode steps run as HIP/CUDA graphs and are not interrupted: an urgent trim during decode waits for the current step (about 13 ms
  on the reference machine). Backends other than CPU and CUDA/HIP are stopped only between micro-batches.

## Hook points in upstream files

Grouped by feature; the exact probe strings are in `touchpoints.toml`.

* **turbo-kv**: `ggml.h` (type and op enums), `ggml.c` (op tables, `GGML_OP_COUNT` assertions), `ggml-cpu/` (type traits, `ops.cpp` WHT), `ggml-common.h`, `ggml-quants.h`, `ggml-cuda/` (`common.cuh`, `convert.cu`, `dequantize.cuh`, `fattn*.cu(h)`, `ggml-cuda.cu`, `set-rows.cu`, `quantize.cu`), cmake lists for the FA instances, `llama-graph.cpp` (query pre-rotation in every `build_attn`), gguf-py constants.
* **scheduler-prefetch**: `ggml-backend.cpp` (inside `ggml_backend_sched_compute_splits`, the host-weight copy loop), `ggml-backend.h`, `llama-context.cpp` (`sched_new()`).
* **moe cache**: `llama-graph.cpp/.h` (`build_moe_ffn` pack path and the `moe_cache` argument), `llama-model.cpp/.h` (`init_moe_expert_cache`, resize, stats; this code lives in `llama-model.cpp` because it needs `llama_model::impl`), `src/models/{qwen35moe,gemma4,laguna,deepseek2}.cpp` (last argument of `build_moe_ffn`), `ggml-cuda/{mmvq,mmid,mmq,mmf,mmvf}.cu`, `llama-model-loader.cpp/.h`, `include/llama.h`, `common/{arg.cpp,common.h}`.
* **governor**: `server.cpp` (`server_prepare_params`, `server_register_routes`), `server-context.cpp` (pause task, resume request, abort callback, `rewind_slice` / `rewind_for_pause`), `server-queue.cpp` and `server-task.cpp` (`seed_resume`), `common/arg.cpp`, `llama-mmap.cpp` (`O_DIRECT`), `ggml-cuda/{common.cuh,ggml-cuda.cu}` (abort callback, allocation log), `llama-context.cpp` (abort between micro-batches), `llama-model.cpp` and `include/llama.h` (release wait in the resize).
* **tests and tools**: `tests/test-backend-ops.cpp` (cases for `ids == -1`, FA shapes, the fused op), `tools/llama-bench` (`--sched-async-cpu`).

## Known untidy places

* `llama-model.cpp` carries about 1,100 lines of the expert cache. They cannot move to their own translation unit without moving the private `llama_model::impl` into a header, which would touch far more of upstream than it saves.
* In `build_moe_ffn` the pack path sits next to upstream's slot-cache code inside `if (!use_moe_packs) { ... }` with unchanged indentation, to keep the diff small.
