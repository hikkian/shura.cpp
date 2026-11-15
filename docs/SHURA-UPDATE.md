# Moving shura.cpp to a newer llama.cpp

Manual, on purpose: there is no automatic pipeline. Count on a few hours of porting plus GPU verification runs. The 1.0 port
(REBASE1) took about a day of porting and ~10 hours of A/B on the card.

## 1. Rebase

```bash
git fetch https://github.com/ggml-org/llama.cpp master:upstream-new      # or a release tag
git checkout shura-v1 && git checkout -b shura-v1-new
git rebase --onto upstream-new <the upstream base of shura-v1> shura-v1-new   # one conflict = one part (commit title)
```

`git config rerere.enabled true` helps. When upstream implemented something we have, take upstream's version and drop ours; write
the dropped part and the reason into `docs/SHURA.md` ("Upstream's parts that replace ours").

## 2. The hook points

```bash
python3 scripts/shura-touchpoints.py          # every probe in touchpoints.toml must be found
```

A missing probe means upstream moved the code: find the new place, fix the hook and the probe. Places that move most often:

* `src/llama-moe-cache.cpp`: upstream's class is the base of our resize; check the constructor (our `layout()`/`apply_layout()` were cut
  out of it), `moe_cache_lru::init` (our `resize_slots`), `copy()` and `copy_experts()`
* `src/llama-context.cpp`: `process_ubatch` (micro-batch lock and lazy reserve), `sched_reserve` (lock, generation, compute snapshot),
  `memory_update` (lock), the ubatch loop of `decode` (abort between ubatches), `graph_params` (cache generation)
* `src/llama-graph.h`: `llm_graph_params::allow_reuse` (cache generation)
* `ggml/src/ggml-cuda/ggml-cuda.cu`: the node loop of `ggml_cuda_graph_evaluate_and_capture` (abort marks), the pools and host buffers
* `ggml/src/ggml-cuda/fattn-vec.cuh`: the GQA-packed kernel (`ncols2`) and its dispatch
* `ggml/src/ggml-backend.cpp`: the host-weight copy loop of `ggml_backend_sched_compute_splits` (expert upload overlap)
* the server: `server_context_impl` (governor members, pause task, `decode()` retry), `handle_completions_impl` (resume), routes

## 3. Build and test without a GPU

```bash
cmake -S . -B build-new -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx1101 \
      -DCMAKE_HIP_COMPILER=$HOME/rocm-10.1/lib/llvm/bin/clang -DCMAKE_PREFIX_PATH=$HOME/rocm-10.1 -DLLAMA_CURL=OFF \
      -DLLAMA_BUILD_TESTS=ON -DGGML_CUDA_FA_QUANTS="q8_0-q8_0;q8_0-q4_0;q4_0-q4_0;q4_0-q8_0;q5_0-q5_0;q5_0-q4_0;f16-f16;bf16-bf16"
cmake --build build-new -j 3
cd build-new && HIP_VISIBLE_DEVICES=-1 ctest            # torch-based Python tests need torch
```

## 4. On the GPU (one model at a time, the VRAM and RAM watchdogs on)

* `test-backend-ops test -b ROCm0` (all ops; `GGML_HIP_FA_GQA_VEC=1 ... -o FLASH_ATTN_EXT` for the GQA kernel)
* `test-moe-cache-resize` on a small MoE model (`test-llama-archs -a qwen35moe -o DIR`), `test-vram-abort` on `-a llama`
* the A/B of REBASE1 (`reports/rebase1/ab/ab.py`, `gov_ab.py`, `analyze.py`): interleaved pairs against the current engine, the rule
  of `reports/rebase1/PREREG.md`

## 5. Notes for the next update

- **KV on virtual memory (`SHURA_KV_VMM`).** The hooks are in `touchpoints.toml` (kv-vmm). After an update check: the hybrid `apply()` still short-circuits (attention first, the recurrent part only if it succeeded), `prepare()` maps the whole batch before any ubatch, and the T=0 tokens with and without VMM match (no governor, cache 0). Re-mapping addresses that kernels used lost writes on gfx1101 / ROCm 10.1; keep grow-only plus the move to a fresh range until a ROCm update is tested.

* KV: upstream's rotated `q4_0`/`q8_0` (live) and `q5_0` (long profile) are in use; a new KV type is measured with
  `reports/rebase1/kv/kldtypes.sh` (`SHURA_PPL_STREAM=1`, 16k and 32k) and needs the GQA kernel for decode speed.
* After the rebase check that `flash_attn_ext_f16` still calls the tile function with a literal `nullptr` without KV rows
  (touchpoint `wmma-null-rows`): `llama-bench -p 1024 -d 100000` must stay at the earlier speed.
* No own ggml type numbers are left in the tree (TurboQuant was dropped). If one comes back, take numbers from 64 up and move them when
  upstream gets close.
