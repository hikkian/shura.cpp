# Moving shura.cpp to a newer llama.cpp

Manual, on purpose: there is no automatic pipeline. Count on a few hours plus one GPU verification run.

## 1. Rebase

```bash
git fetch https://github.com/ggml-org/llama.cpp master:upstream-new      # or: git fetch origin master
git checkout -B upstream upstream-new                                    # the mirror moves forward
git checkout shura && git checkout -b shura-new
git rebase -i upstream                                                   # one conflict = one feature (commit title)
```

Tips: `git config rerere.enabled true`. Conflicts where our block is a pure addition can be resolved by keeping both sides.
When upstream implemented something we have (check the commit title against the PR list), take upstream's version and drop ours:
write the dropped patch and the reason into `reports/REPORT-<card>.md`. Likely candidates to re-check each time:
slot checkpoints, expert cache, `mul_mat_id` skip of `-1`, flash-attention changes, `fwht`, MTP export.

## 2. Check that the hooks survived

```bash
scripts/shura-touchpoints.py          # every probe in touchpoints.toml must be found
```

A missing probe means upstream moved the code; find the new place and fix the hook or the probe.

Things that broke on previous rebases (look for them first):

* `GGML_OP_COUNT` static assertions in `ggml.c` (we add one op; bump both)
* `build_moe_ffn` signature (our `moe_cache` argument is last, after `post_exp_norm_eps`; every model that passes it needs the new arguments)
* new overloads of `build_attn` in `llama-graph.cpp` (add the turbo query rotation to each)
* flash-attention vec dispatch (`ggml_cuda_get_fattn_vec_case`, the instance lists in `ggml/cmake/common.cmake` and `ggml/CMakeLists.txt`)
* the swizzle helpers in `fattn-mma-f16.cuh` (our turbo tile loaders call upstream's `swizzle<stride>()`)
* server refactors (`server_prepare_params`, `server_register_routes`, `task_result_state`, `common_params::hostnames`)

## 3. Build

```bash
export R=~/rocm-7.2.4/opt/rocm-7.2.4        # the isolated ROCm; the system one leaks graph memory
HIPCXX=$R/llvm/bin/clang HIP_PATH=$R cmake -S . -B /dev/shm/shura-build -G Ninja -DGGML_HIP=ON -DGPU_TARGETS=gfx1101 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$R -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_UI=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DGGML_HIP_NO_VMM=ON
nice -n19 ionice -c3 cmake --build /dev/shm/shura-build -j4 --target llama-server llama-bench test-backend-ops
```

Build in `/dev/shm` (the full build writes several GB), delete it afterwards. About 410 steps; do not run two builds at once.
Quick syntax check of a single file before a long build: `g++ -std=c++17 -fsyntax-only -Isrc -Iinclude -Iggml/include -Icommon -Ivendor <file>`.

## 4. Checks (no model needed)

```bash
build/bin/test-backend-ops test -o MUL_MAT_ID        # includes the ids == -1 cases
build/bin/test-backend-ops test -o FLASH_ATTN_EXT
build/bin/test-backend-ops test -o TURBO_WHT
build/bin/test-backend-ops --moe-channel-policy-test
build/bin/test-moe-cache-policy
build/bin/test-vram-governor                          # decision core
```

## 5. Checks with a model (GPU, ask first)

Probes are the same as for any release candidate of the stack: A/B pairs for Tiel and Occamy against the current build, KL against
the reference (limit 0.0385), VRAM at idle and at the ends of the context range, a 40 minute hammer with no GPU page fault in
`journalctl -k`, the governor jump test (+400 MiB taken from the card, reaction within 1 s), a 30 minute memory run. The scripts are in
`reports/vramlogic1/scripts` and `reports/rocm72/scripts` of the stack repository. The VRAM watchdog (free below 1.5 GiB stops the run)
stays on in every run.

## 6. Finish

Tag `shura-YYYYMMDD`, update `docs/SHURA.md` if features changed, record dropped patches, point the stack's build script at the new commit.
