# shura.cpp

**shura.cpp is the inference engine of [Shura](https://github.com/hikkian/shura)**: llama.cpp for running a large MoE model on one
consumer GPU next to a desktop that also needs the card.

shura.cpp 1.0 is [llama.cpp](https://github.com/ggml-org/llama.cpp) **v1.0.0** with only the parts upstream does not have, added on top
of upstream's own mechanisms:

* **live resize of upstream's expert cache** with rollback (a failed allocation restores the previous size, then an empty cache, and the
  server keeps answering), warm refill, an empty state and atomic statistics;
* a **VRAM governor** in the server that grows the cache into free VRAM and gives it back within a fraction of a second when the desktop
  needs the card, with pause/resume of a request when even the smallest cache does not fit;
* the **expert upload overlap** and the **GQA-vectorised decode attention** (HIP) that keep prefill and long-context decode at the
  speed of the earlier engine;
* a pinned host memory limit, an abort check in the CUDA/HIP backend, reasoning-budget extensions (ThinkBrake), failure diagnostics.

What exactly differs from upstream, the recommended settings and the known limits: [docs/SHURA.md](docs/SHURA.md). Updating to a newer
llama.cpp: [docs/SHURA-UPDATE.md](docs/SHURA-UPDATE.md). Sources and licenses: [NOTICE](NOTICE). Upstream's own README:
[docs/UPSTREAM-README.md](docs/UPSTREAM-README.md).

## Build (AMD, ROCm)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx1101 -DLLAMA_CURL=OFF \
      -DGGML_CUDA_FA_QUANTS="q8_0-q8_0;q8_0-q4_0;q4_0-q4_0;q4_0-q8_0;f16-f16;bf16-bf16"
cmake --build build --target llama-server
```

CUDA: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<arch>` instead of the HIP options (kept compiling by CI; the governor needs amdgpu's
sysfs and is unavailable on NVIDIA).

## License

MIT, as llama.cpp. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
