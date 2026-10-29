// PINFAIL1: a refused pinned host allocation falls back to pageable memory with one warning, fails with a message when even that does not fit,
// and a fault inside the GPU runtime ends the process with exit code 70 (other faults keep their crash). Needs a CUDA/HIP device, skips without.
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>

#if defined(__linux__)
#    include <dlfcn.h>
#    include <sys/wait.h>
#    include <unistd.h>
#endif

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "%s:%d: check failed: %s\n", __FILE__, __LINE__, #cond);   \
            return 1;                                                                  \
        }                                                                              \
    } while (0)

typedef void (*host_stats_t)(uint64_t *, uint64_t *);

static const size_t MiB = 1024 * 1024;

static bool is_pinned(ggml_backend_buffer_t buf, ggml_backend_buffer_type_t host_buft) {
    return buf != nullptr && ggml_backend_buffer_get_type(buf) == host_buft;
}

// the buffer works: written, read back
static bool usable(ggml_backend_buffer_t buf) {
    uint8_t * p = (uint8_t *) ggml_backend_buffer_get_base(buf);
    const size_t n = ggml_backend_buffer_get_size(buf);
    p[0] = 1;
    p[n - 1] = 2;
    return p[0] == 1 && p[n - 1] == 2;
}

#if defined(__linux__)
struct dim3_like { unsigned x, y, z; };

// child: pins memory (which installs the guard), then makes the GPU runtime fault; returns the wait status
static int run_child(bool runtime_fault, ggml_backend_buffer_type_t host_buft) {
    fflush(nullptr);
    const pid_t pid = fork();
    if (pid == 0) {
        ggml_backend_buffer_t b = ggml_backend_buft_alloc_buffer(host_buft, 4 * MiB);
        if (b == nullptr) {
            _exit(3);
        }
        if (runtime_fault) {
            using launch_t = int (*)(const void *, dim3_like, dim3_like, void **, size_t, void *);
            launch_t launch = (launch_t) dlsym(RTLD_DEFAULT, "hipLaunchKernel");
            if (launch == nullptr) {
                launch = (launch_t) dlsym(RTLD_DEFAULT, "cudaLaunchKernel");
            }
            if (launch == nullptr) {
                _exit(4);
            }
            launch((const void *) 0x1234, {1, 1, 1}, {1, 1, 1}, nullptr, 0, nullptr);
            _exit(5); // the runtime rejected the bogus handle without faulting: nothing to test
        }
        volatile int * p = nullptr;
        *p = 1;
        _exit(6);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    return status;
}
#endif

int main() {
    ggml_backend_load_all();
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    ggml_backend_buffer_type_t host_buft = dev ? ggml_backend_dev_host_buffer_type(dev) : nullptr;
    if (dev == nullptr || host_buft == nullptr) {
        printf("test-pinned-fallback: no GPU with host buffers, skipped\n");
        return 0;
    }
    host_stats_t stats = (host_stats_t) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_cuda_host_stats");
    if (stats == nullptr) {
        printf("test-pinned-fallback: not a CUDA/HIP device, skipped\n");
        return 0;
    }
    unsetenv("SHURA_FAKE_PIN_FAIL");
    unsetenv("SHURA_FAKE_MEM_AVAILABLE_MIB");
    unsetenv("SHURA_PIN_RESERVE_MIB");
    unsetenv("SHURA_PIN_CHECK");
    uint64_t pinned = 0, denied0 = 0, denied = 0;

    // 1. nothing wrong: pinned
    {
        ggml_backend_buffer_ptr buf(ggml_backend_buft_alloc_buffer(host_buft, 16 * MiB));
        CHECK(is_pinned(buf.get(), host_buft) && usable(buf.get()));
        stats(&pinned, &denied0);
        CHECK(pinned >= 16 * MiB);
    }

    // 2. every pinned allocation fails: pageable memory, still a working host buffer, the refusal is counted
    setenv("SHURA_FAKE_PIN_FAIL", "1", 1);
    {
        ggml_backend_buffer_ptr buf(ggml_backend_buft_alloc_buffer(host_buft, 16 * MiB));
        CHECK(buf != nullptr && !is_pinned(buf.get(), host_buft));
        CHECK(ggml_backend_buffer_is_host(buf.get()) && usable(buf.get()));
        stats(&pinned, &denied);
        CHECK(denied == denied0 + 1);
    }

    // 3. a limit of 8 MiB pinned in total: the 4 MiB is pinned, the 16 MiB after it is not
    setenv("SHURA_FAKE_PIN_FAIL", "after=8", 1);
    {
        uint64_t base = 0, d = 0;
        stats(&base, &d);
        ggml_backend_buffer_ptr small(ggml_backend_buft_alloc_buffer(host_buft, 4 * MiB));
        CHECK(is_pinned(small.get(), host_buft));
        ggml_backend_buffer_ptr big(ggml_backend_buft_alloc_buffer(host_buft, 16 * MiB));
        CHECK(big != nullptr && !is_pinned(big.get(), host_buft) && usable(big.get()));
    }
    unsetenv("SHURA_FAKE_PIN_FAIL");

    // 4. the request does not fit MemAvailable with the reserve: pageable memory (the RAM is there)
    setenv("SHURA_FAKE_MEM_AVAILABLE_MIB", "1500", 1);
    {
        ggml_backend_buffer_ptr buf(ggml_backend_buft_alloc_buffer(host_buft, 600 * MiB));
        CHECK(buf != nullptr && !is_pinned(buf.get(), host_buft) && usable(buf.get()));
    }
    // 5. not even pageable memory would fit: no buffer, the caller gets a clear failure instead of a swap storm
    setenv("SHURA_FAKE_MEM_AVAILABLE_MIB", "700", 1);
    {
        ggml_backend_buffer_ptr buf(ggml_backend_buft_alloc_buffer(host_buft, 600 * MiB));
        CHECK(buf == nullptr);
        // a small buffer is still served
        ggml_backend_buffer_ptr small(ggml_backend_buft_alloc_buffer(host_buft, 4 * MiB));
        CHECK(small != nullptr);
    }
    // 6. the budget check can be turned off
    setenv("SHURA_PIN_CHECK", "0", 1);
    {
        ggml_backend_buffer_ptr buf(ggml_backend_buft_alloc_buffer(host_buft, 16 * MiB));
        CHECK(is_pinned(buf.get(), host_buft));
    }
    unsetenv("SHURA_PIN_CHECK");
    unsetenv("SHURA_FAKE_MEM_AVAILABLE_MIB");

#if defined(__linux__)
    // 7. a fault inside the GPU runtime: exit code 70 with a message; a fault in this code: still a crash
    {
        const int st = run_child(true, host_buft);
        if (WIFEXITED(st) && WEXITSTATUS(st) == 5) {
            printf("test-pinned-fallback: the runtime rejects a bogus kernel handle without faulting, fault guard not exercised\n");
        } else {
            CHECK(WIFEXITED(st) && WEXITSTATUS(st) == 70);
        }
        const int st2 = run_child(false, host_buft);
        CHECK(WIFSIGNALED(st2) && WTERMSIG(st2) == SIGSEGV);
    }
#endif

    printf("test-pinned-fallback: OK\n");
    return 0;
}
