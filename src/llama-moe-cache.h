#pragma once

#include "ggml-backend.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

// keeps recently and repeatedly used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // cpu_share is the share of the misses computed on the CPU instead of uploaded
    // SHURA: resizable: the cache may be empty (size 0, or too small for one token) instead of failing, and its layout may change
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, bool resizable = false);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    // cpu_part: the graph computes the misses in the CPU map of the layer (see get_cpu_map)
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the experts of layer il computed on the CPU: I32 [1, n_expert] in host memory, e for these experts, -1 for the others
    // nullptr if no misses of layer il are computed on the CPU for ubatches of n_tokens tokens
    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // ---- SHURA: observability and run-time settings; safe to call from another thread (atomic snapshot) ----
    struct info {
        size_t   size        = 0;     // bytes of the device banks now
        size_t   host_bytes  = 0;     // bytes of the host experts the cache can serve
        int32_t  n_slots     = 0;     // slots of all groups
        int32_t  n_groups    = 0;     // groups of layers that can be cached
        int32_t  n_groups_on = 0;     // groups with slots now
        bool     empty       = true;
        bool     cpu_part    = false; // created with cpu_share > 0: the share can change at run time
        float    cpu_share   = 0.0f;
        uint64_t generation  = 0;     // bumps at every layout change
        uint64_t hits_small = 0, misses_small = 0, cpu_small = 0, bytes_small = 0; // ubatches of up to 8 tokens
        uint64_t hits_large = 0, misses_large = 0, cpu_large = 0, bytes_large = 0;
        uint64_t copy_hits  = 0, copy_bytes   = 0;                                // large batches: experts copied from the cache
    };
    info     get_info() const;
    size_t   device_size() const; // bytes of the device banks now
    bool     empty() const;
    uint64_t generation() const;
    // false if the cache was created without a CPU part (cpu_share == 0)
    bool     set_cpu_share(float share);

    // ---- SHURA: live resize (design §3.3 В, Г, З). Only through llama_context, which holds the micro-batch lock and has
    // synchronized the backends: nothing running or queued may read the banks. ----
    bool     resizable() const;
    // the smallest size at which every group serves ubatches of n_tokens from the cache (<= 0: the size that caches every expert)
    size_t   size_for_tokens(int64_t n_tokens) const;
    // returns false if the layout did not change; on an allocation failure the previous size, then the empty state
    bool     resize(size_t size);
    // called between the release of the old banks and the allocation of the new ones, with the bytes released
    void     set_release_wait(void (*fn)(size_t freed_bytes, void * data), void * data);
    // test hook: the next n allocations of a non-empty layout fail
    void     set_test_fail_allocs(int32_t n);
    // the host copy of expert e of layer il changed: the cached copy is read again
    bool     invalidate(int32_t il, int32_t e);
    // tests: the LRU lists, slot maps and (contents) the banks agree with each other and with the host experts
    bool     self_check(bool contents, std::string & why) const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
