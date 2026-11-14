#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

// ClassLRU: one LRU list per frequency class, over the experts of a group of layers
// slot_map[il][e] is the slot of expert e of layer il, n_slots or later if not cached, the host slot map tensor
// CRF (combined recency and frequency) = uses of an expert, decayed with a half-life of N tokens, retained across evictions
// class = floor(log2(CRF)), at most n_class - 1; each slot is in the LRU list of the class, as of its last use
// victim: an empty slot, or the least recently used slot of each list not in the current plan() with the max age/2^class
struct moe_cache_lru {
    static constexpr int   n_class  = 8; // floor(log2(1/(1 - 2^(-1/crf_half_life)))) = 7, class of CRF_max for half-life of 128
    static constexpr float crf_half_life = 128.0f; // tokens

    int32_t n_expert = 0;
    int32_t n_slots  = 0;
    int32_t empty    = 0; // slot of the experts that are not cached

    // [n_layer] data of the slot maps, the experts at slot n_slots or later are not cached
    // the experts computed on the CPU read zero slots after the cached ones on the device
    std::vector<int32_t *> slot_map;
    std::vector<int32_t>   key_of;   // [n_slots] key of the expert in the slot, -1 if empty

    // one LRU list per class, the head is the least recently used end
    std::vector<int32_t> prev;    // [n_slots] -1 at the ends
    std::vector<int32_t> next;    // [n_slots]
    int32_t head[n_class];
    int32_t tail[n_class];
    std::vector<int8_t>  list_of; // [n_slots] class list of the slot
    int32_t n_filled = 0;         // slots that hold an expert

    std::vector<float>   crf;      // [n_layer*n_expert]
    std::vector<int64_t> last_use; // [n_layer*n_expert] layer clock at the last use, -1 if not used
    std::vector<int64_t> clock;    // [n_layer] tokens seen by the layer (as added by plan() for each ubatch)

    std::vector<uint32_t> seen; // [n_expert] seen[e] == seen_gen if expert e is selected by the current plan()
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;  // the distinct selected experts
    std::vector<int32_t> misses;
    std::vector<uint64_t> token_zero; // [n_tokens] zero slots used by each token

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        GGML_ASSERT(n_class == (int) std::log2(1.0f/(1.0f - std::exp2(-1.0f/crf_half_life))) + 1);
        this->n_expert = n_expert;
        slot_map.assign(n_layer, nullptr);
        crf.assign((size_t) n_layer*n_expert, 0.0f);
        last_use.assign((size_t) n_layer*n_expert, -1);
        clock.assign(n_layer, 0);
        seen.assign(n_expert, 0);
        resize_slots(n_slots);
    }

    // SHURA: n_slots empty slots in one list; the CRF, last use and clocks of all experts are kept (init() would zero them), so after a
    // resize the experts are ranked as before. The slot maps (owned by the cache) must be set to the new `empty` by the caller.
    // n_slots may be 0: a group that is off.
    void resize_slots(int32_t n) {
        n_slots  = n;
        empty    = n;
        n_filled = 0;
        key_of.assign(n, -1);
        prev.resize(n);
        next.resize(n);
        for (int32_t s = 0; s < n; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n ? s + 1 : -1;
        }
        for (int c = 0; c < n_class; ++c) {
            head[c] = -1;
            tail[c] = -1;
        }
        if (n > 0) {
            head[0] = 0;
            tail[0] = n - 1;
        }
        list_of.assign(n, 0);
    }

    // move slot s to the most recently used end of list c
    void move(int32_t s, int c) {
        const int from = list_of[s];
        if (from == c && s == tail[from]) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head[from] = next[s];
        }
        if (next[s] >= 0) {
            prev[next[s]] = prev[s];
        } else {
            tail[from] = prev[s];
        }

        prev[s] = tail[c];
        next[s] = -1;
        if (tail[c] >= 0) {
            next[tail[c]] = s;
        } else {
            head[c] = s;
        }
        tail[c] = s;
        list_of[s] = c;
    }

    // key = index of an expert of a layer in crf and last_use
    int32_t make_key(int32_t il, int32_t e) const { return il*n_expert + e; }
    int32_t key_layer (int32_t key) const { return key / n_expert; }
    int32_t key_expert(int32_t key) const { return key % n_expert; }

    // for a key after use(): its CRF is >= 1, so log2 is >= 0 and the cast is the floor
    int class_of(int32_t key) const {
        return std::min(n_class - 1, (int) std::log2(crf[key]));
    }

    // the CRF of a key, decayed by the time since its last use
    float decayed_crf(int32_t il, int32_t key) const {
        return crf[key]*std::exp2(-(clock[il] - last_use[key])/crf_half_life);
    }

    // an unused key gets 1 because crf[key] is 0 prior to first use
    void use(int32_t il, int32_t key) {
        crf[key]      = decayed_crf(il, key) + 1.0f;
        last_use[key] = clock[il];
    }

    // the experts selected by this plan() must not be evicted, nor the slots this plan() filled already
    bool is_selected(int32_t il, int32_t key) const {
        if (key < 0) {
            return false; // unused slot
        }
        if (key_layer(key) != il) {
            return false; // seen is per expert index, an expert of another layer can have the same index
        }
        return seen[key_expert(key)] == seen_gen;
    }

    // the least recently used slot of list c that this plan() does not select, -1 if none
    int32_t candidate(int c, int32_t il) const {
        for (int32_t s = head[c]; s >= 0; s = next[s]) {
            if (!is_selected(il, key_of[s])) {
                return s;
            }
        }
        return -1;
    }

    int32_t victim(int32_t il) const {
        int32_t res = -1;
        double res_score = -1.0;
        for (int c = 0; c < n_class; ++c) {
            const int32_t s = candidate(c, il);
            if (s < 0) {
                continue;
            }
            const int32_t key = key_of[s];
            if (key < 0) {
                return s; // unused slot
            }
            const int64_t age   = clock[key_layer(key)] - last_use[key];
            const double  score = std::ldexp((double) age, -c); // age/2^class
            if (score > res_score) {
                res       = s;
                res_score = score;
            }
        }
        if (res < 0) {
            GGML_ABORT("no MoE cache victim");
        }
        return res;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids (n_tokens tokens) in layer il
    // n_cpu(n) of the n misses are left to the CPU and read a zero slot on the device, they are not cached
    // a zero slot is used at most once per token, a miss that does not get one is cached instead
    // returns false if the ids select more distinct experts than there are slots
    template <typename F>
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, int64_t n_tokens, int32_t n_used, int32_t n_zero, F && n_cpu,
            std::vector<fill> & fills, std::vector<fill> & cpu, size_t & n_hit) {
        fills.clear();
        cpu.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // update the CRF of all selected experts and promote hits before choosing victims for misses
        misses.clear();
        for (int32_t e : uniq) {
            const int32_t key = make_key(il, e);
            use(il, key);
            if (slots[e] < n_slots) {
                move(slots[e], class_of(key));
                n_hit++;
            } else {
                misses.push_back(e);
            }
        }

        // the misses with the lowest CRF are left to the CPU, a miss that keeps being used gains CRF and enters the cache
        // all misses are used at the same clock, so their CRF compare without decay
        const size_t n_c = std::min(n_cpu(misses.size()), misses.size());
        if (n_c > 0) {
            std::sort(misses.begin(), misses.end(), [&](int32_t a, int32_t b) {
                const float crf_a = crf[make_key(il, a)];
                const float crf_b = crf[make_key(il, b)];
                return crf_a != crf_b ? crf_a < crf_b : a < b;
            });
            GGML_ASSERT((size_t) n_tokens*n_used == n_ids);
            token_zero.assign(n_tokens, 0);
            size_t n_keep = n_c;
            for (size_t i = 0; i < n_keep; ) {
                const int32_t e = misses[i];
                uint64_t used = 0;
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            used |= token_zero[it];
                        }
                    }
                }
                int32_t z = 0;
                while (z < 64 && (used >> z & 1)) {
                    z++;
                }
                if (z >= n_zero) {
                    std::swap(misses[i], misses[--n_keep]);
                    continue;
                }
                for (int64_t it = 0; it < n_tokens; ++it) {
                    for (int32_t k = 0; k < n_used; ++k) {
                        if (ids[it*n_used + k] == e) {
                            token_zero[it] |= 1ull << z;
                        }
                    }
                }
                slots[e] = n_slots + z;
                cpu.push_back({ e, n_slots + z });
                i++;
            }
            misses.erase(misses.begin(), misses.begin() + n_keep);
        }

        // fill the misses in expert order: victim() takes unused slots, and slots filled by one plan(), in list order,
        // so consecutive experts often get consecutive slots and one upload
        std::sort(misses.begin(), misses.end());
        for (int32_t e : misses) {
            const int32_t s = victim(il);
            if (key_of[s] >= 0) {
                slot_map[key_layer(key_of[s])][key_expert(key_of[s])] = empty;
            } else {
                n_filled++;
            }
            const int32_t key = make_key(il, e);
            key_of[s] = key;
            slots[e]  = s;
            move(s, class_of(key));
            fills.push_back({ e, s });
        }
        clock[il] += n_tokens;
        return true;
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget, fixed at creation (SHURA: by free memory at creation, see layout())

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;              // 0: the group is off (SHURA: a group can be switched on by a later layout)
        moe_cache_lru lru;
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached now
        int32_t ig_cand = -1;               // SHURA: the group of the layer when the group has slots, -1 if the layer is never cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        ggml_tensor * slot_map_cpu = nullptr; // looked up in place of slot_map by the graphs with a CPU part, if cpu_share > 0, its data is not used
        ggml_tensor * cpu_map  = nullptr;   // I32 [1, n_expert] in host memory, if cpu_share > 0
        std::vector<int32_t> cpu_experts;   // experts set in cpu_map
        uint32_t cpu_acc = 0;               // fraction of a miss carried over to the next plan, in 1/65536
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    // SHURA: counters are atomic so that another thread (the server's /vram, the VRAM governor) reads them without a lock
    struct stats {
        std::atomic<uint64_t> hits   {0};
        std::atomic<uint64_t> misses {0};
        std::atomic<uint64_t> cpu    {0}; // misses computed on the CPU
        std::atomic<uint64_t> bytes  {0};
    };

    static constexpr int64_t max_batch = 32;

    int32_t n_expert_used;
    int32_t n_layer;
    bool    resizable = false; // SHURA: the cache may be empty and its size may change (llama_context_params::moe_cache_resizable)

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, std::pair<int32_t, bool>> layer_of; // slot map -> layer, and if the graph that looks it up has a CPU part

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;
    std::vector<moe_cache_lru::fill> cpu;

    std::atomic<float>    cpu_share{0.0f};
    std::atomic<uint32_t> cpu_share_q{0}; // cpu_share in 1/65536, integer so that a simulator can reproduce the rounding
    int32_t  n_zero    = 0; // zero slots per group
    double   split_sum = 0.0;
    bool     no_alloc  = false;

    // SHURA: the geometry as a snapshot for other threads, written by apply_layout()
    std::atomic<size_t>   snap_size{0};
    std::atomic<size_t>   snap_host_bytes{0};
    std::atomic<int32_t>  snap_slots{0};
    std::atomic<int32_t>  snap_groups_on{0};
    std::atomic<uint64_t> generation{0};
    mutable std::mutex    snap_mu;                                  // guards snap_breakdown
    std::map<ggml_backend_buffer_type_t, size_t> snap_breakdown;    // device banks and host slot maps, as memory_breakdown() returns

    // LLAMA_MOE_CACHE_TRACE: the selected experts of each slot lookup, for a simulator of the cache
    FILE * trace = nullptr;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    // with misses on the CPU, the n_zero slots after the cached ones stay zero: the device computes these misses as zero
    // one extra slot at the end, CUDA MMQ can read past the last expert
    size_t alloc_size(const group & g, int32_t n_slots) const {
        const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
        size_t res = 0;
        for (const ggml_tensor * t : g.ref) {
            res += GGML_PAD(t->nb[2]*(n_slots + n_zero + 1), alignment);
        }
        return res;
    }

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share_in, bool resizable) :
            n_expert_used(model.hparams.n_expert_used_max()), n_layer((int32_t) model.layers.size()), resizable(resizable),
            layers(model.layers.size()), cpu_share(std::clamp(cpu_share_in, 0.0f, 1.0f)), no_alloc(no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        n_zero = cpu_share.load() > 0.0f ? std::min<int32_t>(n_expert_used, 64) : 0;

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        // SHURA: the split is taken once, here: after a later release of the banks the free memory would give another split
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        cpu_share_q = (uint32_t) std::lround(cpu_share.load()*65536.0f);
        if (const char * path = getenv("LLAMA_MOE_CACHE_TRACE"); path != nullptr && *path != '\0') {
            trace = fopen(path, "wb");
            if (trace == nullptr) {
                throw std::runtime_error(std::string("failed to open the MoE cache trace ") + path);
            }
            write_trace_header(split_sum);
        }

        const std::vector<int32_t> n_slots = layout(size);
        const bool any_on = std::any_of(n_slots.begin(), n_slots.end(), [](int32_t n) { return n > 0; });
        if (!any_on && !resizable) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        // SHURA: the host slot maps of every group that can be cached exist from the start, also of the groups that are off now,
        // so that a later layout can switch a group on (they are small: three I32 [n_expert] per layer)
        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ std::max<size_t>(n_tensors, 1)*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        size_t n_tensors_host = 0;
        for (const group & g : groups) {
            n_tensors_host += g.layers.size()*(n_zero > 0 ? 3 : 1); // slot maps and CPU map
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig_cand = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = { il, false };
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (n_zero > 0) {
                    l.slot_map_cpu = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.slot_map_cpu, "moe_cache.slot_map_cpu-%d", il);
                    layer_of[l.slot_map_cpu] = { il, true };
                    buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map_cpu), alignment_host);
                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);
                }
            }
            // the LRU state (CRF and last use of every expert of the layers) lives as long as the cache; the slots follow the layout
            g.lru.init(n_layer, g.ref[0]->ne[2], 0);
        }
        buf_host = llama_memory_bufs::alloc_buf(ctx_host.get(), buft_host, no_alloc, "MoE cache slot maps");
        if (!no_alloc) {
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // the CPU maps are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                }
            }
        }
        // as weights, the ops that read the slot maps run on the device and the slot maps are copied with the copy callback
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        size_t host_bytes = 0;
        for (const device & d : devices) {
            host_bytes += d.host_bytes;
        }
        snap_host_bytes = host_bytes;

        apply_layout(n_slots);

        if (const char * fail = getenv("GGML_MOE_CACHE_TEST_FAIL_RESIZE_ALLOCS_ONCE"); fail != nullptr) {
            test_fail_allocs = std::max(0, atoi(fail)); // the next n allocations of a resize fail (test hook)
        }
    }

    // SHURA: the slots of each group for a cache of size bytes (0 = the group is off), as upstream's constructor computed them
    std::vector<int32_t> layout(size_t size) const {
        std::vector<int32_t> res(groups.size(), 0);
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            const device & d = devices[g.id];
            if (d.host_bytes == 0 || split_sum == 0.0) {
                continue;
            }
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            int32_t n = 0;
            while (n < max_slots && alloc_size(g, n + 1) <= budget) {
                n++;
            }
            if (n < n_expert_used) {
                if (size > 0) {
                    LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                }
                n = 0;
            }
            res[ig] = n;
        }
        return res;
    }

    // SHURA: create the banks and the bindings for the slots of each group (n_slots from layout()), with the device buffers unallocated
    // before the call (at creation, or after release_banks()). A group with 0 slots is off: its layers read the host experts.
    void apply_layout(const std::vector<int32_t> & n_slots) {
        GGML_ASSERT(n_slots.size() == groups.size());
        bindings.clear();
        for (device & d : devices) {
            GGML_ASSERT(!d.buf && "apply_layout() with the banks still allocated");
            d.ctx.reset();
            d.buf_size = 0;
        }
        std::vector<size_t> n_tensors(devices.size(), 0);
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            if (n_slots[ig] > 0) {
                n_tensors[groups[ig].id] += groups[ig].ref.size()*(1 + groups[ig].layers.size());
            }
        }
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                ggml_init_params params = {
                    /*.mem_size   =*/ n_tensors[id]*ggml_tensor_overhead(),
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                devices[id].ctx.reset(ggml_init(params));
                if (!devices[id].ctx) {
                    throw std::runtime_error("failed to create the MoE cache context");
                }
            }
        }

        int32_t total_slots = 0;
        int32_t groups_on   = 0;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            g.banks.clear();
            g.n_slots = n_slots[ig];
            g.lru.resize_slots(g.n_slots);
            if (g.n_slots == 0) {
                for (int32_t il : g.layers) {
                    layers[il].ig = -1;
                }
                continue;
            }
            total_slots += g.n_slots;
            groups_on++;
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_zero + 1);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig = (int32_t) ig;
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + n_zero, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            d.buf = llama_memory_bufs::alloc_buf(d.ctx.get(), d.buft, no_alloc, "MoE cache");
            if (!no_alloc) {
                // the zero slots and the extra slot at the end must hold zeros
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            // as weights, the ops that read the banks run on the device
            ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }

        // the slot maps of all groups start empty for the new slots; the CPU maps stay -1 (set by plan() only)
        if (!no_alloc) {
            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    int32_t * slots = (int32_t *) layers[il].slot_map->data;
                    std::fill(slots, slots + layers[il].slot_map->ne[1], g.lru.empty);
                    layers[il].cpu_acc = 0;
                }
            }
        }

        size_t size = 0;
        std::map<ggml_backend_buffer_type_t, size_t> breakdown;
        for (const device & d : devices) {
            if (d.buf) {
                size += d.buf_size;
                breakdown[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
            }
        }
        if (buf_host) {
            breakdown[ggml_backend_buffer_get_type(buf_host.get())] += buf_host_size;
        }
        {
            std::lock_guard<std::mutex> lock(snap_mu);
            snap_breakdown = std::move(breakdown);
        }
        snap_size      = size;
        snap_slots     = total_slots;
        snap_groups_on = groups_on;
        generation++;

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts, %.0f%% of the misses on the CPU%s\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0, 100.0*cpu_share.load(),
                groups_on == 0 ? " (empty)" : "");
            for (const group & g : groups) {
                if (g.id == (int32_t) id && g.n_slots > 0) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%, %.1f tokens worth)\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]),
                        (double) g.n_slots/(g.layers.size()*n_expert_used));
                }
            }
        }
    }

    bool empty() const {
        return snap_groups_on.load() == 0;
    }

    ggml_backend_t backend(int32_t il) const {
        // SHURA: a layer of a group that is off has no device in the cache
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        return devices[groups[layers[il].ig].id].backend;
    }

    // trace format, little endian: "MCT1", then u32 n_layer, n_expert_used, n_group
    // per group: u32 device, n_expert, n_layers, the layers (u32), f64 split, split_sum, u64 group host bytes, device host bytes, alignment,
    //            u32 n_tensors, the bytes of one expert of each tensor (u64)
    // then one record per slot lookup: i32 layer, experts per token, n_ids, the ids (i32)
    void write_trace_header(double split_sum) {
        auto u32 = [&](uint32_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto u64 = [&](uint64_t v) { fwrite(&v, sizeof(v), 1, trace); };
        auto f64 = [&](double   v) { fwrite(&v, sizeof(v), 1, trace); };
        fwrite("MCT1", 1, 4, trace);
        u32(layers.size());
        u32(n_expert_used);
        u32(groups.size());
        for (const group & g : groups) {
            const device & d = devices[g.id];
            u32(g.id);
            u32(g.ref[0]->ne[2]);
            u32(g.layers.size());
            for (int32_t il : g.layers) {
                u32(il);
            }
            f64(d.split);
            f64(split_sum);
            u64(g.host_bytes);
            u64(d.host_bytes);
            u64(ggml_backend_buft_get_alignment(d.buft));
            u32(g.ref.size());
            for (const ggml_tensor * t : g.ref) {
                u64(t->nb[2]);
            }
        }
    }

    void write_trace(int32_t il, int32_t n_used) {
        const int32_t rec[3] = { il, n_used, (int32_t) ids.size() };
        fwrite(rec, sizeof(int32_t), 3, trace);
        fwrite(ids.data(), sizeof(int32_t), ids.size(), trace);
    }

    ~impl() {
        log_stats();
        if (trace != nullptr) {
            fclose(trace);
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        GGML_ASSERT(!cpu_part || l.slot_map_cpu != nullptr);
        return cpu_part ? l.slot_map_cpu : l.slot_map;
    }

    ggml_tensor * get_cpu_map(int32_t il, int64_t n_tokens) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].cpu_map == nullptr || layers[il].ig < 0) {
            return nullptr;
        }
        // the scheduler would run the CPU part on a device that offloads such batches from host weights
        ggml_tensor probe = {};
        probe.op    = GGML_OP_MUL_MAT_ID;
        probe.type  = GGML_TYPE_F32;
        probe.ne[0] = 1;
        probe.ne[1] = n_expert_used;
        probe.ne[2] = n_tokens;
        probe.ne[3] = 1;
        for (const device & d : devices) {
            if (ggml_backend_offload_op(d.backend, &probe)) {
                return nullptr;
            }
        }
        return layers[il].cpu_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        if (layers[b.il].ig < 0) {
            return 0; // SHURA: the group is off
        }
        const group & g = groups[layers[b.il].ig];
        if (backend != devices[g.id].backend) {
            return 0;
        }

        // large batches only read the cache, so the experts used in generation stay in it
        const int32_t * slots = g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] >= g.n_slots) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] < g.n_slots && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il      = it->second.first;
        const bool    has_cpu = it->second.second;
        layer & l = layers[il];
        // SHURA: a graph built for an earlier layout is not reused (generation in the graph params), so a slot map is only copied while
        // its group is on
        GGML_ASSERT(l.ig >= 0 && "MoE cache slot map of a layer whose group is off");
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));
        GGML_ASSERT(sel->op == GGML_OP_RESHAPE); // of the [n_expert_used, n_tokens] ids, see build_moe_cache_slots

        const int32_t n_used = (int32_t) sel->src[0]->ne[0];

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));
        ggml_backend_synchronize(backend);

        if (trace != nullptr) {
            write_trace(il, n_used);
        }

        // part of the misses is computed on the CPU once the cache is full, the carry keeps the share right on average
        const uint32_t share_q = has_cpu && g.lru.n_filled == g.lru.n_slots ? cpu_share_q.load(std::memory_order_relaxed) : 0;
        auto n_cpu = [&](size_t n_miss) {
            const uint64_t acc = (uint64_t) n_miss*share_q + l.cpu_acc;
            l.cpu_acc = acc & 0xffff;
            return (size_t) (acc >> 16);
        };
        size_t n_hit = 0;
        if (!g.lru.plan(il, ids.data(), ids.size(), sel->src[0]->ne[1], n_used, n_zero, n_cpu, fills, cpu, n_hit)) {
            GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
        }
        if (has_cpu) {
            int32_t * cpu_map = (int32_t *) l.cpu_map->data;
            for (int32_t e : l.cpu_experts) {
                cpu_map[e] = -1;
            }
            l.cpu_experts.clear();
            for (const auto & f : cpu) {
                cpu_map[f.expert] = f.expert;
                l.cpu_experts.push_back(f.expert);
            }
        }

        // upload the missing experts, consecutive experts going to consecutive slots are uploaded together
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < fills.size();) {
                size_t n = 1;
                while (i + n < fills.size() && fills[i + n].expert == fills[i].expert + (int32_t) n && fills[i + n].slot == fills[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + fills[i].expert*expert_size, fills[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }

        stats & st = ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += fills.size() + cpu.size();
        st.cpu    += cpu.size();
        st.bytes  += bytes;

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, l.slot_map->data, 0, ggml_nbytes(l.slot_map));

        return true;
    }

    // SHURA: change the CPU share at run time; only for a cache created with a CPU part (cpu_share > 0 reserves the zero slots)
    bool set_cpu_share(float share) {
        if (n_zero == 0) {
            return false;
        }
        share = std::clamp(share, 0.0f, 1.0f);
        cpu_share   = share;
        cpu_share_q = (uint32_t) std::lround(share*65536.0f);
        return true;
    }

    void fill_info(llama_moe_cache::info & res) const {
        res.size        = snap_size.load();
        res.host_bytes  = snap_host_bytes.load();
        res.n_slots     = snap_slots.load();
        res.n_groups    = (int32_t) groups.size();
        res.n_groups_on = snap_groups_on.load();
        res.empty       = res.n_groups_on == 0;
        res.cpu_part    = n_zero > 0;
        res.cpu_share   = cpu_share.load();
        res.generation  = generation.load();
        res.hits_small   = stats_small.hits;   res.misses_small = stats_small.misses; res.cpu_small = stats_small.cpu; res.bytes_small = stats_small.bytes;
        res.hits_large   = stats_large.hits;   res.misses_large = stats_large.misses; res.cpu_large = stats_large.cpu; res.bytes_large = stats_large.bytes;
        res.copy_hits    = stats_copy.hits;    res.copy_bytes   = stats_copy.bytes;
    }

    // ---- SHURA: live resize (design §3.3 В, Г, З). The caller (llama_context) holds the micro-batch lock and has synchronized
    // the scheduler: no graph is running or queued that reads the banks, and no asynchronous upload into them is pending. ----

    std::vector<int32_t> current_slots() const {
        std::vector<int32_t> res(groups.size(), 0);
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            res[ig] = groups[ig].n_slots;
        }
        return res;
    }

    // free the device banks and their contexts; the layers read the host experts until the next apply_layout()
    size_t release_banks() {
        size_t freed = 0;
        bindings.clear();
        for (group & g : groups) {
            g.banks.clear();
            for (int32_t il : g.layers) {
                layers[il].ig = -1;
            }
        }
        for (device & d : devices) {
            if (d.buf) {
                freed += d.buf_size;
            }
            d.buf.reset();
            d.ctx.reset();
            d.buf_size = 0;
        }
        return freed;
    }

    // the test hook: GGML_MOE_CACHE_TEST_FAIL_RESIZE_ALLOCS_ONCE=n at creation, or set_test_fail_allocs(n): the next n allocations of
    // a non-empty layout fail as if the device were out of memory
    std::atomic<int32_t> test_fail_allocs{0};

    void apply_layout_checked(const std::vector<int32_t> & n_slots) {
        const bool any = std::any_of(n_slots.begin(), n_slots.end(), [](int32_t n) { return n > 0; });
        if (any && test_fail_allocs.load() > 0) {
            test_fail_allocs--;
            LLAMA_LOG_WARN("%s: test hook: failing this MoE cache allocation (%d more)\n", __func__, test_fail_allocs.load());
            throw std::runtime_error("test hook: forced MoE cache allocation failure");
        }
        try {
            apply_layout(n_slots);
        } catch (...) {
            release_banks();
            throw;
        }
    }

    // called after the banks were released and before the new ones are allocated, with the bytes freed: the VRAM governor waits
    // there until the card shows the memory as free (on ROCm the release is accounted later than the free call returns)
    void (*release_wait)(size_t freed_bytes, void * data) = nullptr;
    void * release_wait_data = nullptr;

    // change the size: returns false if the layout did not change. On an allocation failure the previous size is allocated again;
    // if that fails too the cache is empty (the layers read the host experts) and the server goes on.
    bool resize(size_t size) {
        if (!resizable) {
            LLAMA_LOG_WARN("%s: the MoE cache was created without --moe-cache-live-resize\n", __func__);
            return false;
        }
        if (groups.empty()) {
            return false;
        }
        const std::vector<int32_t> old_slots = current_slots();
        const std::vector<int32_t> new_slots = layout(size);
        if (new_slots == old_slots) {
            return false;
        }
        const int64_t t0 = ggml_time_us();
        const size_t old_size = snap_size.load();
        const size_t freed = release_banks();
        if (release_wait != nullptr && freed > 0) {
            release_wait(freed, release_wait_data);
        }
        const int64_t t_wait = ggml_time_us();
        const char * outcome = "resized";
        try {
            apply_layout_checked(new_slots);
        } catch (const std::exception & err) {
            LLAMA_LOG_WARN("%s: MoE cache allocation of %.2f MiB failed (%s), restoring %.2f MiB\n", __func__,
                size/1024.0/1024.0, err.what(), old_size/1024.0/1024.0);
            outcome = "restored the previous size";
            try {
                apply_layout_checked(old_slots);
            } catch (const std::exception & err2) {
                LLAMA_LOG_WARN("%s: MoE cache allocation of the previous size failed too (%s), the cache is empty\n", __func__, err2.what());
                outcome = "empty after two failed allocations";
                apply_layout(std::vector<int32_t>(groups.size(), 0)); // allocates nothing
            }
        }
        const size_t n_warm = warm_fill();
        LLAMA_LOG_INFO("%s: MoE cache %s: %.2f -> %.2f MiB, %d slots, %zu experts warm, waited %.1f ms for the release, %.1f ms in all\n",
            __func__, outcome, old_size/1024.0/1024.0, snap_size.load()/1024.0/1024.0, snap_slots.load(), n_warm,
            (t_wait - t0)/1000.0, (ggml_time_us() - t0)/1000.0);
        return true;
    }

    // after a layout change all slots are empty: fill them with the experts of each group by their decayed CRF over all the experts
    // of its layers (also the ones that were not cached), hottest last so that they sit at the recently used end of their class
    size_t warm_fill() {
        if (no_alloc) {
            return 0;
        }
        size_t n_total = 0;
        struct cand {
            float   crf;
            int32_t key;
        };
        std::vector<cand> cands;
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            moe_cache_lru & lru = g.lru;
            cands.clear();
            for (int32_t il : g.layers) {
                for (int32_t e = 0; e < lru.n_expert; ++e) {
                    const int32_t key = lru.make_key(il, e);
                    if (lru.last_use[key] < 0 || lru.crf[key] <= 0.0f) {
                        continue;
                    }
                    cands.push_back({ lru.decayed_crf(il, key), key });
                }
            }
            const size_t n = std::min(cands.size(), (size_t) g.n_slots);
            std::partial_sort(cands.begin(), cands.begin() + n, cands.end(), [](const cand & a, const cand & b) {
                return a.crf != b.crf ? a.crf > b.crf : a.key < b.key;
            });
            ggml_backend_t backend = devices[g.id].backend;
            for (size_t i = n; i-- > 0; ) {
                const int32_t key = cands[i].key;
                const int32_t il  = lru.key_layer(key);
                const int32_t e   = lru.key_expert(key);
                const int32_t s   = (int32_t) (n - 1 - i); // slots 0.. in order
                lru.key_of[s] = key;
                lru.slot_map[il][e] = s;
                lru.move(s, lru.class_of(key));
                lru.n_filled++;
                const layer & l = layers[il];
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const ggml_tensor * w = l.experts[ip];
                    ggml_backend_tensor_set_async(backend, g.banks[ip], (const uint8_t *) w->data + e*w->nb[2], s*w->nb[2], w->nb[2]);
                }
            }
            if (n > 0) {
                ggml_backend_synchronize(backend);
            }
            n_total += n;
        }
        return n_total;
    }

    // design §3.3 З: the host copy of expert e of w changed (a patch map): the cached copy is read again from the host
    bool invalidate(const ggml_tensor * w, int32_t e) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return false;
        }
        const binding & b = it->second;
        const layer & l = layers[b.il];
        if (l.ig < 0) {
            return false;
        }
        group & g = groups[l.ig];
        if (e < 0 || e >= g.lru.n_expert) {
            return false;
        }
        const int32_t s = g.lru.slot_map[b.il][e];
        if (s >= g.n_slots) {
            return true; // not cached: the next upload reads the host copy
        }
        ggml_backend_t backend = devices[g.id].backend;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * x = l.experts[ip];
            ggml_backend_tensor_set(g.banks[ip], (const uint8_t *) x->data + e*x->nb[2], s*x->nb[2], x->nb[2]);
        }
        ggml_backend_synchronize(backend);
        return true;
    }

    // tests: the LRU lists, the slot maps and (optionally) the bank contents agree with each other and with the host experts
    bool self_check(bool contents, std::string & why) const {
        for (size_t ig = 0; ig < groups.size(); ++ig) {
            const group & g = groups[ig];
            const moe_cache_lru & lru = g.lru;
            if (lru.n_slots != g.n_slots) {
                why = "lru slots != group slots";
                return false;
            }
            int32_t filled = 0;
            int32_t in_lists = 0;
            for (int c = 0; c < moe_cache_lru::n_class; ++c) {
                for (int32_t s = lru.head[c]; s >= 0; s = lru.next[s]) {
                    if (lru.list_of[s] != c) {
                        why = "slot in the wrong class list";
                        return false;
                    }
                    if (++in_lists > g.n_slots) {
                        why = "a cycle in the class lists";
                        return false;
                    }
                }
            }
            if (in_lists != g.n_slots) {
                why = "class lists do not hold every slot";
                return false;
            }
            for (int32_t s = 0; s < g.n_slots; ++s) {
                const int32_t key = lru.key_of[s];
                if (key < 0) {
                    continue;
                }
                filled++;
                if (lru.slot_map[lru.key_layer(key)][lru.key_expert(key)] != s) {
                    why = "slot map does not point back to the slot";
                    return false;
                }
            }
            if (filled != lru.n_filled) {
                why = "n_filled does not match the slots";
                return false;
            }
            for (int32_t il : g.layers) {
                const int32_t * slots = lru.slot_map[il];
                for (int32_t e = 0; e < lru.n_expert; ++e) {
                    const int32_t s = slots[e];
                    if (s < g.n_slots && lru.key_of[s] != lru.make_key(il, e)) {
                        why = "slot map points to a slot of another expert";
                        return false;
                    }
                    if (g.n_slots == 0 && s != lru.empty) {
                        why = "a group that is off has a cached expert";
                        return false;
                    }
                }
                if (!contents || g.n_slots == 0) {
                    continue;
                }
                const layer & l = layers[il];
                std::vector<uint8_t> buf;
                for (int32_t e = 0; e < lru.n_expert; ++e) {
                    const int32_t s = slots[e];
                    if (s >= g.n_slots) {
                        continue;
                    }
                    for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                        const ggml_tensor * w = l.experts[ip];
                        buf.resize(w->nb[2]);
                        ggml_backend_tensor_get(g.banks[ip], buf.data(), s*w->nb[2], w->nb[2]);
                        if (memcmp(buf.data(), (const uint8_t *) w->data + e*w->nb[2], w->nb[2]) != 0) {
                            why = "bank slot differs from the host expert";
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    // all expert tensors of layer il
    bool invalidate_layer(int32_t il, int32_t e) {
        if (il < 0 || il >= (int32_t) layers.size()) {
            return false;
        }
        bool ok = !layers[il].experts.empty();
        for (const ggml_tensor * w : layers[il].experts) {
            ok = invalidate(w, e) && ok;
        }
        return ok;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu (%zu on the CPU), hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, (size_t) st.hits.load(), (size_t) st.misses.load(), (size_t) st.cpu.load(), 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", (size_t) stats_copy.hits.load(), stats_copy.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, float cpu_share, bool resizable) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, cpu_share, resizable)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used, bool cpu_part) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used, cpu_part);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

ggml_tensor * llama_moe_cache::get_cpu_map(int32_t il, int64_t n_tokens) const {
    return pimpl->get_cpu_map(il, n_tokens);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    // SHURA: the snapshot written by apply_layout(), safe to read from another thread while the layout changes
    std::lock_guard<std::mutex> lock(pimpl->snap_mu);
    return pimpl->snap_breakdown;
}

size_t llama_moe_cache::device_size() const {
    return pimpl->snap_size.load();
}

bool llama_moe_cache::empty() const {
    return pimpl->empty();
}

uint64_t llama_moe_cache::generation() const {
    return pimpl->generation.load();
}

bool llama_moe_cache::set_cpu_share(float share) {
    return pimpl->set_cpu_share(share);
}

bool llama_moe_cache::resize(size_t size) {
    return pimpl->resize(size);
}

void llama_moe_cache::set_release_wait(void (*fn)(size_t freed_bytes, void * data), void * data) {
    pimpl->release_wait      = fn;
    pimpl->release_wait_data = data;
}

void llama_moe_cache::set_test_fail_allocs(int32_t n) {
    pimpl->test_fail_allocs = std::max(0, n);
}

bool llama_moe_cache::invalidate(int32_t il, int32_t e) {
    return pimpl->invalidate_layer(il, e);
}

bool llama_moe_cache::self_check(bool contents, std::string & why) const {
    return pimpl->self_check(contents, why);
}

bool llama_moe_cache::resizable() const {
    return pimpl->resizable;
}

llama_moe_cache::info llama_moe_cache::get_info() const {
    info res;
    pimpl->fill_info(res);
    return res;
}
