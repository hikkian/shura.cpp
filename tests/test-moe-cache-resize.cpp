// SHURA (REBASE1 step 4): live resize of upstream's expert cache (design §3.3 В, Г, Д, З), on a GPU with a small MoE model whose
// experts are kept in host memory. Not run by ctest: it needs a GPU and a model, e.g.
//     test-llama-archs -a qwen35moe -o /dev/shm/models && test-moe-cache-resize /dev/shm/models/<the qwen35moe file>.gguf
// Checks:
//   1. resize there and back gives the same tokens and logits (deterministic greedy decode);
//   2. every resize bumps the generation, the graph is rebuilt and the cache stays consistent with the host experts;
//   3. the empty state: decoding goes on from the host experts, and the cache comes back;
//   4. a forced allocation failure restores the previous size; a second one leaves the cache empty; decoding goes on;
//   5. after the warm fill the LRU lists, the slot maps and the banks agree (self check with contents);
//   6. invalidate reads a changed host expert again (and the self check notices a changed one that was not invalidated);
//   7. another thread reads the stats and the memory breakdown during resizes and decodes.
#include "llama.h"
#include "../src/llama-ext.h"
#include "../src/llama-model.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); g_fail++; } else { fprintf(stderr, "ok   %s\n", #c); } } while (0)

struct run {
    std::vector<llama_token> tokens;
    std::vector<float> logits;
    bool ok = false;
};

static run decode(llama_context * ctx, int n_vocab, int n_prompt, int n_gen) {
    run r;
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch b = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        b.token[i] = (llama_token) ((i * 7919 + 13) % n_vocab);
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = i == n_prompt - 1;
    }
    b.n_tokens = n_prompt;
    if (llama_decode(ctx, b) != 0) {
        llama_batch_free(b);
        return r;
    }
    llama_batch_free(b);
    llama_pos pos = n_prompt;
    for (int i = 0; i < n_gen; i++) {
        const float * l = llama_get_logits_ith(ctx, -1);
        const llama_token t = (llama_token) (std::max_element(l, l + n_vocab) - l);
        r.tokens.push_back(t);
        if (i == n_gen - 1) {
            r.logits.assign(l, l + n_vocab);
        }
        llama_batch one = llama_batch_init(1, 0, 1);
        one.token[0] = t;
        one.pos[0] = pos++;
        one.n_seq_id[0] = 1;
        one.seq_id[0][0] = 0;
        one.logits[0] = 1;
        one.n_tokens = 1;
        const int rc = llama_decode(ctx, one);
        llama_batch_free(one);
        if (rc != 0) {
            return r;
        }
    }
    r.ok = true;
    return r;
}

static float max_diff(const run & a, const run & b) {
    if (a.logits.size() != b.logits.size() || a.logits.empty()) {
        return INFINITY;
    }
    float m = 0.0f;
    for (size_t i = 0; i < a.logits.size(); i++) {
        m = std::max(m, std::fabs(a.logits[i] - b.logits[i]));
    }
    return m;
}

static llama_moe_cache_info info(llama_context * ctx) {
    llama_moe_cache_info i = {};
    llama_moe_cache_get_info(ctx, &i);
    return i;
}

static bool check(llama_context * ctx, bool contents) {
    char why[256] = {0};
    const bool ok = llama_moe_cache_self_check(ctx, contents, why, sizeof(why));
    if (!ok) {
        fprintf(stderr, "     self check: %s\n", why);
    }
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.gguf (a small MoE model, e.g. from test-llama-archs -a qwen35moe -o DIR)\n", argv[0]);
        return 2;
    }
    llama_backend_init();

    llama_model_tensor_buft_override ov[2] = {
        { "\\.ffn_(up|down|gate|gate_up)_exps", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.tensor_buft_overrides = ov;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "could not load %s\n", argv[1]);
        return 2;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // the host experts and the size of the cache that holds about half of them
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_seq_max = 1;
    cp.moe_cache_size = 0;
    cp.moe_cache_resizable = true;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    CHECK(ctx != nullptr);
    if (!ctx) {
        return 1;
    }
    llama_moe_cache_info i0 = info(ctx);
    fprintf(stderr, "     host experts %.2f MiB, %d groups, empty %d\n", i0.host_bytes / 1048576.0, i0.n_groups, (int) i0.empty);
    CHECK(i0.empty);
    CHECK(i0.n_groups > 0);

    const size_t big   = i0.host_bytes / 2;
    const size_t small = i0.host_bytes / 6;
    const int n_prompt = 24, n_gen = 12;

    // 3. empty from the start: the host experts
    const run r_empty = decode(ctx, n_vocab, n_prompt, n_gen);
    CHECK(r_empty.ok);

    // 1. there and back
    uint64_t gen = info(ctx).generation;
    const size_t s1 = llama_moe_cache_resize(ctx, big);
    CHECK(s1 > 0);
    CHECK(info(ctx).generation > gen);
    const run r1 = decode(ctx, n_vocab, n_prompt, n_gen);
    CHECK(r1.ok);
    CHECK(check(ctx, true));
    fprintf(stderr, "     cache %.2f MiB: hits %llu misses %llu\n", s1 / 1048576.0,
            (unsigned long long) info(ctx).hits_small, (unsigned long long) info(ctx).misses_small);
    fprintf(stderr, "     empty vs cached: max |dlogit| %.3g (host CPU path vs device path), same tokens %d\n",
            max_diff(r_empty, r1), (int) (r_empty.tokens == r1.tokens));

    gen = info(ctx).generation;
    const size_t s2 = llama_moe_cache_resize(ctx, small);
    CHECK(s2 > 0 && s2 < s1);
    CHECK(info(ctx).generation > gen);
    CHECK(check(ctx, true)); // 5. warm fill
    const run r2 = decode(ctx, n_vocab, n_prompt, n_gen);
    CHECK(r2.ok);
    CHECK(check(ctx, true));
    const size_t s1b = llama_moe_cache_resize(ctx, big);
    CHECK(s1b == s1);
    const run r1b = decode(ctx, n_vocab, n_prompt, n_gen);
    CHECK(r1b.ok);
    CHECK(r1b.tokens == r1.tokens);
    fprintf(stderr, "     big -> small -> big: max |dlogit| %.3g; small vs big %.3g\n", max_diff(r1, r1b), max_diff(r1, r2));
    CHECK(max_diff(r1, r1b) == 0.0f);
    CHECK(r2.tokens == r1.tokens);

    // 3. the empty state and back
    CHECK(llama_moe_cache_resize(ctx, 0) == 0);
    CHECK(info(ctx).empty);
    CHECK(check(ctx, false));
    const run r0 = decode(ctx, n_vocab, n_prompt, n_gen);
    CHECK(r0.ok);
    CHECK(max_diff(r0, r_empty) == 0.0f);
    CHECK(llama_moe_cache_resize(ctx, big) == s1);
    const run r1c = decode(ctx, n_vocab, n_prompt, n_gen);
    CHECK(r1c.ok && max_diff(r1, r1c) == 0.0f);

    // 4. allocation failures
    llama_moe_cache_test_fail_allocs(ctx, 1);
    CHECK(llama_moe_cache_resize(ctx, small) == s1);   // restored the previous size
    CHECK(check(ctx, true));
    const run rf1 = decode(ctx, n_vocab, n_prompt, n_gen);
    CHECK(rf1.ok && max_diff(r1, rf1) == 0.0f);
    llama_moe_cache_test_fail_allocs(ctx, 2);
    CHECK(llama_moe_cache_resize(ctx, small) == 0);    // both failed: empty
    CHECK(info(ctx).empty);
    const run rf2 = decode(ctx, n_vocab, n_prompt, n_gen);
    CHECK(rf2.ok && max_diff(rf2, r_empty) == 0.0f);
    CHECK(llama_moe_cache_resize(ctx, big) == s1);
    const run rf3 = decode(ctx, n_vocab, n_prompt, n_gen);
    CHECK(rf3.ok && max_diff(r1, rf3) == 0.0f);

    // 6. invalidate: change one byte of every expert of the first cached layer on the host
    {
        const auto & tmap = llama_internal_get_tensor_map(model);
        ggml_tensor * w = nullptr;
        int il = -1;
        for (const auto & [name, t] : tmap) {
            int l = -1;
            if (sscanf(name.c_str(), "blk.%d.ffn_up_exps.weight", &l) == 1 && t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
                if (w == nullptr || l < il) {
                    w = t;
                    il = l;
                }
            }
        }
        CHECK(w != nullptr);
        if (w != nullptr) {
            const int64_t n_expert = w->ne[2];
            std::vector<uint8_t> saved(n_expert);
            for (int64_t e = 0; e < n_expert; e++) {
                uint8_t * p = (uint8_t *) w->data + e * w->nb[2] + 7;
                saved[e] = *p;
                *p ^= 0x5a;
            }
            CHECK(!check(ctx, true));                  // the checker sees the stale cached copies
            for (int64_t e = 0; e < n_expert; e++) {
                llama_moe_cache_invalidate(ctx, il, (int32_t) e);
            }
            CHECK(check(ctx, true));                   // read again
            for (int64_t e = 0; e < n_expert; e++) {
                ((uint8_t *) w->data)[e * w->nb[2] + 7] = saved[e];
                llama_moe_cache_invalidate(ctx, il, (int32_t) e);
            }
            CHECK(check(ctx, true));
            const run ri = decode(ctx, n_vocab, n_prompt, n_gen);
            CHECK(ri.ok && max_diff(r1, ri) == 0.0f);
        }
    }

    // 7. another thread reads while this one resizes and decodes
    {
        std::atomic<bool> stop{false};
        std::atomic<uint64_t> reads{0};
        std::thread reader([&] {
            while (!stop.load()) {
                llama_moe_cache_info x = {};
                llama_moe_cache_get_info(ctx, &x);
                const auto mb = llama_get_memory_breakdown(ctx);
                size_t cache = 0;
                for (const auto & [buft, d] : mb) {
                    cache += d.moe_cache;
                }
                (void) cache;
                reads++;
            }
        });
        std::thread resizer([&] {
            for (int k = 0; k < 6; k++) {
                llama_moe_cache_resize(ctx, k % 2 ? small : big);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            llama_moe_cache_resize(ctx, big);
        });
        bool all_ok = true;
        for (int k = 0; k < 4; k++) {
            all_ok = decode(ctx, n_vocab, n_prompt, n_gen).ok && all_ok;
        }
        resizer.join();
        stop = true;
        reader.join();
        CHECK(all_ok);
        CHECK(reads.load() > 0);
        CHECK(check(ctx, true));
        const run rt = decode(ctx, n_vocab, n_prompt, n_gen);
        CHECK(rt.ok && max_diff(r1, rt) == 0.0f);
        fprintf(stderr, "     reader made %llu reads during the resizes\n", (unsigned long long) reads.load());
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "%s: %d failed\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
