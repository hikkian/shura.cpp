// SHURA (PATCH1): expansion cards as patches (src/llama-patch.cpp). Not run by ctest: it needs a small MoE model and the cards
// tools/patch/test_cards.py builds for it:
//     test-llama-archs -a '^deepseek2$' -o M && test-llama-archs -a '^deepseek2$' -s 77 -o D
//     tools/patch/test_cards.py M/deepseek2-moe.gguf D/deepseek2-moe.gguf C
//     test-patch M/deepseek2-moe.gguf C            (CPU: weights in host memory)
//     test-patch M/deepseek2-moe.gguf C --gpu      (layers on the GPU, experts in host memory behind a resizable expert cache)
// Checks (the letters of the PATCH1 card):
//   (з) every bad card is rejected by the loader, before anything is written: the model stays byte for byte the base;
//   (а) a card with the base's own bytes: the same tokens and bitwise the same logits as the base;
//   (б) the donor card: every replaced slice holds the donor's bytes in every copy (tensor read back, expert cache self check with
//       contents), the other slices the base's; the logits change; exposure counts the replaced experts;
//       on the GPU: after a resize of the expert cache the slots are read from the patched host experts (self check);
//   (в) removal: the base's bytes come back (hash checked), bitwise the base's logits;
//   (д) no trace: a conversation with one set, then the base: the same as without the first conversation;
//   conflicts: two cards on one slice are rejected and the active set stays; switching between disjoint sets restores what the
//   new set does not cover.
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); g_fail++; } else { fprintf(stderr, "ok   %s\n", #c); } } while (0)

struct run {
    std::vector<llama_token> tokens;
    std::vector<float> logits;
    bool ok = false;
};

static run decode(llama_context * ctx, int n_vocab, int n_prompt, int n_gen, int salt = 0) {
    run r;
    llama_memory_clear(llama_get_memory(ctx), true);
    llama_batch b = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        b.token[i] = (llama_token) ((i * 7919 + 13 + salt * 101) % n_vocab);
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
        r.logits.insert(r.logits.end(), l, l + n_vocab);
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

static bool same(const run & a, const run & b) {
    return a.ok && b.ok && a.tokens == b.tokens && a.logits.size() == b.logits.size() &&
        memcmp(a.logits.data(), b.logits.data(), a.logits.size() * sizeof(float)) == 0;
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

static bool verify(llama_model * model, bool caches) {
    char why[512] = {0};
    const bool ok = llama_model_patch_verify(model, caches, why, sizeof(why));
    if (!ok) {
        fprintf(stderr, "     verify: %s\n", why);
    }
    return ok;
}

static bool set(llama_model * model, std::vector<llama_patch *> packs, std::string * err = nullptr) {
    char why[512] = {0};
    const bool ok = llama_model_patch_set(model, packs.data(), (int32_t) packs.size(), why, sizeof(why));
    if (err) {
        *err = why;
    }
    return ok;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s MODEL.gguf CARD_DIR [--gpu]\n", argv[0]);
        return 2;
    }
    const std::string model_path = argv[1];
    const std::string dir = std::string(argv[2]) + "/";
    const bool gpu = argc > 3 && std::string(argv[3]) == "--gpu";
    llama_backend_init();

    llama_model_tensor_buft_override ov[2] = {
        { "\\.ffn_(up|down|gate|gate_up)_exps", ggml_backend_cpu_buffer_type() },
        { nullptr, nullptr },
    };
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = gpu ? 999 : 0;
    mp.tensor_buft_overrides = gpu ? ov : nullptr;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;   // cards write into the weights: no read-only mapping
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "could not load %s\n", model_path.c_str());
        return 2;
    }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    cp.n_batch = 64;
    cp.n_ubatch = 64;
    cp.n_seq_max = 1;
    cp.no_perf = true;
    if (gpu) {
        cp.moe_cache_size = 0;
        cp.moe_cache_resizable = true;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    CHECK(ctx != nullptr);
    if (!ctx) {
        return 1;
    }
    // a second context on the same model (as the MTP draft context is): its cache must follow the cards too
    llama_context * ctx2 = llama_init_from_model(model, cp);
    CHECK(ctx2 != nullptr);
    size_t big = 0;
    if (gpu) {
        big = llama_moe_cache_size_for_tokens(ctx, -1);
        CHECK(llama_moe_cache_resize(ctx, big / 2) > 0);
        CHECK(llama_moe_cache_resize(ctx2, big) > 0);
    }

    llama_patch_exposure_enable(ctx, true);   // off by default (it costs decode speed)
    const run base  = decode(ctx, n_vocab, 48, 16);
    const run base2 = decode(ctx, n_vocab, 48, 16);
    CHECK(base.ok && same(base, base2));   // the profile is deterministic
    const run base_b = decode(ctx, n_vocab, 40, 12, 1);

    auto load = [&](const char * name, std::string * err = nullptr) -> llama_patch * {
        char why[512] = {0};
        llama_patch * p = llama_patch_load(model, model_path.c_str(), (dir + name + ".gguf").c_str(), why, sizeof(why));
        if (err) {
            *err = why;
        }
        return p;
    };

    // (з) the loader rejects every bad card before writing anything
    const char * bad[] = { "bad_truncated", "bad_otherbase", "bad_layout", "bad_overlap", "bad_range", "bad_type", "bad_shape",
                           "bad_tamper", "bad_notensor", "bad_mtp", "bad_nomanifest" };
    for (const char * b : bad) {
        std::string why;
        llama_patch * p = load(b, &why);
        fprintf(stderr, "     %-15s -> %s\n", b, p ? "LOADED" : why.c_str());
        CHECK(p == nullptr);
        llama_patch_free(p);
    }
    CHECK(same(decode(ctx, n_vocab, 48, 16), base));
    CHECK(std::string(llama_model_patch_set_hash(model)).empty());

    llama_patch * id       = load("id");
    llama_patch * donor    = load("donor");
    llama_patch * donor2   = load("donor2");
    llama_patch * whole    = load("whole");
    llama_patch * conflict = load("conflict");
    CHECK(id && donor && donor2 && whole && conflict);
    if (!(id && donor && donor2 && whole && conflict)) {
        return 1;
    }

    // (а) identity: the base's own bytes
    CHECK(set(model, { id }));
    CHECK(verify(model, gpu));
    CHECK(std::string(llama_model_patch_set_hash(model)).size() == 64);
    CHECK(same(decode(ctx, n_vocab, 48, 16), base));
    {
        int32_t il[64], e[64]; uint64_t c[64];
        const int32_t n = llama_patch_exposure(ctx, 0, il, e, c, 64);
        fprintf(stderr, "     identity card: exposure of %d replaced experts\n", n);
    }

    // (б) the donor card
    llama_patch_exposure_reset(ctx, 0);
    CHECK(set(model, { donor }));
    CHECK(verify(model, gpu));
    const run d1 = decode(ctx, n_vocab, 48, 16);
    fprintf(stderr, "     donor card: max logit change %.4f, tokens %s\n", max_diff(d1, base), d1.tokens == base.tokens ? "same" : "differ");
    CHECK(d1.ok && max_diff(d1, base) > 0.0f);
    {
        int32_t il[256], e[256]; uint64_t c[256];
        const int32_t n = llama_patch_exposure(ctx, 0, il, e, c, 256);
        uint64_t tot = 0;
        bool only_replaced = true;
        for (int32_t i = 0; i < std::min(n, 256); i++) {
            tot += c[i];
            only_replaced = only_replaced && il[i] == 1 && e[i] % 2 == 0;
        }
        fprintf(stderr, "     donor card: %d replaced experts selected, %llu selections\n", n, (unsigned long long) tot);
        CHECK(n > 0 && tot > 0 && only_replaced);
    }
    // the second context on the same model computes with the card as well
    {
        const run x = decode(ctx2, n_vocab, 48, 16);
        CHECK(same(x, d1));
    }
    if (gpu) {
        // a resize after the card: the new slots are read from the patched host experts
        CHECK(llama_moe_cache_resize(ctx, big) > 0);
        CHECK(verify(model, true));
        CHECK(same(decode(ctx, n_vocab, 48, 16), d1));
        CHECK(llama_moe_cache_resize(ctx, big / 3) > 0);
        CHECK(verify(model, true));
        CHECK(same(decode(ctx, n_vocab, 48, 16), d1));
    }

    // conflicts: two cards on one slice; the active set stays
    {
        std::string why;
        CHECK(!set(model, { donor, conflict }, &why));
        fprintf(stderr, "     conflict -> %s\n", why.c_str());
        CHECK(!set(model, { donor, whole }, &why));
        fprintf(stderr, "     whole vs expert -> %s\n", why.c_str());
        CHECK(verify(model, gpu));
        CHECK(same(decode(ctx, n_vocab, 48, 16), d1));
    }

    // switching between disjoint sets: what the new set does not cover gets the base back
    CHECK(set(model, { donor2 }));
    CHECK(verify(model, gpu));
    const run d2 = decode(ctx, n_vocab, 48, 16);
    CHECK(d2.ok && max_diff(d2, base) > 0.0f && max_diff(d2, d1) > 0.0f);
    CHECK(set(model, { donor, donor2 }));
    CHECK(verify(model, gpu));
    const run d12 = decode(ctx, n_vocab, 48, 16);
    CHECK(d12.ok && max_diff(d12, d1) > 0.0f);
    llama_patch_exposure_reset(ctx, 0);
    CHECK(set(model, { whole }));
    CHECK(verify(model, gpu));
    {
        const run w = decode(ctx, n_vocab, 48, 16);
        CHECK(w.ok && max_diff(w, base) > 0.0f);
        int32_t il[256], e[256]; uint64_t c[256];
        const int32_t n = llama_patch_exposure(ctx, 0, il, e, c, 256);
        fprintf(stderr, "     whole card: max logit change %.4f; selected experts:", max_diff(w, base));
        for (int32_t i = 0; i < std::min(n, 256); i++) {
            fprintf(stderr, " %d:%d=%llu", il[i], e[i], (unsigned long long) c[i]);
        }
        fprintf(stderr, "\n");
        CHECK(n > 0);
    }

    // (в) removal
    CHECK(set(model, {}));
    CHECK(verify(model, gpu));
    CHECK(std::string(llama_model_patch_set_hash(model)).empty());
    CHECK(same(decode(ctx, n_vocab, 48, 16), base));
    CHECK(same(decode(ctx2, n_vocab, 48, 16), base));

    // (д) no trace: a conversation with the donor card, then the base, gives what the base gives without it
    CHECK(set(model, { donor }));
    (void) decode(ctx, n_vocab, 40, 12, 1);
    CHECK(set(model, {}));
    CHECK(same(decode(ctx, n_vocab, 40, 12, 1), base_b));

    // free while active is refused by an assert, so the set is emptied first (done above)
    for (llama_patch * p : { id, donor, donor2, whole, conflict }) {
        llama_patch_free(p);
    }
    llama_free(ctx2);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    fprintf(stderr, "%s: %d failed\n", g_fail ? "FAILED" : "PASSED", g_fail);
    return g_fail ? 1 : 0;
}
