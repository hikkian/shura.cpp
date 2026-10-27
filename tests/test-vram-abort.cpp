// VRAMLOGIC3: llama_decode stops for the abort callback inside a ubatch (CPU nodes) and between ubatches; the memory holds
// exactly the ubatches that finished, and the same batch decoded again gives the same logits as a clean run.
// Needs a small model: build test-llama-archs and save one, e.g.
//     test-llama-archs -a llama -o /tmp/models && test-vram-abort /tmp/models/llama-dense.gguf
#include "llama.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

static std::atomic<int> g_calls{0};
static std::atomic<int> g_fire_after{-1};   // -1: never; n: return true from the (n+1)-th call on

static bool abort_cb(void *) {
    const int n = g_calls.fetch_add(1);
    const int after = g_fire_after.load();
    return after >= 0 && n >= after;
}

static llama_batch make_batch(int n_tokens, int n_vocab) {
    llama_batch b = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        b.token[i] = (llama_token) ((i * 7919 + 13) % n_vocab);
        b.pos[i] = i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = i == n_tokens - 1;
    }
    b.n_tokens = n_tokens;
    return b;
}

static std::vector<float> last_logits(llama_context * ctx, int n_vocab) {
    const float * l = llama_get_logits_ith(ctx, -1);
    return std::vector<float>(l, l + n_vocab);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf\n", argv[0]);
        return 2;
    }
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    CHECK(model != nullptr);
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512; cp.n_batch = 256; cp.n_ubatch = 32; cp.n_threads = 1; cp.n_threads_batch = 1;   // one thread: the callback count is exact
    llama_context * ctx = llama_init_from_model(model, cp);
    CHECK(ctx != nullptr);
    llama_set_abort_callback(ctx, abort_cb, nullptr);
    llama_memory_t mem = llama_get_memory(ctx);
    const int n_tokens = 96;   // three ubatches of 32

    // clean run: the reference logits; and how many callback calls one ubatch makes (CPU nodes)
    llama_batch batch = make_batch(n_tokens, n_vocab);
    g_calls = 0; g_fire_after = -1;
    CHECK(llama_decode(ctx, batch) == 0);
    const std::vector<float> ref = last_logits(ctx, n_vocab);
    const int calls_3 = g_calls.load();
    CHECK(llama_memory_seq_pos_max(mem, 0) == n_tokens - 1);
    llama_memory_seq_rm(mem, 0, -1, -1);
    llama_batch one = make_batch(32, n_vocab);
    g_calls = 0;
    CHECK(llama_decode(ctx, one) == 0);
    const int per_ubatch = g_calls.load();
    llama_memory_seq_rm(mem, 0, -1, -1);
    llama_batch_free(one);
    printf("callback calls: %d for 3 ubatches, %d for one\n", calls_3, per_ubatch);
    CHECK(per_ubatch > 1);                                   // the CPU backend checks it per node
    CHECK(calls_3 >= 3 * per_ubatch + 2);                    // plus the checks between ubatches (VRAMLOGIC3)

    // A. stop inside the first ubatch: nothing stays in the memory
    g_calls = 0; g_fire_after = per_ubatch / 2;
    CHECK(llama_decode(ctx, batch) == 2);
    CHECK(llama_memory_seq_pos_max(mem, 0) == -1);

    // B. stop right after the first ubatch (the first call past its nodes is the check between ubatches): the first ubatch stays
    g_calls = 0; g_fire_after = per_ubatch;
    CHECK(llama_decode(ctx, batch) == 2);
    printf("after the stop between ubatches: pos_max = %d\n", (int) llama_memory_seq_pos_max(mem, 0));
    CHECK(llama_memory_seq_pos_max(mem, 0) == 31);

    // C. the server's rollback (from the slice's first position) and the same batch again: the logits of the clean run
    llama_memory_seq_rm(mem, 0, 0, -1);
    g_calls = 0; g_fire_after = -1;
    CHECK(llama_decode(ctx, batch) == 0);
    const std::vector<float> again = last_logits(ctx, n_vocab);
    double max_diff = 0;
    for (int i = 0; i < n_vocab; i++) { max_diff = std::max(max_diff, (double) std::fabs(again[i] - ref[i])); }
    printf("max |logit difference| after the stops and the retry: %g\n", max_diff);
    CHECK(max_diff < 1e-4);
    CHECK(llama_memory_seq_pos_max(mem, 0) == n_tokens - 1);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    printf("test-vram-abort: all passed\n");
    return 0;
}
