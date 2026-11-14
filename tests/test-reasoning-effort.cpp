// THINK3b: the soft target, ThinkBrake and the grace window of the reasoning budget sampler (common/reasoning-budget.cpp).
// State machine only (no vocabulary): token ids stand for words, a boundary token plays the sentence end.
#include "reasoning-budget.h"

#include "llama.h"
#include "ggml.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

static const llama_token T_START = 1;   // <think>
static const llama_token T_END   = 2;   // </think>
static const llama_token T_END2  = 4;   // second token of the two-token end tag used in some tests
static const llama_token T_MSG   = 10;  // the budget message (one token)
static const llama_token T_DOT   = 30;  // a sentence end
static const llama_token T_WORD  = 20;  // a plain word
static const llama_token T_WORD2 = 21;
static const llama_token T_NL    = 40;  // a whitespace-only token ("\n")
static const size_t      V       = 64;

#define CHECK(cond) do { if (!(cond)) { fprintf(stderr, "FAILED %s:%d: %s\n", __FILE__, __LINE__, #cond); GGML_ASSERT(false); } } while (0)

static llama_sampler * make(int32_t budget, const common_reasoning_budget_opts & opts, bool two_token_end = false) {
    const std::vector<llama_tokens> start = {{T_START}};
    const std::vector<llama_tokens> end   = {two_token_end ? llama_tokens{T_END, T_END2} : llama_tokens{T_END}};
    llama_tokens forced = {T_MSG};
    forced.insert(forced.end(), end[0].begin(), end[0].end());
    return common_reasoning_budget_init_ex(nullptr, start, end, forced, budget, REASONING_BUDGET_IDLE, opts);
}

// one step as the server does it: raw logits -> apply -> pick the best -> accept. `top` is the model's preferred token (logit 5),
// the end tag's first token gets z_end. Returns the token that was chosen.
static llama_token step(llama_sampler * s, llama_token top, float z_end) {
    std::vector<llama_token_data> cur;
    for (size_t i = 0; i < V; i++) {
        cur.push_back({(llama_token) i, 0.0f, 0.0f});
    }
    cur[(size_t) top].logit = 5.0f;
    if (top != T_END) {
        cur[(size_t) T_END].logit = z_end;
    }
    llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
    llama_sampler_apply(s, &arr);
    size_t best = 0;
    for (size_t i = 1; i < V; i++) {
        if (cur[i].logit > cur[best].logit) {
            best = i;
        }
    }
    const llama_token t = cur[best].id;
    llama_sampler_accept(s, t);
    return t;
}

static std::vector<llama_token> feed(llama_sampler * s, const std::vector<llama_token> & tokens, float z_end = -10.0f) {
    std::vector<llama_token> out;
    for (llama_token t : tokens) {
        out.push_back(step(s, t, z_end));
    }
    return out;
}

static common_reasoning_budget_opts opts_of(int32_t soft, float margin, int32_t grace) {
    common_reasoning_budget_opts o;
    o.soft_tokens = soft;
    o.brake_margin = margin;
    o.grace_tokens = grace;
    o.boundary_ids = {T_DOT};
    return o;
}

// 1. with the default options the sampler is upstream's, step by step (differential fuzz against the plain init)
static void test_default_is_upstream() {
    std::mt19937 rng(7);
    for (int run = 0; run < 3000; run++) {
        const int32_t budget = (int32_t) (rng() % 12);
        const bool two = rng() % 2;
        const std::vector<llama_tokens> start = {{T_START}};
        const std::vector<llama_tokens> end = {two ? llama_tokens{T_END, T_END2} : llama_tokens{T_END}};
        llama_tokens forced = {T_MSG};
        forced.insert(forced.end(), end[0].begin(), end[0].end());
        llama_sampler * a = common_reasoning_budget_init(nullptr, start, end, forced, budget);
        common_reasoning_budget_opts o;   // all off; the boundary list does not matter without a grace window
        o.boundary_ids = {T_DOT};
        llama_sampler * b = common_reasoning_budget_init_ex(nullptr, start, end, forced, budget, REASONING_BUDGET_IDLE, o);
        for (int i = 0; i < 40; i++) {
            const int r = (int) (rng() % 9);
            const llama_token top = r == 0 ? T_START : r == 1 ? T_END : r == 2 ? T_DOT : r == 3 ? T_END2 : (r % 2 ? T_WORD : T_WORD2);
            const float z = (float) (rng() % 11) - 5.0f;
            const llama_token ta = step(a, top, z);
            const llama_token tb = step(b, top, z);
            CHECK(ta == tb);
            CHECK(common_reasoning_budget_get_state(a) == common_reasoning_budget_get_state(b));
        }
        llama_sampler_free(a);
        llama_sampler_free(b);
    }
    printf("  default options == upstream (3000 fuzz runs): ok\n");
}

// 2. below the soft target the sampler never intervenes; from it on ThinkBrake needs a sentence end and a small margin
static void test_soft_and_brake() {
    auto * s = make(100, opts_of(5, 0.1f, 0));
    feed(s, {T_START});
    // used = 3 (< soft 5), at a sentence end, margin 0 -> still nothing
    feed(s, {T_WORD, T_WORD, T_DOT});
    CHECK(step(s, T_WORD, 4.99f) == T_WORD);                 // the end tag is as likely as the top, but we are below the target
    // two more words: used = 5 (>= soft), last token is a word -> no sentence end -> nothing
    feed(s, {T_WORD});
    CHECK(step(s, T_WORD, 4.99f) == T_WORD);
    // a sentence end now; the end tag is far behind -> nothing
    feed(s, {T_DOT});
    CHECK(step(s, T_WORD, 1.0f) == T_WORD);                // margin 4 > 0.1
    // after the next sentence end the end tag is close behind the top (margin 0.05) -> ThinkBrake
    feed(s, {T_DOT});
    CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_COUNTING);
    CHECK(step(s, T_WORD, 4.95f) == T_END);
    CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_DONE);
    const auto info = common_reasoning_budget_get_info(s);
    CHECK(info.brake_hit && !info.ceiling_hit);
    CHECK(info.reasoning_tokens == 9);
    llama_sampler_free(s);

    // the margin is divided by the temperature: raw gap 0.3 at T 0.6 is 0.5 -> no brake at tau 0.1, brake at tau 0.6
    for (float tau : {0.1f, 0.6f}) {
        auto o = opts_of(1, tau, 0);
        o.brake_temperature = 0.6f;
        auto * t = make(100, o);
        feed(t, {T_START, T_WORD, T_DOT});
        const llama_token got = step(t, T_WORD, 4.7f);
        CHECK(got == (tau < 0.4f ? T_WORD : T_END));
        llama_sampler_free(t);
    }

    // no soft target (-1) -> no ThinkBrake, whatever the margin
    {
        auto * t = make(100, opts_of(-1, 100.0f, 0));
        feed(t, {T_START, T_WORD, T_DOT, T_WORD, T_DOT});
        CHECK(step(t, T_WORD, 4.99f) == T_WORD);
        llama_sampler_free(t);
    }
    // brake_margin < 0 -> off
    {
        auto * t = make(100, opts_of(0, -1.0f, 0));
        feed(t, {T_START, T_WORD, T_DOT});
        CHECK(step(t, T_WORD, 4.99f) == T_WORD);
        llama_sampler_free(t);
    }
    printf("  soft target / ThinkBrake: ok\n");
}

// 3. ThinkBrake with a two-token end tag: the first token is chosen, the second is forced, no message in between
static void test_brake_two_token_tag() {
    auto * s = make(100, opts_of(1, 0.1f, 0), /*two_token_end=*/true);
    feed(s, {T_START, T_WORD, T_DOT});
    CHECK(step(s, T_WORD, 4.99f) == T_END);
    CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_FORCING);
    CHECK(step(s, T_WORD, -10.0f) == T_END2);               // forced although the model wants a word
    CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_DONE);
    CHECK(common_reasoning_budget_get_info(s).brake_hit);
    CHECK(common_reasoning_budget_get_end_match(s) != nullptr);
    CHECK(step(s, T_WORD, -10.0f) == T_WORD);               // passthrough again
    llama_sampler_free(s);
    printf("  ThinkBrake, two-token end tag: ok\n");
}

// 4. the grace window at the ceiling
static void test_grace() {
    // ceiling reached in the middle of a sentence: wait for the sentence end, then message + end tag
    {
        auto * s = make(4, opts_of(-1, -1.0f, 6));
        feed(s, {T_START, T_WORD, T_WORD, T_WORD, T_WORD});         // 4 tokens counted: the ceiling
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_WAITING_UTF8);
        CHECK(step(s, T_WORD, -10.0f) == T_WORD);
        CHECK(step(s, T_DOT, -10.0f) == T_DOT);                    // the sentence ends
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_FORCING);
        CHECK(step(s, T_WORD, -10.0f) == T_MSG);
        CHECK(step(s, T_WORD, -10.0f) == T_END);
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_DONE);
        const auto info = common_reasoning_budget_get_info(s);
        CHECK(info.ceiling_hit && !info.brake_hit && info.grace_used == 2);
        llama_sampler_free(s);
    }
    // the sentence never ends: after grace tokens the cut comes anyway
    {
        auto * s = make(3, opts_of(-1, -1.0f, 4));
        feed(s, {T_START, T_WORD, T_WORD, T_WORD});
        feed(s, {T_WORD, T_WORD, T_WORD});
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_WAITING_UTF8);
        feed(s, {T_WORD});                                          // 4th waiting token
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_FORCING);
        CHECK(step(s, T_WORD, -10.0f) == T_MSG);
        CHECK(step(s, T_WORD, -10.0f) == T_END);
        llama_sampler_free(s);
    }
    // the ceiling lands exactly on a sentence end: no waiting
    {
        auto * s = make(3, opts_of(-1, -1.0f, 8));
        feed(s, {T_START, T_WORD, T_WORD, T_DOT});
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_FORCING);
        CHECK(step(s, T_WORD, -10.0f) == T_MSG);
        llama_sampler_free(s);
    }
    // grace 0 is upstream: cut at once, mid-sentence
    {
        auto * s = make(3, opts_of(-1, -1.0f, 0));
        feed(s, {T_START, T_WORD, T_WORD, T_WORD});
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_FORCING);
        llama_sampler_free(s);
    }
    // the model ends the thought by itself while the grace window waits
    {
        auto * s = make(2, opts_of(-1, -1.0f, 8));
        feed(s, {T_START, T_WORD, T_WORD});
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_WAITING_UTF8);
        CHECK(step(s, T_END, -10.0f) == T_END);
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_DONE);
        CHECK(!common_reasoning_budget_get_info(s).ceiling_hit);
        llama_sampler_free(s);
    }
    printf("  grace window at the ceiling: ok\n");
}

// 5. only inside the thinking block; every new block starts afresh
static void test_scope_and_rearm() {
    auto * s = make(60, opts_of(4, 0.1f, 0));
    // before <think> nothing is touched, however close the end tag is
    CHECK(step(s, T_DOT, 4.99f) == T_DOT);
    CHECK(step(s, T_WORD, 4.99f) == T_WORD);
    CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_IDLE);
    feed(s, {T_START, T_WORD, T_WORD, T_WORD, T_DOT});              // used 4 = soft, at a sentence end
    CHECK(step(s, T_WORD, 4.99f) == T_END);                          // ThinkBrake inside the block
    CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_DONE);
    // after the block: passthrough, even at a sentence end with the end tag close
    CHECK(step(s, T_DOT, 4.99f) == T_DOT);
    CHECK(step(s, T_WORD, 4.99f) == T_WORD);
    // a second <think> block: the counters start again (used 2 < soft 4 at its first sentence end), then the brake works again
    feed(s, {T_START});
    CHECK(common_reasoning_budget_get_info(s).reasoning_tokens == 0);
    feed(s, {T_WORD, T_DOT});
    CHECK(step(s, T_WORD, 4.99f) == T_WORD);                         // used 2 -> 3 after this word
    feed(s, {T_DOT});                                               // used 4, at a sentence end
    CHECK(step(s, T_WORD, 4.99f) == T_END);
    llama_sampler_free(s);
    printf("  scope and re-arm: ok\n");
}

// 6. speculative decoding: the verification step samples position by position (common_sampler_sample_and_accept_n): a draft token
// is kept only while the sampler agrees with it. ThinkBrake and the ceiling force a token; a draft that disagrees ends the loop there.
static std::vector<llama_token> verify(llama_sampler * s, const std::vector<llama_token> & draft, const std::vector<llama_token> & model_top, float z_end) {
    std::vector<llama_token> result;
    size_t i = 0;
    for (; i < draft.size(); i++) {
        const llama_token id = step(s, model_top[i], z_end);
        result.push_back(id);
        if (draft[i] != id) {
            break;
        }
    }
    if (i == draft.size()) {
        result.push_back(step(s, model_top[i], z_end));
    }
    return result;
}

static void test_speculative() {
    // the draft is accepted up to the sentence end, the brake fires at the next position, the draft there is rejected
    {
        auto * s = make(100, opts_of(2, 0.1f, 0));
        feed(s, {T_START, T_WORD, T_WORD});
        const auto r = verify(s, {T_WORD, T_DOT, T_WORD}, {T_WORD, T_DOT, T_WORD, T_WORD}, 4.95f);
        // position 0: the last token (a word) is no sentence end -> nothing; position 1: still a word before it -> nothing;
        // after the T_DOT is accepted the sentence end makes position 2 a brake point -> the end tag, the draft word is rejected
        CHECK((r == std::vector<llama_token>{T_WORD, T_DOT, T_END}));
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_DONE);
        llama_sampler_free(s);
    }
    // the ceiling: the forced message token replaces a drafted word, the rest of the draft is dropped
    {
        auto * s = make(3, opts_of(-1, -1.0f, 0));
        feed(s, {T_START, T_WORD});
        const auto r = verify(s, {T_WORD, T_WORD, T_WORD}, {T_WORD, T_WORD, T_WORD, T_WORD}, -10.0f);
        // budget 3: the first draft word is the 2nd counted token, the second the 3rd (the ceiling); the third position is forced
        CHECK((r == std::vector<llama_token>{T_WORD, T_WORD, T_MSG}));
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_FORCING);
        const auto r2 = verify(s, {T_WORD}, {T_WORD, T_WORD}, -10.0f);
        CHECK((r2 == std::vector<llama_token>{T_END}));
        llama_sampler_free(s);
    }
    // a draft that already is the forced token is simply accepted
    {
        auto * s = make(100, opts_of(1, 0.1f, 0));
        feed(s, {T_START, T_WORD, T_DOT});
        const auto r = verify(s, {T_END, T_WORD}, {T_WORD, T_WORD, T_WORD}, 4.99f);
        CHECK(r.size() == 3 && r[0] == T_END && r[1] == T_WORD);
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_DONE);
        llama_sampler_free(s);
    }
    printf("  speculative decoding (verification step): ok\n");
}

// 7. what counts as a sentence end
static void test_pieces() {
    CHECK(common_reasoning_piece_ends_sentence("."));
    CHECK(common_reasoning_piece_ends_sentence(" done."));
    CHECK(common_reasoning_piece_ends_sentence("?"));
    CHECK(common_reasoning_piece_ends_sentence("!\n"));
    CHECK(common_reasoning_piece_ends_sentence(".)"));
    CHECK(common_reasoning_piece_ends_sentence(".\""));
    CHECK(common_reasoning_piece_ends_sentence("...") );
    CHECK(common_reasoning_piece_ends_sentence("\xE2\x80\xA6"));          // …
    CHECK(common_reasoning_piece_ends_sentence("\xE3\x80\x82"));          // 。
    CHECK(common_reasoning_piece_ends_sentence("\xEF\xBC\x81"));          // ！
    CHECK(common_reasoning_piece_ends_sentence("\xEF\xBC\x9F"));          // ？
    CHECK(common_reasoning_piece_ends_sentence("\xD8\x9F"));              // ؟
    CHECK(common_reasoning_piece_ends_sentence("\xDB\x94"));              // ۔
    CHECK(common_reasoning_piece_ends_sentence("\xE0\xA5\xA4"));          // ।
    CHECK(common_reasoning_piece_ends_sentence("\n\n"));
    CHECK(common_reasoning_piece_ends_sentence("end.\n\n"));
    CHECK(!common_reasoning_piece_ends_sentence("\n"));                   // a single line break is not a sentence end
    CHECK(!common_reasoning_piece_ends_sentence("word"));
    CHECK(!common_reasoning_piece_ends_sentence(","));
    CHECK(!common_reasoning_piece_ends_sentence(":"));
    CHECK(!common_reasoning_piece_ends_sentence(""));
    CHECK(!common_reasoning_piece_ends_sentence("."  , '3'));             // "3." of "3.14"
    CHECK(!common_reasoning_piece_ends_sentence(" .", '7'));
    CHECK(common_reasoning_piece_ends_sentence(".", 'd'));
    CHECK(common_reasoning_piece_ends_sentence("x.", '3'));               // a longer piece is judged by itself
    printf("  sentence ends by text: ok\n");
}

// 7b. an end tag with leading whitespace ("\n</think>", as chat templates give it): ThinkBrake compares the end-of-thinking token,
// not the line break, and a sentence end followed by a line break is still a sentence end
static llama_sampler * make_ws(const std::vector<llama_tokens> & end, const common_reasoning_budget_opts & o) {
    llama_tokens forced = {T_MSG};
    forced.insert(forced.end(), end[0].begin(), end[0].end());
    return common_reasoning_budget_init_ex(nullptr, {{T_START}}, end, forced, 100, REASONING_BUDGET_IDLE, o);
}

static void test_leading_whitespace_tag() {
    auto o = opts_of(2, 0.1f, 0);
    o.space_ids = {T_NL};
    // the template's tags: "\n</think>" first, "</think>" too
    {
        auto * s = make_ws({{T_NL, T_END}, {T_END}}, o);
        feed(s, {T_START, T_WORD, T_WORD, T_DOT});
        // the line break is the likeliest token after the sentence; the end tag is far behind -> no brake (the old code compared
        // the line break, the tag's first token, and fired here)
        std::vector<llama_token_data> cur;
        for (size_t i = 0; i < V; i++) cur.push_back({(llama_token) i, 0.0f, 0.0f});
        cur[T_NL].logit = 9.0f; cur[T_WORD].logit = 5.0f; cur[T_END].logit = -3.0f;
        llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(s, &arr);
        CHECK(arr.data[T_WORD].logit == 5.0f && arr.data[T_NL].logit == 9.0f);  // untouched
        llama_sampler_accept(s, T_NL);
        // after the line break the sentence end still counts; now the end tag is close behind the top -> ThinkBrake ends the thought
        CHECK(step(s, T_WORD, 4.95f) == T_END);
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_DONE);
        CHECK(common_reasoning_budget_get_info(s).brake_hit);
        CHECK(step(s, T_WORD, -10.0f) == T_WORD);
        llama_sampler_free(s);
    }
    // only "\n</think>" exists: the brake picks its second token and nothing is left to force
    {
        auto * s = make_ws({{T_NL, T_END}}, o);
        feed(s, {T_START, T_WORD, T_WORD, T_DOT});
        CHECK(step(s, T_WORD, 4.95f) == T_END);
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_DONE);
        CHECK(common_reasoning_budget_get_info(s).brake_hit);
        CHECK(step(s, T_WORD, -10.0f) == T_WORD);
        llama_sampler_free(s);
    }
    // a word after the line break: no longer a sentence end
    {
        auto * s = make_ws({{T_NL, T_END}, {T_END}}, o);
        feed(s, {T_START, T_WORD, T_WORD, T_DOT, T_NL, T_WORD});
        CHECK(step(s, T_WORD, 4.99f) == T_WORD);
        llama_sampler_free(s);
    }
    printf("  end tag with leading whitespace: ok\n");
}

// 8. counters and shadow mode (FRESH1): where the brake / ceiling fired, or would have fired, without changing a single token
static void test_counters() {
    // ThinkBrake: 9 thinking tokens before the end tag
    {
        auto * s = make(100, opts_of(5, 0.1f, 0));
        feed(s, {T_START, T_WORD, T_WORD, T_DOT, T_WORD, T_WORD, T_WORD, T_DOT, T_WORD, T_DOT});
        CHECK(step(s, T_WORD, 4.95f) == T_END);
        const auto info = common_reasoning_budget_get_info(s);
        CHECK(info.brake_hit && info.first_fire_token == 9 && !info.shadow);
        llama_sampler_free(s);
    }
    // the ceiling with a grace window: the message starts after 6 thinking tokens
    {
        auto * s = make(4, opts_of(-1, -1.0f, 6));
        feed(s, {T_START, T_WORD, T_WORD, T_WORD, T_WORD, T_WORD, T_DOT});
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_FORCING);
        const auto info = common_reasoning_budget_get_info(s);
        CHECK(info.ceiling_hit && info.first_fire_token == 6);
        llama_sampler_free(s);
    }
    // nothing fired: -1
    {
        auto * s = make(100, opts_of(5, 0.1f, 0));
        feed(s, {T_START, T_WORD, T_WORD, T_END});
        const auto info = common_reasoning_budget_get_info(s);
        CHECK(!info.brake_hit && !info.ceiling_hit && info.first_fire_token == -1 && info.reasoning_tokens == 2);
        llama_sampler_free(s);
    }
    // prefill: "<think>" and one token placed by the chat template are not generated, the info counts generated tokens only
    {
        auto * s = make(100, opts_of(5, 0.1f, 0));
        llama_sampler_accept(s, T_START);
        llama_sampler_accept(s, T_WORD);
        common_reasoning_budget_mark_generation_start(s);
        feed(s, {T_WORD, T_WORD, T_DOT, T_WORD, T_DOT});              // 6 counted by the budget, 5 generated
        CHECK(step(s, T_WORD, 4.95f) == T_END);                       // soft 5 counts the prefill token, as the budget does
        const auto info = common_reasoning_budget_get_info(s);
        CHECK(info.reasoning_tokens == 5 && info.first_fire_token == 5);
        llama_sampler_free(s);
    }
    printf("  counters (first intervention, prefill not counted): ok\n");
}

static common_reasoning_budget_opts shadow_of(int32_t soft, float margin, int32_t grace) {
    auto o = opts_of(soft, margin, grace);
    o.shadow = true;
    return o;
}

static void test_shadow() {
    // a. tokens are exactly those of an unlimited upstream sampler, whatever the options (differential fuzz)
    std::mt19937 rng(11);
    for (int run = 0; run < 3000; run++) {
        const int32_t budget = (int32_t) (rng() % 12);
        const bool two = rng() % 2;
        const std::vector<llama_tokens> start = {{T_START}};
        const std::vector<llama_tokens> end = {two ? llama_tokens{T_END, T_END2} : llama_tokens{T_END}};
        llama_tokens forced = {T_MSG};
        forced.insert(forced.end(), end[0].begin(), end[0].end());
        llama_sampler * a = common_reasoning_budget_init(nullptr, start, end, forced, INT32_MAX);
        const auto o = shadow_of((int32_t) (rng() % 6) - 1, (rng() % 2) ? 0.5f : 100.0f, (int32_t) (rng() % 5));
        llama_sampler * b = common_reasoning_budget_init_ex(nullptr, start, end, forced, budget, REASONING_BUDGET_IDLE, o);
        for (int i = 0; i < 40; i++) {
            const int r = (int) (rng() % 9);
            const llama_token top = r == 0 ? T_START : r == 1 ? T_END : r == 2 ? T_DOT : r == 3 ? T_END2 : (r % 2 ? T_WORD : T_WORD2);
            const float z = (float) (rng() % 11) - 5.0f;
            CHECK(step(a, top, z) == step(b, top, z));
            CHECK(common_reasoning_budget_get_state(b) != REASONING_BUDGET_FORCING);
        }
        llama_sampler_free(a);
        llama_sampler_free(b);
    }
    // b. a seeded long thought: the brake would fire at the first sentence end past the soft target where the end tag is close
    {
        auto * s = make(1000, shadow_of(20, 0.1f, 48));
        feed(s, {T_START});
        std::vector<llama_token> thought;
        for (int i = 0; i < 40; i++) {
            thought.push_back(i % 8 == 7 ? T_DOT : T_WORD);      // a sentence end every 8 tokens: after 8, 16, 24, 32, 40
        }
        const auto out = feed(s, thought, 4.95f);              // the end tag is always close behind the top
        CHECK(out == thought);                                  // nothing forced
        auto info = common_reasoning_budget_get_info(s);
        // soft 20: the first sentence end at or past 20 counted tokens is after token 24 -> the end tag would have been token 24
        CHECK(info.shadow && info.brake_hit && !info.ceiling_hit && info.first_fire_token == 24);
        CHECK(info.reasoning_tokens == 40);
        CHECK(step(s, T_END, -10.0f) == T_END);                 // the model ends the thought itself later
        CHECK(common_reasoning_budget_get_state(s) == REASONING_BUDGET_DONE);
        info = common_reasoning_budget_get_info(s);
        CHECK(info.first_fire_token == 24);                     // the first would-be intervention is kept
        llama_sampler_free(s);
    }
    // c. the ceiling in shadow: budget 10, grace 4: words only -> would cut after 10 + 4 tokens; with a sentence end at 12 -> 12
    for (bool dot : {false, true}) {
        auto * s = make(10, shadow_of(-1, -1.0f, 4));
        feed(s, {T_START});
        std::vector<llama_token> thought(30, T_WORD);
        if (dot) {
            thought[11] = T_DOT;
        }
        CHECK(feed(s, thought) == thought);
        const auto info = common_reasoning_budget_get_info(s);
        CHECK(info.ceiling_hit && !info.brake_hit && info.first_fire_token == (dot ? 12 : 14) && info.reasoning_tokens == 30);
        llama_sampler_free(s);
    }
    // d. budget 0 in shadow: would cut at once, nothing forced
    {
        auto * s = make(0, shadow_of(-1, -1.0f, 0));
        CHECK((feed(s, {T_START, T_WORD, T_WORD}) == std::vector<llama_token>{T_START, T_WORD, T_WORD}));
        const auto info = common_reasoning_budget_get_info(s);
        CHECK(info.ceiling_hit && info.first_fire_token == 0);
        llama_sampler_free(s);
    }
    // e. speculative drafts: only accepted tokens are counted (a rejected draft token never reaches accept())
    {
        auto * s = make(1000, shadow_of(-1, -1.0f, 0));
        feed(s, {T_START});
        const auto r = verify(s, {T_WORD, T_WORD, T_WORD}, {T_WORD, T_DOT, T_WORD, T_WORD}, -10.0f);
        CHECK((r == std::vector<llama_token>{T_WORD, T_DOT}));   // the second draft word is rejected, the third never verified
        CHECK(common_reasoning_budget_get_info(s).reasoning_tokens == 2);
        llama_sampler_free(s);
    }
    printf("  shadow mode (3000 fuzz runs == unlimited upstream, would-fire positions, drafts): ok\n");
}

int main(void) {
    printf("Testing reasoning effort (soft target, ThinkBrake, grace):\n");
    test_default_is_upstream();
    test_soft_and_brake();
    test_brake_two_token_tag();
    test_grace();
    test_scope_and_rearm();
    test_speculative();
    test_pieces();
    test_leading_whitespace_tag();
    test_counters();
    test_shadow();
    printf("OK\n");
    return 0;
}
