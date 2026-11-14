#include "reasoning-budget.h"
#include "common.h"
#include "trie.h"
#include "unicode.h"

#include "log.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

struct token_matcher {
    std::vector<llama_tokens> seqs;
    common_aho_corasick ac;
    size_t state = 0;

    token_matcher(const std::vector<llama_tokens> & seqs) : seqs(collect(seqs)), ac(build_trie(this->seqs)) {}

    static std::vector<llama_tokens> collect(const std::vector<llama_tokens> & seqs) {
        std::vector<llama_tokens> res;
        for (const auto & seq : seqs) {
            if (!seq.empty() && std::find(res.begin(), res.end(), seq) == res.end()) {
                res.push_back(seq);
            }
        }
        return res;
    }

    static common_trie build_trie(const std::vector<llama_tokens> & seqs) {
        common_trie t;
        for (const auto & seq : seqs) {
            t.insert(std::vector<uint32_t>(seq.begin(), seq.end()));
        }
        return t;
    }

    // returns the index into seqs of the longest sequence ending at this token, or -1
    int32_t advance(llama_token token) {
        state = ac.next(state, (uint32_t) token);
        const int32_t p = ac.match_pattern(state);
        if (p >= 0) {
            state = 0;
        }
        return p;
    }

    void reset() { state = 0; }
};

struct common_reasoning_budget_ctx {
    const llama_vocab * vocab;

    token_matcher start_matcher;
    token_matcher end_matcher;
    llama_tokens forced_tokens;

    int32_t budget;           // maximum tokens in reasoning block
    int32_t remaining;        // tokens remaining in budget

    common_reasoning_budget_state state;

    // for forcing
    size_t force_pos;         // next position in forced_tokens to force

    int32_t end_match;        // index into end_matcher.seqs of the sequence that transitioned to DONE, -1 if none

    // THINK3b: soft target, ThinkBrake, grace window at the ceiling (all off with the default options)
    common_reasoning_budget_opts opts;
    int32_t used          = 0;      // tokens counted inside the current thinking block
    int32_t grace_left    = 0;      // tokens the grace window may still wait for a sentence boundary
    bool    at_boundary   = false;  // the last accepted token ended a sentence
    char    last_char     = ' ';    // last character of the last accepted token's text
    bool    brake_armed   = false;  // apply() chose the end tag for this step; accept() commits it if that token is sampled
    bool    brake_forcing = false;  // FORCING the end tag alone (ThinkBrake), not the message + end tag
    bool    brake_would   = false;  // shadow mode: apply() found that ThinkBrake would fire at this step
    int32_t fire_pos      = -1;     // shadow mode: thinking tokens before the step where it would fire
    bool    shadow_done   = false;  // shadow mode: the first would-be intervention is recorded; the real thought ends there
    size_t  brake_seq     = 0;      // ThinkBrake: the end sequence it forces (index into end_matcher.seqs)
    size_t  brake_pos     = 0;      //   and the position of its end-of-thinking token in it (past leading whitespace)
    bool    brake_ready   = false;  //   brake_seq / brake_pos were chosen
    int32_t block_no      = 0;      // thinking blocks started so far
    int32_t gen_block     = -1;     // the block that was open when generation started (its prefill tokens are not reported)
    int32_t gen_offset    = 0;      // thinking tokens of that block that came from the prompt
    common_reasoning_budget_info info = {};

    // the sequence FORCING walks through
    const llama_tokens & forcing_seq() const {
        return brake_forcing ? end_matcher.seqs[brake_seq] : forced_tokens;
    }
};

static bool utf8_ends_with(const std::string & s, size_t e, const char * suffix) {
    const size_t n = strlen(suffix);
    return e >= n && s.compare(e - n, n, suffix) == 0;
}

bool common_reasoning_piece_ends_sentence(const std::string & piece, char prev_char) {
    const size_t n = piece.size();
    if (n >= 2 && piece.compare(n - 2, 2, "\n\n") == 0) {
        return true; // a blank line ends a thought
    }
    size_t e = n;
    // trailing spaces, line ends, then closing quotes and brackets
    for (bool again = true; again && e > 0; ) {
        again = false;
        const char c = piece[e - 1];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == ')' || c == ']' || c == '}' || c == '"' || c == '\'') {
            e--; again = true;
            continue;
        }
        for (const char * q : { "\xC2\xBB", "\xE2\x80\x9D", "\xE2\x80\x99", "\xEF\xBC\x89", "\xE3\x80\x8D", "\xE3\x80\x8F" }) {
            if (utf8_ends_with(piece, e, q)) {
                e -= strlen(q); again = true;
                break;
            }
        }
    }
    if (e == 0) {
        return false;
    }
    const char c = piece[e - 1];
    if (c == '!' || c == '?') {
        return true;
    }
    if (c == '.') {
        // a lone "." right after a digit is a decimal point or a list number ("3.14", "2."), not a full stop
        size_t b = 0;
        while (b < e && piece[b] == ' ') b++;
        const bool lone = (e - b == 1);
        if (lone && prev_char >= '0' && prev_char <= '9') {
            return false;
        }
        return true;
    }
    for (const char * t : { "\xE2\x80\xA6", "\xE3\x80\x82", "\xEF\xBC\x81", "\xEF\xBC\x9F", "\xEF\xBC\x8E",
                            "\xD8\x9F", "\xDB\x94", "\xE0\xA5\xA4", "\xE0\xA5\xA5" }) {
        if (utf8_ends_with(piece, e, t)) {
            return true;
        }
    }
    return false;
}

// a token whose text is only spaces / tabs / line ends (or one the caller listed as such)
static bool common_reasoning_budget_is_space(const common_reasoning_budget_ctx * ctx, llama_token token) {
    if (std::find(ctx->opts.space_ids.begin(), ctx->opts.space_ids.end(), token) != ctx->opts.space_ids.end()) {
        return true;
    }
    if (ctx->vocab == nullptr) {
        return false;
    }
    const std::string piece = common_token_to_piece(ctx->vocab, token, false);
    return !piece.empty() && piece.find_first_not_of(" \t\r\n") == std::string::npos;
}

// ThinkBrake compares the end-of-thinking token, not leading whitespace of the tag: chat templates often end the thought with
// "\n</think>", and a line break is likely after any sentence. Prefer a tag variant without leading whitespace.
static void common_reasoning_budget_pick_brake_token(common_reasoning_budget_ctx * ctx) {
    if (ctx->brake_ready || ctx->end_matcher.seqs.empty()) {
        return;
    }
    ctx->brake_ready = true;
    bool found = false;
    for (size_t s = 0; s < ctx->end_matcher.seqs.size() && !found; s++) {
        const auto & seq = ctx->end_matcher.seqs[s];
        size_t k = 0;
        while (k + 1 < seq.size() && common_reasoning_budget_is_space(ctx, seq[k])) {
            k++;
        }
        if (k == 0) {
            ctx->brake_seq = s; ctx->brake_pos = 0; found = true;
        } else if (s == 0) {
            ctx->brake_seq = 0; ctx->brake_pos = k;   // fallback: the first tag past its leading whitespace
        }
    }
}

static llama_token common_reasoning_budget_brake_token(const common_reasoning_budget_ctx * ctx) {
    return ctx->end_matcher.seqs[ctx->brake_seq][ctx->brake_pos];
}

static const char * common_reasoning_budget_name(const struct llama_sampler * /*smpl*/) {
    return "reasoning-budget";
}

static void common_reasoning_budget_block_start(common_reasoning_budget_ctx * ctx) {
    ctx->used          = 0;
    ctx->grace_left    = 0;
    ctx->at_boundary   = false;
    ctx->last_char     = ' ';
    ctx->brake_armed   = false;
    ctx->brake_forcing = false;
    ctx->brake_would   = false;
    ctx->fire_pos      = -1;
    ctx->shadow_done   = false;
    ctx->block_no++;
    ctx->info          = {};
    ctx->info.shadow   = ctx->opts.shadow;
}

// shadow mode: record the first would-be intervention once; afterwards only the natural end of the thought is tracked
static void common_reasoning_budget_shadow_fire(common_reasoning_budget_ctx * ctx, bool brake, int32_t pos) {
    if (ctx->shadow_done) {
        return;
    }
    ctx->shadow_done = true;
    if (brake) {
        ctx->info.brake_hit = true;
    } else {
        ctx->info.ceiling_hit = true;
    }
    ctx->info.first_fire_token = pos;
    ctx->state = REASONING_BUDGET_COUNTING;
    COM_TRC("shadow: %s would fire after %d thinking tokens\n", brake ? "ThinkBrake" : "the ceiling", pos);
}

static void common_reasoning_budget_accept(struct llama_sampler * smpl, llama_token token) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    switch (ctx->state) {
        case REASONING_BUDGET_IDLE:
        {
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                common_reasoning_budget_block_start(ctx);
                COM_TRC("activated, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    if (ctx->opts.shadow) {
                        common_reasoning_budget_shadow_fire(ctx, false, 0);
                    } else {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        COM_TRC("%s", "budget=0, forcing immediately\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_COUNTING:
        case REASONING_BUDGET_WAITING_UTF8:
        {
            const bool armed = ctx->brake_armed;
            const bool would = ctx->brake_would;
            ctx->brake_armed = false;
            ctx->brake_would = false;

            const int32_t match = ctx->end_matcher.advance(token);
            if (match >= 0) {
                if (would) {
                    // shadow: the model ended the thought itself at the very step the brake would have ended it
                    common_reasoning_budget_shadow_fire(ctx, true, ctx->fire_pos);
                }
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                if (armed && ctx->state == REASONING_BUDGET_DONE) {
                    ctx->info.brake_hit = true; // ThinkBrake chose the (single-token) end tag
                    if (ctx->info.first_fire_token < 0) {
                        ctx->info.first_fire_token = ctx->used;
                    }
                }
                COM_TRC("%s", "deactivated (natural end)\n");
                break;
            }

            bool utf8_complete = true;
            std::string piece;
            if (ctx->vocab != nullptr) {
                piece = common_token_to_piece(ctx->vocab, token, false);
                utf8_complete = common_utf8_is_complete(piece);
            }

            // sentence boundary: by the token's text, or a token the caller named
            const bool listed = std::find(ctx->opts.boundary_ids.begin(), ctx->opts.boundary_ids.end(), token) != ctx->opts.boundary_ids.end();
            const bool space  = !listed && common_reasoning_budget_is_space(ctx, token);
            if (space) {
                // whitespace after a sentence end keeps it a sentence end (".", then "\n"); a blank line is one by itself
                ctx->at_boundary = ctx->at_boundary || (!piece.empty() && common_reasoning_piece_ends_sentence(piece, ctx->last_char));
            } else {
                ctx->at_boundary = listed || (!piece.empty() && common_reasoning_piece_ends_sentence(piece, ctx->last_char));
                if (!piece.empty()) {
                    ctx->last_char = piece.back();
                }
            }
            ctx->used++;
            ctx->info.reasoning_tokens = ctx->used;

            if (ctx->opts.shadow) {
                // shadow mode: the same decisions as below, recorded instead of forced
                if (would) {
                    common_reasoning_budget_shadow_fire(ctx, true, ctx->fire_pos);
                }
                if (ctx->shadow_done) {
                    break;
                }
                if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                    ctx->grace_left--;
                    ctx->info.grace_used++;
                    if (utf8_complete && (ctx->at_boundary || ctx->grace_left <= 0)) {
                        common_reasoning_budget_shadow_fire(ctx, false, ctx->used);
                    }
                } else if (--ctx->remaining <= 0) {
                    if (utf8_complete && (ctx->opts.grace_tokens <= 0 || ctx->at_boundary)) {
                        common_reasoning_budget_shadow_fire(ctx, false, ctx->used);
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->grace_left = ctx->opts.grace_tokens;
                    }
                }
                break;
            }

            if (armed && token == common_reasoning_budget_brake_token(ctx)) {
                // ThinkBrake picked the end-of-thinking token and it was sampled: the rest of the tag follows, without the message
                ctx->state = REASONING_BUDGET_FORCING;
                ctx->brake_forcing = true;
                ctx->force_pos = ctx->brake_pos + 1;
                ctx->info.brake_hit = true;
                if (ctx->info.first_fire_token < 0) {
                    ctx->info.first_fire_token = ctx->used - 1;
                }
                if (ctx->force_pos >= ctx->forcing_seq().size()) {
                    // the end-of-thinking token was the tag's last one (its leading whitespace was skipped): nothing left to force
                    ctx->state = REASONING_BUDGET_DONE;
                    ctx->end_match = (int32_t) ctx->brake_seq;
                }
                COM_TRC("%s", "ThinkBrake: end tag started, forcing the rest of it\n");
                break;
            }

            if (ctx->state == REASONING_BUDGET_WAITING_UTF8) {
                ctx->grace_left--;
                ctx->info.grace_used++;
                if (utf8_complete && (ctx->at_boundary || ctx->grace_left <= 0)) {
                    ctx->state = REASONING_BUDGET_FORCING;
                    ctx->force_pos = 0;
                    ctx->info.ceiling_hit = true;
                    if (ctx->info.first_fire_token < 0) {
                        ctx->info.first_fire_token = ctx->used;
                    }
                    ctx->end_matcher.reset();
                    COM_TRC("%s", "UTF-8 complete / sentence end / grace over, now forcing end sequence\n");
                }
            } else if (ctx->state == REASONING_BUDGET_COUNTING) {
                ctx->remaining--;
                if (ctx->remaining <= 0) {
                    if (utf8_complete && (ctx->opts.grace_tokens <= 0 || ctx->at_boundary)) {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        ctx->info.ceiling_hit = true;
                        if (ctx->info.first_fire_token < 0) {
                            ctx->info.first_fire_token = ctx->used;
                        }
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, forcing end sequence\n");
                    } else {
                        ctx->state = REASONING_BUDGET_WAITING_UTF8;
                        ctx->grace_left = ctx->opts.grace_tokens;
                        ctx->end_matcher.reset();
                        COM_TRC("%s", "budget exhausted, waiting for UTF-8 completion / a sentence end\n");
                    }
                }
            }
            break;
        }
        case REASONING_BUDGET_FORCING:
        {
            // track the end sequence within forced_tokens so it is also reported on DONE
            const int32_t match = ctx->end_matcher.advance(token);
            ctx->force_pos++;
            // an end tag that is already in the prompt (e.g. "<think></think>") also ends forcing
            if (match >= 0 || ctx->force_pos >= ctx->forcing_seq().size()) {
                ctx->state = REASONING_BUDGET_DONE;
                ctx->end_match = match;
                COM_TRC("%s", "forced sequence complete, done\n");
            }
            break;
        }
        case REASONING_BUDGET_DONE:
            // Re-arm on a new start tag: some models emit multiple <think> blocks
            // per response, and each should get a fresh budget window.
            if (ctx->start_matcher.advance(token) >= 0) {
                ctx->state = REASONING_BUDGET_COUNTING;
                ctx->remaining = ctx->budget;
                ctx->end_matcher.reset();
                ctx->end_match = -1;
                common_reasoning_budget_block_start(ctx);
                COM_TRC("re-activated on new start tag, budget=%d tokens\n", ctx->budget);

                if (ctx->remaining <= 0) {
                    if (ctx->opts.shadow) {
                        common_reasoning_budget_shadow_fire(ctx, false, 0);
                    } else {
                        ctx->state = REASONING_BUDGET_FORCING;
                        ctx->force_pos = 0;
                        COM_TRC("%s", "budget=0, forcing immediately\n");
                    }
                }
            }
            break;
    }
}

// ThinkBrake: should the thought end here? At a sentence boundary past the soft target, the first token of the end tag is about as
// likely as the best other token: (logit_best - logit_end) / T <= margin (a log-probability margin, arXiv 2510.00546).
static bool common_reasoning_budget_brake_fires(const common_reasoning_budget_ctx * ctx, const llama_token_data_array * cur_p) {
    if (ctx->state != REASONING_BUDGET_COUNTING || ctx->opts.soft_tokens < 0 || ctx->opts.brake_margin < 0.0f ||
        !ctx->at_boundary || ctx->used < ctx->opts.soft_tokens || ctx->end_matcher.seqs.empty()) {
        return false;
    }

    const llama_token end0 = common_reasoning_budget_brake_token(ctx);
    float z_end = -INFINITY, z_top = -INFINITY;
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id == end0) {
            z_end = cur_p->data[i].logit;
        } else if (cur_p->data[i].logit > z_top) {
            z_top = cur_p->data[i].logit;
        }
    }
    if (z_end == -INFINITY) {
        return false; // the end tag is not a candidate (a sampler before us removed it): leave the thought alone
    }
    const float t = ctx->opts.brake_temperature > 0.0f ? ctx->opts.brake_temperature : 1.0f;
    return (z_top - z_end) / t <= ctx->opts.brake_margin;
}

static void common_reasoning_budget_apply(struct llama_sampler * smpl, llama_token_data_array * cur_p) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    if (ctx->state == REASONING_BUDGET_COUNTING) {
        common_reasoning_budget_pick_brake_token(ctx);
    }

    if (ctx->state == REASONING_BUDGET_COUNTING && ctx->opts.shadow) {
        // shadow mode: only note that the brake would fire here; the logits stay untouched
        ctx->brake_would = !ctx->shadow_done && common_reasoning_budget_brake_fires(ctx, cur_p);
        if (ctx->brake_would) {
            ctx->fire_pos = ctx->used;
        }
        return;
    }

    if (ctx->state == REASONING_BUDGET_COUNTING) {
        // nothing is changed in the state here: accept() commits ThinkBrake only if the end tag's first token is the one sampled
        ctx->brake_armed = common_reasoning_budget_brake_fires(ctx, cur_p);
        if (ctx->brake_armed) {
            const llama_token end0 = common_reasoning_budget_brake_token(ctx);
            for (size_t i = 0; i < cur_p->size; i++) {
                if (cur_p->data[i].id != end0) {
                    cur_p->data[i].logit = -INFINITY;
                }
            }
        }
        return;
    }

    if (ctx->state != REASONING_BUDGET_FORCING) {
        // passthrough — don't modify logits
        return;
    }

    const llama_tokens & seq = ctx->forcing_seq();
    if (ctx->force_pos >= seq.size()) {
        return;
    }

    const llama_token forced = seq[ctx->force_pos];

    // set all logits to -inf except the forced token
    for (size_t i = 0; i < cur_p->size; i++) {
        if (cur_p->data[i].id != forced) {
            cur_p->data[i].logit = -INFINITY;
        }
    }
}

static void common_reasoning_budget_reset(struct llama_sampler * smpl) {
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    ctx->state = REASONING_BUDGET_IDLE;
    ctx->remaining = ctx->budget;
    ctx->start_matcher.reset();
    ctx->end_matcher.reset();
    ctx->force_pos = 0;
    ctx->end_match = -1;
    common_reasoning_budget_block_start(ctx);
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab * vocab, const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs, const llama_tokens & forced_tokens,
        int32_t budget, common_reasoning_budget_state initial_state, const common_reasoning_budget_opts & opts);

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl);

static void common_reasoning_budget_free(struct llama_sampler * smpl) {
    delete (common_reasoning_budget_ctx *) smpl->ctx;
}

static struct llama_sampler_i common_reasoning_budget_i = {
    /* .name              = */ common_reasoning_budget_name,
    /* .accept            = */ common_reasoning_budget_accept,
    /* .apply             = */ common_reasoning_budget_apply,
    /* .reset             = */ common_reasoning_budget_reset,
    /* .clone             = */ common_reasoning_budget_clone,
    /* .free              = */ common_reasoning_budget_free,
    /* .backend_init      = */ nullptr,
    /* .backend_accept    = */ nullptr,
    /* .backend_apply     = */ nullptr,
    /* .backend_set_input = */ nullptr,
    /* .backend_reset     = */ nullptr,
    /* .copy_state        = */ nullptr,
};

static struct llama_sampler * common_reasoning_budget_clone(const struct llama_sampler * smpl) {
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx(*ctx)
    );
}

static struct llama_sampler * common_reasoning_budget_init_state(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        const common_reasoning_budget_opts & opts) {
    // promote COUNTING with budget <= 0 to FORCING (not in shadow mode, which never forces)
    if (initial_state == REASONING_BUDGET_COUNTING && budget <= 0 && !opts.shadow) {
        initial_state = REASONING_BUDGET_FORCING;
    }

    return llama_sampler_init(
        /* .iface = */ &common_reasoning_budget_i,
        /* .ctx   = */ new common_reasoning_budget_ctx {
            /* .vocab         = */ vocab,
            /* .start_matcher = */ token_matcher(start_seqs),
            /* .end_matcher   = */ token_matcher(end_seqs),
            /* .forced_tokens = */ forced_tokens,
            /* .budget        = */ budget,
            /* .remaining     = */ budget,
            /* .state         = */ initial_state,
            /* .force_pos     = */ 0,
            /* .end_match     = */ -1,
            /* .opts          = */ opts,
        }
    );
}

struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state) {
    return common_reasoning_budget_init_state(vocab, start_seqs, end_seqs, forced_tokens, budget, initial_state, common_reasoning_budget_opts{});
}

struct llama_sampler * common_reasoning_budget_init_ex(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        const common_reasoning_budget_opts & opts) {
    return common_reasoning_budget_init_state(vocab, start_seqs, end_seqs, forced_tokens, budget, initial_state, opts);
}

common_reasoning_budget_info common_reasoning_budget_get_info(const struct llama_sampler * smpl) {
    if (!smpl) {
        return {};
    }
    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    common_reasoning_budget_info info = ctx->info;
    if (ctx->gen_block == ctx->block_no && ctx->gen_offset > 0) {
        info.reasoning_tokens = std::max(0, info.reasoning_tokens - ctx->gen_offset);
        if (info.first_fire_token >= 0) {
            info.first_fire_token = std::max(0, info.first_fire_token - ctx->gen_offset);
        }
    }
    return info;
}

void common_reasoning_budget_mark_generation_start(struct llama_sampler * smpl) {
    if (!smpl) {
        return;
    }
    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;
    const bool open = ctx->state != REASONING_BUDGET_IDLE && ctx->state != REASONING_BUDGET_DONE;
    ctx->gen_block  = open ? ctx->block_no : -1;
    ctx->gen_offset = open ? ctx->used : 0;
}

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl) {
    if (!smpl) {
        return REASONING_BUDGET_IDLE;
    }
    return ((const common_reasoning_budget_ctx *)smpl->ctx)->state;
}

const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl) {
    if (!smpl) {
        return nullptr;
    }

    const auto * ctx = (const common_reasoning_budget_ctx *) smpl->ctx;
    if (ctx->end_match < 0) {
        return nullptr;
    }

    return &ctx->end_matcher.seqs[ctx->end_match];
}

bool common_reasoning_budget_force(struct llama_sampler * smpl) {
    if (!smpl) {
        return false;
    }

    auto * ctx = (common_reasoning_budget_ctx *) smpl->ctx;

    // only a sampler that is actively counting down the budget may be forced;
    // any other state (idle, already forcing/waiting, or done) is left untouched
    if (ctx->state != REASONING_BUDGET_COUNTING) {
        return false;
    }

    ctx->state = REASONING_BUDGET_FORCING;
    ctx->force_pos = 0;
    ctx->brake_forcing = false;
    ctx->end_matcher.reset();
    COM_TRC("%s", "forced into forcing state (manual transition)\n");

    return true;
}
