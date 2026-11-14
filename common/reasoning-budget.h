#pragma once

#include "llama.h"

#include "common.h"

#include <cstdint>
#include <string>
#include <vector>

enum common_reasoning_budget_state {
    REASONING_BUDGET_IDLE,         // waiting for start sequence
    REASONING_BUDGET_COUNTING,     // counting down tokens
    REASONING_BUDGET_FORCING,      // forcing budget message + end sequence
    REASONING_BUDGET_WAITING_UTF8, // budget exhausted, waiting for UTF-8 completion
    REASONING_BUDGET_DONE,         // passthrough forever
};

// Creates a reasoning budget sampler that limits token generation inside a
// reasoning block (e.g. between <think> and </think>).
//
// State machine: IDLE -> COUNTING -> WAITING_UTF8 -> FORCING -> DONE
//   IDLE:         passthrough, watching for a start sequence
//   COUNTING:     counting down remaining tokens, watching for a natural end sequence
//   WAITING_UTF8: budget exhausted, allowing tokens to complete a UTF-8 sequence
//   FORCING:      forces forced_tokens token-by-token (all other logits -> -inf)
//   DONE:         passthrough forever
//
// Parameters:
//   vocab          - vocabulary (used for UTF-8 boundary detection; can be nullptr)
//   start_seqs     - token sequences, any of which activates counting
//   end_seqs       - token sequences, any of which naturally deactivates
//   forced_tokens  - token sequence forced when budget expires
//   budget         - max tokens allowed in the reasoning block
//   initial_state  - initial state
//
struct llama_sampler * common_reasoning_budget_init(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state = REASONING_BUDGET_IDLE);

// THINK3b (shura.cpp): effort levels on the server side. All of it is off by default (the sampler above then behaves exactly like
// upstream); see docs/EFFORT.md and docs/SHURA.md.
//
//   soft_tokens   soft target of the thinking block, in tokens; -1 = off. Below it the sampler never intervenes. ThinkBrake
//                 (below) only works from this point on, so with -1 there is no ThinkBrake.
//   brake_margin  ThinkBrake (arXiv 2510.00546): between the soft target and the ceiling, at a sentence boundary inside the
//                 thinking block, compare the logit of the end-of-thinking token (the first token of the end tag that is not
//                 whitespace; a variant of the tag without leading whitespace is preferred) with the best other token; if
//                 (logit_best - logit_end) / brake_temperature <= brake_margin, force the end tag at once, without the
//                 budget message (the thought ended by itself). The paper's default is 0.1 (Qwen3, R1-7B; 2.5 for Phi-4).
//                 < 0 = off.
//   brake_temperature  the sampling temperature the margin is scaled by (the paper compares log-probabilities, which are
//                 logit differences divided by T); the sampler sees raw logits.
//   grace_tokens  at the ceiling, wait up to this many more tokens for a sentence boundary before forcing the message and the
//                 end tag, so the cut lands between sentences; 0 = cut at once (upstream).
//   boundary_ids  tokens that always count as a sentence end (tests, and models whose tokenizer has a dedicated token);
//                 every other token is judged by its text (common_reasoning_piece_ends_sentence).
struct common_reasoning_budget_opts {
    int32_t                soft_tokens       = -1;
    float                  brake_margin      = -1.0f;
    float                  brake_temperature = 1.0f;
    int32_t                grace_tokens      = 0;
    std::vector<llama_token> boundary_ids;
    // tokens that count as whitespace-only (tests; with a vocabulary the token's text decides). A sentence end followed by
    // whitespace is still a sentence end, and ThinkBrake looks past leading whitespace of the end tag ("\n</think>").
    std::vector<llama_token> space_ids;
    // shadow mode (FRESH1): ThinkBrake and the ceiling are evaluated and reported in common_reasoning_budget_info as "would
    // have fired", but nothing is ever forced: the sampled tokens are exactly those of a sampler without these options
    bool                   shadow            = false;
};

struct llama_sampler * common_reasoning_budget_init_ex(
        const struct llama_vocab        * vocab,
        const std::vector<llama_tokens> & start_seqs,
        const std::vector<llama_tokens> & end_seqs,
        const llama_tokens              & forced_tokens,
        int32_t                           budget,
        common_reasoning_budget_state     initial_state,
        const common_reasoning_budget_opts & opts);

// What happened in the thinking block (for the server's timings and for calibration): tokens counted inside the block (the
// current one, or the last one that ended), and how it ended.
struct common_reasoning_budget_info {
    int32_t reasoning_tokens = 0;
    bool    ceiling_hit      = false; // the budget ran out: the message + end tag were forced
    bool    brake_hit        = false; // ThinkBrake ended the thought
    int32_t grace_used       = 0;     // tokens the grace window waited at the ceiling
    // number of thinking tokens before the first intervention (the token where the end tag / the message began, or would
    // have begun in shadow mode); -1 = none
    int32_t first_fire_token = -1;
    bool    shadow           = false; // ceiling_hit / brake_hit mean "would have fired"; nothing was forced
};

common_reasoning_budget_info common_reasoning_budget_get_info(const struct llama_sampler * smpl);

// Call once the prompt's prefill tokens are accepted (e.g. "<think>\n" placed by the chat template): the info above then counts
// generated thinking tokens only (the budget itself still counts from the start tag, as upstream does).
void common_reasoning_budget_mark_generation_start(struct llama_sampler * smpl);

// True when the text of a token ends a sentence: . ! ? … 。！？ ؟ ۔ । (optionally followed by closing quotes/brackets and
// spaces), or a blank line (\n\n). prev_char is the last character of the previous token, to tell "3." in "3.14" from a full
// stop (a lone "." after a digit is not an end).
bool common_reasoning_piece_ends_sentence(const std::string & piece, char prev_char = ' ');

common_reasoning_budget_state common_reasoning_budget_get_state(const struct llama_sampler * smpl);

// The end sequence that transitioned the sampler to DONE, or nullptr if none
// was recorded. Cleared when a new start sequence re-arms the sampler.
const llama_tokens * common_reasoning_budget_get_end_match(const struct llama_sampler * smpl);

// Manually transition the reasoning budget sampler into the FORCING state.
// Returns true if the transition occurred.
bool common_reasoning_budget_force(struct llama_sampler * smpl);
