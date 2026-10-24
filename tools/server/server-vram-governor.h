#pragma once

// VRAMLOGIC1: decision logic of the VRAM governor. Pure arithmetic, no GPU or server dependencies,
// so it can be tested with a scripted card (tests/test-vram-governor.cpp). All sizes are MiB.
//
// The governor owns the expert cache: it keeps card free VRAM >= reserve by dropping cache chunks
// (a chunk is chunk_slots cache slots), and grows the cache into the room above reserve + hysteresis
// only while the server is idle. When the cache is at its minimum and the card is still short, it asks
// for a pause with a checkpoint (the gateway then restarts the server in a smaller layout).

#include <algorithm>
#include <cstdint>
#include <string>

struct vram_policy {
    int32_t reserve_mib     = 1536;
    int32_t hysteresis_mib  = 256;
    int32_t grow_seconds    = 30;
    int32_t pause_floor_mib = 768;
    int32_t pause_after_seconds = 2; // cache at its minimum and free < reserve for this long: pause (0 = off)
    int32_t stall_slack_mib = 48;    // a shortfall up to this much does not count as a stall (the reserve is a comfort figure)
    int32_t min_chunks      = 0;
    int32_t max_chunks      = -1;     // -1 = unlimited (JSON null); 0 = cache fully released ("game")
    bool    prefill_yield   = false;
    int32_t poll_ms         = 200;
};

enum class vram_state {
    WARMUP,
    FULL,
    TRIM,
    MIN,
    GROW,
    PAUSING,
    PAUSED,
};

static inline const char * vram_state_name(vram_state s) {
    switch (s) {
        case vram_state::WARMUP:  return "WARMUP";
        case vram_state::FULL:    return "FULL";
        case vram_state::TRIM:    return "TRIM";
        case vram_state::MIN:     return "MIN";
        case vram_state::GROW:    return "GROW";
        case vram_state::PAUSING: return "PAUSING";
        case vram_state::PAUSED:  return "PAUSED";
    }
    return "?";
}

struct vram_inputs {
    double  now_s       = 0;  // monotonic seconds
    int32_t free_mib    = 0;  // card free VRAM
    bool    ready       = false;
    bool    busy        = false; // a request is being processed
    bool    big_prefill = false; // a large new prompt is being processed (prefill_yield policy)
};

struct vram_cache_geometry {
    int32_t chunks      = 0;  // current number of chunks
    int32_t chunk_slots = 4;
    int32_t chunk_mib   = 52;
    int32_t max_chunks  = 0;  // hard limit from the model (n_expert / chunk_slots)
};

enum class vram_action_kind { NONE, SET_CHUNKS, PAUSE };

struct vram_action {
    vram_action_kind kind = vram_action_kind::NONE;
    int32_t chunks = 0;
    std::string reason;
};

class vram_governor_core {
public:
    vram_policy policy;
    vram_state  state = vram_state::WARMUP;
    std::string last_reason;

    // counters
    uint64_t trims = 0, chunks_dropped = 0, grows = 0, chunks_grown = 0, pauses = 0;
    uint64_t below_reserve_events = 0;
    double   below_reserve_max_s  = 0;

    // Decide the next action from the current observation. The caller applies it and then calls
    // note_applied() with the resulting chunk count (the actual result may differ on a failed resize).
    vram_action step(const vram_inputs & in, const vram_cache_geometry & g) {
        vram_action act;
        const int32_t cap = effective_max(g);
        const int32_t floor_chunks = std::min(policy.min_chunks, cap);
        track_below(in);

        if (state == vram_state::PAUSING || state == vram_state::PAUSED) {
            return act;
        }
        if (!in.ready) {
            state = vram_state::WARMUP;
            return act;
        }
        if (state == vram_state::WARMUP) {
            state = vram_state::FULL;
            calm_since_ = in.now_s - policy.grow_seconds; // the first fill after READY does not wait
            last_reason = "ready";
        }

        // 0. prefill yield: the cache slows a large prefill down (hot/cold split path), so release it for the duration
        //    of the prefill and refill at once afterwards
        if (policy.prefill_yield && in.big_prefill) {
            yielding_ = true;
            if (g.chunks > floor_chunks) {
                return set_chunks(act, floor_chunks, "prefill_yield", vram_state::TRIM, g, in);
            }
            return act;
        }
        if (yielding_) {
            yielding_ = false;
            refill_now_ = true; // grow back at once, even though the request is still decoding
            calm_since_ = in.now_s - policy.grow_seconds; // no calm period after a yield
            last_reason = "prefill_done";
        }

        // 1. policy cap (game mode, lowered max_chunks)
        if (g.chunks > cap) {
            return set_chunks(act, cap, "policy_cap", vram_state::TRIM, g, in);
        }

        // 2. short on free VRAM: drop chunks
        if (in.free_mib < policy.reserve_mib) {
            calm_since_ = in.now_s;
            const int32_t need = policy.reserve_mib - in.free_mib;
            const int32_t drop = (need + g.chunk_mib - 1) / std::max(1, g.chunk_mib);
            const int32_t target = std::max(floor_chunks, g.chunks - drop);
            if (target < g.chunks) {
                low_polls_ = 0;
                return set_chunks(act, target, "short_" + std::to_string(need) + "mib", vram_state::TRIM, g, in);
            }
            // nothing left to drop
            state = vram_state::MIN;
            last_reason = "min_short_" + std::to_string(need) + "mib";
            if (need <= policy.stall_slack_mib) {
                stall_since_ = -1;
            } else if (stall_since_ < 0) {
                stall_since_ = in.now_s;
            }
            if (in.free_mib < policy.pause_floor_mib) {
                ++low_polls_;
            } else {
                low_polls_ = 0;
            }
            const bool floor_hit = low_polls_ >= 2;
            const bool stalled = policy.pause_after_seconds > 0 && stall_since_ >= 0 && in.now_s - stall_since_ >= policy.pause_after_seconds;
            if (floor_hit || stalled) {
                act.kind = vram_action_kind::PAUSE;
                act.reason = floor_hit ? "pause_floor" : "short_stall";
                state = vram_state::PAUSING;
                ++pauses;
                last_reason = act.reason;
            }
            return act;
        }
        low_polls_ = 0;
        stall_since_ = -1;

        // 3. grow into the room above reserve + hysteresis, only when idle and calm
        const int32_t room = in.free_mib - policy.reserve_mib - policy.hysteresis_mib;
        if ((!in.busy || refill_now_) && room >= g.chunk_mib && g.chunks < cap &&
                in.now_s - calm_since_ >= policy.grow_seconds) {
            refill_now_ = false;
            const int32_t add = std::min(cap - g.chunks, room / std::max(1, g.chunk_mib));
            if (add > 0) {
                return set_chunks(act, g.chunks + add, "room_" + std::to_string(room) + "mib", vram_state::GROW, g, in);
            }
        }

        // 4. steady state
        if (g.chunks <= floor_chunks && in.free_mib < policy.reserve_mib + policy.hysteresis_mib) {
            state = floor_chunks > 0 ? vram_state::FULL : vram_state::MIN;
        } else if (state == vram_state::MIN || state == vram_state::TRIM || state == vram_state::GROW) {
            state = vram_state::FULL;
        }
        return act;
    }

    // report the chunk count after an applied (or failed) action
    void note_applied(const vram_action & act, int32_t before, int32_t after, double now_s) {
        if (act.kind != vram_action_kind::SET_CHUNKS) {
            return;
        }
        if (after < before) {
            ++trims;
            chunks_dropped += uint64_t(before - after);
        } else if (after > before) {
            ++grows;
            chunks_grown += uint64_t(after - before);
        }
        if (act.chunks < before && act.reason != "prefill_yield") {
            calm_since_ = now_s; // a trim restarts the calm period (hysteresis against flapping)
        }
    }

    void set_state(vram_state s, const std::string & why) {
        state = s;
        last_reason = why;
    }

    void reset_calm(double now_s) { calm_since_ = now_s; }

private:
    double  calm_since_ = 0;
    int32_t low_polls_  = 0;
    double  stall_since_ = -1;
    bool    yielding_ = false;
    bool    refill_now_ = false;
    bool    below_       = false;
    double  below_since_ = 0;

    int32_t effective_max(const vram_cache_geometry & g) const {
        const int32_t m = policy.max_chunks < 0 ? g.max_chunks : std::min(policy.max_chunks, g.max_chunks);
        return std::max(0, m);
    }

    vram_action & set_chunks(vram_action & act, int32_t target, const std::string & why, vram_state st,
            const vram_cache_geometry &, const vram_inputs &) {
        act.kind = vram_action_kind::SET_CHUNKS;
        act.chunks = target;
        act.reason = why;
        state = st;
        last_reason = why;
        return act;
    }

    void track_below(const vram_inputs & in) {
        if (in.free_mib < policy.reserve_mib) {
            if (!below_) {
                below_ = true;
                below_since_ = in.now_s;
                ++below_reserve_events;
            }
            below_reserve_max_s = std::max(below_reserve_max_s, in.now_s - below_since_);
        } else {
            below_ = false;
        }
    }
};
