#pragma once

// VRAMLOGIC1: decision logic of the VRAM governor. Pure arithmetic, no GPU or server dependencies,
// so it can be tested with a scripted card (tests/test-vram-governor.cpp). All sizes are MiB.
//
// The governor owns the expert cache: it keeps card free VRAM >= reserve by dropping cache chunks
// (a chunk is chunk_slots cache slots), and grows the cache into the room above reserve + hysteresis
// only while the server is idle. When the cache is at its minimum and the card is still short, it asks
// for a pause with a checkpoint (the gateway then restarts the server in a smaller layout).
//
// VRAMLOGIC2: the reserve is the floor (never below); the working target is reserve + buffer. Trimming
// starts as soon as free VRAM drops below the target, by the whole deficit in one resize, so a desk burst
// up to the buffer never reaches the floor; the cache grows only into the room above target + hysteresis.
// A trim with free VRAM already below the floor is urgent: the runtime may interrupt the running ubatch
// for it. Below the target the governor polls fast (poll_fast_ms) instead of poll_ms.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <deque>
#include <string>
#include <vector>

struct vram_policy {
    int32_t reserve_mib     = 1536;  // the floor: free VRAM is never meant to go below it
    int32_t buffer_mib      = 256;   // VRAMLOGIC2: working target = reserve + buffer; trimming starts below the target
                                     // (VRAMLOGIC3: 256 base; the gateway raises it to 512 while the desk is active)
    int32_t hysteresis_mib  = 256;
    int32_t grow_seconds    = 30;
    int32_t pause_floor_mib = 768;
    int32_t pause_after_ms  = 2000;  // cache at its minimum and free < reserve for this long: pause (0 = off); VRAMLOGIC2: ms,
                                     // so the gateway can pause before a 1 s stand watchdog (the old field was whole seconds)
    int32_t stall_slack_mib = 48;    // a shortfall up to this much does not count as a stall (the reserve is a comfort figure)
    int32_t min_chunks      = 0;
    int32_t max_chunks      = -1;     // -1 = unlimited (JSON null); 0 = cache fully released ("game")
    bool    prefill_yield   = false;
    int32_t poll_ms         = 200;
    int32_t poll_fast_ms    = 50;     // VRAMLOGIC2: polling period while free VRAM is below the target or falling fast
    int32_t fast_drop_mib   = 128;    // a drop of this much between two polls switches to the fast period for the next poll
    bool    trim_abort      = true;   // VRAMLOGIC2: an urgent trim may interrupt the running ubatch (the runtime decides how)
    int32_t sleep_level     = 0;      // MEMV2: 0 = awake, 1 = light sleep (the cache is released, the model stays in VRAM)

    int32_t target_mib() const { return reserve_mib + buffer_mib; }
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
    bool urgent = false;   // VRAMLOGIC2: a trim decided with free VRAM already below the floor
};


// MEMV2: journal of the concessions the server made to the desk (what it gave back, when, why, how long the card stayed short).
// In memory only (a ring of the last events plus running totals), never written to disk.
struct vram_journal_event {
    uint64_t id = 0;
    double   wall_s = 0;          // unix time
    double   mono_s = 0;          // the governor's monotonic clock
    std::string kind;             // trim | grow | prefill_yield | pause | policy | sleep | wake
    std::string reason;
    std::string state;            // governor state after the event
    int32_t  chunks_before = 0, chunks_after = 0;
    int32_t  mib = 0;             // VRAM given back (negative) or taken (positive) by the cache change
    int32_t  free_mib = 0;        // card free VRAM when it was decided
    double   recovered_s = -1;    // trim: how long free VRAM stayed below the reserve (-1 = not recovered yet / not applicable)
    std::string note;             // policy events: what the caller said (the gateway's reason, e.g. "game")
};

struct vram_journal_totals {
    uint64_t yields = 0;          // trims that were a concession (short on VRAM or the policy cap, not prefill yield)
    uint64_t yielded_mib = 0;
    uint64_t recovered = 0;       // yields after which free VRAM got back to the reserve
    double   recovered_sum_s = 0;
    double   recovered_max_s = 0;
    uint64_t pauses = 0;
    uint64_t sleeps = 0;
};

class vram_journal {
public:
    size_t capacity = 512;
    vram_journal_totals totals;

    static double wall_now() {
        return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    }

    vram_journal_event & add(vram_journal_event ev, double mono_s) {
        ev.id = next_id_++;
        ev.mono_s = mono_s;
        if (ev.wall_s <= 0) { ev.wall_s = wall_now(); }
        events_.push_back(std::move(ev));
        while (events_.size() > capacity) { events_.pop_front(); }
        return events_.back();
    }

    // free VRAM is back at the reserve: close the open concessions made since the shortfall began
    void mark_recovered(double shortfall_since_mono_s, double now_mono_s) {
        const double took = now_mono_s - shortfall_since_mono_s;
        for (auto it = events_.rbegin(); it != events_.rend() && it->mono_s >= shortfall_since_mono_s - 1e-6; ++it) {
            if (it->kind == "trim" && it->recovered_s < 0) {
                it->recovered_s = std::max(0.0, took);
                ++totals.recovered;
                totals.recovered_sum_s += it->recovered_s;
                totals.recovered_max_s = std::max(totals.recovered_max_s, it->recovered_s);
            }
        }
    }

    std::vector<vram_journal_event> since(uint64_t after_id, size_t limit) const {
        std::vector<vram_journal_event> out;
        for (const auto & e : events_) {
            if (e.id > after_id) { out.push_back(e); }
        }
        if (limit > 0 && out.size() > limit) { out.erase(out.begin(), out.end() - limit); }
        return out;
    }

    uint64_t last_id() const { return next_id_ - 1; }
    size_t size() const { return events_.size(); }

private:
    std::deque<vram_journal_event> events_;
    uint64_t next_id_ = 1;
};

class vram_governor_core {
public:
    vram_policy policy;
    vram_state  state = vram_state::WARMUP;
    std::string last_reason;
    vram_journal journal;

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
        falling_ = have_prev_free_ && prev_free_mib_ - in.free_mib >= policy.fast_drop_mib;
        prev_free_mib_ = in.free_mib;
        have_prev_free_ = true;
        last_free_mib_ = in.free_mib;
        last_chunk_mib_ = g.chunk_mib;
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

        // 2. short on free VRAM: below the working target drop the whole deficit at once (VRAMLOGIC2)
        const int32_t target_mib = policy.target_mib();
        if (in.free_mib < target_mib) {
            calm_since_ = in.now_s;
            const int32_t need = target_mib - in.free_mib;
            const int32_t drop = (need + g.chunk_mib - 1) / std::max(1, g.chunk_mib);
            const int32_t target = std::max(floor_chunks, g.chunks - drop);
            if (target < g.chunks) {
                low_polls_ = 0;
                set_chunks(act, target, "short_" + std::to_string(need) + "mib", vram_state::TRIM, g, in);
                act.urgent = in.free_mib < policy.reserve_mib;   // already below the floor: do not wait for the ubatch
                return act;
            }
            // nothing left to drop: MIN (a signal to the gateway); the pause rules look at the floor, not the target
            state = vram_state::MIN;
            last_reason = "min_short_" + std::to_string(need) + "mib";
            const int32_t floor_need = policy.reserve_mib - in.free_mib;
            if (floor_need <= policy.stall_slack_mib) {
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
            const bool stalled = policy.pause_after_ms > 0 && stall_since_ >= 0 && in.now_s - stall_since_ >= policy.pause_after_ms / 1000.0 - 1e-9;
            if (floor_hit || stalled) {
                act.kind = vram_action_kind::PAUSE;
                act.reason = floor_hit ? "pause_floor" : "short_stall";
                state = vram_state::PAUSING;
                ++pauses;
                last_reason = act.reason;
                note_pause(act.reason, in.now_s, in.free_mib);
            }
            return act;
        }
        low_polls_ = 0;
        stall_since_ = -1;

        // 3. grow into the room above target + hysteresis, only when idle and calm
        const int32_t room = in.free_mib - target_mib - policy.hysteresis_mib;
        if ((!in.busy || refill_now_) && room >= g.chunk_mib && g.chunks < cap &&
                in.now_s - calm_since_ >= policy.grow_seconds) {
            refill_now_ = false;
            const int32_t add = std::min(cap - g.chunks, room / std::max(1, g.chunk_mib));
            if (add > 0) {
                return set_chunks(act, g.chunks + add, "room_" + std::to_string(room) + "mib", vram_state::GROW, g, in);
            }
        }

        // 4. steady state
        if (g.chunks <= floor_chunks && in.free_mib < target_mib + policy.hysteresis_mib) {
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
        if (after != before) {
            vram_journal_event ev;
            const bool yield = act.reason == "prefill_yield";
            ev.kind = after < before ? (yield ? "prefill_yield" : "trim") : "grow";
            ev.reason = act.reason;
            ev.state = vram_state_name(state);
            ev.chunks_before = before;
            ev.chunks_after = after;
            ev.mib = (after - before) * last_chunk_mib_;
            ev.free_mib = last_free_mib_;
            if (ev.kind == "trim") {
                ++journal.totals.yields;
                journal.totals.yielded_mib += uint64_t(before - after) * uint64_t(std::max(0, last_chunk_mib_));
            }
            journal.add(std::move(ev), now_s);
        }
        if (act.chunks < before && act.reason != "prefill_yield") {
            calm_since_ = now_s; // a trim restarts the calm period (hysteresis against flapping)
        }
    }

    // MEMV2: journal entries for things that are not cache resizes
    void note_pause(const std::string & why, double now_s, int32_t free_mib) {
        vram_journal_event ev;
        ev.kind = "pause";
        ev.reason = why;
        ev.state = vram_state_name(state);
        ev.free_mib = free_mib;
        ++journal.totals.pauses;
        journal.add(std::move(ev), now_s);
    }

    // a new policy was applied; `note` is the caller's reason (the gateway says what it saw on the desk)
    void note_policy(const vram_policy & before, const vram_policy & after, const std::string & note, double now_s) {
        vram_journal_event ev;
        ev.state = vram_state_name(state);
        ev.free_mib = last_free_mib_;
        ev.note = note;
        if (after.sleep_level != before.sleep_level) {
            ev.kind = after.sleep_level > before.sleep_level ? "sleep" : "wake";
            ev.reason = "sleep_l" + std::to_string(after.sleep_level);
            if (after.sleep_level > before.sleep_level) { ++journal.totals.sleeps; }
            journal.add(ev, now_s);
        }
        if (after.reserve_mib != before.reserve_mib || after.max_chunks != before.max_chunks || after.prefill_yield != before.prefill_yield ||
                after.min_chunks != before.min_chunks) {
            ev.kind = "policy";
            ev.reason = "reserve=" + std::to_string(after.reserve_mib) + " max_chunks=" + std::to_string(after.max_chunks);
            journal.add(ev, now_s);
        }
    }

    void set_state(vram_state s, const std::string & why) {
        state = s;
        last_reason = why;
    }

    // VRAMLOGIC2: how long to sleep before the next poll. Fast while free VRAM is below target + hysteresis (there is nothing
    // to grow into and a burst may be under way) or fell by fast_drop_mib since the previous poll; the normal period otherwise.
    int32_t poll_interval_ms() const {
        const int32_t fast = std::max(20, std::min(policy.poll_fast_ms, policy.poll_ms));
        if (state == vram_state::WARMUP || state == vram_state::PAUSING || state == vram_state::PAUSED) {
            return std::max(20, policy.poll_ms);
        }
        if (falling_ || last_free_mib_ < policy.target_mib() + policy.hysteresis_mib) {
            return fast;
        }
        return std::max(20, policy.poll_ms);
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
    int32_t last_free_mib_  = 0;
    int32_t last_chunk_mib_ = 0;
    int32_t prev_free_mib_  = 0;   // VRAMLOGIC2: the previous poll's free VRAM (for the falling-fast rule)
    bool    have_prev_free_ = false;
    bool    falling_        = false;

    int32_t effective_max(const vram_cache_geometry & g) const {
        if (policy.sleep_level >= 1) { return 0; }
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
            if (below_) {
                journal.mark_recovered(below_since_, in.now_s);
            }
            below_ = false;
        }
    }
};
