// VRAMLOGIC1: scripted-card tests of the governor decision logic (no GPU).
#include "../tools/server/server-vram-governor.h"

#include <cstdio>
#include <cstdlib>

#define CHECK(c) do { if (!(c)) { fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); exit(1); } } while (0)

struct sim {
    vram_governor_core gov;
    vram_cache_geometry g;
    double t = 0;
    int32_t desk_mib = 0;       // everything on the card that is not the cache
    int32_t total_mib = 12282;
    bool busy = false, ready = true, big = false;
    int32_t free_mib() const { return total_mib - desk_mib - g.chunks * g.chunk_mib; }
    vram_action tick(double dt = 0.2) {
        t += dt;
        vram_inputs in; in.now_s = t; in.free_mib = free_mib(); in.ready = ready; in.busy = busy; in.big_prefill = big;
        vram_action a = gov.step(in, g);
        if (a.kind == vram_action_kind::SET_CHUNKS) {
            const int32_t before = g.chunks;
            g.chunks = a.chunks;
            gov.note_applied(a, before, g.chunks, t);
        }
        return a;
    }
};

// VRAMLOGIC2: the scenarios 1..10 were written for a floor of 1536 with no buffer and keep that geometry (buffer 0);
// the buffer has its own scenarios (11..17).
static sim make(int32_t desk, int32_t buffer = 0) {
    sim s;
    s.g.chunks = 0; s.g.chunk_slots = 4; s.g.chunk_mib = 52; s.g.max_chunks = 64;
    s.desk_mib = desk;
    s.gov.policy.buffer_mib = buffer;
    return s;
}

int main() {
    // 1. fills the room above reserve + hysteresis on the first poll after READY, then holds
    {
        sim s = make(9000);                   // free 3282 with an empty cache
        s.ready = false; s.tick();
        CHECK(s.gov.state == vram_state::WARMUP);
        s.ready = true; s.tick();
        CHECK(s.g.chunks == (3282 - 1536 - 256) / 52);
        CHECK(s.free_mib() >= 1536 + 256 - 0 && s.free_mib() < 1536 + 256 + 52);
        for (int i = 0; i < 100; i++) { s.tick(); }
        CHECK(s.g.chunks == (3282 - 1536 - 256) / 52);   // no flapping
        CHECK(s.gov.trims == 0);
    }
    // 2. a desktop jump is absorbed by dropping chunks within one poll, no pause
    {
        sim s = make(9000); s.tick(); const int32_t full = s.g.chunks;
        s.desk_mib += 700;                    // free = ~1792 - 700 + ... below reserve
        CHECK(s.free_mib() < 1536);
        s.tick();
        CHECK(s.free_mib() >= 1536);
        CHECK(s.g.chunks < full);
        CHECK(s.gov.state == vram_state::TRIM);
        CHECK(s.gov.pauses == 0);
        CHECK(s.gov.below_reserve_max_s <= 0.2 + 1e-9);
    }
    // 3. no regrowth for grow_seconds after a trim, then regrow when idle; never while busy
    {
        sim s = make(9000); s.tick(); s.desk_mib += 700; s.tick(); const int32_t after_trim = s.g.chunks;
        s.desk_mib -= 700;                    // desk went back
        for (int i = 0; i < 100; i++) { s.tick(); }          // 20 s < 30 s calm
        CHECK(s.g.chunks == after_trim);
        for (int i = 0; i < 70; i++) { s.tick(); }           // 34 s total
        CHECK(s.g.chunks > after_trim);
        sim b = make(9000); b.tick(); b.desk_mib += 700; b.tick(); b.desk_mib -= 700; b.busy = true;
        const int32_t keep = b.g.chunks;
        for (int i = 0; i < 400; i++) { b.tick(); }
        CHECK(b.g.chunks == keep);            // busy: no growth
    }
    // 4. cache exhausted and the card still short: pause only below the floor, after two polls
    {
        sim s = make(9000); s.tick();
        s.desk_mib = 12282 - 1000 - 0;        // free = 1000 even with an empty cache? cache keeps shrinking
        for (int i = 0; i < 3; i++) { s.tick(); }
        CHECK(s.g.chunks == 0);
        CHECK(s.gov.pauses == 0 || s.gov.state == vram_state::PAUSING);
        // free 1000 > pause floor 768: MIN, no pause
        CHECK(s.gov.state == vram_state::MIN);
        s.desk_mib = 12282 - 700;             // free 700 < 768
        vram_action a = s.tick(); CHECK(a.kind == vram_action_kind::NONE);
        a = s.tick(); CHECK(a.kind == vram_action_kind::PAUSE);
        CHECK(s.gov.state == vram_state::PAUSING);
    }
    // 4b. short for pause_after_seconds with the cache at its minimum: pause even above the floor
    {
        sim s = make(9000); s.tick();
        s.desk_mib = 12282 - 1000;            // free 1000 with an empty cache (floor 768 not hit)
        s.tick();
        vram_action a;
        for (int i = 0; i < 30; i++) { a = s.tick(); if (a.kind == vram_action_kind::PAUSE) break; }
        CHECK(a.kind == vram_action_kind::PAUSE);
        CHECK(a.reason == "short_stall");
        CHECK(s.t < 3.5);                     // 2 s + the polls it took to get to the minimum
        sim off = make(9000); off.gov.policy.pause_after_ms = 0; off.tick();
        off.desk_mib = 12282 - 1000;
        for (int i = 0; i < 200; i++) { CHECK(off.tick().kind != vram_action_kind::PAUSE); }
    }
    // 4c. a shortfall within the slack is not a stall
    {
        sim s = make(9000); s.tick();
        s.desk_mib = 12282 - 1536 + 30;        // short by 30 MiB with an empty cache
        for (int i = 0; i < 100; i++) { CHECK(s.tick().kind != vram_action_kind::PAUSE); }
        s.desk_mib = 12282 - 1536 + 100;       // short by 100 MiB
        vram_action a; for (int i = 0; i < 30; i++) { a = s.tick(); if (a.kind == vram_action_kind::PAUSE) break; }
        CHECK(a.kind == vram_action_kind::PAUSE);
    }
    // 5. game mode (max_chunks = 0) drops everything and keeps it off; policy cap raise allows regrowth
    {
        sim s = make(9000); s.tick(); CHECK(s.g.chunks > 0);
        s.gov.policy.max_chunks = 0; s.tick();
        CHECK(s.g.chunks == 0);
        for (int i = 0; i < 400; i++) { s.tick(); }
        CHECK(s.g.chunks == 0);
        s.gov.policy.max_chunks = -1;
        for (int i = 0; i < 200; i++) { s.tick(); }
        CHECK(s.g.chunks > 0);
    }
    // 6. min_chunks is a floor for trimming
    {
        sim s = make(9000); s.gov.policy.min_chunks = 5; s.tick();
        s.desk_mib += 5000;
        for (int i = 0; i < 5; i++) { s.tick(); }
        CHECK(s.g.chunks == 5);
    }
    // 6b. prefill_yield: release during a big prefill, refill at once afterwards (no calm period)
    {
        sim s = make(9000); s.gov.policy.prefill_yield = true; s.tick(); const int32_t full = s.g.chunks; CHECK(full > 0);
        s.busy = true; s.big = true; s.tick();
        CHECK(s.g.chunks == 0);
        for (int i = 0; i < 50; i++) { s.tick(); }
        CHECK(s.g.chunks == 0);
        s.big = false; s.busy = true; s.tick();          // the request moved on to decoding: still busy
        CHECK(s.g.chunks == full);
        CHECK(s.gov.trims == 1);              // counted, but the calm timer is not restarted
        sim off = make(9000); off.tick(); off.big = true; off.busy = true;
        for (int i = 0; i < 20; i++) { off.tick(); }
        CHECK(off.g.chunks > 0);              // policy off: the cache stays
    }
    // 7. hard geometry limit
    {
        sim s = make(2000); s.g.max_chunks = 10; s.tick();
        CHECK(s.g.chunks == 10);
    }
    // 8. MEMV2 journal: a concession is recorded with its size and how long the card stayed short
    {
        sim s = make(9000); s.tick(); const int32_t full = s.g.chunks;
        CHECK(s.gov.journal.size() == 1);                      // the first fill is a "grow"
        s.desk_mib += 700;
        s.tick();
        const auto ev = s.gov.journal.since(0, 0);
        CHECK(ev.size() == 2);
        CHECK(ev[1].kind == "trim" && ev[1].chunks_before == full && ev[1].chunks_after < full);
        CHECK(ev[1].mib < 0 && ev[1].free_mib < 1536 && ev[1].reason.rfind("short_", 0) == 0);
        s.tick();                                               // free is back at the reserve: the concession is closed
        const auto ev2 = s.gov.journal.since(1, 0);
        CHECK(ev2.size() == 1 && ev2[0].recovered_s >= 0.0 && ev2[0].recovered_s <= 0.5);
        CHECK(s.gov.journal.totals.yields == 1 && s.gov.journal.totals.recovered == 1);
        CHECK(s.gov.journal.totals.yielded_mib == uint64_t(full - s.g.chunks) * 52);
        // since() returns only newer events and respects the limit
        CHECK(s.gov.journal.since(s.gov.journal.last_id(), 0).empty());
        CHECK(s.gov.journal.since(0, 1).size() == 1);
    }
    // 8b. the ring is bounded and ids keep growing
    {
        vram_journal j; j.capacity = 8;
        for (int i = 0; i < 20; i++) { vram_journal_event e; e.kind = "trim"; j.add(e, i); }
        CHECK(j.size() == 8 && j.last_id() == 20 && j.since(0, 0).front().id == 13);
    }
    // 8c. prefill yield is journaled but is not a concession to the desk
    {
        sim s = make(9000); s.gov.policy.prefill_yield = true; s.tick();
        s.busy = true; s.big = true; s.tick();
        s.big = false; s.tick();
        const auto ev = s.gov.journal.since(0, 0);
        CHECK(ev.size() == 3 && ev[1].kind == "prefill_yield" && ev[2].kind == "grow");
        CHECK(s.gov.journal.totals.yields == 0);
    }
    // 9. light sleep: the cache goes to zero and stays there, waking lets it grow back; policy changes are journaled with the caller's note
    {
        sim s = make(9000); s.tick(); const int32_t full = s.g.chunks; CHECK(full > 0);
        vram_policy before = s.gov.policy;
        s.gov.policy.sleep_level = 1;
        s.gov.note_policy(before, s.gov.policy, "game_started", s.t);
        s.tick();
        CHECK(s.g.chunks == 0 && s.gov.last_reason == "policy_cap");
        for (int i = 0; i < 100; i++) { s.tick(); }
        CHECK(s.g.chunks == 0);
        before = s.gov.policy;
        s.gov.policy.sleep_level = 0;
        s.gov.note_policy(before, s.gov.policy, "game_ended", s.t);
        for (int i = 0; i < 200; i++) { s.tick(); }             // grow_seconds = 30 s of calm
        CHECK(s.g.chunks == full);
        const auto ev = s.gov.journal.since(0, 0);
        bool saw_sleep = false, saw_wake = false;
        for (const auto & e : ev) {
            if (e.kind == "sleep") { saw_sleep = true; CHECK(e.note == "game_started" && e.reason == "sleep_l1"); }
            if (e.kind == "wake")  { saw_wake = true;  CHECK(e.note == "game_ended"); }
        }
        CHECK(saw_sleep && saw_wake && s.gov.journal.totals.sleeps == 1);
    }
    // 10. a pause is journaled with the reason and the free VRAM it was decided at
    {
        sim s = make(9000); s.tick();
        s.desk_mib += 12000;
        vram_action a;
        for (int i = 0; i < 20 && a.kind != vram_action_kind::PAUSE; i++) { a = s.tick(); }
        CHECK(a.kind == vram_action_kind::PAUSE);
        const auto ev = s.gov.journal.since(0, 0);
        CHECK(ev.back().kind == "pause" && ev.back().reason == a.reason && s.gov.journal.totals.pauses == 1);
    }
    // 11. VRAMLOGIC2: with a buffer the trim starts below the working target, by the whole deficit, and the floor is not touched
    {
        sim s = make(9000, 512); s.tick();                     // free 3282 - cache: fills to target + hysteresis
        CHECK(s.g.chunks == (3282 - 2048 - 256) / 52);
        CHECK(s.free_mib() >= 2048 + 256);
        const int32_t full = s.g.chunks;
        s.desk_mib += 400;                                      // a burst within the buffer: above the floor
        CHECK(s.free_mib() >= 1536);
        vram_action a = s.tick();
        CHECK(a.kind == vram_action_kind::SET_CHUNKS && !a.urgent);   // trimmed at once, not urgent: the floor was never crossed
        CHECK(s.free_mib() >= 2048 && s.free_mib() < 2048 + 52);     // one resize restored the target exactly
        CHECK(s.g.chunks < full);
        CHECK(s.gov.below_reserve_events == 0);
    }
    // 12. a Brave-sized burst (+1700 MiB in one poll) with a full cache: one urgent resize gives the whole deficit back
    {
        sim s = make(7000, 512); s.tick();                     // free 5282 with an empty cache: a big cache
        CHECK(s.g.chunks == (5282 - 2048 - 256) / 52);
        s.desk_mib += 1700;                                     // free before the poll ~604, below the floor
        CHECK(s.free_mib() < 1536);
        vram_action a = s.tick();
        CHECK(a.kind == vram_action_kind::SET_CHUNKS && a.urgent);
        CHECK(s.free_mib() >= 2048);                            // back at the target after one resize
        CHECK(s.gov.pauses == 0);
        CHECK(s.gov.below_reserve_max_s <= 0.2 + 1e-9);        // the floor was crossed for one poll at most
        a = s.tick();
        CHECK(a.kind == vram_action_kind::NONE);                // nothing more to do: no second resize
    }
    // 13. a saw-tooth within the buffer (+-100 MiB) never trims and never grows: no flapping
    {
        sim s = make(9000, 512); s.tick(); const int32_t full = s.g.chunks;
        for (int i = 0; i < 300; i++) {
            s.desk_mib += (i % 2 == 0) ? 100 : -100;
            CHECK(s.tick().kind == vram_action_kind::NONE);
        }
        CHECK(s.g.chunks == full && s.gov.trims == 0);
    }
    // 14. a slow rise: trims as the target is crossed, each one restoring the target; the floor stays untouched
    {
        sim s = make(9000, 512); s.tick();
        for (int i = 0; i < 40; i++) {
            s.desk_mib += 30;                                   // 150 MiB/s
            s.tick();
            CHECK(s.free_mib() >= 1536);
            CHECK(s.free_mib() >= 2048 || s.g.chunks == 0);     // the target is restored by every trim while there is cache
        }
        CHECK(s.gov.below_reserve_events == 0);
    }
    // 15. between the floor and the target with an empty cache: MIN, but no stall and no pause (the pause rules are about the floor)
    {
        sim s = make(9000, 512); s.tick();
        s.desk_mib = 12282 - 1800;                              // free 1800 with an empty cache: below the target, above the floor
        for (int i = 0; i < 60; i++) { CHECK(s.tick().kind != vram_action_kind::PAUSE); }
        CHECK(s.g.chunks == 0 && s.gov.state == vram_state::MIN && s.gov.pauses == 0);
        s.desk_mib = 12282 - 1400;                              // free 1400: below the floor by 136 (more than the slack)
        vram_action a; for (int i = 0; i < 30; i++) { a = s.tick(); if (a.kind == vram_action_kind::PAUSE) break; }
        CHECK(a.kind == vram_action_kind::PAUSE && a.reason == "short_stall");
    }
    // 16. the poll period: normal with room above the target, fast below target + hysteresis or after a fast drop
    {
        sim s = make(7000, 512); s.tick();
        CHECK(s.gov.poll_interval_ms() == 200);                 // room above the target (the cache just filled)
        s.desk_mib += 150; s.tick();                            // a drop of 150 >= fast_drop_mib: fast
        CHECK(s.gov.poll_interval_ms() == 50);
        s.tick();                                               // the trim put free back at the target: below target + hysteresis, still fast
        CHECK(s.gov.poll_interval_ms() == 50);
        s.desk_mib -= 150; s.gov.policy.grow_seconds = 0; s.tick(); s.tick();
        CHECK(s.gov.poll_interval_ms() == 200);                 // room again (the cache grew back above target + hysteresis)
        sim w = make(9000, 512); CHECK(w.gov.poll_interval_ms() == 200);   // WARMUP: the normal period
    }
    // 17. a 2 GiB/s ramp (the largest SAGE1 burst) with the resize taking 100 ms: with 50 ms polls and the buffer the floor
    //     is never crossed; with 200 ms polls the desk takes poll + resize worth of VRAM (600 MiB) before the first chunk is back,
    //     more than the buffer. The pure core has no latency, so the test moves the desk while the resize "runs".
    {
        for (int poll_ms : {50, 200}) {
            sim s = make(7000, 512); s.tick();
            int32_t min_free = s.free_mib();
            const double dt = poll_ms / 1000.0, resize_s = 0.1, rate = 2000;
            for (double t = 0; t < 1.0 - 1e-9; t += dt) {
                s.desk_mib += (int32_t) (rate * dt);            // the desk grows between two polls
                min_free = std::min(min_free, s.free_mib());    // what the card sees just before the poll
                s.t += dt;
                vram_inputs in; in.now_s = s.t; in.free_mib = s.free_mib(); in.ready = true;
                vram_action a = s.gov.step(in, s.g);
                if (a.kind == vram_action_kind::SET_CHUNKS) {
                    s.desk_mib += (int32_t) (rate * resize_s);  // the desk keeps growing while the resize runs
                    min_free = std::min(min_free, s.free_mib());
                    const int32_t before = s.g.chunks; s.g.chunks = a.chunks; s.gov.note_applied(a, before, s.g.chunks, s.t);
                    s.desk_mib -= (int32_t) (rate * resize_s);  // (accounted again by the next poll's growth)
                }
            }
            if (poll_ms == 50) { CHECK(min_free >= 1536); } else { CHECK(min_free < 1536); }
            CHECK(s.g.chunks >= 0 && s.gov.pauses == 0);
        }
    }
    // 18. VRAMLOGIC2: a sub-second stall pause (the gateway's default 250 ms) fires before a 1 s stand watchdog: with fast polls
    //     the decision lands 250..300 ms after the first poll that saw the floor short with the cache at its minimum
    {
        sim s = make(9000, 512); s.gov.policy.pause_after_ms = 250; s.gov.policy.pause_floor_mib = 0; s.tick();
        s.desk_mib = 12282 - 1300;                              // free 1300 with an empty cache: 236 below the floor (> slack)
        double first_short = -1; vram_action a;
        for (int i = 0; i < 40; i++) {
            a = s.tick(0.05);
            if (first_short < 0 && s.gov.state == vram_state::MIN) { first_short = s.t; }
            if (a.kind == vram_action_kind::PAUSE) { break; }
        }
        CHECK(a.kind == vram_action_kind::PAUSE && a.reason == "short_stall");
        CHECK(first_short >= 0 && s.t - first_short >= 0.25 - 1e-9 && s.t - first_short <= 0.30 + 1e-9);
        sim off = make(9000, 512); off.gov.policy.pause_after_ms = 0; off.gov.policy.pause_floor_mib = 0; off.tick();
        off.desk_mib = 12282 - 1300;
        for (int i = 0; i < 100; i++) { CHECK(off.tick(0.05).kind != vram_action_kind::PAUSE); }
    }
    printf("test-vram-governor: all passed\n");
    return 0;
}
