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

static sim make(int32_t desk) {
    sim s;
    s.g.chunks = 0; s.g.chunk_slots = 4; s.g.chunk_mib = 52; s.g.max_chunks = 64;
    s.desk_mib = desk;
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
        sim off = make(9000); off.gov.policy.pause_after_seconds = 0; off.tick();
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
    printf("test-vram-governor: all passed\n");
    return 0;
}
