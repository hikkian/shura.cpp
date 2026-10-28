#include "../src/llama-moe-cache-policy.h"
#include "../src/llama-moe-cache-two-channel-policy.h"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <unordered_map>
#include <vector>

static std::vector<int32_t> ranking(int32_t n) {
    std::vector<int32_t> result(n);
    for (int32_t i = 0; i < n; ++i) result[i] = i;
    return result;
}

int main() {
    const auto order = ranking(32);
    std::unordered_map<int32_t, uint64_t> recency;
    for (int32_t i = 0; i < 32; ++i) recency[i] = 32 - i;

    llama_moe_cache_hybrid_policy policy;
    assert(policy.initialize(order, recency, 16, 50, 1));
    assert(policy.pinned_slots() == 8);
    assert(policy.slot_to_expert().size() == 16);
    for (int32_t i = 0; i < 16; ++i) assert(policy.slot_to_expert()[i] == i);

    // A token with multiple misses inserts at most one expert, and only in the LRU tail.
    auto changed = policy.observe({20, 21}, 1, 2);
    assert(changed.size() == 1);
    assert(changed[0].slot == 15);
    assert(changed[0].old_expert == 15);
    assert(changed[0].new_expert == 20);
    assert(policy.stats().hits == 0);
    assert(policy.stats().misses == 2);
    assert(policy.stats().insertions == 1);
    for (int32_t i = 0; i < 8; ++i) assert(policy.slot_to_expert()[i] == i);

    // The next token may insert another cold expert; a recently-used tail item survives.
    changed = policy.observe({20, 21}, 1, 2);
    assert(changed.size() == 1);
    assert(changed[0].new_expert == 21);
    assert(changed[0].slot == 14);
    assert(policy.stats().hits == 1);
    assert(policy.stats().misses == 3);
    assert(policy.stats().insertions == 2);
    assert(policy.slot_to_expert()[15] == 20);

    // Compression sheds the dynamic tail before any pinned slot.
    auto resized = policy.preview_resize(8);
    assert(resized.size() == 8);
    for (int32_t i = 0; i < 8; ++i) assert(resized[i] == i);
    policy.commit_resize(8);
    assert(policy.pinned_slots() == 8);
    resized = policy.preview_resize(4);
    assert(resized.size() == 4);
    for (int32_t i = 0; i < 4; ++i) assert(resized[i] == i);
    policy.commit_resize(4);
    assert(policy.pinned_slots() == 4);

    // Regrowth restores the configured pinned set before rebuilding the profile tail.
    policy.commit_resize(16);
    assert(policy.pinned_slots() == 8);
    for (int32_t i = 0; i < 8; ++i) assert(policy.slot_to_expert()[i] == i);
    assert(policy.slot_to_expert()[8] == 8);

    // A zero-slot interval retains rankings and can grow back to a valid pack.
    policy.commit_resize(0);
    assert(policy.slots() == 0);
    policy.commit_resize(16);
    assert(policy.slots() == 16);
    assert(policy.pinned_slots() == 8);

    llama_moe_cache_hybrid_policy two_per_layer;
    assert(two_per_layer.initialize(order, recency, 16, 50, 2));
    changed = two_per_layer.observe({20, 21, 22}, 1, 3);
    assert(changed.size() == 2);
    assert(two_per_layer.stats().insertions == 2);

    llama_moe_cache_hybrid_policy lazy_zero;
    assert(lazy_zero.initialize(order, recency, 0, 75, 1));
    assert(lazy_zero.preview_resize(20).size() == 20);
    lazy_zero.commit_resize(20);
    assert(lazy_zero.pinned_slots() == 15);
    assert(lazy_zero.slots() == 20);

    // MEMV2 (LFU) in the two-channel policy: the session's frequency decides which miss is uploaded and which resident stays
    {
#define LFU_CHECK(c) do { if (!(c)) { std::cerr << "FAIL line " << __LINE__ << ": " #c "\n"; return 1; } } while (0)
        auto warm = [](llama_moe_channel_policy & p) {
            // three timed plans so the cost model may upload (the first sample is skipped); uploads are cheap here
            for (int i = 0; i < 3; ++i) {
                LFU_CHECK(p.make_plan({14, 15}, 1, 2));
                p.observe(400.0, 40.0);
            }
            return 0;
        };
        llama_moe_channel_policy lfu;
        LFU_CHECK(lfu.initialize(16, 4, ranking(16)));
        lfu.set_frequency(true, 32.0);
        LFU_CHECK(lfu.frequency_on());
        // the seed is the profile's order: expert 0 is the most frequent, 15 the least
        LFU_CHECK(lfu.frequency(0) > lfu.frequency(1) && lfu.frequency(1) > lfu.frequency(15) && lfu.frequency(15) > 0.0);
        if (warm(lfu)) return 1;
        // a session that keeps routing to 10 and 11: they become residents
        for (int i = 0; i < 40; ++i) {
            LFU_CHECK(lfu.make_plan({10, 11}, 1, 2));
            lfu.observe(400.0, 40.0);
        }
        auto resident = [](const llama_moe_channel_policy & p, int32_t e) {
            for (int32_t x : p.slots()) if (x == e) return true;
            return false;
        };
        LFU_CHECK(resident(lfu, 10) && resident(lfu, 11));
        // a one-off miss rarer than every resident does not evict one: it is served from a transient slot
        const auto before = lfu.slots();
        LFU_CHECK(lfu.make_plan({3, 13}, 1, 2));
        for (const auto & up : lfu.latest().uploads) {
            if (up.expert == 13) LFU_CHECK(!up.persistent);
        }
        LFU_CHECK(resident(lfu, 10) && resident(lfu, 11));
        // the counts halve every half-life: 64 tokens of other experts leave 10 at about a quarter of what it had
        const double f10 = lfu.frequency(10);
        for (int i = 0; i < 64; ++i) {
            LFU_CHECK(lfu.make_plan({4, 5}, 1, 2));
            lfu.observe(400.0, 40.0);
        }
        LFU_CHECK(lfu.frequency(10) < f10 * 0.26 && lfu.frequency(10) > f10 * 0.24);

        // the same session under the plain policy: the one-off miss takes a resident's slot (least recently used)
        llama_moe_channel_policy lru;
        LFU_CHECK(lru.initialize(16, 4, ranking(16)));
        if (warm(lru)) return 1;
        for (int i = 0; i < 40; ++i) {
            LFU_CHECK(lru.make_plan({10, 11}, 1, 2));
            lru.observe(400.0, 40.0);
        }
        LFU_CHECK(lru.make_plan({3, 13}, 1, 2));
        bool persisted = false;
        for (const auto & up : lru.latest().uploads) persisted |= up.expert == 13 && up.persistent;
        LFU_CHECK(persisted);
        LFU_CHECK(!lru.frequency_on() && lru.frequency(10) == 0.0);

        // a copy (the runtime's resize stages one) keeps the learned counts
        llama_moe_channel_policy staged = lfu;
        LFU_CHECK(staged.frequency(10) == lfu.frequency(10));
        LFU_CHECK(staged.commit_resize(staged.preview_resize(6)));
        LFU_CHECK(staged.frequency(10) == lfu.frequency(10));
        std::cout << "LFU: frequent experts stay resident, one-off misses go transient, counts halve per half-life\n";
#undef LFU_CHECK
    }

    std::cout << "test-moe-cache-policy: all tests OK\n";
    return 0;
}
