#include "../src/llama-moe-cache-policy.h"

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

    std::cout << "test-moe-cache-policy: all tests OK\n";
    return 0;
}
