#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// CPU-only residency policy for one MoE layer. Tensor copies and map writes are
// deliberately left to the model layer so they can be synchronized with its backend.
class llama_moe_cache_hybrid_policy {
public:
    struct slot_update {
        int32_t slot;
        int32_t old_expert;
        int32_t new_expert;
    };

    struct counters {
        uint64_t hits = 0;
        uint64_t misses = 0;
        uint64_t insertions = 0;
    };

    bool initialize(const std::vector<int32_t> & ranking,
                    const std::unordered_map<int32_t, uint64_t> & last_seen,
                    int32_t slots, int32_t pinned_percent, int32_t max_insertions_per_layer_token) {
        if (slots < 0 || pinned_percent < 0 || pinned_percent > 100 || max_insertions_per_layer_token < 1) {
            return false;
        }
        ranking_ = ranking;
        last_seen_ = last_seen;
        clock_ = 0;
        for (const auto & item : last_seen_) {
            clock_ = std::max(clock_, item.second);
        }
        pinned_percent_ = pinned_percent;
        max_insertions_per_layer_token_ = max_insertions_per_layer_token;
        if (slots > 0) {
            pinned_limit_ = slots * pinned_percent_ / 100;
        }
        slot_to_expert_ = preview_resize(slots);
        pinned_slots_ = slots > 0 ? std::min<int32_t>(pinned_limit_, slots) : 0;
        return true;
    }

    // Process routes after the graph has completed. IDs are flattened with the
    // expert-k axis contiguous: [token 0 experts..., token 1 experts..., ...].
    std::vector<slot_update> observe(const std::vector<int32_t> & routed_ids,
                                     int32_t n_tokens, int32_t experts_per_token) {
        if (n_tokens <= 0 || experts_per_token <= 0 ||
                routed_ids.size() != size_t(n_tokens) * size_t(experts_per_token)) {
            return {};
        }
        const auto before = slot_to_expert_;
        const std::unordered_set<int32_t> resident_at_batch_start(before.begin(), before.end());
        for (const int32_t expert : routed_ids) {
            if (resident_at_batch_start.count(expert)) {
                ++stats_.hits;
            } else {
                ++stats_.misses;
            }
        }
        for (int32_t token = 0; token < n_tokens; ++token) {
            observe_token(routed_ids.data() + size_t(token) * experts_per_token, experts_per_token);
        }
        std::vector<slot_update> result;
        for (size_t slot = 0; slot < slot_to_expert_.size(); ++slot) {
            if (before[slot] != slot_to_expert_[slot]) {
                result.push_back({int32_t(slot), before[slot], slot_to_expert_[slot]});
            }
        }
        return result;
    }

    std::vector<int32_t> preview_resize(int32_t target_slots) const {
        if (target_slots < 0) {
            return {};
        }
        int32_t pinned_limit = pinned_limit_;
        if (pinned_limit < 0 && target_slots > 0) {
            pinned_limit = target_slots * pinned_percent_ / 100;
        }
        const int32_t pinned = std::min(std::max(0, pinned_limit), target_slots);
        std::vector<int32_t> result(target_slots, -1);
        std::unordered_set<int32_t> selected;
        for (int32_t i = 0; i < pinned && size_t(i) < ranking_.size(); ++i) {
            result[i] = ranking_[i];
            selected.insert(result[i]);
        }

        std::vector<int32_t> tail;
        for (int32_t slot = pinned_slots_; slot < int32_t(slot_to_expert_.size()); ++slot) {
            const int32_t expert = slot_to_expert_[slot];
            if (expert >= 0 && !selected.count(expert)) {
                tail.push_back(expert);
                selected.insert(expert);
            }
        }
        std::sort(tail.begin(), tail.end(), [&](int32_t a, int32_t b) {
            const auto a_seen = last_seen_.count(a) ? last_seen_.at(a) : 0;
            const auto b_seen = last_seen_.count(b) ? last_seen_.at(b) : 0;
            return a_seen != b_seen ? a_seen > b_seen : a < b;
        });

        int32_t out = pinned;
        const int32_t tail_capacity = target_slots - pinned;
        for (int32_t expert : tail) {
            if (out >= pinned + tail_capacity) {
                break;
            }
            result[out++] = expert;
        }
        for (int32_t i = pinned; out < target_slots && size_t(i) < ranking_.size(); ++i) {
            const int32_t expert = ranking_[i];
            if (selected.insert(expert).second) {
                result[out++] = expert;
            }
        }
        return result;
    }

    void commit_resize(int32_t target_slots) {
        if (target_slots < 0) {
            return;
        }
        if (pinned_limit_ < 0 && target_slots > 0) {
            pinned_limit_ = target_slots * pinned_percent_ / 100;
        }
        slot_to_expert_ = preview_resize(target_slots);
        pinned_slots_ = std::min(std::max(0, pinned_limit_), target_slots);
    }

    int32_t slots() const { return int32_t(slot_to_expert_.size()); }
    int32_t pinned_slots() const { return pinned_slots_; }
    const std::vector<int32_t> & slot_to_expert() const { return slot_to_expert_; }
    const counters & stats() const { return stats_; }

    void record_copies(uint64_t n, uint64_t bytes) {
        copies_ += n;
        copied_bytes_ += bytes;
    }
    uint64_t copies() const { return copies_; }
    uint64_t copied_bytes() const { return copied_bytes_; }

private:
    int32_t find_slot(const std::unordered_set<int32_t> & active) const {
        for (int32_t slot = pinned_slots_; slot < int32_t(slot_to_expert_.size()); ++slot) {
            if (slot_to_expert_[slot] == -1) {
                return slot;
            }
        }
        int32_t victim_slot = -1;
        uint64_t oldest = std::numeric_limits<uint64_t>::max();
        int32_t oldest_expert = std::numeric_limits<int32_t>::max();
        for (int32_t slot = pinned_slots_; slot < int32_t(slot_to_expert_.size()); ++slot) {
            const int32_t candidate = slot_to_expert_[slot];
            if (candidate < 0 || active.count(candidate)) {
                continue;
            }
            const uint64_t seen = last_seen_.count(candidate) ? last_seen_.at(candidate) : 0;
            if (seen < oldest || (seen == oldest && candidate < oldest_expert)) {
                oldest = seen;
                oldest_expert = candidate;
                victim_slot = slot;
            }
        }
        return victim_slot;
    }

    bool contains(int32_t expert) const {
        return std::find(slot_to_expert_.begin(), slot_to_expert_.end(), expert) != slot_to_expert_.end();
    }

    void observe_token(const int32_t * ids, int32_t n_ids) {
        ++clock_;
        std::unordered_set<int32_t> active;
        std::vector<int32_t> candidates;
        for (int32_t i = 0; i < n_ids; ++i) {
            const int32_t expert = ids[i];
            if (!contains(expert) && std::find(candidates.begin(), candidates.end(), expert) == candidates.end()) {
                candidates.push_back(expert);
            }
            active.insert(expert);
            last_seen_[expert] = clock_;
        }

        int32_t inserted = 0;
        for (const int32_t expert : candidates) {
            if (inserted >= max_insertions_per_layer_token_) {
                break;
            }
            const int32_t slot = find_slot(active);
            if (slot < 0) {
                break;
            }
            slot_to_expert_[slot] = expert;
            last_seen_[expert] = clock_;
            ++inserted;
            ++stats_.insertions;
        }
    }

    std::vector<int32_t> ranking_;
    std::vector<int32_t> slot_to_expert_;
    std::unordered_map<int32_t, uint64_t> last_seen_;
    int32_t pinned_percent_ = 50;
    int32_t pinned_limit_ = -1;
    int32_t pinned_slots_ = 0;
    int32_t max_insertions_per_layer_token_ = 1;
    uint64_t clock_ = 0;
    uint64_t copies_ = 0;
    uint64_t copied_bytes_ = 0;
    counters stats_;
};
