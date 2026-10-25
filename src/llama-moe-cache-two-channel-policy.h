#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

class llama_moe_channel_policy {
public:
    static constexpr int transient_slots = 4;

    struct upload {
        int32_t expert;
        int32_t slot;
        bool persistent;
    };

    struct plan {
        std::vector<int32_t> ids;
        std::vector<int32_t> slot_map;
        std::vector<upload> uploads;
        int n_tokens = 0;
        int cpu_unique = 0;
        int hits = 0;
        int uploaded = 0;
        int cpu = 0;
    };

    struct costs {
        double cpu_base_us = 15.0;
        double cpu_expert_us = 100.0;
        double gpu_base_us = 30.0;
        double gpu_upload_us = 60.0;
        uint64_t samples = 0;
    };

    bool initialize(int experts, int slots, const std::vector<int32_t> & ranking) {
        if (experts <= 0 || experts > 4096 || slots < 0 || slots > experts) return false;
        std::vector<uint8_t> seen(experts, 0);
        for (int32_t expert : ranking) {
            if (expert < 0 || expert >= experts || seen[expert]) return false;
            seen[expert] = 1;
        }
        slot_to_expert.clear();
        n_experts = experts;
        seed_order = ranking;
        rank_value.assign(experts, 0);
        for (size_t i = 0; i < ranking.size(); ++i) {
            rank_value[ranking[i]] = int(ranking.size() - i);
        }
        counts.resize(experts);
        initial_hits.resize(experts);
        candidates.reserve(experts);
        last.ids.reserve(8 * experts);
        last.slot_map.resize(experts);
        last.uploads.reserve(experts);
        epoch = 0;
        timing = {};
        return commit_resize(preview_resize(slots));
    }

    std::vector<int32_t> preview_resize(int slots) const {
        if (slots < 0 || slots > n_experts) return {};
        std::vector<int32_t> order;
        std::vector<uint8_t> seen(n_experts, 0);
        order.reserve(slots);
        for (int32_t expert : slot_to_expert) {
            if (expert >= 0 && !seen[expert] && int(order.size()) < slots) {
                order.push_back(expert);
                seen[expert] = 1;
            }
        }
        for (int32_t expert : seed_order) {
            if (!seen[expert] && int(order.size()) < slots) {
                order.push_back(expert);
                seen[expert] = 1;
            }
        }
        order.resize(slots, -1);
        return order;
    }

    bool commit_resize(const std::vector<int32_t> & order) {
        if (int(order.size()) > n_experts) return false;
        std::vector<uint8_t> seen(n_experts, 0);
        for (int32_t expert : order) {
            if (expert < -1 || expert >= n_experts || (expert >= 0 && seen[expert])) return false;
            if (expert >= 0) seen[expert] = 1;
        }
        n_slots = int(order.size());
        slot_to_expert = order;
        expert_to_slot.assign(n_experts, -1);
        ages.assign(n_slots, epoch);
        protected_slots.resize(n_slots);
        for (int slot = 0; slot < n_slots; ++slot) {
            if (order[slot] >= 0) expert_to_slot[order[slot]] = slot;
        }
        last.n_tokens = 0;
        last.uploads.clear();
        return true;
    }

    bool make_plan(const std::vector<int32_t> & ids, int tokens, int used) {
        if (tokens < 1 || tokens > 8 || used < 1 || used > n_experts ||
                ids.size() != size_t(tokens) * size_t(used)) return false;
        for (int32_t expert : ids) if (expert < 0 || expert >= n_experts) return false;
        epoch += tokens;
        std::fill(counts.begin(), counts.end(), 0);
        std::fill(initial_hits.begin(), initial_hits.end(), 0);
        std::fill(protected_slots.begin(), protected_slots.end(), 0);
        candidates.clear();
        last.ids = ids;
        last.n_tokens = tokens;
        last.hits = last.uploaded = last.cpu = last.cpu_unique = 0;
        last.uploads.clear();
        int protected_count = 0;
        for (int32_t expert : ids) {
            ++counts[expert];
            const int slot = expert_to_slot[expert];
            if (slot >= 0) {
                ++last.hits;
                initial_hits[expert] = 1;
                if (!protected_slots[slot]) {
                    protected_slots[slot] = 1;
                    ++protected_count;
                }
                ages[slot] = epoch;
            }
        }
        for (int expert = 0; expert < n_experts; ++expert) {
            if (counts[expert] > 0 && !initial_hits[expert]) candidates.push_back(expert);
        }
        std::sort(candidates.begin(), candidates.end(), [&](int32_t a, int32_t b) {
            return rank_value[a] != rank_value[b] ? rank_value[a] > rank_value[b] : a < b;
        });
        const int capacity = n_slots > 0 ? n_slots - protected_count + transient_slots : 0;
        const int maximum = std::min(int(candidates.size()), capacity);
        const auto & cost = timing[tokens];
        int uploads = 0;
        if (cost.samples == 2) {
            uploads = std::min(1, maximum);
        } else if (cost.samples > 2) {
            double best = std::numeric_limits<double>::infinity();
            for (int u = 0; u <= maximum; ++u) {
                const double cpu_us = cost.cpu_base_us + (candidates.size() - u) * cost.cpu_expert_us;
                const double gpu_us = cost.gpu_base_us + u * cost.gpu_upload_us;
                const double wall_us = std::max(cpu_us, gpu_us);
                if (wall_us < best) {
                    best = wall_us;
                    uploads = u;
                }
            }
        }
        int scratch = 0;
        for (int i = 0; i < uploads; ++i) {
            const int32_t expert = candidates[i];
            int victim = -1;
            for (int slot = 0; slot < n_slots; ++slot) {
                if (protected_slots[slot]) continue;
                if (victim < 0 || slot_to_expert[slot] < 0 ||
                        (slot_to_expert[victim] >= 0 && ages[slot] < ages[victim])) victim = slot;
                if (slot_to_expert[slot] < 0) break;
            }
            if (victim >= 0) {
                const int32_t old = slot_to_expert[victim];
                if (old >= 0) expert_to_slot[old] = -1;
                slot_to_expert[victim] = expert;
                expert_to_slot[expert] = victim;
                protected_slots[victim] = 1;
                ages[victim] = epoch;
                last.uploads.push_back({expert, victim, true});
            } else if (scratch < transient_slots) {
                last.uploads.push_back({expert, n_slots + scratch++, false});
            }
        }
        std::copy(expert_to_slot.begin(), expert_to_slot.end(), last.slot_map.begin());
        for (const auto & entry : last.uploads) last.slot_map[entry.expert] = entry.slot;
        for (int expert = 0; expert < n_experts; ++expert) {
            if (counts[expert] == 0) continue;
            if (last.slot_map[expert] < 0) {
                last.cpu += counts[expert];
                ++last.cpu_unique;
            } else if (!initial_hits[expert]) {
                last.uploaded += counts[expert];
            }
        }
        return last.hits + last.uploaded + last.cpu == int(ids.size());
    }

    void observe(double cpu_us, double gpu_us) {
        if (last.n_tokens < 1 || !std::isfinite(cpu_us) || !std::isfinite(gpu_us) ||
                cpu_us <= 0 || gpu_us <= 0) return;
        auto & cost = timing[last.n_tokens];
        // The first graph may include backend capture and thread-pool startup.
        if (cost.samples++ == 0) return;
        auto update = [](double & value, double sample) {
            value = 0.8 * value + 0.2 * std::clamp(sample, 1.0, 10000.0);
        };
        if (last.cpu_unique > 0) {
            update(cost.cpu_expert_us, (cpu_us - cost.cpu_base_us) / last.cpu_unique);
        } else {
            update(cost.cpu_base_us, cpu_us);
        }
        if (!last.uploads.empty()) {
            update(cost.gpu_upload_us, (gpu_us - cost.gpu_base_us) / last.uploads.size());
        } else {
            update(cost.gpu_base_us, gpu_us);
        }
    }

    const plan & latest() const { return last; }
    const std::vector<int32_t> & slots() const { return slot_to_expert; }
    const costs & estimate(int tokens) const { return timing.at(tokens); }
    int capacity() const { return n_slots == 0 ? 0 : n_slots + transient_slots; }

private:
    int n_experts = 0;
    int n_slots = 0;
    uint64_t epoch = 0;
    std::vector<int32_t> seed_order, rank_value, slot_to_expert, expert_to_slot, candidates;
    std::vector<int> counts;
    std::vector<uint8_t> initial_hits, protected_slots;
    std::vector<uint64_t> ages;
    std::array<costs, 9> timing;
    plan last;
};
