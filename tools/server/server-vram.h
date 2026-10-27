#pragma once

// VRAMLOGIC1: runtime of the VRAM governor inside llama-server (thread, sysfs, expert-cache actuator, /vram JSON).
// The decision logic is in server-vram-governor.h.

#include "server-vram-governor.h"

#include "json.h"

#include <atomic>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct common_params;
struct llama_model;

typedef void (*pfn_pool_freeze)(bool);
typedef void (*pfn_pool_stats)(uint64_t *, uint64_t *);

class server_vram {
public:
    using json = common_json;

    struct host_parts { int32_t checkpoints_mib = 0, prompt_cache_mib = 0; };
    void set_host_callback(std::function<host_parts()> cb) { host_cb_ = std::move(cb); }

    struct fixed_parts { int32_t weights = 0, kv = 0, sched = 0, draft = 0; }; // GPU buffers, MiB, cache excluded by the caller

    // busy: true while a request is being processed; model: the loaded target model; fixed: ggml buffer sizes
    server_vram(const common_params & params, const llama_model * model, std::function<bool()> busy,
            std::function<fixed_parts()> fixed);
    ~server_vram();

    bool available() const { return card_total_mib_ > 0 && geometry_ok_; }
    void set_big_prefill_callback(std::function<bool()> cb) { big_prefill_ = std::move(cb); }
    void start();
    void stop();


    json status();                                       // GET /vram
    json policy_json();                                  // GET /vram/policy
    json journal_json(uint64_t since_id, size_t limit);  // GET /vram/journal (MEMV2)
    bool set_policy(const json & body, std::string & err); // POST /vram/policy
    json set_chunks_manual(int32_t chunks, std::string & err); // POST /moe-cache/chunks (test hooks)
    void simulate(int32_t free_delta_mib, double seconds); // POST /vram/simulate (test hooks)

    // pause with a checkpoint (by the governor or POST /vram/pause): the server saves the active slots, tells the
    // clients and exits with code 75. The callback posts the task into the server queue.
    void set_pause_callback(std::function<void(const std::string & reason, const std::string & checkpoint)> cb) { pause_cb_ = std::move(cb); }
    bool request_pause(const std::string & reason, const std::string & checkpoint);
    bool test_hooks() const { return test_hooks_; }
    int32_t n_poll_ms() const { return core_.policy.poll_ms; }

    // VRAMLOGIC2: an urgent trim (free VRAM below the floor) asks the running ubatch to stop at its next graph node, so the
    // resize does not wait for the end of a long prefill ubatch. Register with llama_set_abort_callback(ctx, abort_callback, this);
    // the server retries the interrupted slice (llama_decode removed its positions from the memory).
    // VRAMLOGIC3: a pending pause stops a running prefill slice too (the server then rolls the slice back, saves and exits); a decode
    // step is short and is never cut for a pause, because its sampled token may already be on its way to the client.
    static bool abort_callback(void * data) {
        if (data == nullptr) { return false; }
        const auto * v = static_cast<server_vram *>(data);
        return v->abort_wanted_.load(std::memory_order_relaxed) ||
               (v->pause_pending_.load(std::memory_order_relaxed) && v->prefill_in_flight_.load(std::memory_order_relaxed));
    }
    uint64_t n_ubatch_interrupts() const { return ubatch_interrupts_.load(); }
    void note_ubatch_interrupted() { ++ubatch_interrupts_; }
    bool pause_pending() const { return pause_pending_.load(); }
    void set_prefill_in_flight(bool on) { prefill_in_flight_.store(on, std::memory_order_relaxed); }
    void note_pause_cut() { ++pause_cuts_; }

private:
    void loop();
    void warmup();
    void warm_request(int n_prompt, int n_predict);
    bool read_card(int32_t & free_mib, int32_t & total_mib, int32_t & used_mib);
    int32_t read_proc_vram_mib();
    void refresh_geometry(vram_cache_geometry & g);
    bool apply_chunks(int32_t chunks, const std::string & why, double now_s, bool urgent = false);
    // VRAMLOGIC3: inside a resize, after the old bank is freed: wait until the card shows the VRAM back (sysfs), at most release_wait_ms
    static void release_wait_cb(size_t released_bytes, void * data) { static_cast<server_vram *>(data)->wait_released(released_bytes); }
    void wait_released(size_t released_bytes);
    json policy_locked() const;

    const llama_model * model_;
    std::function<bool()> busy_;
    std::function<bool()> big_prefill_;
    std::function<fixed_parts()> fixed_cb_;
    std::function<host_parts()> host_cb_;
    typedef void (*pfn_host_limit)(uint64_t);
    typedef void (*pfn_host_stats)(uint64_t *, uint64_t *);
    pfn_host_limit host_limit_ = nullptr;
    pfn_host_stats host_stats_ = nullptr;
    int32_t host_pinned_max_mib_ = 0;
    json host_json();
    std::string host_ = "127.0.0.1";
    int port_ = 8080;
    int n_ubatch_ = 512, n_ctx_ = 4096, n_vocab_hint_ = 100000;
    bool warmup_enabled_ = true;
    double warm_seconds_ = 0;
    std::vector<std::string> warm_shapes_;
    std::thread warm_thread_;
    std::atomic<bool> core_ready_{false};
    pfn_pool_freeze  pool_freeze_  = nullptr;
    pfn_pool_stats   pool_stats_   = nullptr;

    std::mutex mu_;                 // guards core_, geometry, counters, window
    vram_governor_core core_;
    vram_cache_geometry geom_;
    bool geometry_ok_ = false;
    size_t slot_bytes_ = 0;
    int32_t max_slots_ = 0;
    int32_t chunk_slots_ = 4;

    std::string mem_used_path_, mem_total_path_;
    int32_t card_total_mib_ = 0;
    int32_t last_free_mib_ = 0;
    int32_t window_min_free_ = INT32_MAX;
    int32_t proc_vram_mib_ = 0;
    double  human_need_ = -1;       // MEMV2: what the gateway last said about the desk, 0..1 (-1 = never said), informational
    uint64_t last_resize_ms_ = 0;
    double   uptime0_s_ = 0;

    std::atomic<bool> ready_{false}; // warm-up finished
    std::atomic<bool> pool_frozen_{false};
    std::atomic<uint64_t> pool_overflow_{0};
    std::function<void(const std::string &, const std::string &)> pause_cb_;
    std::atomic<bool> pause_requested_{false};
    std::atomic<bool> abort_wanted_{false};        // VRAMLOGIC2: set while an urgent trim waits for the cache lock
    std::atomic<bool> pause_pending_{false};       // VRAMLOGIC3: a pause task is posted and has not run yet
    std::atomic<bool> prefill_in_flight_{false};   // VRAMLOGIC3: the running llama_process is a prompt-only slice
    std::atomic<uint64_t> pause_cuts_{0};          // prefill slices cut short for a pause
    std::atomic<int32_t> release_baseline_free_{-1};   // VRAMLOGIC3: card free VRAM just before a governor resize (-1 = no wait:
                                                       // a resize outside apply_chunks, e.g. the legacy slots endpoint)
    int32_t release_wait_ms_ = 250;                // the longest wait for a released bank to show on the card
    double  last_release_wait_ms_ = 0;             // the last wait and how much came back in it (for the log and /vram)
    int32_t last_release_seen_mib_ = 0;
    std::atomic<uint64_t> ubatch_interrupts_{0};   // how many ubatches the server retried because of it
    std::atomic<bool> stop_{false};
    std::thread thread_;
    bool test_hooks_ = false;
    int32_t sim_delta_mib_ = 0;
    double  sim_until_s_ = 0;
    std::string sysfs_card_;
    std::string marker_dir_;
};
