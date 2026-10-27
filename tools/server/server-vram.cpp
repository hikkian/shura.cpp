#include "server-vram.h"

#include "common.h"
#include <cpp-httplib/httplib.h>
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <set>
#include <sstream>

static double mono_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool read_int64_file(const std::string & path, int64_t & out) {
    std::ifstream f(path);
    if (!f) { return false; }
    f >> out;
    return bool(f);
}

static std::string read_text_file(const std::string & path) {
    std::ifstream f(path);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// find an amdgpu card: the one with the largest VRAM among vendor 0x1002
static std::string find_card_dir() {
    std::string best;
    int64_t best_total = 0;
    DIR * d = opendir("/sys/class/drm");
    if (!d) { return best; }
    while (dirent * e = readdir(d)) {
        const std::string name = e->d_name;
        if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) { continue; }
        const std::string dir = "/sys/class/drm/" + name + "/device";
        int64_t total = 0;
        if (!read_int64_file(dir + "/mem_info_vram_total", total)) { continue; }
        if (total > best_total) {
            best_total = total;
            best = dir;
        }
    }
    closedir(d);
    return best;
}

server_vram::server_vram(const common_params & params, const llama_model * model, std::function<bool()> busy,
        std::function<fixed_parts()> fixed)
    : model_(model), busy_(std::move(busy)), fixed_cb_(std::move(fixed)) {
    host_ = params.hostnames.empty() ? "127.0.0.1" : params.hostnames[0];
    port_ = params.port;
    n_ubatch_ = params.n_ubatch;
    n_ctx_ = params.n_ctx;
    warmup_enabled_ = getenv("LLAMA_VRAM_NO_WARMUP") == nullptr;
    core_.policy.reserve_mib     = params.vram_reserve_mib;
    core_.policy.buffer_mib      = params.vram_buffer_mib;
    core_.policy.poll_fast_ms    = params.vram_poll_fast_ms;
    core_.policy.trim_abort      = params.vram_trim_abort;
    core_.policy.hysteresis_mib  = params.vram_hysteresis_mib;
    core_.policy.grow_seconds    = params.vram_grow_seconds;
    core_.policy.pause_floor_mib = params.vram_pause_floor_mib;
    core_.policy.pause_after_ms  = params.vram_pause_after_ms;
    core_.policy.min_chunks      = params.vram_min_chunks;
    core_.policy.poll_ms         = params.vram_poll_ms;
    chunk_slots_ = std::max(1, params.moe_cache_chunk_slots);
    test_hooks_  = params.vram_test_hooks;
    sysfs_card_  = params.vram_sysfs_card;
    marker_dir_  = params.slot_save_path;
    host_pinned_max_mib_ = params.host_pinned_max_mib;
    uptime0_s_   = mono_s();

    std::string dir = sysfs_card_ == "auto" ? find_card_dir() : sysfs_card_;
    if (!dir.empty()) {
        mem_used_path_  = dir + "/mem_info_vram_used";
        mem_total_path_ = dir + "/mem_info_vram_total";
        int64_t total = 0;
        if (read_int64_file(mem_total_path_, total)) {
            card_total_mib_ = (int32_t) (total / (1024 * 1024));
        }
    }
    int32_t slots = 0, max_slots = 0;
    size_t slot_bytes = 0;
    if (llama_model_moe_cache_info(model_, &slots, &slot_bytes, &max_slots) && slot_bytes > 0) {
        slot_bytes_ = slot_bytes;
        max_slots_  = max_slots;
        geometry_ok_ = true;
        refresh_geometry(geom_);
    }
    if (!available()) {
        LOG_WRN("vram governor: unavailable (card_total_mib=%d, expert cache geometry %s)\n",
                card_total_mib_, geometry_ok_ ? "ok" : "missing");
    } else {
        LOG_INF("vram governor: card %d MiB, reserve %d MiB, hysteresis %d MiB, chunk %d slots = %d MiB, up to %d chunks\n",
                card_total_mib_, core_.policy.reserve_mib, core_.policy.hysteresis_mib, chunk_slots_,
                geom_.chunk_mib, geom_.max_chunks);
    }
}

server_vram::~server_vram() { stop(); }

void server_vram::start() {
    if (!available() || thread_.joinable()) { return; }
    // pool freeze/overflow hooks of the GPU backend (absent in CPU-only builds)
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) { continue; }
        ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(dev);
        pool_freeze_ = (pfn_pool_freeze) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_pool_freeze");
        pool_stats_  = (pfn_pool_stats)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_pool_stats");
        host_limit_  = (pfn_host_limit)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_set_limit");
        host_stats_  = (pfn_host_stats)  ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_host_stats");
        if (host_limit_ && host_pinned_max_mib_ > 0) { host_limit_((uint64_t) host_pinned_max_mib_ * 1024 * 1024); }
        break;
    }
    thread_ = std::thread([this] { loop(); });
    if (warmup_enabled_) {
        warm_thread_ = std::thread([this] { warmup(); });
    } else {
        core_ready_.store(true);
        ready_.store(true);
    }
}

// one synthetic request through the real server path (slot, sampler, speculative drafting), random token ids
void server_vram::warm_request(int n_prompt, int n_predict) {
    json prompt = json::array();
    uint32_t x = 12345u;
    for (int i = 0; i < n_prompt; i++) {
        x = x * 1664525u + 1013904223u;
        prompt.push_back(1000 + (int) ((x >> 8) % 30000));
    }
    json body = {{"prompt", prompt}, {"n_predict", n_predict}, {"temperature", 0.0}, {"cache_prompt", false}, {"stream", false}};
    httplib::Client cli(host_, port_);
    cli.set_read_timeout(900, 0);
    cli.set_connection_timeout(10, 0);
    auto r = cli.Post("/completion", body.dump(), "application/json");
    if (!r || r->status != 200) {
        LOG_WRN("vram governor: warm-up request (%d prompt tokens) failed: %s\n", n_prompt,
                r ? std::to_string(r->status).c_str() : "no response");
    }
}

void server_vram::warmup() {
    const double t0 = mono_s();
    {   // wait until the HTTP server is up
        httplib::Client cli(host_, port_);
        cli.set_connection_timeout(2, 0);
        for (int i = 0; i < 600 && !stop_.load(); i++) {
            auto r = cli.Get("/health");
            if (r && r->status == 200) { break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    int n_prompt = std::max(16, std::min(2 * n_ubatch_, n_ctx_ - 256));
    if (const char * e = getenv("LLAMA_VRAM_WARM_TOKENS")) { n_prompt = std::max(8, atoi(e)); }  // stand experiments
    const bool skip_second = getenv("LLAMA_VRAM_WARM_SKIP2") != nullptr;
    const bool skip_first  = getenv("LLAMA_VRAM_WARM_SKIP1") != nullptr;
    // 1. worst-case shapes with the cache empty: the pools reach their size (this also covers the cold path)
    if (!skip_first) { warm_request(n_prompt, 48); }
    warm_shapes_ = {"prefill_" + std::to_string(n_prompt), "decode_draft"};
    // 2. let the governor fill the cache, then run the same shapes at the final cache size; if the pools grew,
    //    the governor trims the cache back to the reserve
    core_ready_.store(true);
    for (int i = 0; i < 100 && !stop_.load(); i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        std::lock_guard<std::mutex> lk(mu_);
        if (core_.state == vram_state::FULL) { break; }
    }
    if (!skip_second) { warm_request(n_prompt, 48); }
    std::this_thread::sleep_for(std::chrono::milliseconds(2 * std::max(50, core_.policy.poll_ms)));
    llama_model_moe_cache_reset_learning(model_); // the warm-up routed random tokens: do not keep what it taught the channel policy
    if (pool_freeze_) { pool_freeze_(true); pool_frozen_.store(true); }
    warm_seconds_ = mono_s() - t0;
    ready_.store(true);
    LOG_INF("vram governor: warm-up done in %.1f s; pools frozen, cache %d chunks\n", warm_seconds_, geom_.chunks);
}

void server_vram::stop() {
    stop_.store(true);
    if (warm_thread_.joinable()) { warm_thread_.join(); }
    if (thread_.joinable()) { thread_.join(); }
}

void server_vram::refresh_geometry(vram_cache_geometry & g) {
    int32_t slots = 0, max_slots = 0;
    size_t slot_bytes = 0;
    llama_model_moe_cache_info(model_, &slots, &slot_bytes, &max_slots);
    g.chunk_slots = chunk_slots_;
    g.chunk_mib   = (int32_t) (((size_t) chunk_slots_ * slot_bytes_ + (1024 * 1024 - 1)) / (1024 * 1024));
    g.chunks      = (slots + chunk_slots_ / 2) / chunk_slots_;
    g.max_chunks  = max_slots_ / chunk_slots_;
}

bool server_vram::read_card(int32_t & free_mib, int32_t & total_mib, int32_t & used_mib) {
    int64_t used = 0, total = 0;
    if (!read_int64_file(mem_used_path_, used) || !read_int64_file(mem_total_path_, total)) {
        return false;
    }
    total_mib = (int32_t) (total / (1024 * 1024));
    used_mib  = (int32_t) (used / (1024 * 1024));
    free_mib  = total_mib - used_mib;
    return true;
}

// VRAM of this process: sum of drm-memory-vram over distinct drm clients in /proc/self/fdinfo
int32_t server_vram::read_proc_vram_mib() {
    DIR * d = opendir("/proc/self/fdinfo");
    if (!d) { return 0; }
    std::set<std::string> seen;
    int64_t kib = 0;
    while (dirent * e = readdir(d)) {
        if (e->d_name[0] == '.') { continue; }
        const std::string text = read_text_file(std::string("/proc/self/fdinfo/") + e->d_name);
        if (text.find("drm-driver:\tamdgpu") == std::string::npos) { continue; }
        auto field = [&](const char * key) -> std::string {
            const size_t p = text.find(key);
            if (p == std::string::npos) { return ""; }
            const size_t b = p + strlen(key);
            return text.substr(b, text.find('\n', b) - b);
        };
        const std::string id = field("drm-client-id:\t");
        if (!id.empty() && !seen.insert(id).second) { continue; }
        const std::string v = field("drm-memory-vram:\t");
        if (!v.empty()) { kib += atoll(v.c_str()); }
    }
    closedir(d);
    return (int32_t) (kib / 1024);
}

bool server_vram::apply_chunks(int32_t chunks, const std::string & why, double now_s, bool urgent) {
    const int32_t before = geom_.chunks;
    const int32_t slots = chunks * chunk_slots_;
    int32_t old_slots = 0, new_slots = 0;
    size_t released = 0, allocated = 0;
    const double t0 = mono_s();
    // VRAMLOGIC2: the resize takes the cache lock, which a running ubatch holds until it ends. An urgent trim (the card is
    // already below the floor) raises the abort flag: the ubatch stops at its next graph node (CPU splits check the callback
    // per node), the lock is handed to the waiting resize, and the server retries that slice afterwards. The flag is down
    // again before the retry can start computing, because the retry waits for this very lock.
    const bool interrupt = urgent && chunks < before && core_.policy.trim_abort;
    if (interrupt) { abort_wanted_.store(true, std::memory_order_relaxed); }
    const bool ok = llama_model_moe_cache_resize(model_, slots, &old_slots, &new_slots, &released, &allocated);
    if (interrupt) { abort_wanted_.store(false, std::memory_order_relaxed); }
    const double ms = (mono_s() - t0) * 1000.0;
    last_resize_ms_ = (uint64_t) ms;
    refresh_geometry(geom_);
    LOG_INF("vram governor: cache %d -> %d chunks (%d -> %d slots, %s%s) in %.0f ms, free %d MiB%s\n",
            before, geom_.chunks, old_slots, new_slots, why.c_str(), interrupt ? ", urgent" : "", ms, last_free_mib_, ok ? "" : " [resize failed]");
    if (!ok) {
        core_.reset_calm(now_s);
    }
    return ok;
}

void server_vram::loop() {
    while (!stop_.load()) {
        int poll_ms = 200;
        {
            std::lock_guard<std::mutex> lk(mu_);
            poll_ms = core_.poll_interval_ms();    // VRAMLOGIC2: fast below the target or while free VRAM falls
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
        int32_t free_mib = 0, total_mib = 0, used_mib = 0;
        if (!read_card(free_mib, total_mib, used_mib)) { continue; }
        const double now = mono_s();
        std::string pause_why;
        {
        std::lock_guard<std::mutex> lk(mu_);
        if (sim_until_s_ > now) { free_mib += sim_delta_mib_; }
        last_free_mib_ = free_mib;
        window_min_free_ = std::min(window_min_free_, free_mib);

        vram_inputs in;
        in.now_s = now; in.free_mib = free_mib; in.ready = core_ready_.load(); in.busy = busy_ ? busy_() : false; in.big_prefill = big_prefill_ ? big_prefill_() : false;
        refresh_geometry(geom_);
        const vram_action act = core_.step(in, geom_);
        if (act.kind == vram_action_kind::SET_CHUNKS && act.chunks != geom_.chunks) {
            const int32_t before = geom_.chunks;
            apply_chunks(act.chunks, act.reason, now, act.urgent);
            core_.note_applied(act, before, geom_.chunks, now);
        } else if (act.kind == vram_action_kind::PAUSE) {
            pause_why = act.reason;
            LOG_WRN("vram governor: pause requested (%s), free %d MiB with the cache at its minimum\n",
                    act.reason.c_str(), free_mib);
        }
        }
        if (!pause_why.empty()) {
            request_pause(pause_why, "");
        }
    }
}

bool server_vram::request_pause(const std::string & reason, const std::string & checkpoint) {
    if (!pause_cb_ || pause_requested_.exchange(true)) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lk(mu_);
        core_.set_state(vram_state::PAUSING, reason);
        if (reason != "pause_floor" && reason != "short_stall") { // the governor's own pauses were journaled in step()
            core_.note_pause(reason, mono_s(), last_free_mib_);
        }
    }
    // marker for the gateway: why the process is about to exit with code 75 and what the card looked like
    if (!marker_dir_.empty()) {
        try {
            json m = status();
            m["exit_reason"] = reason;
            m["exit_code"] = 75;
            std::ofstream f(marker_dir_ + "vram-exit.json");
            f << m.dump();
        } catch (...) {
        }
    }
    pause_cb_(reason, checkpoint.empty() ? "vram-pause.bin" : checkpoint);
    return true;
}

server_vram::json server_vram::policy_locked() const {
    const auto & p = core_.policy;
    return {
        {"reserve_mib", p.reserve_mib}, {"buffer_mib", p.buffer_mib}, {"target_mib", p.target_mib()}, {"hysteresis_mib", p.hysteresis_mib},
        {"grow_seconds", p.grow_seconds},
        {"pause_floor_mib", p.pause_floor_mib}, {"pause_after_ms", p.pause_after_ms}, {"pause_after_seconds", p.pause_after_ms / 1000.0},
        {"min_chunks", p.min_chunks},
        {"max_chunks", p.max_chunks < 0 ? json(nullptr) : json(p.max_chunks)},
        {"prefill_yield", p.prefill_yield}, {"poll_ms", p.poll_ms}, {"poll_fast_ms", p.poll_fast_ms}, {"trim_abort", p.trim_abort},
        {"sleep_level", p.sleep_level},
    };
}

server_vram::json server_vram::policy_json() {
    std::lock_guard<std::mutex> lk(mu_);
    return policy_locked();
}

bool server_vram::set_policy(const json & body, std::string & err) {
    if (!body.is_object()) { err = "body must be a JSON object"; return false; }
    vram_policy p;
    {
        std::lock_guard<std::mutex> lk(mu_);
        p = core_.policy;
    }
    auto num = [&](const char * key, int32_t lo, int32_t hi, int32_t & out) -> bool {
        if (!body.contains(key)) { return true; }
        const auto & v = body[key];
        if (!v.is_number_integer()) { err = std::string(key) + " must be an integer"; return false; }
        const int64_t x = v.get<int64_t>();
        if (x < lo || x > hi) { err = std::string(key) + " out of range"; return false; }
        out = (int32_t) x;
        return true;
    };
    static const std::set<std::string> known = {"reserve_mib", "buffer_mib", "hysteresis_mib", "grow_seconds", "pause_floor_mib",
        "min_chunks", "max_chunks", "prefill_yield", "poll_ms", "poll_fast_ms", "trim_abort", "pause_after_seconds", "pause_after_ms",
        "sleep_level", "human_need", "note"};
    for (auto it = body.begin(); it != body.end(); ++it) {
        if (!known.count(it.key())) { err = "unknown field " + it.key(); return false; }
    }
    if (!num("reserve_mib", 256, 4096, p.reserve_mib) || !num("buffer_mib", 0, 2048, p.buffer_mib) ||
        !num("hysteresis_mib", 0, 1024, p.hysteresis_mib) ||
        !num("grow_seconds", 0, 600, p.grow_seconds) || !num("pause_floor_mib", 0, 2048, p.pause_floor_mib) ||
        !num("min_chunks", 0, 64, p.min_chunks) || !num("pause_after_ms", 0, 600000, p.pause_after_ms) || !num("poll_ms", 50, 1000, p.poll_ms) ||
        !num("poll_fast_ms", 20, 1000, p.poll_fast_ms) || !num("sleep_level", 0, 1, p.sleep_level)) {
        return false;
    }
    if (body.contains("pause_after_seconds") && !body.contains("pause_after_ms")) {
        // the VRAMLOGIC1 field, now also fractional (0.25 = 250 ms); pause_after_ms wins when both are given
        const auto & v = body["pause_after_seconds"];
        if (!v.is_number() || v.get<double>() < 0.0 || v.get<double>() > 600.0) { err = "pause_after_seconds must be a number 0..600"; return false; }
        p.pause_after_ms = (int32_t) std::lround(v.get<double>() * 1000.0);
    }
    if (body.contains("trim_abort")) {
        if (!body["trim_abort"].is_boolean()) { err = "trim_abort must be a boolean"; return false; }
        p.trim_abort = body["trim_abort"].get<bool>();
    }
    double human_need = -2;
    if (body.contains("human_need")) {
        const auto & v = body["human_need"];
        if (!v.is_number() || v.get<double>() < 0.0 || v.get<double>() > 1.0) { err = "human_need must be a number 0..1"; return false; }
        human_need = v.get<double>();
    }
    std::string note;
    if (body.contains("note")) {
        if (!body["note"].is_string() || body["note"].get<std::string>().size() > 200) { err = "note must be a string up to 200 characters"; return false; }
        note = body["note"].get<std::string>();
    }
    if (body.contains("max_chunks")) {
        const auto & v = body["max_chunks"];
        if (v.is_null()) {
            p.max_chunks = -1;
        } else if (v.is_number_integer() && v.get<int64_t>() >= 0 && v.get<int64_t>() <= 64) {
            p.max_chunks = (int32_t) v.get<int64_t>();
        } else {
            err = "max_chunks must be null or an integer 0..64";
            return false;
        }
    }
    if (body.contains("prefill_yield")) {
        if (!body["prefill_yield"].is_boolean()) { err = "prefill_yield must be a boolean"; return false; }
        p.prefill_yield = body["prefill_yield"].get<bool>();
    }
    std::lock_guard<std::mutex> lk(mu_);
    core_.note_policy(core_.policy, p, note, mono_s());
    core_.policy = p;
    if (human_need > -1) { human_need_ = human_need; }
    return true;
}

server_vram::json server_vram::journal_json(uint64_t since_id, size_t limit) {
    std::lock_guard<std::mutex> lk(mu_);
    json events = json::array();
    const double mono_now = mono_s(), wall_now = vram_journal::wall_now();
    for (const auto & e : core_.journal.since(since_id, limit)) {
        events.push_back({
            {"id", e.id}, {"t", std::round((wall_now - (mono_now - e.mono_s)) * 1000.0) / 1000.0}, {"kind", e.kind}, {"reason", e.reason},
            {"state", e.state}, {"chunks_before", e.chunks_before}, {"chunks_after", e.chunks_after}, {"mib", e.mib},
            {"free_mib", e.free_mib}, {"recovered_s", e.recovered_s < 0 ? json(nullptr) : json(std::round(e.recovered_s * 100.0) / 100.0)},
            {"note", e.note},
        });
    }
    const auto & t = core_.journal.totals;
    return {{"events", events}, {"last_id", core_.journal.last_id()},
            {"totals", {{"yields", t.yields}, {"yielded_mib", t.yielded_mib}, {"recovered", t.recovered},
                        {"recovered_mean_s", t.recovered ? std::round(t.recovered_sum_s / double(t.recovered) * 100.0) / 100.0 : 0.0},
                        {"recovered_max_s", std::round(t.recovered_max_s * 100.0) / 100.0}, {"pauses", t.pauses}, {"sleeps", t.sleeps}}},
            {"human_need", human_need_ < 0 ? json(nullptr) : json(human_need_)}, {"uptime_s", std::round((mono_now - uptime0_s_) * 10.0) / 10.0}};
}

server_vram::json server_vram::set_chunks_manual(int32_t chunks, std::string & err) {
    std::lock_guard<std::mutex> lk(mu_);
    refresh_geometry(geom_);
    if (chunks < 0 || chunks > geom_.max_chunks) { err = "chunks out of range"; return json::object(); }
    const bool ok = chunks == geom_.chunks || apply_chunks(chunks, "manual", mono_s());
    if (!ok) { err = "resize failed"; }
    return {{"chunks", geom_.chunks}, {"slots", geom_.chunks * chunk_slots_}, {"resize_ms", last_resize_ms_}};
}

void server_vram::simulate(int32_t free_delta_mib, double seconds) {
    std::lock_guard<std::mutex> lk(mu_);
    sim_delta_mib_ = free_delta_mib;
    sim_until_s_ = mono_s() + seconds;
}

// host (system RAM) side of the server, for the RAM budget of the gateway
server_vram::json server_vram::host_json() {
    auto kib_of = [](const std::string & text, const char * key) -> int64_t {
        const size_t p = text.find(key);
        if (p == std::string::npos) { return 0; }
        return atoll(text.c_str() + p + strlen(key));
    };
    const std::string st = read_text_file("/proc/self/status");
    const std::string mi = read_text_file("/proc/meminfo");
    auto psi = [](const char * line_key, const char * field) -> double {
        const std::string t = read_text_file("/proc/pressure/memory");
        const size_t l = t.find(line_key);
        if (l == std::string::npos) { return -1.0; }
        const size_t f = t.find(field, l);
        if (f == std::string::npos) { return -1.0; }
        return atof(t.c_str() + f + strlen(field));
    };
    uint64_t pinned = 0, denied = 0;
    if (host_stats_) { host_stats_(&pinned, &denied); }
    host_parts hp = host_cb_ ? host_cb_() : host_parts();
    const int64_t anon = kib_of(st, "RssAnon:") / 1024, file = kib_of(st, "RssFile:") / 1024, shmem = kib_of(st, "RssShmem:") / 1024, lck = kib_of(st, "VmLck:") / 1024;
    return {
        {"process", {{"rss_anon_mib", anon}, {"rss_file_mib", file}, {"rss_shmem_mib", shmem}, {"locked_mib", lck},
                     {"resident_fixed_mib", anon + shmem + lck}}},
        {"pinned_mib", (int64_t) (pinned / (1024 * 1024))}, {"pinned_denied", denied}, {"pinned_max_mib", host_pinned_max_mib_},
        {"checkpoints_mib", hp.checkpoints_mib}, {"prompt_cache_mib", hp.prompt_cache_mib},
        {"system", {{"mem_available_mib", kib_of(mi, "MemAvailable:") / 1024}, {"cached_mib", kib_of(mi, "Cached:") / 1024},
                    {"swap_used_mib", (kib_of(mi, "SwapTotal:") - kib_of(mi, "SwapFree:")) / 1024}, {"swap_total_mib", kib_of(mi, "SwapTotal:") / 1024},
                    {"shmem_mib", kib_of(mi, "Shmem:") / 1024},
                    {"psi_some_avg10", psi("some", "avg10=")}, {"psi_some_avg60", psi("some", "avg60=")}, {"psi_full_avg10", psi("full", "avg10=")}}},
    };
}

server_vram::json server_vram::status() {
    int32_t free_now = 0, total_now = 0, used_now = 0;
    const bool have_card = read_card(free_now, total_now, used_now);
    const int32_t proc = read_proc_vram_mib();
    std::lock_guard<std::mutex> lk(mu_);
    refresh_geometry(geom_);
    if (have_card && sim_until_s_ > mono_s()) { free_now += sim_delta_mib_; }
    const int32_t win_min = window_min_free_ == INT32_MAX ? free_now : std::min(window_min_free_, free_now);
    window_min_free_ = INT32_MAX;
    const int32_t cache_mib = geom_.chunks * geom_.chunk_mib;
    fixed_parts fp = fixed_cb_ ? fixed_cb_() : fixed_parts();
    fp.weights = std::max(0, fp.weights - cache_mib); // the cache buffers are counted with the model by ggml
    const int32_t known = fp.weights + fp.kv + fp.sched + fp.draft;
    uint64_t pool_overflows = 0, pool_bytes = 0;
    if (pool_stats_) { pool_stats_(&pool_overflows, &pool_bytes); }
    // "pool" is the measured remainder: the process's VRAM minus the cache and the sized parts
    const int32_t pool = std::max(0, proc - cache_mib - known);
    const int32_t fixed_total = std::max(0, proc - cache_mib);
    return {
        {"state", vram_state_name(core_.state)},
        {"reason", core_.last_reason},
        {"ready", ready_.load()},
        {"pool_frozen", pool_frozen_.load()},
        {"card", {{"total_mib", total_now}, {"free_mib", free_now}, {"min_free_mib_window", win_min}, {"source", have_card ? "sysfs" : "none"}}},
        {"fixed_mib", {{"weights", fp.weights}, {"kv", fp.kv}, {"draft", fp.draft}, {"pool", pool},
                       {"sched", fp.sched}, {"vision", 0}, {"total", fixed_total}, {"pool_backend_mib", (int64_t) (pool_bytes / (1024 * 1024))}}},
        {"cache", {{"enabled", geom_.chunks > 0}, {"chunk_slots", geom_.chunk_slots}, {"chunk_mib", geom_.chunk_mib},
                   {"chunks", geom_.chunks}, {"slots", geom_.chunks * geom_.chunk_slots}, {"mib", cache_mib},
                   {"target_chunks", geom_.chunks}, {"max_chunks", core_.policy.max_chunks < 0 ? json(nullptr) : json(core_.policy.max_chunks)},
                   {"last_resize_ms", last_resize_ms_}, {"ubatch_interrupts", ubatch_interrupts_.load()}}},
        {"policy", policy_locked()},
        {"journal", {{"last_id", core_.journal.last_id()}, {"yields", core_.journal.totals.yields}, {"yielded_mib", core_.journal.totals.yielded_mib},
                     {"recovered_max_s", std::round(core_.journal.totals.recovered_max_s * 100.0) / 100.0}}},
        {"human_need", human_need_ < 0 ? json(nullptr) : json(human_need_)},
        {"counters", {{"trims", core_.trims}, {"chunks_dropped", core_.chunks_dropped}, {"grows", core_.grows},
                      {"chunks_grown", core_.chunks_grown}, {"pauses", core_.pauses},
                      {"pool_overflow", pool_overflows},
                      {"free_below_reserve_events", core_.below_reserve_events},
                      {"free_below_reserve_max_s", std::round(core_.below_reserve_max_s * 100.0) / 100.0}}},
        {"warmup", {{"done", ready_.load()}, {"seconds", std::round(warm_seconds_ * 10.0) / 10.0}, {"shapes", warm_shapes_}}},
        {"host", host_json()},
        {"uptime_s", std::round((mono_s() - uptime0_s_) * 10.0) / 10.0},
    };
}
