// SHURA (PATCH1): expansion cards as patches; see llama-patch.h
#include "llama-patch.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-context.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <set>
#include <unordered_map>

//
// SHA-256 (FIPS 180-4), small and self-contained: the card format is checked with it at every load and removal
//

namespace {

struct sha256_ctx {
    uint32_t h[8] = { 0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19 };
    uint8_t  buf[64];
    size_t   n   = 0;
    uint64_t len = 0;

    static uint32_t rotr(uint32_t x, int r) { return (x >> r) | (x << (32 - r)); }

    void block(const uint8_t * p) {
        static const uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
            0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
            0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08,
            0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2 };
        uint32_t w[64];
        for (int i = 0; i < 16; i++) {
            w[i] = (uint32_t) p[4*i] << 24 | (uint32_t) p[4*i + 1] << 16 | (uint32_t) p[4*i + 2] << 8 | (uint32_t) p[4*i + 3];
        }
        for (int i = 16; i < 64; i++) {
            const uint32_t s0 = rotr(w[i-15], 7) ^ rotr(w[i-15], 18) ^ (w[i-15] >> 3);
            const uint32_t s1 = rotr(w[i-2], 17) ^ rotr(w[i-2], 19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; i++) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    }

    void update(const void * data, size_t size) {
        const uint8_t * p = (const uint8_t *) data;
        len += size;
        if (n > 0) {
            const size_t m = std::min(size, 64 - n);
            memcpy(buf + n, p, m);
            n += m; p += m; size -= m;
            if (n == 64) {
                block(buf);
                n = 0;
            }
        }
        while (size >= 64) {
            block(p);
            p += 64; size -= 64;
        }
        if (size > 0) {
            memcpy(buf, p, size);
            n = size;
        }
    }

    std::string hex() {
        const uint64_t bits = len * 8;
        const uint8_t pad = 0x80;
        update(&pad, 1);
        const uint8_t zero = 0;
        while (n != 56) {
            update(&zero, 1);
        }
        uint8_t l[8];
        for (int i = 0; i < 8; i++) {
            l[i] = (uint8_t) (bits >> (56 - 8*i));
        }
        update(l, 8);
        static const char * dig = "0123456789abcdef";
        std::string s(64, '0');
        for (int i = 0; i < 8; i++) {
            for (int j = 0; j < 4; j++) {
                const uint8_t v = (uint8_t) (h[i] >> (24 - 8*j));
                s[8*i + 2*j]     = dig[v >> 4];
                s[8*i + 2*j + 1] = dig[v & 15];
            }
        }
        return s;
    }
};

struct file_reader {
    FILE * f = nullptr;
    explicit file_reader(const std::string & path) { f = fopen(path.c_str(), "rb"); }
    ~file_reader() { if (f) { fclose(f); } }
    bool read_at(size_t offset, void * dst, size_t size) const {
        if (!f || fseeko(f, (off_t) offset, SEEK_SET) != 0) {
            return false;
        }
        return fread(dst, 1, size, f) == size;
    }
};

// "blk.<n>." -> n, otherwise -1
int32_t block_of(const std::string & name) {
    int32_t il = -1;
    int     n  = 0;
    if (sscanf(name.c_str(), "blk.%d.%n", &il, &n) == 1 && n > 0) {
        return il;
    }
    return -1;
}

bool is_expert_tensor(const std::string & name) {
    return name.find("_exps") != std::string::npos;
}

std::string layout_of_gguf(const gguf_context * g) {
    sha256_ctx h;
    char line[256];
    const int64_t n = gguf_get_n_tensors(g);
    for (int64_t i = 0; i < n; i++) {
        // ne is not in the gguf API without a ggml context: the type and the size in bytes are, and with the name they pin the layout
        const size_t sz = gguf_get_tensor_size(g, i);
        snprintf(line, sizeof(line), "%s\t%d\t%zu\n", gguf_get_tensor_name(g, i), (int) gguf_get_tensor_type(g, i), sz);
        h.update(line, strlen(line));
    }
    return h.hex();
}

void set_err(char * err, size_t err_size, const std::string & msg) {
    if (err && err_size > 0) {
        snprintf(err, err_size, "%s", msg.c_str());
    }
}

struct slice_key {
    std::string target;
    int32_t     expert;
    bool operator<(const slice_key & o) const { return target != o.target ? target < o.target : expert < o.expert; }
};

} // namespace

std::string llama_patch_sha256_hex(const void * data, size_t len) {
    sha256_ctx h;
    h.update(data, len);
    return h.hex();
}

llama_patch::~llama_patch() {
    if (ctx) {
        ggml_free(ctx);
    }
    if (gguf) {
        gguf_free(gguf);
    }
}

// the base file's metadata (offsets for removal, the layout hash); read once per model
static bool patch_state_init(llama_model & model, const std::string & base_path, std::string & why) {
    if (!model.patch) {
        model.patch = std::make_shared<llama_patch_state>();
    }
    llama_patch_state & st = *model.patch;
    if (!st.base_path.empty()) {
        if (st.base_path != base_path) {
            why = "the model was opened for cards with another base file: " + st.base_path;
            return false;
        }
        return true;
    }
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(base_path.c_str(), gp);
    if (!g) {
        why = "cannot read the base file " + base_path;
        return false;
    }
    // the base file must be the model's file: every model tensor is in it with the same type and size
    std::map<std::string, size_t> offs;
    for (int64_t i = 0; i < gguf_get_n_tensors(g); i++) {
        offs[gguf_get_tensor_name(g, i)] = gguf_get_tensor_offset(g, i);
    }
    for (const auto & [name, t] : model.tensors_by_name) {
        const int64_t id = gguf_find_tensor(g, name.c_str());
        if (id < 0 || gguf_get_tensor_type(g, id) != t->type || gguf_get_tensor_size(g, id) != ggml_nbytes(t)) {
            gguf_free(g);
            why = "the base file " + base_path + " is not the model's file (tensor " + name + ")";
            return false;
        }
    }
    st.base_layout      = layout_of_gguf(g);
    st.base_data_offset = gguf_get_data_offset(g);
    st.base_offsets     = std::move(offs);
    st.base_path        = base_path;
    st.n_expert         = (int32_t) model.hparams.n_expert;
    st.replaced.assign((size_t) model.hparams.n_layer_all * std::max<int32_t>(st.n_expert, 1), 0);
    gguf_free(g);
    return true;
}

std::string llama_patch_layout_sha256(const llama_model & model) {
    return model.patch ? model.patch->base_layout : std::string();
}

// all checks of a card, before anything is written (PATCH1 §2)
static llama_patch * patch_load(llama_model & model, const std::string & base_path, const std::string & pack_path, std::string & why) {
    if (!patch_state_init(model, base_path, why)) {
        return nullptr;
    }
    llama_patch_state & st = *model.patch;

    auto p = std::make_unique<llama_patch>();
    p->path  = pack_path;
    p->model = &model;
    p->state = model.patch;
    {
        std::string n = pack_path.substr(pack_path.find_last_of('/') + 1);
        if (n.size() > 5 && n.compare(n.size() - 5, 5, ".gguf") == 0) {
            n.resize(n.size() - 5);
        }
        p->name = n;
    }

    // a truncated or damaged file is rejected whole by the GGUF reader (sizes, offsets, the data blob against the file size)
    gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &p->ctx };
    p->gguf = gguf_init_from_file(pack_path.c_str(), gp);
    if (!p->gguf || !p->ctx) {
        why = "not a valid card file (damaged or truncated GGUF): " + pack_path;
        return nullptr;
    }
    const gguf_context * g = p->gguf;

    auto get_str = [&](const char * key, std::string & out) -> bool {
        const int64_t id = gguf_find_key(g, key);
        if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_STRING) {
            return false;
        }
        out = gguf_get_val_str(g, id);
        return true;
    };
    {
        const int64_t id = gguf_find_key(g, "shura.patch.version");
        if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_UINT32 || gguf_get_val_u32(g, id) != 1) {
            why = "not a card (shura.patch.version 1 missing): " + pack_path;
            return nullptr;
        }
    }
    if (!get_str("shura.patch.manifest", p->manifest) || p->manifest.empty() || p->manifest[0] != '{') {
        why = "the card has no manifest";
        return nullptr;
    }
    if (!get_str("shura.patch.base_layout_sha256", p->layout_sha256)) {
        why = "the card has no base layout hash";
        return nullptr;
    }
    if (p->layout_sha256 != st.base_layout) {
        why = "the card was made for another base (layout " + p->layout_sha256.substr(0, 12) + " != " + st.base_layout.substr(0, 12) + ")";
        return nullptr;
    }

    const int64_t n = gguf_get_n_tensors(g);
    auto arr = [&](const char * key, gguf_type type) -> int64_t {
        const int64_t id = gguf_find_key(g, key);
        if (id < 0 || gguf_get_kv_type(g, id) != GGUF_TYPE_ARRAY || gguf_get_arr_type(g, id) != type || (int64_t) gguf_get_arr_n(g, id) != n) {
            return -1;
        }
        return id;
    };
    const int64_t k_tgt = arr("shura.patch.targets",     GGUF_TYPE_STRING);
    const int64_t k_exp = arr("shura.patch.experts",     GGUF_TYPE_INT32);
    const int64_t k_bsh = arr("shura.patch.base_sha256", GGUF_TYPE_STRING);
    if (n <= 0 || k_tgt < 0 || k_exp < 0 || k_bsh < 0) {
        why = "the card's slice table is missing or does not match its tensors";
        return nullptr;
    }

    std::unordered_map<std::string, ggml_tensor *> by_name;
    for (const auto & [name, t] : model.tensors_by_name) {
        by_name[name] = t;
    }
    const int32_t n_layer_all = (int32_t) model.hparams.n_layer_all;
    const int32_t first_mtp   = n_layer_all - (int32_t) model.hparams.n_layer_nextn;
    const int32_t * experts   = (const int32_t *) gguf_get_arr_data(g, k_exp);

    std::map<std::string, std::set<int32_t>> claimed;   // target -> experts (-1: the whole tensor)
    size_t total = 0;
    for (int64_t i = 0; i < n; i++) {
        llama_patch_slice s;
        s.target      = gguf_get_arr_str(g, k_tgt, i);
        s.expert      = experts[i];
        s.base_sha256 = gguf_get_arr_str(g, k_bsh, i);
        s.il          = block_of(s.target);
        const std::string where = "slice " + std::to_string(i) + " (" + s.target + (s.expert >= 0 ? " expert " + std::to_string(s.expert) : "") + "): ";

        char tname[32];
        snprintf(tname, sizeof(tname), "slice.%" PRId64, i);
        ggml_tensor * src = ggml_get_tensor(p->ctx, tname);
        if (!src || std::string(gguf_get_tensor_name(g, i)) != tname) {
            why = where + "tensor " + tname + " missing or out of order";
            return nullptr;
        }
        if ((s.il >= first_mtp && s.il < n_layer_all && model.hparams.n_layer_nextn > 0) || s.target.find("nextn") != std::string::npos) {
            why = where + "the MTP head block is not a patch target";
            return nullptr;
        }
        auto it = by_name.find(s.target);
        if (it == by_name.end()) {
            why = where + "no such tensor in the base";
            return nullptr;
        }
        ggml_tensor * t = it->second;
        if (src->type != t->type) {
            why = where + std::string("type ") + ggml_type_name(src->type) + " != base " + ggml_type_name(t->type);
            return nullptr;
        }
        if (s.expert >= 0) {
            if (!is_expert_tensor(s.target) || ggml_n_dims(t) != 3) {
                why = where + "an expert slice of a tensor that is not a stacked expert tensor";
                return nullptr;
            }
            if (s.expert >= t->ne[2]) {
                why = where + "expert index out of range (" + std::to_string(t->ne[2]) + " experts)";
                return nullptr;
            }
            if (src->ne[0] != t->ne[0] || src->ne[1] != t->ne[1] || src->ne[2] != 1 || src->ne[3] != 1) {
                why = where + "shape does not match one expert of the base tensor";
                return nullptr;
            }
            s.offset = (size_t) s.expert * t->nb[2];
            s.size   = t->nb[2];
        } else if (s.expert == -1) {
            for (int d = 0; d < GGML_MAX_DIMS; d++) {
                if (src->ne[d] != t->ne[d]) {
                    why = where + "shape does not match the base tensor";
                    return nullptr;
                }
            }
            s.offset = 0;
            s.size   = ggml_nbytes(t);
        } else {
            why = where + "bad expert index";
            return nullptr;
        }
        if (ggml_nbytes(src) != s.size || s.offset > ggml_nbytes(t) || s.size > ggml_nbytes(t) - s.offset) {
            why = where + "byte range outside the base tensor";
            return nullptr;
        }
        // overlap: the same slice twice, or a whole tensor together with any of its experts
        auto & cl = claimed[s.target];
        if (cl.count(s.expert) || (s.expert >= 0 && cl.count(-1)) || (s.expert == -1 && !cl.empty())) {
            why = where + "overlaps another slice of the card";
            return nullptr;
        }
        cl.insert(s.expert);
        // the model's copy must be writable memory in the base format
        if (!t->buffer || t->extra != nullptr || t->view_src != nullptr) {
            why = where + "the base tensor is not in a plain buffer (repacked or a view)";
            return nullptr;
        }
        if (model.data_is_mapped(t->data)) {
            why = where + "the base tensor is mapped read-only from the file (load the model without mmap, e.g. --load-mode dio)";
            return nullptr;
        }
        s.dst  = t;
        s.data = (const uint8_t *) src->data;
        s.sha256 = llama_patch_sha256_hex(s.data, s.size);
        total += s.size;
        p->slices.push_back(std::move(s));
    }

    // the base bytes each slice replaces: read from the base file, hash checked (this is also the base content check)
    {
        file_reader fr(st.base_path);
        std::vector<uint8_t> buf;
        for (const auto & s : p->slices) {
            auto it = st.base_offsets.find(s.target);
            if (it == st.base_offsets.end()) {
                why = s.target + ": not in the base file";
                return nullptr;
            }
            buf.resize(s.size);
            if (!fr.read_at(st.base_data_offset + it->second + s.offset, buf.data(), s.size)) {
                why = s.target + ": cannot read the base bytes";
                return nullptr;
            }
            if (llama_patch_sha256_hex(buf.data(), s.size) != s.base_sha256) {
                why = s.target + (s.expert >= 0 ? " expert " + std::to_string(s.expert) : "") + ": the base bytes differ from the card's record (another base)";
                return nullptr;
            }
        }
    }

    // the content hash: the slice table and the bytes; the manifest must name it
    sha256_ctx h;
    for (const auto & s : p->slices) {
        const std::string line = s.target + "\t" + std::to_string(s.expert) + "\t" + s.sha256 + "\n";
        h.update(line.data(), line.size());
    }
    p->content_sha256 = h.hex();
    std::string declared;
    if (!get_str("shura.patch.content_sha256", declared) || declared != p->content_sha256) {
        why = "the card's content hash (shura.patch.content_sha256) does not match its slices: computed " + p->content_sha256.substr(0, 12) +
            ", declared " + (declared.empty() ? std::string("none") : declared.substr(0, 12)) + " (damaged or edited card)";
        return nullptr;
    }
    LLAMA_LOG_INFO("%s: card '%s': %zu slices, %.1f MiB, content %s\n", __func__, p->name.c_str(), p->slices.size(), total/1048576.0,
        p->content_sha256.substr(0, 12).c_str());
    return p.release();
}

llama_patch * llama_patch_load(const llama_model * model, const char * base_path, const char * pack_path, char * err, size_t err_size) {
    std::string why;
    llama_patch * p = nullptr;
    try {
        p = model && base_path && pack_path ? patch_load(const_cast<llama_model &>(*model), base_path, pack_path, why) : nullptr;
    } catch (const std::exception & ex) {
        why = ex.what();
    }
    if (!p) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, why.c_str());
        set_err(err, err_size, why.empty() ? "bad arguments" : why);
    }
    return p;
}

void llama_patch_free(llama_patch * patch) {
    if (!patch) {
        return;
    }
    // the model may already be gone (its state is held weakly): then nothing is active any more
    if (auto st = patch->state.lock()) {
        std::lock_guard<std::mutex> lk(st->mu);
        for (const auto & a : st->applied) {
            GGML_ASSERT(a.get() != patch && "a card is freed while it is active: make another set active first");
        }
    }
    delete patch;
}

const char * llama_patch_name(const llama_patch * p)         { return p ? p->name.c_str() : ""; }
const char * llama_patch_content_hash(const llama_patch * p) { return p ? p->content_sha256.c_str() : ""; }
const char * llama_patch_manifest(const llama_patch * p)     { return p ? p->manifest.c_str() : ""; }
int32_t      llama_patch_n_slices(const llama_patch * p)     { return p ? (int32_t) p->slices.size() : 0; }

static bool patch_set(llama_model & model, llama_patch ** packs, int32_t n, std::string & why) {
    if (n > 0 && !model.patch) {
        why = "no card was loaded for this model";
        return false;
    }
    if (!model.patch) {
        return true;   // the base, and nothing was ever applied
    }
    llama_patch_state & st = *model.patch;
    std::lock_guard<std::mutex> lk(st.mu);

    // the new state: one owner per slice (PATCH1 §5)
    std::map<slice_key, const llama_patch_slice *> want;
    std::map<std::string, std::string> whole_owner;
    std::vector<llama_patch *> set;
    for (int32_t i = 0; i < n; i++) {
        llama_patch * p = packs[i];
        if (!p || p->model != &model) {
            why = "card " + std::to_string(i) + " was loaded for another model";
            return false;
        }
        for (const auto & s : p->slices) {
            const slice_key k = { s.target, s.expert };
            auto conflict = [&](const llama_patch_slice * o) {
                for (const auto * q : set) {
                    for (const auto & qs : q->slices) {
                        if (&qs == o) {
                            why = "cards '" + q->name + "' and '" + p->name + "' both replace " + s.target +
                                (s.expert >= 0 ? " expert " + std::to_string(s.expert) : "") +
                                ": a slice belongs to one card (train the second over the first and release it as a new version)";
                            return;
                        }
                    }
                }
                why = "two cards replace " + s.target;
            };
            if (auto it = want.find(k); it != want.end()) {
                conflict(it->second);
                return false;
            }
            // whole tensor against an expert of it, either way
            if (auto it = want.find({ s.target, -1 }); it != want.end()) {
                conflict(it->second);
                return false;
            }
            if (s.expert == -1) {
                for (auto it = want.lower_bound({ s.target, INT32_MIN }); it != want.end() && it->first.target == s.target; ++it) {
                    conflict(it->second);
                    return false;
                }
            }
            want[k] = &s;
        }
        set.push_back(p);
    }

    // the current state
    std::map<slice_key, const llama_patch_slice *> have;
    for (const auto & p : st.applied) {
        for (const auto & s : p->slices) {
            have[{ s.target, s.expert }] = &s;
        }
    }

    // what to write: new slices whose bytes differ from what is there; what to restore: old slices the new set does not cover
    std::vector<const llama_patch_slice *> to_write;
    std::vector<const llama_patch_slice *> to_restore;
    for (const auto & [k, s] : want) {
        auto it = have.find(k);
        if (it == have.end() || it->second->sha256 != s->sha256) {
            to_write.push_back(s);
        }
    }
    for (const auto & [k, s] : have) {
        if (!want.count(k)) {
            to_restore.push_back(s);
        }
    }

    // the base bytes for the restore: read and checked before anything is written
    std::vector<std::vector<uint8_t>> base_bytes(to_restore.size());
    {
        file_reader fr(st.base_path);
        for (size_t i = 0; i < to_restore.size(); i++) {
            const auto * s = to_restore[i];
            base_bytes[i].resize(s->size);
            if (!fr.read_at(st.base_data_offset + st.base_offsets.at(s->target) + s->offset, base_bytes[i].data(), s->size) ||
                    llama_patch_sha256_hex(base_bytes[i].data(), s->size) != s->base_sha256) {
                why = s->target + ": the base file no longer has the original bytes (changed on disk?): nothing was written";
                return false;
            }
        }
    }

    // the first write of a slice: the bytes in memory must be the base bytes the card records (the loaded model is the base file)
    {
        std::vector<uint8_t> live;
        for (const auto * s : to_write) {
            // only tensors no card has touched yet: elsewhere the bytes may be another card's (checked when it was written)
            bool touched = false;
            for (auto it = st.touched.lower_bound(s->target + "#"); it != st.touched.end() && it->first.compare(0, s->target.size() + 1, s->target + "#") == 0; ++it) {
                touched = true;
                break;
            }
            for (const auto & [k, h] : have) {
                touched = touched || k.target == s->target;
            }
            if (touched) {
                continue;
            }
            live.resize(s->size);
            ggml_backend_tensor_get(s->dst, live.data(), s->offset, s->size);
            if (llama_patch_sha256_hex(live.data(), s->size) != s->base_sha256) {
                why = s->target + (s->expert >= 0 ? " expert " + std::to_string(s->expert) : "") +
                    ": the loaded model does not hold the base bytes the card records (another file was loaded?): nothing was written";
                return false;
            }
        }
    }

    // the experts to read again in every expert cache
    std::set<std::pair<int32_t, int32_t>> changed;
    auto note = [&](const llama_patch_slice * s) {
        if (s->il < 0 || !is_expert_tensor(s->target)) {
            return;
        }
        if (s->expert >= 0) {
            changed.insert({ s->il, s->expert });
        } else {
            for (int32_t e = 0; e < (int32_t) s->dst->ne[2]; e++) {
                changed.insert({ s->il, e });
            }
        }
    };
    for (const auto * s : to_write)   { note(s); }
    for (const auto * s : to_restore) { note(s); }

    const int64_t t0 = ggml_time_us();
    // every context of the model: its micro-batch lock, its queues drained (no graph and no upload may read the old bytes)
    std::vector<std::unique_lock<std::recursive_mutex>> locks;
    for (llama_context * c : st.contexts) {
        locks.push_back(c->patch_lock());
    }
    size_t bytes = 0;
    for (size_t i = 0; i < to_restore.size(); i++) {
        ggml_backend_tensor_set(to_restore[i]->dst, base_bytes[i].data(), to_restore[i]->offset, to_restore[i]->size);
        bytes += to_restore[i]->size;
    }
    for (const auto * s : to_write) {
        ggml_backend_tensor_set(s->dst, s->data, s->offset, s->size);
        bytes += s->size;
    }
    for (llama_context * c : st.contexts) {
        c->patch_changed(changed);
    }

    // the new state
    st.applied.clear();
    std::string name, hashes;
    for (auto * p : set) {
        // the set does not own the packs: a non-owning shared_ptr keeps the list type simple
        st.applied.push_back(std::shared_ptr<llama_patch>(p, [](llama_patch *) {}));
        name   += (name.empty() ? "" : "+") + p->name;
        hashes += p->content_sha256 + "\n";
    }
    st.set_name = name;
    st.set_hash = set.empty() ? std::string() : llama_patch_sha256_hex(hashes.data(), hashes.size());
    std::fill(st.replaced.begin(), st.replaced.end(), 0);
    std::set<int32_t> layers;
    for (const auto & [k, s] : want) {
        if (s->il < 0 || !is_expert_tensor(s->target) || st.n_expert <= 0) {
            continue;
        }
        layers.insert(s->il);
        for (int32_t e = s->expert >= 0 ? s->expert : 0; e < (s->expert >= 0 ? s->expert + 1 : st.n_expert); e++) {
            st.replaced[(size_t) s->il * st.n_expert + e] = 1;
        }
    }
    st.layers.assign(layers.begin(), layers.end());
    for (const auto & [k, s] : have) {
        st.touched[k.target + "#" + std::to_string(k.expert)] = { s->dst, s->offset, s->size, s->base_sha256 };
    }
    for (const auto & [k, s] : want) {
        st.touched[k.target + "#" + std::to_string(k.expert)] = { s->dst, s->offset, s->size, s->base_sha256 };
    }
    st.gen++;
    locks.clear();
    LLAMA_LOG_INFO("%s: active cards: '%s' (%s): %zu slices written, %zu restored, %.1f MiB, %zu experts re-read in %zu contexts, %.1f ms\n",
        __func__, st.set_name.c_str(), st.set_hash.substr(0, 12).c_str(), to_write.size(), to_restore.size(), bytes/1048576.0,
        changed.size(), st.contexts.size(), (ggml_time_us() - t0)/1000.0);
    return true;
}

bool llama_model_patch_set(llama_model * model, llama_patch ** packs, int32_t n, char * err, size_t err_size) {
    std::string why;
    bool ok = false;
    try {
        ok = model && n >= 0 && (n == 0 || packs) && patch_set(*model, packs, n, why);
    } catch (const std::exception & ex) {
        why = ex.what();
    }
    if (!ok) {
        LLAMA_LOG_ERROR("%s: %s\n", __func__, why.c_str());
        set_err(err, err_size, why.empty() ? "bad arguments" : why);
    }
    return ok;
}

const char * llama_model_patch_set_hash(const llama_model * model) {
    return model && model->patch ? model->patch->set_hash.c_str() : "";
}

const char * llama_model_patch_set_name(const llama_model * model) {
    return model && model->patch ? model->patch->set_name.c_str() : "";
}

bool llama_model_patch_verify(llama_model * model, bool check_caches, char * err, size_t err_size) {
    if (!model || !model->patch) {
        return true;
    }
    llama_patch_state & st = *model->patch;
    std::lock_guard<std::mutex> lk(st.mu);
    // what each touched region must hold: the base bytes from the file with every active slice that falls into it laid over
    std::vector<const llama_patch_slice *> active;
    for (const auto & p : st.applied) {
        for (const auto & s : p->slices) {
            active.push_back(&s);
        }
    }
    file_reader fr(st.base_path);
    std::vector<uint8_t> want, have;
    for (const auto & [k, t] : st.touched) {
        const std::string target = k.substr(0, k.rfind('#'));
        want.resize(t.size);
        have.resize(t.size);
        if (!fr.read_at(st.base_data_offset + st.base_offsets.at(target) + t.offset, want.data(), t.size)) {
            set_err(err, err_size, k + ": cannot read the base file");
            return false;
        }
        if (llama_patch_sha256_hex(want.data(), t.size) != t.base_sha256) {
            set_err(err, err_size, k + ": the base file changed on disk");
            return false;
        }
        bool covered = false;
        for (const auto * s : active) {
            if (s->dst != t.dst) {
                continue;
            }
            const size_t lo = std::max(s->offset, t.offset), hi = std::min(s->offset + s->size, t.offset + t.size);
            if (lo < hi) {
                memcpy(want.data() + (lo - t.offset), s->data + (lo - s->offset), hi - lo);
                covered = true;
            }
        }
        ggml_backend_tensor_get(t.dst, have.data(), t.offset, t.size);
        if (memcmp(want.data(), have.data(), t.size) != 0) {
            set_err(err, err_size, k + ": the tensor does not hold " + (covered ? "the active cards' bytes" : "the base bytes"));
            return false;
        }
    }
    if (check_caches) {
        for (llama_context * c : st.contexts) {
            std::string why;
            if (c->get_moe_cache() != nullptr && !c->moe_cache_self_check(true, why)) {
                set_err(err, err_size, "expert cache: " + why);
                return false;
            }
        }
    }
    return true;
}

void llama_patch_register_context(llama_model & model, llama_context * ctx) {
    if (!model.patch) {
        model.patch = std::make_shared<llama_patch_state>();
    }
    std::lock_guard<std::mutex> lk(model.patch->mu);
    model.patch->contexts.push_back(ctx);
}

void llama_patch_unregister_context(llama_model & model, llama_context * ctx) {
    if (!model.patch) {
        return;
    }
    std::lock_guard<std::mutex> lk(model.patch->mu);
    auto & v = model.patch->contexts;
    v.erase(std::remove(v.begin(), v.end(), ctx), v.end());
}
