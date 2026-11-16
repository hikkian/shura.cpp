#pragma once

// SHURA (PATCH1): expansion cards delivered as patches. A card (pack) is a GGUF file with slices of the base model's tensors in the
// base's own format (a whole tensor, or one expert of a stacked expert tensor). Applying a set of cards writes the slices over the base
// in memory, in every copy the engine keeps (host tensor, device tensor, the expert cache banks of every context of the model);
// removing it reads the original bytes back from the base file and checks them against the hash the card recorded. The base file
// on disk is never written. The graph and the tensor sizes stay the same. All checks run before the first byte is written.

#include "llama.h"

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct ggml_tensor;
struct ggml_context;
struct gguf_context;
struct llama_model;
struct llama_context;

// SHA-256 (lowercase hex) of a buffer: the card format records hashes of the replaced and the original bytes
std::string llama_patch_sha256_hex(const void * data, size_t len);

struct llama_patch_slice {
    std::string   target;          // base tensor name
    int32_t       expert = -1;     // expert index (the last dimension of a stacked expert tensor), -1: the whole tensor
    int32_t       il     = -1;     // block index of the target, -1 if not a block tensor
    ggml_tensor * dst    = nullptr;// the model's tensor
    size_t        offset = 0;      // bytes into the tensor
    size_t        size   = 0;      // bytes
    const uint8_t * data = nullptr;// the card's bytes (owned by the pack)
    std::string   sha256;          // of the card's bytes
    std::string   base_sha256;     // of the base bytes it replaces (checked at load and at removal)
};

struct llama_patch {
    std::string name;              // file name without the directory and ".gguf"
    std::string path;
    std::string content_sha256;    // over the slice table and the slice bytes, in order
    std::string manifest;          // JSON text of the manifest (data only)
    std::string layout_sha256;     // the base layout the card was built for
    const llama_model * model = nullptr;
    std::vector<llama_patch_slice> slices;
    gguf_context * gguf = nullptr;
    ggml_context * ctx  = nullptr;
    ~llama_patch();
};

// the base layout hash: names, types and shapes of all tensors of the model, in the order of the file
std::string llama_patch_layout_sha256(const llama_model & model);

// per-model state (lives in llama_model::patch)
struct llama_patch_state {
    std::mutex mu;
    std::vector<llama_context *> contexts;                   // every context created on this model (main, MTP draft, ...)
    std::vector<std::shared_ptr<llama_patch>> applied;       // the active set, in order
    std::string set_hash;                                    // "" when the base is active
    std::string set_name;                                    // names joined by '+'
    uint64_t    gen = 0;                                     // bumps at every set change (graphs are rebuilt)
    // replaced[il * n_expert + e]: expert e of block il is replaced by the active set (whole-tensor targets mark all experts)
    std::vector<uint8_t> replaced;
    int32_t n_expert = 0;
    std::vector<int32_t> layers;                             // blocks with replaced experts
    bool layer_replaced(int32_t il) const { return std::binary_search(layers.begin(), layers.end(), il); }
    // every slice any card of this model ever replaced: where it lives and the base bytes' hash (verify checks the base came back)
    struct touched_slice { ggml_tensor * dst; size_t offset; size_t size; std::string base_sha256; };
    std::map<std::string, touched_slice> touched;            // "target#expert"
    // the base file (for removal): path, layout hash, data offset and tensor offsets
    std::string base_path;
    std::string base_layout;
    size_t base_data_offset = 0;
    std::map<std::string, size_t> base_offsets;
};

void llama_patch_register_context  (llama_model & model, llama_context * ctx);
void llama_patch_unregister_context(llama_model & model, llama_context * ctx);

// exposure: how often each replaced expert was selected, per sequence; filled from the router's top-k outputs
struct llama_patch_exposure_state {
    std::map<int32_t, std::vector<uint64_t>> by_seq;         // seq -> counts [n_layer * n_expert]
    struct pending {
        std::vector<int32_t> ids;                            // [n_tokens * n_used]
        std::vector<int32_t> seq;                            // seq of each token
        int32_t il = 0, n_used = 0, stride = 0;
    };
    std::vector<pending> queued;                             // read asynchronously, counted at the next synchronize
};
