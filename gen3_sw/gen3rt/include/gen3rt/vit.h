// SmolVLM2's vision encoder (SigLIP: 12 layers, 768 wide, 12 heads, MLP 3072 with GELU(tanh), LayerNorm) and its
// projector (pixel shuffle 4 x 4 -> 12288, fc to the text width) on the Gen3 Node v3, the hybrid way of
// docs/gen3/node_v3.md s1.3:
//   CPU   LayerNorms, residual adds, the biases of the attention output and of fc2, pixel shuffle (all FP32)
//   Node  patch embedding (pixels split into hi / lo K segments, bias by an extra K group), q / k / v and fc1 (bias by
//         an extra K group: A [1, 1, 1, 0 ..], W [b_hi, b_mid, b_lo, 0 ..]), fc1's GELU, the K / V^T images (k and v
//         appended a tile = a 32-key band at a time), attention (K / V^T of the 1,024 patches in device memory,
//         streamed; three commands of four heads), o (three K segments chained by FP32 C), fc2, the projector (two
//         K segments chained by FP32 C); FP32 D for everything the CPU adds to
// Tiles of 32 patches (the 16K-word UB); per layer the phases run over all tiles: CPU LN1 -> Node q, k, v (K / V^T
// images) -> Node attention, o -> CPU residual, LN2 -> Node fc1, fc2 -> CPU residual. The tiles run in groups
// (G3RT_VIT_GROUPS, default 4): the CPU's pass over one group (residual, LayerNorm, A quantize and pack, in one go)
// overlaps the Node's phase on the others.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "gen3rt/device.h"
#include "gen3rt/llm.h"
#include "gen3rt/program.h"

namespace g3rt {

struct VitConfig {
    int n_embd = 768, n_ff = 3072, n_head = 12, dh = 64, n_layer = 12;
    int n_patch = 1024, patch_k = 768;      // patches of 3 x 16 x 16
    int shuffle = 4, proj_out = 576;        // pixel shuffle factor, projector output width
    float ln_eps = 1e-6f;
};

// FP32 host weights (rows = output features)
struct VitLayerWeights {
    const float *ln1_w, *ln1_b, *q_w, *q_b, *k_w, *k_b, *v_w, *v_b, *o_w, *o_b;
    const float *ln2_w, *ln2_b, *fc1_w, *fc1_b, *fc2_w, *fc2_b;
};
struct VitWeights {
    const float *patch_w, *patch_b, *pos;   // patch_w [n_embd][patch_k], pos [n_patch][n_embd]
    std::vector<VitLayerWeights> layers;
    const float *post_w, *post_b, *proj_w;  // proj_w [proj_out][n_embd * shuffle^2]
};

class Vit {
public:
    Vit(Device& dev, const VitConfig& cfg);
    void upload(const VitWeights& w);
    // patches [n_patch][patch_k] (im2col rows, normalised pixels) -> embeddings [n_patch / shuffle^2][proj_out]
    void encode(const float* patches, float* out);
    // Cache one exact normalized patch tensor per immutable configuration/weight upload.
    void set_encode_cache(bool enabled) { cache_enabled_ = enabled; cached_patches_.clear(); cached_output_.clear(); }
    uint64_t cache_hits() const { return cache_hits_; }
    uint64_t cache_misses() const { return cache_misses_; }
    const VitConfig& cfg() const { return cfg_; }
    double node_seconds() const { return node_s_; }
    // host time of encode by stage: 0 the passes between the Node's phases (residual adds, layer norms, A quantize
    // and pack), 1 the other A staging (patches, projector), 2 the K / V^T images (0: the Node appends them), 3
    // device memory made visible (D to the host, A to the Node)
    const double* host_seconds() const { return host_s_; }
    // profiles: every command reports, the Node's cycles are added to p per command role (board only)
    void set_profile(Profile* p) { profile_ = p; prog_.report_all = p != nullptr; }

private:
    struct Layer {
        WeightImage q, k, v, o[3], fc1, fc2;
        std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b, o_b, fc2_b;
    };
    Device& dev_;
    VitConfig cfg_;
    Program prog_;
    WeightImage patch_, proj_[2];
    std::vector<float> patch_b_, pos_, post_w_, post_b_;
    std::vector<Layer> layers_;
    uint64_t a_st_ = 0, q_st_ = 0, d_st_ = 0;                      // per-tile staging in device memory
    uint64_t kimg_ = 0, vimg_ = 0;                                  // K / V^T images of the layer, 12 heads
    uint32_t kv_words_ = 0;
    int stream_slot_ = 0;
    double node_s_ = 0, host_s_[4] = {};
    Profile* profile_ = nullptr;
    bool cache_enabled_ = true;
    uint64_t cache_hits_ = 0, cache_misses_ = 0;
    std::vector<float> cached_patches_, cached_output_;
    std::vector<g3::Word> d_copy_, a_copy_;      // D and A staging of a group (a device without a host mapping)

    uint16_t stream(const WeightImage& w);
    void run();
    // A rows (MXFP8, P8) of a tile from FP32 rows, with the bias group [1, 1, 1, 0 ..] when `bias`
    std::vector<g3::Word> stage_words(const float* x, int m, int k, bool bias, bool hilo) const;
    void stage_tiles(const float* x, int k, bool hilo, int t0, int nt);
    void host_pass(float* x, const float* add, const float* bias, const std::vector<float>& w,
                   const std::vector<float>& b, int t0, int nt, float* to);
};

}  // namespace g3rt
