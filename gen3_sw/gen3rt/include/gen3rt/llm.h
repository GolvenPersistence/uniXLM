// SmolVLM2 (llama architecture) text model on the Gen3 Node v3: weights converted to MXFP8 P32 images in device
// memory, the KV cache in device memory (K rows and V^T cells per layer and KV head), and the Node program of a
// decode step (docs/gen3/node_v3.md s6.1):
//
//   per layer   MOVE_IN the K band and V^T cell of position t (one per KV head) -> rebind them as APPEND at row t % 32
//               q = RoPE(XG W_q) (9 heads)   k -> K tail, v -> V^T tail (fold consumers: XG with 1/rms at XG + 1)
//               MOVE_OUT the tails, FENCE; ATTN over t + 1 keys streamed from the cache (one command, 3 KV heads)
//               X2 = O W_o + X (FP32 rows), X2G = MXFP8(X2 g1), 1/rms (fold producer)
//               H = SiLU(X2G W_g) (X2G W_u)  (fold consumer)
//               X3 = H W_d + X2, X3G = MXFP8(X3 g0'), 1/rms (producer for the next layer; after the last layer
//               g0' = the output norm)
//   lm_head     fold consumer with an FP32 D, 1,024 columns per GEMM -> logits in device memory
// Layer 0's inputs (x, x g0, 1/rms) are formed on the host as a fold producer would (fold_producer_v3).
//
// Models beyond one command per projection (other Llama / Qwen2 shapes) keep this program and split it by the
// command limits (an ATTN takes at most 4 KV heads and a group of at most 4; a GEMM at most 4,064 columns):
//   attention   the KV heads in runs of up to 4, each group in rounds of up to 4: one q GEMM and one ATTN per
//               (run, round); W_q's rows are stored in that order, and the o projection is a chain of K slices
//               (one per ATTN, W_o's columns stored slice after slice) joined by FP32 C
//   gate / up   parts of whole 32-row bands (contiguous in the interleaved image); down is a chain of K slices
//   RoPE        NEOX pairs (i, i + dh / 2) become the Node's adjacent pairs by permuting W_q / W_k rows within each
//               head (q . k and the cache are unchanged by a common permutation)
// A model that needs none of this gets the same images and command stream as before.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "gen3rt/device.h"
#include "gen3rt/profile.h"
#include "gen3rt/program.h"

namespace g3rt {

struct LlmConfig {
    int n_embd = 576, n_ff = 1536, n_head = 9, n_kv = 3, dh = 64, n_layer = 30, vocab = 49280;
    float rope_base = 100000.0f;
    bool rope_neox = false;         // RoPE pairs (i, i + dh / 2) (llama.cpp NEOX, Qwen2); else adjacent pairs
    int eps_profile = 2;            // 1e-5
    int ctx = 2048;                 // KV capacity (multiple of 32, at most 2048)
    int lm_chunk = 1024;            // lm_head columns per GEMM (multiple of 64)
    // the command limits the program splits by (the Node's by default; smaller values exercise the splitting on
    // small models): q heads per KV head and KV heads per ATTN, gate / up columns per GEMM (a multiple of 32)
    int max_group = 4, max_kv_heads = 4, max_ffn_columns = 4064;
    // Qwen3: each head's q and k RMS-normalized (gain [dh], epsilon qk_eps) before RoPE. The Node has no such step:
    // q / k / v of a layer come out in FP32, the host normalizes, rotates and quantizes q and k (qk_host).
    bool qk_norm = false;
    float qk_eps = 1e-6f;
};

// FP32 host weights, rows = output features (ggml ne1), columns = input features (ggml ne0)
struct LayerWeights {
    const float *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr;
    const float *wg = nullptr, *wu = nullptr, *wd = nullptr;
    const float *g_attn = nullptr, *g_ffn = nullptr;
    const float *g_q = nullptr, *g_k = nullptr;     // q / k norm gains [dh] (qk_norm)
};

struct WeightImage {
    uint64_t addr = 0;
    uint32_t words = 0;
};

struct HostTiming {
    double input_ms = 0, build_ms = 0, submit_ms = 0, sync_ms = 0, read_ms = 0;
    uint64_t tiles = 0, commands = 0;
};

class Llm {
public:
    Llm(Device& dev, const LlmConfig& cfg);
    void upload_layer(int l, const LayerWeights& w);
    void upload_output(const float* g_out, const float* lm_head);
    // one tensor at a time (ggml-gen3: as llama.cpp loads them). role: attn_q, attn_k, attn_v, attn_output, ffn_gate,
    // ffn_up, ffn_down (rows x cols FP32), attn_norm, ffn_norm, attn_q_norm, attn_k_norm (cols); layer -1:
    // output_norm, output (lm_head).
    // Returns false for a role it does not know. gate / up are combined when both are in.
    bool upload_tensor(int layer, const std::string& role, const float* w, int rows, int cols);
    void save_weights(const std::string& path, const std::string& model_sha256);
    void load_weights(const std::string& path, const std::string& model_sha256);
    bool complete() const;              // every weight is in
    // an image converted earlier (upload_matrix_image / upload_f32_row_image), by the roles of upload_tensor. A
    // model whose program reorders W_q / W_k / W_o / W_d gets that image rewritten in place: it then belongs to
    // this Llm.
    bool attach(int layer, const std::string& role, const WeightImage& img);
    // logits of the rows `rows` of the last tile of a prefill (rows of the tile's m; empty: the last row)
    void prefill_rows(const float* x, int n, int pos, const std::vector<int>& out_rows, float* logits);
    // one decode step: x = the token's embedding [n_embd] at position pos; logits [vocab]
    void decode(const float* x, int pos, float* logits);
    // n tokens at positions pos.. in tiles of up to 32 that end at band ends (x: n rows of embeddings); the last
    // token's logits, if asked
    void prefill(const float* x, int n, int pos, float* logits);
    // one tile of m <= 32 tokens inside one 32-token band (m == 1 or a tile that starts a band: the KV append on the
    // Node; else merged by the host after each layer's q/k/v)
    void forward(const float* x, int m, int pos, float* logits);
    const LlmConfig& cfg() const { return cfg_; }
    // profiles: every command reports, the Node's cycles are added to p per command role (board only)
    void set_profile(Profile* p) { profile_ = p; prog_.report_all = p != nullptr; }
    // MXFP8 P32 image of an FP32 matrix [rows][cols] in device memory
    WeightImage upload_matrix(const float* w, int rows, int cols);
    WeightImage upload_f32_row(const float* v, int n);
    void set_host_timing(bool enabled) { host_timing_ = enabled; }
    void reset_host_timing() { timing_ = {}; }
    const HostTiming& host_timing() const { return timing_; }
    size_t device_bytes() const { return dev_bytes_; }
    int tile() const { return tile_; }          // tokens per prefill tile the UB map holds

private:
    struct LayerImages {
        WeightImage wq, wk, wv, wo, wgu, wd, g_attn, g_ffn, wg, wu;
        WeightImage g_q, g_k;           // q / k norm gains (qk_norm)
        WeightImage g_qa, g_ka;         // the gains in W_q / W_k row order (qk_norm on the Node; qk_gains)
        uint64_t kimg = 0, vimg = 0;     // KV head 0's K / V^T image; head h at + h * kv_words * 32
    };
    // Per-model storage; preserve the accepted 576-wide layout verbatim.
    struct MemoryMap {
        uint32_t XGA=0, XGB=600, RA=1200, RB=1208, GA=1216, GB=1288, STREAMB=1360;
        uint32_t XA=1424, XB=3728, QB=6032, OB=6632, TB=7232, HB=8032;
        uint32_t KTB=9616, VTB=9872, UB_END=10640, LOGA=1424, LOGB=5520;
        uint32_t XG_ST=2304, R_ST=2904, T_ST=2912, KR_ST=3712, VR_ST=3912, ST_END=4680;
        uint8_t I_KT=0x30, I_VT=0x34, I_KA=0x38, I_VA=0x3C;
        // qk_norm: the layer's start binds one region in turn (a BIND waits for the readers before it, R5): FP32 q of
        // each attention part (QF), FP32 k (KF), the V^T cells (VTB), the K bands (KTB) with the host's k after them
        // (KN). In staging from QF_ST the m rows of each part's q, then of k (QF_ST .. KF_ST + its extent: room for a
        // tile), MXFP8 q and k at Q8_ST, K8_ST.
        uint32_t QF=0, KF=0, KN=0, QF_ST=0, KF_ST=0, Q8_ST=0, K8_ST=0;
        // the FFN parts one after another (gate / up of a part, then its link of the down chain), sharing HB
        bool ffn_one_h = false;
        // the K bands and V^T cells (K rows and V rows: no hw_kv) in one region, k and v in turn
        bool kv_one = false;
        // qk_norm on the Node: FP32 RoPE tables of q and k (the tile's rows, from T32_ST) and their gain rows
        uint32_t TQB=0, GQB=0, TKB=0, GKB=0, T32_ST=0;
    } map_;
    // One q GEMM + ATTN per part: KV heads kv0 .. kv0 + kvs - 1, q heads kv * group + r0 .. + grp - 1 of each
    // (KV-head-major, the ATTN's order). ub: its Q / O at QB / OB + ub; wq_off: its rows in W_q (words); W_o's K
    // slice at wo_off (words).
    struct AttnPart {
        int kv0 = 0, kvs = 0, r0 = 0, grp = 0;
        uint32_t ub = 0, wq_off = 0, wo_off = 0, wo_words = 0;
    };
    // gate / up bands band0 .. band0 + bands - 1; its H at HB + ub; W_d's K slice at wd_off (words)
    struct FfnPart { int band0 = 0, bands = 0; uint32_t ub = 0, wd_off = 0, wd_words = 0; };
    std::vector<AttnPart> attn_;
    std::vector<FfnPart> ffn_;
    int x_region_ = 0;                  // the F32 region of the current X: 0 XA, 1 XB (a K chain alternates them)
    void plan();
    void configure_memory();
    // W_q / W_k / W_o / W_d as the program reads them (rows in attention-part and RoPE order, K slices);
    // arranged(role): not the plain P32 image
    bool arranged(const std::string& role) const;
    std::vector<g3::Word> arrange(const std::string& role, const uint8_t* codes, const uint8_t* scales, int rows,
                                  int cols) const;
    WeightImage upload_weight(const std::string& role, const float* w, int rows, int cols);
    void rearrange(const std::string& role, const WeightImage& img);
    std::string weight_layout() const;  // the weight cache's tag of arranged images ("" when none is)
    Device& dev_;
    LlmConfig cfg_;
    Program prog_;
    std::vector<LayerImages> layers_;
    WeightImage g_out_, lm_;
    WeightImage ident_;                 // qk_norm: the KVD x KVD identity of the GEMM that appends the host's k
    uint32_t kv_words_ = 0;             // one KV head's K image (= its V^T image) in words
    uint64_t stage_ = 0, logits_ = 0;   // per-step inputs, logits
    uint64_t zero_ = 0;                 // a zero K band / V^T cell
    uint64_t logits_rows_ = 0;          // logits of several rows (prefill_rows)
    size_t logits_rows_cap_ = 0;
    size_t dev_bytes_ = 0;
    int stream_slot_ = 0;
    int prefill_tile_ = 32;
    int tile_ = 32;                     // tokens of the UB map (configure_memory): 32, or 16 / 8 for a wide model
    Profile* profile_ = nullptr;
    bool host_timing_ = false;
    HostTiming timing_;
    // Per-instance scratch: one Llm is driven by one request at a time.
    struct InputScratch {
        std::vector<uint32_t> xb, g, xg, xd, r, tt;
        std::vector<uint8_t> c, s, xc, xs, tc, ts;
        std::vector<g3::Word> gw, wx, wxg, wr;
        std::vector<float> tr;
    } input_;
    std::vector<uint32_t> norm0_;
    // qk_norm: per layer the q gains then the k gains [2 dh] in W_q / W_k row order; per position the cos, sin of
    // the dh / 2 rotated pairs
    std::vector<std::vector<float>> qk_gain_;
    std::vector<std::vector<float>> qk_rope_;
    // qk_norm on the Node (R56 and later, the default; G3RT_QK_NODE=0 the host's): q and k normalized per head by
    // their GEMMs, no host step
    bool qk_node_ = false;
    // Config is immutable; each key identifies exactly (position, tile length).
    std::vector<std::vector<g3::Word>> rope_cache_;

    int tile_len(int pos, int left) const;
    uint8_t next_weight_stream();
    void host_inputs(const float* x, int m, int pos);
    void build_inputs(int m);
    void build_qkv(int l, int m, int pos, bool hw_kv);
    void merge_kv(int l, int m, int pos);
    void insert_kv(int l, int m, int pos, const uint8_t* kc, const uint8_t* ks, const uint32_t* v);
    std::vector<uint8_t> build_qkv_f32(int l, int m, bool hw_kv);
    void qk_host(int l, int m, int pos, bool hw_kv);
    void qk_rope_rows(int pos, int m);
    void qk_gains(int l);
    void build_v_append(int l, int m, int pos);
    void build_k_append(int l, int m, int pos);
    void build_q_in(int m);
    void build_rest(int l, int m, int pos);
    void build_lm_head(int m, const std::vector<std::pair<int, uint64_t>>& rows);
    int lm_cols(int m) const;
    WeightImage interleave_bands(const WeightImage& a, const WeightImage& b, int rows, int cols);
    std::vector<WeightImage*> weight_slots();
    std::vector<std::string> weight_roles() const;
    std::vector<uint32_t> weight_sizes() const;
    void run(bool final_sync);
    void run_until(const std::vector<uint8_t>& tags);

    // R55: a streamed weight in quarters (by default; G3RT_WEIGHT_QUARTERS=0 for R54 and older images). A role's
    // segments are what its GEMM commands read (segments()); band b of a segment sits at quarter b % 4, position b / 4,
    // the quarters `stride` 16 KiB units apart (odd: the four streams of a two-window GEMM in four DDR banks). An image
    // not placed so (attached, or the setting off) has its bands in order, as the weight cache keeps every image.
    struct Segment { int bands = 0; uint32_t band_words = 0; uint64_t addr = 0; uint32_t stride = 0; };
    std::map<uint64_t, std::vector<Segment>> placed_;   // image address -> its segments, in quarters
    // segments() of the streamed roles, set once the program is planned; addr: the segment's first word in an image
    // with its bands in order
    std::map<std::string, std::vector<Segment>> shapes_;
    struct Span { uint64_t addr = 0; uint32_t words = 0, quarters = 0; };   // a GEMM's B as its binding takes it
    std::vector<Segment> segments(const std::string& role) const;
    std::vector<Segment> quarter_layout(const std::string& role, size_t& bytes) const;
    void write_quarters(const Segment& s, const g3::Word* bands);
    WeightImage place(const std::string& role, const std::vector<g3::Word>& image);
    WeightImage upload_plain(const std::string& role, const float* w, int rows, int cols);
    Span span(const WeightImage& w, const std::string& role, int seg, int band0, int bands) const;
    uint16_t stream(const Span& s);
};

// MXFP8 P32 image of an FP32 matrix [rows][cols] in device memory; an FP32 row
WeightImage upload_matrix_image(Device& dev, const float* w, int rows, int cols);
WeightImage upload_f32_row_image(Device& dev, const float* v, int n);

WeightImage upload_gate_up_image(Device& dev, const float* gate, const float* up, int rows, int cols);

// RoPE table of one position (llama.cpp NORM mode, ggml rope_cache_init with ext_factor 0): {cos, sin} of the
// pairs, interleaved, n_dims values
void rope_row(int pos, int n_dims, float base, float* out);

}  // namespace g3rt
