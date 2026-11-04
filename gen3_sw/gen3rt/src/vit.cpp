#include "gen3rt/vit.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "g3emu_kernels.h"
#include "gen3rt/layout.h"
#include "gen3rt/parallel.h"
#include "gen3rt/quantize_fast.h"

namespace g3rt {
namespace {

uint32_t fbits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

// adds the scope's wall time to `acc`
struct ScopeTimer {
    double& acc;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    ~ScopeTimer() { acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
};
constexpr int TILE = 32;
// UB map of a tile (words, of 16384); the phases rebind the regions:
//   A        A operand: the projector's half (6,336), the patch embedding's (1,624), q / k / v and fc1 (832)
//   Q        q of the tile, 3 groups of 4 heads (3 x 264)
//   KA / O   the tile's K bands of the 12 heads (12 x 72) in the q / k / v phase, the attention output (3 x 264) in
//            the attention phase
//   VA / H   the tile's V^T cells of the 12 heads (12 x 72) in the q / k / v phase; fc1's output (3,168, from Q) in
//            the MLP phase
//   D_A, D_B FP32 results (3,072; the projector's 2,304)
//   A1, H1   the second A and H of double buffering (vit_buffers): A1 in the q / k / v and MLP phases, A1P in the
//            patch embedding's (which alternates D_A and D_B as well)
constexpr uint32_t STREAM_B = 0, A_B = 64, Q_B = 1688, KA_B = 2480, O_B = 2480, VA_B = 3344, H_B = 1688;
constexpr uint32_t D_A = 6400, D_B = 9472, UB_END = 12544;
constexpr uint32_t A1_B = 4864, A1P_B = 1688, H1_B = 5696;
// staging per tile (words): A, q (3 groups), D
constexpr uint64_t A_ST = 1624, Q_ST = 792, D_ST = 3072;
constexpr uint8_t I_A = 0x50, I_A1 = 0x51, I_Q = 0x54, I_O = 0x58, I_DA = 0x5E, I_DB = 0x5F, I_H = 0x60, I_H1 = 0x61;
constexpr uint8_t I_KA = 0x70, I_VA = 0x80;     // arrays of 12 (a head each)
constexpr uint8_t I_KS = 0xF9, I_VS = 0xFA;
// The Node evicts the binding that its owner record names (index, generation) when a BIND claims the granules. This
// encoder and the language model (Llm) keep their own generations and share indices (0x61: H1 here, O of attention
// part 1 there), so a granule left owned by one of them can name a live binding of the other with an equal
// generation, which the next BIND over the granule evicts (SmolVLM2-500M: Llm's O, a GEMM's BINDING). Each encode
// claims all it uses, [STREAM_B, UB_END) (the language model's streams are elsewhere), under an index neither uses
// before it starts and after its last command: the granules it takes over and leaves behind are owned by I_UB only.
constexpr uint8_t I_UB = 0xE0;
const uint8_t WEIGHT_STREAMS[4] = {0xF8, 0xFB, 0xFC, 0xFD};

// the three MXFP8 terms of a bias value at the scale of its largest
void bias_terms(float b, uint8_t& scale, uint8_t c[3]) {
    const uint32_t bb = fbits(b);
    scale = g3::choose_scale(bb & 0x7FFFFFFFu);
    uint32_t r = bb;
    for (int i = 0; i < 3; ++i) {
        c[i] = g3::quantize_e4m3(r, scale);
        r = g3::f32_sub(r, g3::decode_e4m3(c[i], scale));
    }
}

// MXFP8 P32 image of W [rows][cols] (cols % 32 == 0), optionally twice (pixels hi / lo) and with a bias group
WeightImage weight_image(Device& dev, const float* w, int rows, int cols, const float* bias, bool twice) {
    std::vector<uint32_t> bits(static_cast<size_t>(rows) * cols);
    std::memcpy(bits.data(), w, bits.size() * 4);
    std::vector<uint8_t> c0, s0;
    quantize(bits.data(), rows, cols, c0, s0);
    const int g0 = cols / 32;
    const int K = cols * (twice ? 2 : 1) + (bias ? 32 : 0), G = K / 32;
    std::vector<uint8_t> codes(static_cast<size_t>(rows) * K, 0), scales(static_cast<size_t>(rows) * G, 127);
    for (int r = 0; r < rows; ++r) {
        for (int t = 0; t < (twice ? 2 : 1); ++t) {
            std::memcpy(&codes[size_t(r) * K + size_t(t) * cols], &c0[size_t(r) * cols], size_t(cols));
            std::memcpy(&scales[size_t(r) * G + size_t(t) * g0], &s0[size_t(r) * g0], size_t(g0));
        }
        if (bias) {
            uint8_t sc, c[3];
            bias_terms(bias[r], sc, c);
            const size_t k0 = size_t(r) * K + size_t(K - 32);
            codes[k0] = c[0]; codes[k0 + 1] = c[1]; codes[k0 + 2] = c[2];
            scales[size_t(r) * G + size_t(G - 1)] = sc;
        }
    }
    const std::vector<Word> words = Layout(rows, K, true).pack(codes.data(), scales.data());
    WeightImage img;
    img.words = uint32_t(words.size());
    img.addr = dev.alloc(words.size() * 32);
    dev.write(img.addr, words.data(), words.size() * 32);
    return img;
}

// columns [c0, c0 + n) of W [rows][cols]
std::vector<float> columns(const float* w, int rows, int cols, int c0, int n) {
    std::vector<float> out(static_cast<size_t>(rows) * n);
    for (int r = 0; r < rows; ++r) std::memcpy(&out[size_t(r) * n], &w[size_t(r) * cols + c0], size_t(n) * 4);
    return out;
}

// ggml_norm of four rows (sums in double, in index order) then * w + b. The four rows' sums are interleaved, so that
// a row's additions, which keep their order, do not wait for each other.
void ln4(float* const x[4], float* const y[4], int E, const float* w, const float* b, float eps) {
    double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    for (int i = 0; i < E; ++i) {
        s0 += double(x[0][i]); s1 += double(x[1][i]); s2 += double(x[2][i]); s3 += double(x[3][i]);
    }
    const float m0 = float(s0 / E), m1 = float(s1 / E), m2 = float(s2 / E), m3 = float(s3 / E);
    double q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    for (int i = 0; i < E; ++i) {
        const float v0 = x[0][i] - m0, v1 = x[1][i] - m1, v2 = x[2][i] - m2, v3 = x[3][i] - m3;
        y[0][i] = v0; y[1][i] = v1; y[2][i] = v2; y[3][i] = v3;
        q0 += double(v0 * v0); q1 += double(v1 * v1); q2 += double(v2 * v2); q3 += double(v3 * v3);
    }
    const double q[4] = {q0, q1, q2, q3};
    for (int k = 0; k < 4; ++k) {
        const float variance = float(q[k] / E);
        const float scale = 1.0f / std::sqrt(variance + eps);
        float* yk = y[k];
        for (int i = 0; i < E; ++i) yk[i] = yk[i] * scale * w[i] + b[i];
    }
}

// A (and H) buffers of the tile phases (G3RT_VIT_BUFFERS, 1 or 2; default 2): with two, a tile's A comes in while the
// tile before is still computing on the other
int vit_buffers() {
    static const int n = [] {
        const char* v = std::getenv("G3RT_VIT_BUFFERS");
        const int b = v ? std::atoi(v) : 2;
        if (b != 1 && b != 2) throw std::invalid_argument("G3RT_VIT_BUFFERS: 1 or 2");
        return b;
    }();
    return n;
}

// groups of tiles in the pipeline (G3RT_VIT_GROUPS, a divisor of the tiles; default 4)
int vit_groups(int tiles) {
    static const int groups = [] {
        const char* v = std::getenv("G3RT_VIT_GROUPS");
        return v ? std::atoi(v) : 4;
    }();
    if (groups < 1 || tiles % groups) throw std::invalid_argument("G3RT_VIT_GROUPS: a divisor of the tiles");
    return groups;
}

}  // namespace

Vit::Vit(Device& dev, const VitConfig& cfg) : dev_(dev), cfg_(cfg) {
    static_assert(UB_END <= 16384, "UB map");
    const int NP = cfg_.n_patch, NT = NP / TILE;
    if (NP % TILE || cfg_.n_embd != 768 || cfg_.dh != 64 || cfg_.n_head != 12 || cfg_.n_ff != 3072)
        throw std::invalid_argument("vision encoder: the UB map is laid out for SigLIP-B (768, 12 x 64, 3072)");
    kv_words_ = uint32_t(Layout(NP, cfg_.dh, true).words());
    kimg_ = dev_.alloc(size_t(cfg_.n_head) * kv_words_ * 32);
    vimg_ = dev_.alloc(size_t(cfg_.n_head) * kv_words_ * 32);
    // the projector's two A halves use the A staging of tiles 0..7
    a_st_ = dev_.alloc(size_t(std::max<uint64_t>(uint64_t(NT) * A_ST, 2 * 6336)) * 32);
    q_st_ = dev_.alloc(size_t(NT) * Q_ST * 32);
    d_st_ = dev_.alloc(size_t(NT) * D_ST * 32);
}

void Vit::upload(const VitWeights& w) {
    cached_patches_.clear(); cached_output_.clear();
    const int E = cfg_.n_embd, F = cfg_.n_ff, NH = cfg_.n_head, dh = cfg_.dh;
    patch_ = weight_image(dev_, w.patch_w, E, cfg_.patch_k, w.patch_b, true);
    pos_.assign(w.pos, w.pos + size_t(cfg_.n_patch) * E);
    post_w_.assign(w.post_w, w.post_w + E);
    post_b_.assign(w.post_b, w.post_b + E);
    const int PK = E * cfg_.shuffle * cfg_.shuffle;
    const std::vector<float> p0 = columns(w.proj_w, cfg_.proj_out, PK, 0, PK / 2);
    const std::vector<float> p1 = columns(w.proj_w, cfg_.proj_out, PK, PK / 2, PK / 2);
    proj_[0] = weight_image(dev_, p0.data(), cfg_.proj_out, PK / 2, nullptr, false);
    proj_[1] = weight_image(dev_, p1.data(), cfg_.proj_out, PK / 2, nullptr, false);
    layers_.resize(w.layers.size());
    const int HG = 4 * dh;                           // a group of four heads
    for (size_t l = 0; l < w.layers.size(); ++l) {
        const VitLayerWeights& s = w.layers[l];
        Layer& L = layers_[l];
        L.q = weight_image(dev_, s.q_w, E, E, s.q_b, false);
        L.k = weight_image(dev_, s.k_w, E, E, s.k_b, false);
        L.v = weight_image(dev_, s.v_w, E, E, s.v_b, false);
        for (int g = 0; g < NH / 4; ++g) {
            const std::vector<float> og = columns(s.o_w, E, E, g * HG, HG);
            L.o[g] = weight_image(dev_, og.data(), E, HG, nullptr, false);
        }
        L.fc1 = weight_image(dev_, s.fc1_w, F, E, s.fc1_b, false);
        L.fc2 = weight_image(dev_, s.fc2_w, E, F, nullptr, false);
        L.ln1_w.assign(s.ln1_w, s.ln1_w + E); L.ln1_b.assign(s.ln1_b, s.ln1_b + E);
        L.ln2_w.assign(s.ln2_w, s.ln2_w + E); L.ln2_b.assign(s.ln2_b, s.ln2_b + E);
        L.o_b.assign(s.o_b, s.o_b + E); L.fc2_b.assign(s.fc2_b, s.fc2_b + E);
    }
    patch_b_.assign(w.patch_b, w.patch_b + E);
}

uint16_t Vit::stream(const WeightImage& w) {
    const uint8_t s = WEIGHT_STREAMS[stream_slot_];
    stream_slot_ = (stream_slot_ + 1) % 4;
    BindArgs a{s, 0, uint32_t(STREAM_B + 8 * (s - 0xF8)), 8, Order::P32, Kind::Stream};
    a.src = w.addr; a.length = w.words;
    return prog_.bind(a).b();
}

void Vit::run() {
    const uint32_t t0 = profile_ ? dev_.cycles() : 0;
    const auto w0 = std::chrono::steady_clock::now();      // node_s_: submit to completion, the host waiting
    dev_.submit(prog_.cmds);
    dev_.sync();
    node_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    const std::vector<g3::Completion> cpl = dev_.take_completions();
    if (profile_) profile_->add(prog_.cmds, prog_.roles, t0, cpl);
    prog_.clear();
    for (const g3::Completion& c : cpl)
        if (c.status != g3::ST_OK)
            throw std::runtime_error(std::string("Node command failed (vision): tag ") + std::to_string(c.tag) + " " +
                                     g3::status_name(c.status));
}

// The host's part between two Node phases for tiles t0 .. t0 + nt, in one pass over each tile (a host thread item),
// four rows at a time: x = D + add (the positions) or x += D + bias, in the order of the sums before; the layer norm
// (w, b) of the rows (ln4); then either their MXFP8 codes and scales as the A of the tile (P8, with the bias group
// [1, 1, 1, 0 ..], scale 2^0) in its A staging, or (to) the normalized rows. D and the A staging are used in place
// where the device has a host mapping (the board), else through copies.
void Vit::host_pass(float* x, const float* add, const float* bias, const std::vector<float>& w,
                    const std::vector<float>& b, int t0, int nt, float* to) {
    const int E = cfg_.n_embd, K = E + 32, G = K / 32;
    const size_t dbytes = size_t(nt) * D_ST * 32, abytes = size_t(nt) * A_ST * 32;
    const uint64_t daddr = d_st_ + uint64_t(t0) * D_ST * 32, aaddr = a_st_ + uint64_t(t0) * A_ST * 32;
    const Word* d = nullptr;
    Word* a = nullptr;
    {
        ScopeTimer tm{host_s_[3]};
        d = static_cast<const Word*>(dev_.read_in_place(daddr, dbytes));
        if (!d) {
            d_copy_.resize(size_t(nt) * D_ST);
            dev_.read(daddr, d_copy_.data(), dbytes);
            d = d_copy_.data();
        }
    }
    bool copy = false;
    if (!to) {
        a = static_cast<Word*>(dev_.write_in_place(aaddr, abytes));
        if (!a) { a_copy_.assign(size_t(nt) * A_ST, Word{}); a = a_copy_.data(); copy = true; }
    }
    const Layout al(TILE, K);
    const int aw = int(round8(uint32_t(al.words())));
    {
        ScopeTimer tm{host_s_[0]};
        parallel_for(nt, [&](int i) {
            const size_t row0 = (size_t(t0) + size_t(i)) * TILE;
            std::vector<uint8_t> codes(to ? 0 : size_t(TILE) * K, 0), scales(to ? 0 : size_t(TILE) * G, 127), c, s;
            std::vector<float> dr(size_t(4) * E), yr4(size_t(4) * E);
            std::vector<uint32_t> bits(size_t(4) * E);
            for (int r0 = 0; r0 < TILE; r0 += 4) {
                f32_unpack(d + size_t(i) * D_ST + size_t(r0) * size_t(E / 8), 4, E, reinterpret_cast<uint32_t*>(dr.data()));
                float* xr[4];
                float* yr[4];
                for (int k = 0; k < 4; ++k) {
                    const size_t row = row0 + size_t(r0 + k);
                    float* xk = x + row * E;
                    const float* dk = &dr[size_t(k) * E];
                    if (bias) for (int j = 0; j < E; ++j) xk[j] += dk[j] + bias[j];
                    else for (int j = 0; j < E; ++j) xk[j] = dk[j] + add[row * E + size_t(j)];
                    xr[k] = xk;
                    yr[k] = to ? to + row * E : &yr4[size_t(k) * E];
                }
                ln4(xr, yr, E, w.data(), b.data(), cfg_.ln_eps);
                if (to) continue;
                std::memcpy(bits.data(), yr4.data(), bits.size() * 4);
                quantize_fast(bits.data(), 4, E, c, s);
                for (int k = 0; k < 4; ++k) {
                    uint8_t* ck = &codes[size_t(r0 + k) * K];
                    std::memcpy(ck, &c[size_t(k) * E], size_t(E));
                    std::memcpy(&scales[size_t(r0 + k) * G], &s[size_t(k) * (E / 32)], size_t(E / 32));
                    ck[E] = ck[E + 1] = ck[E + 2] = 0x38;          // 1.0
                }
            }
            if (to) return;
            Word* at = a + size_t(i) * A_ST;
            al.pack_into(codes.data(), scales.data(), at);
            std::memset(static_cast<void*>(at + al.words()), 0, sizeof(Word) * size_t(aw - al.words()));
        });
    }
    if (!to) {
        ScopeTimer tm{host_s_[3]};
        if (copy) dev_.write(aaddr, a, abytes);
        else dev_.publish(aaddr, abytes);
    }
}

// m rows of FP32 x [m][k] as the A operand (P8) of an m-row tile (the leading panels of a TILE-row image; the
// projector's tile is 16 rows when its output does not fit UB at 32): codes; with `hilo` the residual
// x - decode(codes) as a second K segment; with `bias` the group [1, 1, 1, 0 ..] (scale 2^0)
std::vector<Word> Vit::stage_words(const float* x, int m, int k, bool bias, bool hilo) const {
    std::vector<uint32_t> bits(static_cast<size_t>(m) * k);
    std::memcpy(bits.data(), x, bits.size() * 4);
    std::vector<uint8_t> c0, s0, c1, s1;
    quantize_fast(bits.data(), m, k, c0, s0);
    const int g0 = k / 32;
    if (hilo) {
        std::vector<uint32_t> lo(bits.size()), d(bits.size());
        dequantize(c0.data(), s0.data(), m, k, d.data());
        for (size_t i = 0; i < bits.size(); ++i) lo[i] = g3::f32_sub(bits[i], d[i]);
        quantize_fast(lo.data(), m, k, c1, s1);
    }
    const int K = k * (hilo ? 2 : 1) + (bias ? 32 : 0), G = K / 32;
    std::vector<uint8_t> codes(static_cast<size_t>(TILE) * K, 0), scales(static_cast<size_t>(TILE) * G, 127);
    for (int r = 0; r < m; ++r) {
        std::memcpy(&codes[size_t(r) * K], &c0[size_t(r) * k], size_t(k));
        std::memcpy(&scales[size_t(r) * G], &s0[size_t(r) * g0], size_t(g0));
        if (hilo) {
            std::memcpy(&codes[size_t(r) * K + size_t(k)], &c1[size_t(r) * k], size_t(k));
            std::memcpy(&scales[size_t(r) * G + size_t(g0)], &s1[size_t(r) * g0], size_t(g0));
        }
        if (bias) {
            const size_t k0 = size_t(r) * K + size_t(K - 32);
            codes[k0] = codes[k0 + 1] = codes[k0 + 2] = 0x38;      // 1.0
        }
    }
    return Layout(m, K).pack(codes.data(), scales.data());
}

// the A of tiles t0 .. t0 + nt (rows TILE * t .. of x) with the bias group, in parallel, into their A staging (in
// place, or one transfer)
void Vit::stage_tiles(const float* x, int k, bool hilo, int t0, int nt) {
    const uint64_t addr = a_st_ + uint64_t(t0) * A_ST * 32;
    const size_t bytes = size_t(nt) * A_ST * 32;
    Word* a = static_cast<Word*>(dev_.write_in_place(addr, bytes));
    std::vector<Word> copy;
    if (!a) { copy.resize(size_t(nt) * A_ST); a = copy.data(); }
    parallel_for(nt, [&](int i) {
        const std::vector<Word> w = stage_words(x + size_t(t0 + i) * TILE * k, TILE, k, true, hilo);
        if (w.size() > A_ST) throw std::logic_error("vision encoder: A staging");
        Word* at = a + std::ptrdiff_t(size_t(i) * A_ST);
        std::copy(w.begin(), w.end(), at);
        std::memset(static_cast<void*>(at + w.size()), 0, sizeof(Word) * (A_ST - w.size()));   // the padding
    });
    if (copy.empty()) dev_.publish(addr, bytes);
    else dev_.write(addr, copy.data(), bytes);
}

void Vit::encode(const float* patches, float* out) {
    const size_t patch_count = size_t(cfg_.n_patch) * cfg_.patch_k;
    const size_t output_count = size_t(cfg_.n_patch / (cfg_.shuffle * cfg_.shuffle)) * cfg_.proj_out;
    if (cache_enabled_ && cached_patches_.size() == patch_count &&
        std::memcmp(cached_patches_.data(), patches, patch_count * sizeof(float)) == 0) {
        std::memcpy(out, cached_output_.data(), output_count * sizeof(float));
        ++cache_hits_; return;
    }
    ++cache_misses_;
    Program& p = prog_;
    const int E = cfg_.n_embd, F = cfg_.n_ff, NH = cfg_.n_head, dh = cfg_.dh, NP = cfg_.n_patch, NT = NP / TILE;
    const int NG = NH / 4, HG = 4 * dh, PK0 = cfg_.patch_k;
    if (size_t(D_ST) * 8 != size_t(TILE) * E) throw std::logic_error("vision encoder: D staging");
    std::vector<float> x(static_cast<size_t>(NP) * E), y(x.size());
    const uint32_t a_words_x = round8(uint32_t(Layout(TILE, E + 32).words()));
    const uint32_t a_words_p = round8(uint32_t(Layout(TILE, 2 * PK0 + 32).words()));
    if (a_words_p > A_ST) throw std::logic_error("vision encoder: patch A staging");
    const uint32_t qw = round8(uint32_t(Layout(TILE, dh, false, 4).words()));         // 264
    // a head's K band (32 keys, P32) and V^T cells (32 keys x 64 dims) of a tile: 66 words, 72 in UB
    const uint32_t band = uint32_t(Layout(32, dh, true).words()), tail = round8(band);
    if (KA_B + uint32_t(NH) * tail > VA_B || VA_B + uint32_t(NH) * tail > D_A || vt_words(dh, TILE) != band)
        throw std::logic_error("vision encoder: K / V^T bands");
    const uint32_t hw = round8(uint32_t(Layout(TILE, F).words()));
    // The tiles run in groups (vit_groups), so that the host's work on one group (residual, layer norm, A staging)
    // overlaps the Node's on the others. Each group's commands of a phase end with a reporting FENCE; the host waits
    // for the one it needs next. The staging of a tile (A, q, D) is its own, and a group's next commands go to the
    // Node only after the host has read and restaged that group, so the two never touch the same tile.
    const int groups = vit_groups(NT), GT = NT / groups;
    const bool two = vit_buffers() == 2;
    if (A1P_B + a_words_p > D_A || A1_B < VA_B + uint32_t(NH) * tail || A1_B < H_B + hw || A1_B + a_words_x > H1_B ||
        H1_B + hw > D_B)
        throw std::logic_error("vision encoder: the second A / H");
    auto patch = [&](int t0) {                          // A = [hi, lo, bias], F32 D
        p.role("vit_patch");
        for (int t = t0; t < t0 + GT; ++t) {
            const bool odd = two && (t & 1);
            Bound A = p.bind({odd ? I_A1 : I_A, 0, odd ? A1P_B : A_B, a_words_p, Order::P8, Kind::Static, true});
            p.move_in(A.b(), 0, a_words_p, a_st_ + uint64_t(t) * A_ST * 32, true);
            Bound D = p.bind({odd ? I_DB : I_DA, 0, odd ? D_B : D_A, f32_extent(TILE, E), Order::F32});
            GemmArgs g;
            g.a = A.b(); g.b = stream(patch_); g.d = D.b(); g.m = TILE; g.n = E; g.k = 2 * PK0 + 32;
            p.gemm(g);
            p.move_out(D.b(), 0, uint32_t(TILE * E / 8), d_st_ + uint64_t(t) * D_ST * 32);
        }
        return p.fence(true);
    };
    // q (three groups of four heads) to the q staging; k and v appended to the tile's K bands and V^T cells (a tile
    // is a band: 32 rows, from row 0), which go to the K / V^T images of the layer. The attention streams the images
    // after a FENCE (below).
    auto qkv = [&](const Layer& L, int t0) {
        p.role("vit_qkv");
        for (int t = t0; t < t0 + GT; ++t) {
            const bool odd = two && (t & 1);
            Bound A = p.bind({odd ? I_A1 : I_A, 0, odd ? A1_B : A_B, a_words_x, Order::P8, Kind::Static, true});
            p.move_in(A.b(), 0, a_words_x, a_st_ + uint64_t(t) * A_ST * 32, true);
            for (int g = 0; g < NG; ++g) {
                Bound Q = p.bind({uint8_t(I_Q + g), 0, Q_B + qw * uint32_t(g), qw});
                const uint32_t wband = uint32_t(Layout(32, E + 32, true).words());
                WeightImage wg{L.q.addr + uint64_t(g) * (HG / 32) * wband * 32, uint32_t(HG / 32) * wband};
                GemmArgs gq;
                gq.a = A.b(); gq.b = stream(wg); gq.d = Q.b(); gq.m = TILE; gq.n = HG; gq.k = E + 32; gq.d_heads = 4;
                p.gemm(gq);
                p.move_out(Q.b(), 0, qw, q_st_ + (uint64_t(t) * Q_ST + uint64_t(qw) * g) * 32);
            }
            const uint64_t off = uint64_t(t) * band * 32;
            auto append = [&](uint8_t idx, uint32_t base, Order order, const WeightImage& w, uint64_t img) {
                const Bound B = p.bind({idx, 0, base, uint32_t(NH) * tail, order, Kind::Append, false, NH});
                GemmArgs g;
                g.a = A.b(); g.b = stream(w); g.d = B.b(); g.m = TILE; g.n = E; g.k = E + 32; g.d_heads = NH;
                p.gemm(g);
                for (int h = 0; h < NH; ++h) p.move_out(B.b(h), 0, band, img + uint64_t(h) * kv_words_ * 32 + off);
            };
            append(I_KA, KA_B, Order::P32, L.k, kimg_);
            append(I_VA, VA_B, Order::VT, L.v, vimg_);
        }
    };
    auto attn_o = [&](const Layer& L, int t0) {         // attention (three of four heads), o (three K segments)
        for (int t = t0; t < t0 + GT; ++t) {
            p.role("vit_attn");
            for (int g = 0; g < NG; ++g) {
                Bound Q = p.bind({uint8_t(I_Q + g), 0, Q_B + qw * uint32_t(g), qw, Order::P8, Kind::Static, true});
                p.move_in(Q.b(), 0, qw, q_st_ + (uint64_t(t) * Q_ST + uint64_t(qw) * g) * 32, true);
                BindArgs ks{I_KS, 0, uint32_t(STREAM_B + 8 * (I_KS - 0xF8)), 8, Order::P32, Kind::Stream};
                ks.src = kimg_ + uint64_t(4 * g) * kv_words_ * 32; ks.length = kv_words_;
                BindArgs vs{I_VS, 0, uint32_t(STREAM_B + 8 * (I_VS - 0xF8)), 8, Order::VT, Kind::Stream};
                vs.src = vimg_ + uint64_t(4 * g) * kv_words_ * 32; vs.length = kv_words_;
                Bound KS = p.bind(ks), VS = p.bind(vs);
                Bound O = p.bind({uint8_t(I_O + g), 0, O_B + qw * uint32_t(g), qw});
                AttnArgs at;
                at.q = Q.b(); at.k = KS.b(); at.v = VS.b(); at.o = O.b(); at.tokens = TILE; at.q_heads = 4; at.group = 1;
                at.keys = NP; at.delta = NP; at.dh = dh;               // every key visible
                p.attn(at);
            }
            p.role("vit_o");
            uint16_t prev = 0;
            for (int g = 0; g < NG; ++g) {
                const bool a = (g & 1) == 0;
                Bound D = p.bind({a ? I_DA : I_DB, 0, a ? D_A : D_B, f32_extent(TILE, E), Order::F32});
                GemmArgs go;
                go.a = binding(uint8_t(I_O + g), p.gen_of(uint8_t(I_O + g))); go.b = stream(L.o[g]); go.c = prev;
                go.d = D.b(); go.m = TILE; go.n = E; go.k = HG; go.a_heads = 4; go.epi = prev ? Epi::Add : Epi::None;
                p.gemm(go);
                prev = D.b();
            }
            p.move_out(prev, 0, uint32_t(TILE * E / 8), d_st_ + uint64_t(t) * D_ST * 32);
        }
        return p.fence(true);
    };
    auto mlp = [&](const Layer& L, int t0) {            // fc1 (bias group, GELU), fc2 (FP32 D)
        for (int t = t0; t < t0 + GT; ++t) {
            const bool odd = two && (t & 1);
            Bound A = p.bind({odd ? I_A1 : I_A, 0, odd ? A1_B : A_B, a_words_x, Order::P8, Kind::Static, true});
            p.move_in(A.b(), 0, a_words_x, a_st_ + uint64_t(t) * A_ST * 32, true);
            p.role("vit_fc1");
            Bound H = p.bind({odd ? I_H1 : I_H, 0, odd ? H1_B : H_B, hw});
            GemmArgs g1;
            g1.a = A.b(); g1.b = stream(L.fc1); g1.d = H.b(); g1.m = TILE; g1.n = F; g1.k = E + 32; g1.epi = Epi::Gelu;
            p.gemm(g1);
            p.role("vit_fc2");
            Bound D = p.bind({I_DB, 0, D_B, f32_extent(TILE, E), Order::F32});
            GemmArgs g2;
            g2.a = H.b(); g2.b = stream(L.fc2); g2.d = D.b(); g2.m = TILE; g2.n = E; g2.k = F;
            p.gemm(g2);
            p.move_out(D.b(), 0, uint32_t(TILE * E / 8), d_st_ + uint64_t(t) * D_ST * 32);
        }
        return p.fence(true);
    };
    // the commands built so far go to the Node (a profile runs them to completion here: run)
    auto submit = [&] {
        if (profile_) { run(); return; }
        dev_.submit(p.cmds);
        p.clear();
    };
    // until the FENCE `tag` has completed. The completions taken with it may hold later FENCEs' (a group's that has
    // finished too): they are counted in `arrived`, so that the wait for them returns at once.
    std::array<int, 256> arrived{};
    auto wait = [&](uint8_t tag) {
        if (profile_) return;
        if (arrived[tag]) { --arrived[tag]; return; }
        const auto w0 = std::chrono::steady_clock::now();
        dev_.wait(tag);
        node_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
        for (const g3::Completion& cp : dev_.take_completions()) {
            if (cp.status != g3::ST_OK)
                throw std::runtime_error(std::string("Node command failed (vision): tag ") + std::to_string(cp.tag) + " " +
                                         g3::status_name(cp.status));
            ++arrived[cp.tag];
        }
        --arrived[tag];
    };
    std::vector<uint8_t> done(static_cast<size_t>(groups));
    p.bind({I_UB, 0, STREAM_B, UB_END - STREAM_B});    // the granules from the language model's bindings (I_UB)
    // ---- patch embedding (+ positions), then the first layer's LN1 and q / k / v ----
    for (int g = 0; g < groups; ++g) {
        { ScopeTimer tm{host_s_[1]}; stage_tiles(patches, PK0, true, g * GT, GT); }
        done[size_t(g)] = patch(g * GT);
        submit();
    }
    for (int g = 0; g < groups; ++g) {
        wait(done[size_t(g)]);
        host_pass(x.data(), pos_.data(), nullptr, layers_[0].ln1_w, layers_[0].ln1_b, g * GT, GT, nullptr);
        qkv(layers_[0], g * GT);
        submit();
    }
    for (size_t l = 0; l < layers_.size(); ++l) {
        const Layer& L = layers_[l];
        const Layer* next = l + 1 < layers_.size() ? &layers_[l + 1] : nullptr;
        p.fence();                                      // the K / V^T images written before the attention streams them
        for (int g = 0; g < groups; ++g) done[size_t(g)] = attn_o(L, g * GT);
        submit();
        for (int g = 0; g < groups; ++g) {              // x += o + bias, LN2
            wait(done[size_t(g)]);
            host_pass(x.data(), nullptr, L.o_b.data(), L.ln2_w, L.ln2_b, g * GT, GT, nullptr);
            done[size_t(g)] = mlp(L, g * GT);
            submit();
        }
        for (int g = 0; g < groups; ++g) {              // x += fc2 + bias, the next layer's LN1 (the last: post-LN)
            wait(done[size_t(g)]);
            if (!next) {
                host_pass(x.data(), nullptr, L.fc2_b.data(), post_w_, post_b_, g * GT, GT, y.data());
                continue;
            }
            host_pass(x.data(), nullptr, L.fc2_b.data(), next->ln1_w, next->ln1_b, g * GT, GT, nullptr);
            qkv(*next, g * GT);
            submit();
        }
    }
    // ---- CPU: pixel shuffle (ggml's reshape / permute order of build_patch_merge_permute) ----
    const int S = cfg_.shuffle, W = int(std::lround(std::sqrt(double(NP)))), PK = E * S * S, NO = NP / (S * S);
    std::vector<float> sh(static_cast<size_t>(NO) * PK);
    parallel_for(W, [&](int yy) {
        for (int xx = 0; xx < W; ++xx) {
            const int row = xx / S + (W / S) * (yy / S);
            const int c0 = E * (xx % S) + E * S * (yy % S);
            std::memcpy(&sh[size_t(row) * PK + size_t(c0)], &y[size_t(yy * W + xx) * E], size_t(E) * 4);
        }
    });
    // ---- Node: projector, K in two halves chained by FP32 C; the A halves of every projector tile staged at once,
    // one program ----
    const int PO = cfg_.proj_out;
    int projector_tile = TILE;
    while (projector_tile>8 && f32_extent(projector_tile,PO)>D_B-D_A) projector_tile/=2;
    if (PO<=0 || PO%32 || NO%projector_tile || f32_extent(projector_tile,PO)>D_B-D_A)
        throw std::invalid_argument("vision encoder: projector output does not fit UB");
    const int NPT = NO / projector_tile;
    const uint32_t ahw = round8(uint32_t(Layout(projector_tile, PK / 2).words()));
    const uint32_t ow = uint32_t(projector_tile * PO / 8);
    if (A_B + ahw > D_A || uint64_t(2 * NPT) * ahw > std::max<uint64_t>(uint64_t(NT) * A_ST, 2 * 6336) ||
        uint64_t(NPT) * ow > uint64_t(NT) * D_ST)
        throw std::logic_error("vision encoder: projector A");
    {
        ScopeTimer tm{host_s_[1]};
        std::vector<Word> all(static_cast<size_t>(2 * NPT) * ahw);
        parallel_for(2 * NPT, [&](int i) {
            const int t = i / 2, hlf = i % 2;
            std::vector<float> half(static_cast<size_t>(projector_tile) * (PK / 2));
            for (int r = 0; r < projector_tile; ++r)
                std::memcpy(&half[size_t(r) * (PK / 2)], &sh[(size_t(t) * projector_tile + r) * PK + size_t(hlf) * (PK / 2)], size_t(PK / 2) * 4);
            const std::vector<Word> w = stage_words(half.data(), projector_tile, PK / 2, false, false);
            if (w.size() > ahw) throw std::logic_error("vision encoder: projector A staging");
            std::copy(w.begin(), w.end(), all.begin() + std::ptrdiff_t(size_t(i) * ahw));
        });
        dev_.write(a_st_, all.data(), all.size() * 32);
    }
    p.role("vit_proj");
    for (int t = 0; t < NPT; ++t) {
        uint16_t prev = 0;
        for (int hlf = 0; hlf < 2; ++hlf) {
            Bound A = p.bind({I_A, 0, A_B, ahw, Order::P8, Kind::Static, true});
            p.move_in(A.b(), 0, ahw, a_st_ + uint64_t(2 * t + hlf) * ahw * 32, true);
            Bound D = p.bind({hlf ? I_DB : I_DA, 0, hlf ? D_B : D_A, f32_extent(projector_tile, PO), Order::F32});
            GemmArgs gp;
            gp.a = A.b(); gp.b = stream(proj_[hlf]); gp.c = prev; gp.d = D.b(); gp.m = projector_tile; gp.n = PO; gp.k = PK / 2;
            gp.epi = prev ? Epi::Add : Epi::None;
            p.gemm(gp);
            prev = D.b();
        }
        p.move_out(prev, 0, ow, d_st_ + uint64_t(t) * ow * 32);
    }
    p.bind({I_UB, 0, STREAM_B, UB_END - STREAM_B});    // and back (I_UB)
    run();
    std::vector<Word> o(static_cast<size_t>(NPT) * ow);
    dev_.read(d_st_, o.data(), o.size() * 32);
    for (int t = 0; t < NPT; ++t)
        f32_unpack(&o[size_t(t) * ow], projector_tile, PO, reinterpret_cast<uint32_t*>(out + size_t(t) * projector_tile * PO));
    if (cache_enabled_) {
        // Publish only after successful execution; upload() invalidates both buffers.
        cached_patches_.assign(patches, patches + patch_count);
        cached_output_.assign(out, out + output_count);
    }
}

}  // namespace g3rt
