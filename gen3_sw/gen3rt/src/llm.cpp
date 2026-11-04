#include "gen3rt/llm.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "g3emu_kernels.h"
#include "gen3rt/layout.h"
#include "gen3rt/parallel.h"
#include "gen3rt/quantize_fast.h"

namespace g3rt {
namespace {

struct HostTimer {
    using Clock = std::chrono::steady_clock;
    double* total;
    Clock::time_point start;
    explicit HostTimer(double* t) : total(t) { if (total) start = Clock::now(); }
    ~HostTimer() { if (total) *total += std::chrono::duration<double, std::milli>(Clock::now() - start).count(); }
};

uint32_t fbits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }

constexpr int MAX_TILE = 32;
constexpr int MAX_N = 4064;             // GEMM columns per command (the board's limit, below the field's 4,096)
constexpr int MAX_PARTS = 16;           // attention / FFN parts (binding ranges below)
constexpr int MAX_ATTN_HEADS = 16;      // q heads per ATTN (plan()); R51 / R52 need 9 (see plan())
// Binding ranges for >4 KV heads are selected per instance in configure_memory().
constexpr uint8_t I_X2 = 0x10, I_X3 = 0x20;
constexpr uint8_t I_Q = 0x40, I_O = 0x41, I_H = 0x42, I_T = 0x44, I_KR = 0x46, I_VR = 0x47, I_LOG = 0x48;
constexpr uint8_t I_P = 0x4C;           // partial sums of a K chain: I_P, I_P + 1 alternately
constexpr uint8_t I_QN = 0x50, I_ON = 0x60, I_HN = 0x70;   // Q / O of attention part c > 0, H of FFN part > 0
constexpr uint8_t I_KS = 0xF9, I_VS = 0xFA;
constexpr uint8_t I_QF = 0xC0, I_KF = 0x4A;    // qk_norm: FP32 q of attention part c at I_QF + c, FP32 k
constexpr uint8_t I_TK = 0x4E;                 // qk_norm on the Node: k's RoPE table (its gain at + 1; q's at I_T + 1)
const uint8_t WEIGHT_STREAMS[4] = {0xF8, 0xFB, 0xFC, 0xFD};

uint8_t q_index(size_t c) { return c ? uint8_t(I_QN + c) : I_Q; }
uint8_t o_index(size_t c) { return c ? uint8_t(I_ON + c) : I_O; }
uint8_t h_index(size_t c) { return c ? uint8_t(I_HN + c) : I_H; }

// part i of n of `total` (sizes differ by at most one)
int split_at(int total, int n, int i) { return int(int64_t(total) * i / n); }

// threads of the host's per-KV-head work (G3RT_HOST_THREADS 1..4; default: the cores, at most 4)
int host_threads() {
    static const int count = [] {
        if (const char* v = std::getenv("G3RT_HOST_THREADS")) {
            char* end = nullptr;
            const long n = std::strtol(v, &end, 10);
            if (end == v || *end || n < 1 || n > 4) throw std::invalid_argument("G3RT_HOST_THREADS: integer 1..4");
            return int(n);
        }
        return int(std::min(4u, std::max(1u, std::thread::hardware_concurrency())));
    }();
    return count;
}

// host_threads() - 1 worker threads, started once and kept: a prefill tile's per-head work (a few hundred us) is
// not worth starting threads for each time. run(n, f): f(0) .. f(n - 1), item i on thread i % T (T = min(n, threads),
// this one included); one caller at a time.
class HostPool {
public:
    explicit HostPool(int threads) {
        for (int t = 1; t < threads; ++t) workers_.emplace_back([this, t] { serve(t); });
    }
    ~HostPool() {
        { std::lock_guard<std::mutex> lock(m_); stop_ = true; ++job_; }
        start_.notify_all();
        for (std::thread& w : workers_) w.join();
    }
    void run(int n, const std::function<void(int)>& f) {
        std::lock_guard<std::mutex> caller(run_m_);
        const int T = std::min(n, int(workers_.size()) + 1);
        {
            std::lock_guard<std::mutex> lock(m_);
            f_ = &f; n_ = n; T_ = T; left_ = T - 1; ++job_;
        }
        start_.notify_all();
        for (int i = 0; i < n; i += T) f(i);
        std::unique_lock<std::mutex> lock(m_);
        done_.wait(lock, [this] { return left_ == 0; });
    }
private:
    void serve(int t) {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(m_);
        for (;;) {
            start_.wait(lock, [&] { return job_ != seen; });
            seen = job_;
            if (stop_) return;
            if (t >= T_) continue;                      // not needed for this one
            const std::function<void(int)>& f = *f_;
            const int n = n_, T = T_;
            lock.unlock();
            for (int i = t; i < n; i += T) f(i);
            lock.lock();
            if (--left_ == 0) done_.notify_one();
        }
    }
    std::vector<std::thread> workers_;
    std::mutex run_m_, m_;
    std::condition_variable start_, done_;
    const std::function<void(int)>* f_ = nullptr;
    int n_ = 0, T_ = 1, left_ = 0;
    uint64_t job_ = 0;
    bool stop_ = false;
};

// qk_norm models' q / k RMSNorm on the Node: images R56 and later, the default. R55 and older have no GEMM qk_norm
// and need G3RT_QK_NODE=0 (the host's, qk_host).
bool qk_on_node() {
    static const bool on = [] {
        const char* v = std::getenv("G3RT_QK_NODE");
        if (v && std::strcmp(v, "0") && std::strcmp(v, "1")) throw std::invalid_argument("G3RT_QK_NODE: 0 or 1");
        return !v || std::strcmp(v, "1") == 0;
    }();
    return on;
}

// streamed weights in quarters: images R55 and later. R54 and older take [106:95] of a STREAM BIND as source address
// bits and need G3RT_WEIGHT_QUARTERS=0.
bool weight_quarters() {
    static const bool on = [] {
        const char* v = std::getenv("G3RT_WEIGHT_QUARTERS");
        if (v && std::strcmp(v, "0") && std::strcmp(v, "1")) throw std::invalid_argument("G3RT_WEIGHT_QUARTERS: 0 or 1");
        return !v || std::strcmp(v, "1") == 0;
    }();
    return on;
}

// a segment's quarter stride, 16 KiB units: its quarter (ceil(bands / 4) bands) rounded up, made odd
uint32_t quarter_units(int bands, uint32_t band_words) {
    const uint64_t bytes = uint64_t((bands + 3) / 4) * band_words * 32;
    return uint32_t((bytes + 16383) / 16384) | 1u;
}

}  // namespace

void parallel_for(int n, const std::function<void(int)>& f) {
    if (n <= 1 || host_threads() == 1) { for (int i = 0; i < n; ++i) f(i); return; }
    static HostPool pool(host_threads());
    pool.run(n, f);
}

void rope_row(int pos, int n_dims, float base, float* out) {
    const float theta_scale = powf(base, -2.0f / float(n_dims));
    float theta = float(pos);
    for (int i0 = 0; i0 < n_dims; i0 += 2) {
        out[i0] = cosf(theta) * 1.0f;
        out[i0 + 1] = sinf(theta) * 1.0f;
        theta *= theta_scale;
    }
}

// Attention: KV heads in runs of up to max_kv_heads (an ATTN takes 4), the group in rounds of up to max_group (its
// group field: 4); FFN: gate / up in parts of at most max_ffn_columns. One part each when the model fits one command.
void Llm::plan() {
    const int E = cfg_.n_embd, NK = cfg_.n_kv, D = cfg_.dh, G = cfg_.n_head / cfg_.n_kv;
    const int max_g = cfg_.max_group, max_kv = cfg_.max_kv_heads, max_f = cfg_.max_ffn_columns;
    if (max_g < 1 || max_g > 4 || max_kv < 1 || max_kv > 4 || max_f < 32 || max_f > MAX_N || max_f % 32)
        throw std::invalid_argument("command limits: group 1..4, KV heads 1..4, FFN columns 32..4064 (x 32)");
    const uint32_t band_e = uint32_t(Layout(32, E, true).words());      // a 32-row band of a K = E matrix
    // At most MAX_ATTN_HEADS q heads per ATTN. 16 (4 KV heads x group 4, MicroLlama) need R53: on R51 / R52 the
    // GEMM head counts (q's D, o's A) wrap to 0 at 16 (docs/reports/20261023_attn_16_heads); those images need 9.
    const int runs = (NK + max_kv - 1) / max_kv, kv_run = (NK + runs - 1) / runs;
    const int group = std::max(1, std::min(max_g, MAX_ATTN_HEADS / kv_run)), rounds = (G + group - 1) / group;
    attn_.clear();
    uint32_t ub = 0, wq = 0, wo = 0;
    for (int r = 0; r < runs; ++r)
        for (int k = 0; k < rounds; ++k) {
            AttnPart a;
            a.kv0 = split_at(NK, runs, r); a.kvs = split_at(NK, runs, r + 1) - a.kv0;
            a.r0 = split_at(G, rounds, k); a.grp = split_at(G, rounds, k + 1) - a.r0;
            const int heads = a.kvs * a.grp;
            if (heads * D > MAX_N) throw std::invalid_argument("attention part wider than a GEMM");
            a.ub = ub; a.wq_off = wq; a.wo_off = wo;
            a.wo_words = 33u * uint32_t(E / 32) * uint32_t(heads * D / 32);
            ub += round8(uint32_t(Layout(tile_, D, false, heads).words()));
            wq += uint32_t(heads * D / 32) * band_e;
            wo += a.wo_words;
            attn_.push_back(a);
        }
    const int bands = cfg_.n_ff / 32, parts = (bands + max_f / 32 - 1) / (max_f / 32);
    ffn_.clear();
    ub = 0; uint32_t wd = 0;
    for (int i = 0; i < parts; ++i) {
        FfnPart f;
        f.band0 = split_at(bands, parts, i); f.bands = split_at(bands, parts, i + 1) - f.band0;
        f.ub = ub; f.wd_off = wd;
        f.wd_words = 33u * uint32_t(E / 32) * uint32_t(f.bands);
        ub += round8(uint32_t(Layout(tile_, 32 * f.bands).words()));
        wd += f.wd_words;
        ffn_.push_back(f);
    }
    if (attn_.size() > MAX_PARTS || ffn_.size() > MAX_PARTS) throw std::invalid_argument("too many parts");
}

void Llm::configure_memory() {
    const int E=cfg_.n_embd, F=cfg_.n_ff, NH=cfg_.n_head, NK=cfg_.n_kv, D=cfg_.dh;
    // o / down / lm_head write FP32 D (columns a multiple of 64, at most MAX_N); ATTN dh <= 256
    if (E<=0 || E%64 || E>MAX_N || F<=0 || F%32 ||
        D<=0 || D%32 || D>256 || NH<1 || NH>16 || NK<1 || NK>16 ||
        NH%NK || NK*D>MAX_N || cfg_.n_layer<1 || cfg_.vocab<64 || cfg_.vocab%64)
        throw std::invalid_argument("unsupported Llama dimensions for the R51 command format");
    if (cfg_.lm_chunk<64 || cfg_.lm_chunk>1024 || cfg_.lm_chunk%64)
        throw std::invalid_argument("lm_chunk: a multiple of 64, at most 1024");
    tile_=MAX_TILE;
    // qk_norm on the Node: R56 and later (default; G3RT_QK_NODE=0 for older images), heads of 64 or 128 columns,
    // the q / k epsilon the fold's
    qk_node_=cfg_.qk_norm && qk_on_node() && (D==64 || D==128) &&
             fbits(cfg_.qk_eps)==g3::epsilon_bits(cfg_.eps_profile);
    plan();
    if (E==576 && F==1536 && NH==9 && NK==3 && D==64 && attn_.size()==1 && ffn_.size()==1 && !cfg_.qk_norm) return;
    uint32_t next=0;
    auto alloc=[&](uint32_t words) { const uint32_t base=next; next+=round8(words); return base; };
    const uint32_t tail=uint32_t(NK)*round8(uint32_t(Layout(32,D,true).words()));
    uint32_t xg=0, kr=0, q=0, qf=0;
    // The UB holds the buffers of a tile of tile_ tokens: 32, or 16 / 8 for a model too wide for that (prefill then
    // runs in tiles of at most tile_ tokens; decode is one token either way). Diagnosis: G3RT_UB_TILE=16 / 8 starts
    // the search there (a model that fits 32 tokens on a smaller map).
    if (const char* v=std::getenv("G3RT_UB_TILE")) {
        tile_=std::atoi(v);
        if (tile_!=8 && tile_!=16 && tile_!=32) throw std::invalid_argument("G3RT_UB_TILE must be 8, 16, or 32");
    }
    if (qk_node_ && D==128) tile_=std::min(tile_,16);     // a q / k RMSNorm GEMM of 128-column heads: m <= 16
    for (;; tile_/=2) {
        if (tile_<8) throw std::invalid_argument("model's 8-token tile exceeds the 512 KiB UB");
        plan();
        bool fits=false;
        xg=uint32_t(Layout(tile_,E).words());
        kr=uint32_t(Layout(tile_,D,false,NK).words());
        uint32_t h=0, h1=0;
        q=0; qf=0;
        for (const AttnPart& a : attn_) q+=round8(uint32_t(Layout(tile_,D,false,a.kvs*a.grp).words()));
        for (const AttnPart& a : attn_) qf+=round8(f32_extent(tile_,a.kvs*a.grp*D));
        for (const FfnPart& f : ffn_) {
            const uint32_t w=round8(uint32_t(Layout(tile_,32*f.bands).words()));
            h+=w; h1=std::max(h1,w);
        }
        const uint32_t kt=std::max(tail,round8(kr)), vt=std::max(tail,f32_extent(tile_,NK*D));
        // qk_norm: the layer's start in one region (map_.QF): a part's FP32 q, FP32 k, the V^T cells, the K bands
        // with the host's k
        uint32_t start=0;
        for (const AttnPart& a : attn_) start=std::max(start,round8(f32_extent(tile_,a.kvs*a.grp*D)));
        const uint32_t kn=round8(uint32_t(Layout(tile_,NK*D).words()));
        start=std::max({start,round8(f32_extent(tile_,NK*D)),round8(vt),round8(kt)+kn});
        // If the tile does not fit: (1) X2G / X3G and their 1/rms share one region: a fold producer's BIND then
        // waits for the readers of the other (q / k / v of the layer, gate / up), which have finished or are about
        // to; (2) the FFN parts run one after another (ffn_one_h), sharing one H; (3) the K bands and the V^T cells
        // share one region, k and v appended in turn (kv_one).
        for (int option=0; option<4 && !fits; ++option) {
            const bool shared=option>=1;
            map_.ffn_one_h=option>=2 && ffn_.size()>1;
            map_.kv_one=option>=3 && !(cfg_.qk_norm && !qk_node_);   // (the host's q / k path shares them anyway)
            next=0;
            map_.XGA=alloc(xg); map_.XGB=shared ? map_.XGA : alloc(xg);
            map_.RA=alloc(8); map_.RB=shared ? map_.RA : alloc(8);
            map_.GA=alloc(uint32_t(E/8)); map_.GB=alloc(uint32_t(E/8)); map_.STREAMB=alloc(64);
            map_.XA=alloc(f32_extent(tile_,E)); map_.XB=alloc(f32_extent(tile_,E));
            map_.QB=alloc(q); map_.OB=alloc(q); map_.TB=alloc(uint32_t(Layout(tile_,D).words()));
            map_.HB=alloc(map_.ffn_one_h ? h1 : h);
            if (cfg_.qk_norm && !qk_node_) {
                map_.QF=map_.KF=map_.VTB=map_.KTB=alloc(start);
                map_.KN=map_.KTB+round8(kt);
            } else if (map_.kv_one) {
                map_.KTB=map_.VTB=alloc(std::max(kt,vt));
            } else {
                map_.KTB=alloc(kt);
                map_.VTB=alloc(vt);
            }
            if (qk_node_) {
                map_.TQB=alloc(f32_extent(tile_,D)); map_.GQB=alloc(uint32_t(D/8));
                map_.TKB=alloc(f32_extent(tile_,D)); map_.GKB=alloc(uint32_t(D/8));
            }
            map_.LOGA=map_.XA; map_.LOGB=map_.LOGA+f32_extent(tile_,cfg_.lm_chunk);
            map_.UB_END=std::max(next,map_.LOGB+f32_extent(tile_,cfg_.lm_chunk));
            fits=map_.UB_END<=16384;
        }
        if (fits) break;
    }
    next=f32_extent(tile_,E);
    map_.XG_ST=alloc(xg); map_.R_ST=alloc(8);
    map_.T_ST=alloc(uint32_t(Layout(tile_,D).words()));
    map_.KR_ST=alloc(kr); map_.VR_ST=alloc(f32_extent(tile_,NK*D));
    if (qk_node_) map_.T32_ST=alloc(f32_extent(tile_,D));
    else if (cfg_.qk_norm) {
        map_.QF_ST=alloc(qf); map_.KF_ST=alloc(f32_extent(tile_,NK*D));
        map_.Q8_ST=alloc(q); map_.K8_ST=alloc(round8(uint32_t(Layout(tile_,NK*D).words())));
    }
    map_.ST_END=next;
    if (NK>4) { map_.I_KT=0x80; map_.I_VT=0x90; map_.I_KA=0xA0; map_.I_VA=0xB0; }
}

Llm::Llm(Device& dev, const LlmConfig& cfg) : dev_(dev), cfg_(cfg) {
    configure_memory();
    for (const char* role : {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down", "output", "ident"}) {
        std::vector<Segment> segs = segments(role);
        uint64_t at = 0;
        for (Segment& g : segs) { g.addr = at; at += uint64_t(g.bands) * g.band_words; }
        shapes_[role] = segs;
    }
    if (const char* value=std::getenv("G3RT_LLM_PREFILL_TILE")) {
        const std::string v(value);
        if (v!="8" && v!="16" && v!="32")
            throw std::invalid_argument("G3RT_LLM_PREFILL_TILE must be 8, 16, or 32");
        prefill_tile_=std::atoi(value);
    }
    prefill_tile_=std::min(prefill_tile_, tile_);
    if (cfg_.ctx < 32 || cfg_.ctx % 32 || cfg_.ctx > 2048) throw std::invalid_argument("ctx: multiple of 32, at most 2048");
    layers_.resize(static_cast<size_t>(cfg_.n_layer));
    rope_cache_.resize(size_t(cfg_.ctx) * 32);
    kv_words_ = uint32_t(Layout(cfg_.ctx, cfg_.dh, true).words());
    for (auto& L : layers_) {
        L.kimg = dev_.alloc(static_cast<size_t>(cfg_.n_kv) * kv_words_ * 32);
        L.vimg = dev_.alloc(static_cast<size_t>(cfg_.n_kv) * kv_words_ * 32);
        dev_bytes_ += 2 * size_t(cfg_.n_kv) * kv_words_ * 32;
    }
    if (cfg_.lm_chunk > 1024 || cfg_.lm_chunk % 64) throw std::invalid_argument("lm_chunk: a multiple of 64, at most 1024");
    stage_ = dev_.alloc(map_.ST_END * 32);
    if (cfg_.qk_norm && !qk_node_) {         // the host's k appended by an identity GEMM (build_k_append)
        const int KVD = cfg_.n_kv * cfg_.dh;
        std::vector<float> eye(size_t(KVD) * KVD, 0.0f);
        for (int i = 0; i < KVD; ++i) eye[size_t(i) * KVD + i] = 1.0f;
        ident_ = upload_plain("ident", eye.data(), KVD, KVD);
    }
    // a zero K band / V^T cell: the start of a band
    const std::vector<Word> zero(round8(uint32_t(Layout(32, cfg_.dh, true).words())));
    zero_ = dev_.alloc(zero.size() * 32);
    dev_.write(zero_, zero.data(), zero.size() * 32);
    logits_ = dev_.alloc(static_cast<size_t>(cfg_.vocab) * 4 + 4096);
}

WeightImage upload_matrix_image(Device& dev, const float* w, int rows, int cols) {
    std::vector<uint32_t> bits(static_cast<size_t>(rows) * cols);
    std::memcpy(bits.data(), w, bits.size() * 4);
    std::vector<uint8_t> codes, scales;
    quantize(bits.data(), rows, cols, codes, scales);
    const std::vector<Word> words = Layout(rows, cols, true).pack(codes.data(), scales.data());
    WeightImage img;
    img.words = uint32_t(words.size());
    img.addr = dev.alloc(words.size() * 32);
    dev.write(img.addr, words.data(), words.size() * 32);
    return img;
}

WeightImage upload_f32_row_image(Device& dev, const float* v, int n) {
    std::vector<uint32_t> bits(static_cast<size_t>(n));
    std::memcpy(bits.data(), v, size_t(n) * 4);
    const std::vector<Word> words = f32_pack(bits.data(), 1, n);
    WeightImage img;
    img.words = uint32_t(words.size());
    img.addr = dev.alloc(words.size() * 32);
    dev.write(img.addr, words.data(), words.size() * 32);
    return img;
}

bool Llm::attach(int layer, const std::string& role, const WeightImage& img) {
    if (layer < 0) {
        if (role == "output_norm") g_out_ = img;
        else if (role == "output") lm_ = img;
        else return false;
        return true;
    }
    if (layer >= cfg_.n_layer) return false;
    LayerImages& L = layers_[size_t(layer)];
    if (layer == 0 && role == "attn_norm") norm0_.clear();
    if (role == "ffn_gate_up") L.wgu = img;
    else if (role == "attn_q") L.wq = img;
    else if (role == "attn_k") L.wk = img;
    else if (role == "attn_v") L.wv = img;
    else if (role == "attn_output") L.wo = img;
    else if (role == "ffn_gate") L.wg = img;
    else if (role == "ffn_up") L.wu = img;
    else if (role == "ffn_down") L.wd = img;
    else if (role == "attn_norm") L.g_attn = img;
    else if (role == "ffn_norm") L.g_ffn = img;
    else return false;
    if (arranged(role)) rearrange(role, img);
    if ((role == "ffn_gate" || role == "ffn_up") && L.wg.words && L.wu.words)
        L.wgu = interleave_bands(L.wg, L.wu, cfg_.n_ff, cfg_.n_embd);
    return true;
}

bool Llm::arranged(const std::string& role) const {
    if (role == "attn_q") return cfg_.rope_neox || attn_.size() > 1;
    if (role == "attn_k") return cfg_.rope_neox;
    if (role == "attn_output") return attn_.size() > 1;
    if (role == "ffn_down") return ffn_.size() > 1;
    return false;
}

// Rows: W_q's heads in attention-part order (KV-head-major within a part), W_q / W_k rows within a head for NEOX RoPE
// (row 2i <- i, 2i + 1 <- i + dh / 2: the Node rotates adjacent pairs). Columns: W_o one K slice per attention part
// (its heads in the same order; V and so O are not permuted), W_d one per FFN part; the slices' P32 images one after
// another. Quantization is per row and 32-column group, so this equals quantizing the rearranged matrix.
std::vector<Word> Llm::arrange(const std::string& role, const uint8_t* codes, const uint8_t* scales, int rows,
                               int cols) const {
    const int D = cfg_.dh, G = cfg_.n_head / cfg_.n_kv, cg = cols / 32;
    auto head_row = [&](int d) { return cfg_.rope_neox ? (d & 1 ? d / 2 + D / 2 : d / 2) : d; };
    std::vector<int> order(static_cast<size_t>(rows));
    std::vector<std::vector<int>> slices;
    for (int r = 0; r < rows; ++r) order[size_t(r)] = r;
    auto all_groups = [&] { std::vector<int> g(static_cast<size_t>(cg)); for (int j = 0; j < cg; ++j) g[size_t(j)] = j; return g; };
    if (role == "attn_q") {
        int r = 0;
        for (const AttnPart& a : attn_)
            for (int kv = a.kv0; kv < a.kv0 + a.kvs; ++kv)
                for (int i = 0; i < a.grp; ++i)
                    for (int d = 0; d < D; ++d) order[size_t(r++)] = (kv * G + a.r0 + i) * D + head_row(d);
        slices.push_back(all_groups());
    } else if (role == "attn_k") {
        for (int r = 0; r < rows; ++r) order[size_t(r)] = r / D * D + head_row(r % D);
        slices.push_back(all_groups());
    } else if (role == "attn_output") {
        for (const AttnPart& a : attn_) {
            std::vector<int> s;
            for (int kv = a.kv0; kv < a.kv0 + a.kvs; ++kv)
                for (int i = 0; i < a.grp; ++i)
                    for (int j = 0; j < D / 32; ++j) s.push_back((kv * G + a.r0 + i) * (D / 32) + j);
            slices.push_back(s);
        }
    } else if (role == "ffn_down") {
        for (const FfnPart& f : ffn_) {
            std::vector<int> s;
            for (int j = f.band0; j < f.band0 + f.bands; ++j) s.push_back(j);
            slices.push_back(s);
        }
    } else slices.push_back(all_groups());
    std::vector<Word> out;
    std::vector<uint8_t> c, s;
    for (const std::vector<int>& groups : slices) {
        const int n = int(groups.size());
        c.resize(size_t(rows) * size_t(n) * 32); s.resize(size_t(rows) * size_t(n));
        for (int r = 0; r < rows; ++r)
            for (int j = 0; j < n; ++j) {
                const size_t src = size_t(order[size_t(r)]), g = size_t(groups[size_t(j)]);
                std::memcpy(&c[(size_t(r) * n + j) * 32], &codes[src * size_t(cols) + g * 32], 32);
                s[size_t(r) * n + j] = scales[src * size_t(cg) + g];
            }
        const std::vector<Word> w = Layout(rows, n * 32, true).pack(c.data(), s.data());
        out.insert(out.end(), w.begin(), w.end());
    }
    return out;
}

// the shapes of the arranged roles
namespace {
std::pair<int, int> weight_shape(const LlmConfig& c, const std::string& role) {
    const int QD = c.n_head * c.dh;
    if (role == "attn_q") return {QD, c.n_embd};
    if (role == "attn_k") return {c.n_kv * c.dh, c.n_embd};
    if (role == "attn_output") return {c.n_embd, QD};
    return {c.n_embd, c.n_ff};          // ffn_down
}
}  // namespace

// an image in device memory, plain P32 -> arranged, in place (the same number of words)
void Llm::rearrange(const std::string& role, const WeightImage& img) {
    const auto shape = weight_shape(cfg_, role);
    const Layout lay(shape.first, shape.second, true);
    if (img.words != uint32_t(lay.words())) throw std::invalid_argument("image of " + role + ": unexpected size");
    std::vector<Word> w(img.words);
    dev_.read(img.addr, w.data(), w.size() * 32);
    std::vector<uint8_t> codes(size_t(shape.first) * size_t(shape.second)), scales(codes.size() / 32);
    lay.unpack(w.data(), codes.data(), scales.data());
    const std::vector<Word> out = arrange(role, codes.data(), scales.data(), shape.first, shape.second);
    if (out.size() != w.size()) throw std::logic_error("arranged image size");
    dev_.write(img.addr, out.data(), out.size() * 32);
}

WeightImage Llm::upload_weight(const std::string& role, const float* w, int rows, int cols) {
    if (!arranged(role)) return upload_plain(role, w, rows, cols);
    std::vector<uint32_t> bits(static_cast<size_t>(rows) * cols);
    std::memcpy(bits.data(), w, bits.size() * 4);
    std::vector<uint8_t> codes, scales;
    quantize(bits.data(), rows, cols, codes, scales);
    return place(role, arrange(role, codes.data(), scales.data(), rows, cols));
}

std::string Llm::weight_layout() const {
    if (!cfg_.rope_neox && attn_.size() == 1 && ffn_.size() == 1) return "";
    std::string s = std::string("neox=") + (cfg_.rope_neox ? "1" : "0") + " attn=";
    for (const AttnPart& a : attn_)
        s += std::to_string(a.kv0) + ":" + std::to_string(a.kvs) + ":" + std::to_string(a.r0) + ":" + std::to_string(a.grp) + ",";
    s += " ffn=";
    for (const FfnPart& f : ffn_) s += std::to_string(f.band0) + ":" + std::to_string(f.bands) + ",";
    return s;
}

WeightImage Llm::upload_matrix(const float* w, int rows, int cols) { return upload_plain("", w, rows, cols); }

// the P32 image of an FP32 matrix, placed for `role` (place)
WeightImage Llm::upload_plain(const std::string& role, const float* w, int rows, int cols) {
    std::vector<uint32_t> bits(static_cast<size_t>(rows) * cols);
    std::memcpy(bits.data(), w, bits.size() * 4);
    std::vector<uint8_t> codes, scales;
    quantize(bits.data(), rows, cols, codes, scales);
    return place(role, Layout(rows, cols, true).pack(codes.data(), scales.data()));
}

WeightImage Llm::upload_f32_row(const float* v, int n) {
    std::vector<uint32_t> bits(static_cast<size_t>(n));
    std::memcpy(bits.data(), v, size_t(n) * 4);
    const std::vector<Word> words = f32_pack(bits.data(), 1, n);
    WeightImage img;
    img.words = uint32_t(words.size());
    img.addr = dev_.alloc(words.size() * 32);
    dev_.write(img.addr, words.data(), words.size() * 32);
    dev_bytes_ += words.size() * 32;
    return img;
}

namespace {
// gate / up for SILU_MUL as one P32 image (see interleave_bands)
std::vector<Word> gate_up_words(const float* gate_w, const float* up_w, int rows, int cols) {
    std::vector<uint32_t> bits(size_t(rows) * cols);
    std::vector<uint8_t> codes, scales;
    const Layout lay(rows, cols, true);
    std::memcpy(bits.data(), gate_w, bits.size() * 4);
    quantize(bits.data(), rows, cols, codes, scales);
    const auto gate = lay.pack(codes.data(), scales.data());
    std::memcpy(bits.data(), up_w, bits.size() * 4);
    quantize(bits.data(), rows, cols, codes, scales);
    const auto up = lay.pack(codes.data(), scales.data());
    const size_t band = size_t(Layout(32, cols, true).words());
    std::vector<Word> gu(gate.size() + up.size());
    for (int t = 0; t < rows / 32; ++t) {
        std::memcpy(gu.data() + size_t(2 * t) * band, gate.data() + size_t(t) * band, band * 32);
        std::memcpy(gu.data() + size_t(2 * t + 1) * band, up.data() + size_t(t) * band, band * 32);
    }
    return gu;
}
}  // namespace

WeightImage upload_gate_up_image(Device& dev, const float* gate_w, const float* up_w, int rows, int cols) {
    const std::vector<Word> gu = gate_up_words(gate_w, up_w, rows, cols);
    WeightImage image{dev.alloc(gu.size()*32),uint32_t(gu.size())};
    dev.write(image.addr,gu.data(),gu.size()*32);
    return image;
}

void Llm::upload_layer(int l, const LayerWeights& w) {
    const int E = cfg_.n_embd, F = cfg_.n_ff, QD = cfg_.n_head * cfg_.dh, KVD = cfg_.n_kv * cfg_.dh;
    LayerImages& L = layers_.at(static_cast<size_t>(l));
    if (l == 0) norm0_.clear();
    L.wq = upload_weight("attn_q", w.wq, QD, E);
    L.wk = upload_weight("attn_k", w.wk, KVD, E);
    L.wv = upload_plain("attn_v", w.wv, KVD, E);
    L.wo = upload_weight("attn_output", w.wo, E, QD);
    L.wgu = place("ffn_gate_up", gate_up_words(w.wg, w.wu, F, E));
    L.wd = upload_weight("ffn_down", w.wd, E, F);
    L.g_attn = upload_f32_row(w.g_attn, E);
    L.g_ffn = upload_f32_row(w.g_ffn, E);
    if (cfg_.qk_norm) {
        L.g_q = upload_f32_row(w.g_q, cfg_.dh);
        L.g_k = upload_f32_row(w.g_k, cfg_.dh);
        qk_gain_.clear();
        L.g_qa = L.g_ka = {};
    }
}

// gate / up for SILU_MUL: rows 64 t .. 64 t + 31 = gate rows 32 t .., rows 64 t + 32 .. = up rows 32 t ..; in P32 images
// a 32-row band is contiguous, so the combined image alternates the bands of the two
WeightImage Llm::interleave_bands(const WeightImage& a, const WeightImage& b, int rows, int cols) {
    const uint32_t band = uint32_t(Layout(32, cols, true).words());
    const int nb = rows / 32;
    std::vector<Word> gu(2 * size_t(nb) * band);
    for (int t = 0; t < nb; ++t) {
        dev_.read(a.addr + uint64_t(t) * band * 32, &gu[size_t(2 * t) * band], band * 32);
        dev_.read(b.addr + uint64_t(t) * band * 32, &gu[size_t(2 * t + 1) * band], band * 32);
    }
    return place("ffn_gate_up", gu);
}

bool Llm::upload_tensor(int layer, const std::string& role, const float* w, int rows, int cols) {
    if (layer < 0) {
        if (role == "output_norm") g_out_ = upload_f32_row(w, cols);
        else if (role == "output") lm_ = upload_plain(role, w, rows, cols);
        else return false;
        return true;
    }
    if (layer >= cfg_.n_layer) return false;
    LayerImages& L = layers_[size_t(layer)];
    if (layer == 0 && role == "attn_norm") norm0_.clear();
    if (role == "attn_q") L.wq = upload_weight(role, w, rows, cols);
    else if (role == "attn_k") L.wk = upload_weight(role, w, rows, cols);
    else if (role == "attn_v") L.wv = upload_plain(role, w, rows, cols);
    else if (role == "attn_output") L.wo = upload_weight(role, w, rows, cols);
    else if (role == "ffn_gate") L.wg = upload_matrix(w, rows, cols);
    else if (role == "ffn_up") L.wu = upload_matrix(w, rows, cols);
    else if (role == "ffn_down") L.wd = upload_weight(role, w, rows, cols);
    else if (role == "attn_norm") L.g_attn = upload_f32_row(w, cols);
    else if (role == "ffn_norm") L.g_ffn = upload_f32_row(w, cols);
    else if (role == "attn_q_norm") L.g_q = upload_f32_row(w, cols);
    else if (role == "attn_k_norm") L.g_k = upload_f32_row(w, cols);
    else return false;
    if (role == "attn_q_norm" || role == "attn_k_norm") { qk_gain_.clear(); L.g_qa = L.g_ka = {}; }
    if ((role == "ffn_gate" || role == "ffn_up") && L.wg.words && L.wu.words)
        L.wgu = interleave_bands(L.wg, L.wu, cfg_.n_ff, cfg_.n_embd);
    return true;
}

bool Llm::complete() const {
    for (const LayerImages& L : layers_)
        if (!L.wq.words || !L.wk.words || !L.wv.words || !L.wo.words || !L.wgu.words || !L.wd.words ||
            !L.g_attn.words || !L.g_ffn.words || (cfg_.qk_norm && (!L.g_q.words || !L.g_k.words)))
            return false;
    return g_out_.words && lm_.words;
}

void Llm::upload_output(const float* g_out, const float* lm_head) {
    g_out_ = upload_f32_row(g_out, cfg_.n_embd);
    lm_ = upload_plain("output", lm_head, cfg_.vocab, cfg_.n_embd);
}

uint8_t Llm::next_weight_stream() {
    const uint8_t s = WEIGHT_STREAMS[stream_slot_];
    stream_slot_ = (stream_slot_ + 1) % 4;
    return s;
}

// Inputs of a tile of m tokens at positions pos.. (staging, words): x rows (FP32, m E / 8) at 0; x g0 (P8) at
// map_.XG_ST; 1/rms (8) at map_.R_ST; RoPE rows (P8; q: n_head copies, k: n_kv copies) at TQ_ST, TK_ST. Layer 0's x g0 and
// 1/rms are formed as a fold producer forms them (fold_producer_v3 with y = x): MXFP8(FP32(x g0)), 1/rms of MXFP8(x).
namespace {
// words padded with zeros to a multiple of 8 (the binding's extent)
std::vector<Word> pad8(std::vector<Word> w) { w.resize(round8(uint32_t(w.size()))); return w; }
}

void Llm::host_inputs(const float* x, int m, int pos) {
    const int E = cfg_.n_embd, dh = cfg_.dh;
    auto& v = input_;
    v.xb.resize(size_t(m) * E); v.xg.resize(v.xb.size()); v.xd.resize(v.xb.size());
    std::memcpy(v.xb.data(), x, v.xb.size() * 4);
    if (norm0_.empty()) {
        v.gw.resize(size_t(E / 8)); norm0_.resize(size_t(E));
        dev_.read(layers_[0].g_attn.addr, v.gw.data(), v.gw.size() * 32);
        f32_unpack(v.gw.data(), 1, E, norm0_.data());
    }
    for (size_t i = 0; i < v.xb.size(); ++i) v.xg[i] = g3::f32_mul(v.xb[i], norm0_[i % size_t(E)]);
    quantize(v.xg.data(), m, E, v.c, v.s);
    quantize(v.xb.data(), m, E, v.xc, v.xs);
    dequantize(v.xc.data(), v.xs.data(), m, E, v.xd.data());
    v.r.resize(size_t(m));
    for (int i = 0; i < m; ++i) v.r[size_t(i)] = g3::inv_rms_row(&v.xd[size_t(i) * E], E, cfg_.eps_profile);
    v.wx.resize(v.xb.size() / 8);
    for (size_t i = 0; i < v.xb.size(); ++i) v.wx[i / 8].set_u32(i % 8, v.xb[i]);
    const Layout lay(m, E);
    v.wxg.resize(round8(uint32_t(lay.words())));
    std::fill(v.wxg.begin(), v.wxg.end(), Word{});
    lay.pack_into(v.c.data(), v.s.data(), v.wxg.data());
    v.wr.assign(8, Word{});
    for (int i = 0; i < m; ++i) v.wr[size_t(i / 8)].set_u32(i % 8, v.r[size_t(i)]);
    auto& wt = rope_cache_.at(size_t(pos) * 32 + size_t(m - 1));
    if (wt.empty()) {
        v.tt.resize(size_t(m) * dh); v.tr.resize(size_t(dh));
        for (int i = 0; i < m; ++i) {
            rope_row(pos + i, dh, cfg_.rope_base, v.tr.data());
            for (int k = 0; k < dh; ++k) v.tt[size_t(i) * dh + k] = fbits(v.tr[size_t(k)]);
        }
        quantize(v.tt.data(), m, dh, v.tc, v.ts);
        wt = pad8(Layout(m, dh).pack(v.tc.data(), v.ts.data()));
    }
    dev_.write(stage_, v.wx.data(), v.wx.size() * 32);
    dev_.write(stage_ + map_.XG_ST * 32, v.wxg.data(), v.wxg.size() * 32);
    dev_.write(stage_ + map_.R_ST * 32, v.wr.data(), v.wr.size() * 32);
    dev_.write(stage_ + map_.T_ST * 32, wt.data(), wt.size() * 32);
    if (qk_node_) {                     // the q / k RMSNorm GEMMs' RoPE table: FP32 rows of cos, sin pairs
        qk_rope_rows(pos, m);
        v.tt.resize(size_t(m) * dh);
        for (int i = 0; i < m; ++i) std::memcpy(&v.tt[size_t(i) * dh], qk_rope_[size_t(pos + i)].data(), size_t(dh) * 4);
        const std::vector<Word> t32 = f32_pack(v.tt.data(), m, dh);
        dev_.write(stage_ + uint64_t(map_.T32_ST) * 32, t32.data(), t32.size() * 32);
    }
}

uint16_t Llm::stream(const Span& w) {
    const uint8_t s = next_weight_stream();
    BindArgs a{s, 0, uint32_t(map_.STREAMB + 8 * (s - 0xF8)), 8, Order::P32, Kind::Stream};
    a.src = w.addr; a.length = w.words; a.quarters = w.quarters;
    return prog_.bind(a).b();
}

// A streamed role's segments in image order {bands, band words}: one per GEMM command that reads it (W_q, W_o: per
// attention part; gate / up, W_d: per FFN part; lm_head: a decode chunk, a prefill chunk being an aligned part of
// one); none for other roles
std::vector<Llm::Segment> Llm::segments(const std::string& role) const {
    const int E = cfg_.n_embd, dh = cfg_.dh, KVD = cfg_.n_kv * dh;
    const uint32_t be = 33u * uint32_t(E / 32);         // a band of a K = E matrix
    std::vector<Segment> s;
    if (role == "attn_q")
        for (const AttnPart& a : attn_) s.push_back({a.kvs * a.grp * dh / 32, be});
    else if (role == "attn_k" || role == "attn_v") s.push_back({KVD / 32, be});
    else if (role == "attn_output")
        for (const AttnPart& a : attn_) s.push_back({E / 32, 33u * uint32_t(a.kvs * a.grp * dh / 32)});
    else if (role == "ffn_gate_up")
        for (const FfnPart& f : ffn_) s.push_back({2 * f.bands, be});
    else if (role == "ffn_down")
        for (const FfnPart& f : ffn_) s.push_back({E / 32, 33u * uint32_t(f.bands)});
    else if (role == "output") {
        const int c = lm_cols(1) / 32, nb = cfg_.vocab / 32;
        for (int b = 0; b < nb; b += c) s.push_back({std::min(c, nb - b), be});
    } else if (role == "ident") s.push_back({KVD / 32, 33u * uint32_t(KVD / 32)});
    return s;
}

// the role's segments in quarters (strides set, addresses from 0; bytes: their extent), or none: the setting off, a
// role that is not streamed, or lm_head chunks that do not start at a multiple of 4 bands (lm_chunk % 128)
std::vector<Llm::Segment> Llm::quarter_layout(const std::string& role, size_t& bytes) const {
    bytes = 0;
    if (!weight_quarters() || (role == "output" && cfg_.lm_chunk % 128)) return {};
    std::vector<Segment> segs = segments(role);
    for (Segment& s : segs) {
        s.stride = quarter_units(s.bands, s.band_words);
        s.addr = bytes;
        bytes += 4 * size_t(s.stride) * 16384;
    }
    return segs;
}

// a segment's bands (in order) to its four quarters
void Llm::write_quarters(const Segment& s, const Word* bands) {
    std::vector<Word> q;
    for (int r = 0; r < 4 && r < s.bands; ++r) {
        q.clear();
        for (int b = r; b < s.bands; b += 4)
            q.insert(q.end(), bands + size_t(b) * s.band_words, bands + size_t(b + 1) * s.band_words);
        dev_.write(s.addr + uint64_t(r) * s.stride * 16384, q.data(), q.size() * 32);
    }
}

// a P32 image (its bands in order) into device memory: in quarters for a streamed role (quarter_layout), else as it is
WeightImage Llm::place(const std::string& role, const std::vector<Word>& image) {
    WeightImage img;
    img.words = uint32_t(image.size());
    size_t bytes = 0;
    std::vector<Segment> segs = quarter_layout(role, bytes);
    if (segs.empty()) {
        img.addr = dev_.alloc(image.size() * 32);
        dev_.write(img.addr, image.data(), image.size() * 32);
        dev_bytes_ += image.size() * 32;
        return img;
    }
    img.addr = dev_.alloc(bytes);
    size_t from = 0;
    for (Segment& s : segs) {
        s.addr += img.addr;
        write_quarters(s, &image[from]);
        from += size_t(s.bands) * s.band_words;
    }
    if (from != image.size()) throw std::logic_error("segments of " + role + ": not the image");
    placed_[img.addr] = segs;
    dev_bytes_ += bytes;
    return img;
}

// bands band0 .. band0 + bands - 1 of segment seg of a role's image, as a stream binding takes them
Llm::Span Llm::span(const WeightImage& w, const std::string& role, int seg, int band0, int bands) const {
    const auto it = placed_.find(w.addr);
    if (it != placed_.end()) {
        const Segment& s = it->second.at(size_t(seg));
        if (band0 % 4 || band0 + bands > s.bands) throw std::logic_error(role + ": bands outside a quartered segment");
        return {s.addr + uint64_t(band0 / 4) * s.band_words * 32, uint32_t(bands) * s.band_words, s.stride};
    }
    const Segment& s = shapes_.at(role).at(size_t(seg));
    return {w.addr + (s.addr + uint64_t(band0) * s.band_words) * 32, uint32_t(bands) * s.band_words, 0};
}

void Llm::build_inputs(int m) {
    Program& p = prog_;
    const int E = cfg_.n_embd;
    p.role("inputs");
    const uint8_t g = p.next_gen(I_X3);
    x_region_ = 0;
    Bound X = p.bind({I_X3, g, map_.XA, f32_extent(m, E), Order::F32, Kind::Static, true});
    Bound XG = p.bind({uint8_t(I_X3 + 1), g, map_.XGA, round8(uint32_t(Layout(m, E).words())), Order::P8, Kind::Static, true});
    Bound R = p.bind({uint8_t(I_X3 + 2), g, map_.RA, 8, Order::P8, Kind::Static, true});
    p.move_in(X.b(), 0, uint32_t(m * E / 8), stage_, true);
    p.move_in(XG.b(), 0, XG.extent, stage_ + map_.XG_ST * 32, true);
    p.move_in(R.b(), 0, 8, stage_ + map_.R_ST * 32, true);
    Bound T = p.bind({I_T, 0, map_.TB, round8(uint32_t(Layout(m, cfg_.dh).words())), Order::P8, Kind::Static, true});
    p.move_in(T.b(), 0, T.extent, stage_ + map_.T_ST * 32, true);
    // Diagnosis (docs/reports/20261023_clb_resources): G3RT_INPUT_FENCE=1 lets the inputs land before the first
    // GEMM reads them, instead of that GEMM following the MOVE_IN watermarks.
    static const bool input_fence = [] {
        const char* v = std::getenv("G3RT_INPUT_FENCE");
        return v && std::strcmp(v, "1") == 0;
    }();
    if (input_fence) p.fence();
}

// q, k, v of a layer. hw_kv (m == 1, or a tile that starts a band: pos % 32 == 0): the K band and V^T cell of
// position pos come in (a band that starts: zeros, so no row of an earlier context stays in it), k and v append the
// m rows to them, they go back, a FENCE orders the cache. Otherwise k (P8 rows) and v (FP32 rows) go to the staging
// area (the last MOVE_OUT reports), for the host to merge (merge_kv). kv_one: K and V^T (K rows and V rows) in one
// region, k's in and out before v's. qk_norm on the Node: the q and k GEMMs normalize each head (an FP32 RoPE table
// with the layer's gain row at idx + 1 each).
void Llm::build_qkv(int l, int m, int pos, bool hw_kv) {
    Program& p = prog_;
    if (qk_node_) qk_gains(l);
    const LayerImages& L = layers_[size_t(l)];
    const int E = cfg_.n_embd, NKV = cfg_.n_kv, dh = cfg_.dh;
    const int KVD = NKV * dh;
    const uint32_t band = uint32_t(Layout(32, dh, true).words()), tail = round8(band);
    const int j = pos / 32, r = pos % 32;
    const uint16_t xg = binding(uint8_t(I_X3 + 1), p.gen_of(uint8_t(I_X3 + 1)));
    // RoPE's C: the tile's P8 table (T), or the FP32 table (from T32_ST) and the gain row at idx + 1, C's generation
    auto rope_c = [&](uint8_t idx, uint32_t tb, uint32_t gb, const WeightImage& gain) {
        if (!qk_node_) return binding(I_T, p.gen_of(I_T));
        Bound T = p.bind({idx, 0, tb, f32_extent(m, dh), Order::F32, Kind::Static, true});
        p.move_in(T.b(), 0, uint32_t(m * dh / 8), stage_ + uint64_t(map_.T32_ST) * 32, true);
        Bound G = p.bind({uint8_t(idx + 1), T.gen, gb, round8(uint32_t(dh / 8)), Order::F32, Kind::Static, true});
        p.move_in(G.b(), 0, uint32_t(dh / 8), gain.addr, true);
        return T.b();
    };
    p.role("q");
    GemmArgs g;
    g.a = xg; g.c = rope_c(I_T, map_.TQB, map_.GQB, L.g_qa); g.m = m; g.k = E; g.epi = Epi::Rope; g.fold = true;
    g.eps = cfg_.eps_profile; g.qk_norm = qk_node_;
    for (size_t c = 0; c < attn_.size(); ++c) {         // one q GEMM per attention part (its rows of W_q)
        const AttnPart& a = attn_[c];
        const int heads = a.kvs * a.grp;
        Bound Q = p.bind({q_index(c), 0, map_.QB + a.ub, round8(uint32_t(Layout(m, dh, false, heads).words()))});
        g.b = stream(span(L.wq, "attn_q", int(c), 0, heads * dh / 32));
        g.d = Q.b(); g.n = heads * dh; g.d_heads = heads;
        p.gemm(g);
    }
    const uint16_t tk = qk_node_ ? rope_c(I_TK, map_.TKB, map_.GKB, L.g_ka) : g.c;
    GemmArgs gk = g, gv = g;
    gk.c = tk; gk.n = KVD; gk.d_heads = NKV;
    gv.c = 0; gv.epi = Epi::None; gv.qk_norm = false; gv.n = KVD; gv.d_heads = NKV;
    auto k_gemm = [&](uint16_t d) { p.role("k"); gk.b = stream(span(L.wk, "attn_k", 0, 0, KVD / 32)); gk.d = d; p.gemm(gk); };
    auto v_gemm = [&](uint16_t d) { p.role("v"); gv.b = stream(span(L.wv, "attn_v", 0, 0, KVD / 32)); gv.d = d; p.gemm(gv); };
    if (hw_kv) {
        const uint64_t off = uint64_t(j) * band * 32;
        auto cell_in = [&](uint8_t idx, uint32_t base, Order order, uint64_t img) {
            Bound B = p.bind({idx, 0, base, uint32_t(NKV) * tail, order, Kind::Static, true, NKV});
            for (int h = 0; h < NKV; ++h)
                p.move_in(B.b(h), 0, band, r == 0 ? zero_ : img + uint64_t(h) * kv_words_ * 32 + off, true);
        };
        auto append = [&](uint8_t idx, uint32_t base, Order order) {
            BindArgs a{idx, 0, base, uint32_t(NKV) * tail, order, Kind::Append, false, NKV};
            a.rows = uint32_t(r);
            return p.bind(a);
        };
        auto cell_out = [&](const Bound& B, uint64_t img) {
            p.role("kv_out");
            for (int h = 0; h < NKV; ++h) p.move_out(B.b(h), 0, band, img + uint64_t(h) * kv_words_ * 32 + off);
        };
        p.role("kv_in");
        if (map_.kv_one) {
            cell_in(map_.I_KT, map_.KTB, Order::P32, L.kimg);
            const Bound KA = append(map_.I_KA, map_.KTB, Order::P32);
            k_gemm(KA.b());
            cell_out(KA, L.kimg);
            p.role("kv_in");
            cell_in(map_.I_VT, map_.VTB, Order::VT, L.vimg);   // its BIND waits for the K bands' MOVE_OUTs
            const Bound VA = append(map_.I_VA, map_.VTB, Order::VT);
            v_gemm(VA.b());
            cell_out(VA, L.vimg);
        } else {
            cell_in(map_.I_KT, map_.KTB, Order::P32, L.kimg);
            cell_in(map_.I_VT, map_.VTB, Order::VT, L.vimg);
            const Bound KA = append(map_.I_KA, map_.KTB, Order::P32), VA = append(map_.I_VA, map_.VTB, Order::VT);
            k_gemm(KA.b());
            v_gemm(VA.b());
            cell_out(KA, L.kimg);
            cell_out(VA, L.vimg);
        }
        p.fence();
        return;
    }
    gv.d_heads = 1;
    const uint32_t kw = uint32_t(Layout(m, dh, false, NKV).words());
    auto k_rows = [&] {
        const Bound KR = p.bind({I_KR, 0, map_.KTB, round8(kw)});
        k_gemm(KR.b());
        return KR;
    };
    auto v_rows = [&] {
        const Bound VR = p.bind({I_VR, 0, map_.VTB, f32_extent(m, KVD), Order::F32});
        v_gemm(VR.b());
        return VR;
    };
    if (map_.kv_one) {
        const Bound KR = k_rows();
        p.role("kv_out");
        p.move_out(KR.b(), 0, kw, stage_ + map_.KR_ST * 32);
        const Bound VR = v_rows();                       // its BIND waits for the K rows' MOVE_OUT
        p.role("kv_out");
        p.move_out(VR.b(), 0, uint32_t(m * KVD / 8), stage_ + map_.VR_ST * 32, true);
        return;
    }
    const Bound KR = k_rows(), VR = v_rows();
    p.role("kv_out");
    p.move_out(KR.b(), 0, kw, stage_ + map_.KR_ST * 32);
    p.move_out(VR.b(), 0, uint32_t(m * KVD / 8), stage_ + map_.VR_ST * 32, true);
}

// The host's insertion of a tile's K rows and V values into the cache of layer l, token by token as the Node's
// decode append does it (K: the row's codes and scales; V^T: a cell's first token starts the cell, a later one
// rewrites it: scale per dim = the larger of the old and the token's, old codes re-rounded to it), so a prefill
// leaves the same cache as decode steps would.
void Llm::merge_kv(int l, int m, int pos) {
    const int NKV = cfg_.n_kv, dh = cfg_.dh, G = dh / 32, KVD = NKV * dh;
    const Layout kl(m, dh, false, NKV);
    std::vector<Word> kw(static_cast<size_t>(kl.words())), vw(static_cast<size_t>(m) * KVD / 8);
    dev_.read(stage_ + map_.KR_ST * 32, kw.data(), kw.size() * 32);
    dev_.read(stage_ + map_.VR_ST * 32, vw.data(), vw.size() * 32);
    std::vector<uint8_t> kc(static_cast<size_t>(NKV) * m * dh), ks(static_cast<size_t>(NKV) * m * G);
    kl.unpack(kw.data(), kc.data(), ks.data());
    std::vector<uint32_t> v(static_cast<size_t>(m) * KVD);
    f32_unpack(vw.data(), m, KVD, v.data());
    insert_kv(l, m, pos, kc.data(), ks.data(), v.data());
}

// K rows (codes [NKV m][dh], scales [NKV m][dh / 32], KV-head-major) and V values ([m][NKV dh] FP32) of a tile into the
// cache of layer l (merge_kv). Per band: the KV heads' K bands and V^T cells read, rewritten on up to
// host_threads() threads (a head each), written back.
void Llm::insert_kv(int l, int m, int pos, const uint8_t* kc, const uint8_t* ks, const uint32_t* v) {
    const LayerImages& L = layers_[size_t(l)];
    const int NKV = cfg_.n_kv, dh = cfg_.dh, G = dh / 32, KVD = NKV * dh;
    const Layout band_l(32, dh, true);
    const uint32_t band = uint32_t(band_l.words());
    std::vector<Word> kw(size_t(NKV) * band), vw(size_t(NKV) * band);
    for (int i = 0; i < m;) {
        const int j = (pos + i) / 32;
        int e = i;
        while (e < m && (pos + e) / 32 == j) ++e;                // tokens i..e-1 in band j
        const uint64_t off = uint64_t(j) * band * 32;
        for (int h = 0; h < NKV; ++h) {
            dev_.read(L.kimg + uint64_t(h) * kv_words_ * 32 + off, &kw[size_t(h) * band], band * 32);
            dev_.read(L.vimg + uint64_t(h) * kv_words_ * 32 + off, &vw[size_t(h) * band], band * 32);
        }
        parallel_for(NKV, [&](int h) {
            // K rows
            for (int t = i; t < e; ++t)
                band_l.set_row(&kw[size_t(h) * band], (pos + t) % 32, &kc[(size_t(h) * m + t) * dh],
                               &ks[(size_t(h) * m + t) * G]);
            // V^T cell, a dim at a time
            std::vector<uint8_t> vc(32u * size_t(dh)), vs(static_cast<size_t>(dh));
            vt_unpack(&vw[size_t(h) * band], dh, 32, vc.data(), vs.data());
            for (int d = 0; d < dh; ++d) {
                uint8_t* row = &vc[size_t(d) * 32];
                uint8_t s = vs[size_t(d)];
                for (int t = i; t < e; ++t) {
                    const int k = (pos + t) % 32;
                    const uint32_t x = v[size_t(t) * KVD + size_t(h) * dh + d];
                    if (k == 0) {
                        s = g3::choose_scale(x & 0x7FFFFFFFu);
                        std::memset(row, 0, 32);
                        row[0] = g3::quantize_e4m3(x, s);
                        continue;
                    }
                    uint64_t any = 0;                            // the codes' bits, OR-ed (zero: every code +-0)
                    for (int q = 0; q < 32; q += 8) { uint64_t u; std::memcpy(&u, row + q, 8); any |= u; }
                    const uint8_t sn = g3::vt_tail_scale(s, (any & 0x7F7F7F7F7F7F7F7Full) == 0, x);
                    if (sn != s)                                 // vt_requant_code keeps every code when the scale stays
                        for (int q = 0; q < 32; ++q) row[q] = g3::vt_requant_code(row[q], s, sn);
                    row[k] = g3::quantize_e4m3(x, sn);
                    s = sn;
                }
                vs[size_t(d)] = s;
            }
            const std::vector<Word> cw = vt_pack(vc.data(), vs.data(), dh, 32);
            std::copy(cw.begin(), cw.end(), vw.begin() + std::ptrdiff_t(size_t(h) * band));
        });
        for (int h = 0; h < NKV; ++h) {
            dev_.write(L.kimg + uint64_t(h) * kv_words_ * 32 + off, &kw[size_t(h) * band], band * 32);
            dev_.write(L.vimg + uint64_t(h) * kv_words_ * 32 + off, &vw[size_t(h) * band], band * 32);
        }
        i = e;
    }
}

// qk_norm: q (per attention part), k and (no hw_kv: v) of a layer in FP32 (fold consumers of X g0, no epilogue), moved
// out to the staging area for qk_host. hw_kv: every MOVE_OUT reports, their tags returned (run_until waits for them
// while the Node runs build_v_append); else the last one reports.
std::vector<uint8_t> Llm::build_qkv_f32(int l, int m, bool hw_kv) {
    Program& p = prog_;
    const LayerImages& L = layers_[size_t(l)];
    const int E = cfg_.n_embd, dh = cfg_.dh, KVD = cfg_.n_kv * dh;
    GemmArgs g;
    g.a = binding(uint8_t(I_X3 + 1), p.gen_of(uint8_t(I_X3 + 1))); g.m = m; g.k = E; g.epi = Epi::None; g.fold = true;
    g.eps = cfg_.eps_profile;
    std::vector<uint8_t> moved;
    uint32_t st = map_.QF_ST;           // the m rows of each part's q, then k, one after another (one read)
    p.role("q");
    for (size_t c = 0; c < attn_.size(); ++c) {
        const AttnPart& a = attn_[c];
        const int heads = a.kvs * a.grp;
        Bound Q = p.bind({uint8_t(I_QF + c), 0, map_.QF, f32_extent(m, heads * dh), Order::F32});   // a part at a time
        g.b = stream(span(L.wq, "attn_q", int(c), 0, heads * dh / 32));
        g.d = Q.b(); g.n = heads * dh;
        p.gemm(g);
        const uint8_t t = p.move_out(Q.b(), 0, uint32_t(m * heads * dh / 8), stage_ + uint64_t(st) * 32, hw_kv);
        if (hw_kv) moved.push_back(t);
        st += uint32_t(m * heads * dh / 8);
    }
    p.role("k");
    Bound K = p.bind({I_KF, 0, map_.KF, f32_extent(m, KVD), Order::F32});
    g.b = stream(span(L.wk, "attn_k", 0, 0, KVD / 32)); g.d = K.b(); g.n = KVD;
    p.gemm(g);
    p.role("kv_out");
    const uint8_t t = p.move_out(K.b(), 0, uint32_t(m * KVD / 8), stage_ + uint64_t(st) * 32, hw_kv);
    if (hw_kv) { moved.push_back(t); return moved; }
    p.role("v");
    Bound V = p.bind({I_VR, 0, map_.VTB, f32_extent(m, KVD), Order::F32});
    g.b = stream(span(L.wv, "attn_v", 0, 0, KVD / 32)); g.d = V.b();
    p.gemm(g);
    p.role("kv_out");
    p.move_out(V.b(), 0, uint32_t(m * KVD / 8), stage_ + uint64_t(map_.VR_ST) * 32, true);
    return moved;
}

// qk_norm: every layer's q and k gains [2 dh] in W_q / W_k row order (qk_gain_; arrange's head_row); and for layer
// l >= 0 (the Node normalizing q and k) its two as FP32 rows in device memory (g_qa, g_ka)
void Llm::qk_gains(int l) {
    const int dh = cfg_.dh, half = dh / 2;
    if (qk_gain_.empty()) {
        std::vector<Word> w(size_t(dh / 8));
        std::vector<uint32_t> b(static_cast<size_t>(dh));
        qk_gain_.resize(size_t(cfg_.n_layer));
        for (int i = 0; i < cfg_.n_layer; ++i) {
            std::vector<float>& gain = qk_gain_[size_t(i)];
            gain.resize(size_t(2 * dh));
            for (int s = 0; s < 2; ++s) {
                dev_.read((s ? layers_[size_t(i)].g_k : layers_[size_t(i)].g_q).addr, w.data(), w.size() * 32);
                f32_unpack(w.data(), 1, dh, b.data());
                for (int r = 0; r < dh; ++r) {
                    const int d = cfg_.rope_neox ? (r & 1 ? r / 2 + half : r / 2) : r;    // arrange's head_row
                    std::memcpy(&gain[size_t(s * dh + r)], &b[size_t(d)], 4);
                }
            }
        }
    }
    if (l < 0) return;
    LayerImages& L = layers_[size_t(l)];
    if (L.g_qa.words) return;
    L.g_qa = upload_f32_row(&qk_gain_[size_t(l)][0], dh);
    L.g_ka = upload_f32_row(&qk_gain_[size_t(l)][size_t(dh)], dh);
}

// qk_norm: the cos, sin of the dh / 2 rotated pairs at positions pos .. pos + m - 1 (qk_rope_), in double rounded once
void Llm::qk_rope_rows(int pos, int m) {
    const int dh = cfg_.dh, half = dh / 2;
    if (qk_rope_.size() < size_t(pos + m)) qk_rope_.resize(size_t(cfg_.ctx));
    for (int i = 0; i < m; ++i) {
        std::vector<float>& cs = qk_rope_[size_t(pos + i)];
        if (!cs.empty()) continue;
        cs.resize(size_t(dh));
        for (int k = 0; k < half; ++k) {
            const double a = double(pos + i) * std::pow(double(cfg_.rope_base), -2.0 * k / dh);
            cs[size_t(2 * k)] = float(std::cos(a));
            cs[size_t(2 * k + 1)] = float(std::sin(a));
        }
    }
}

// qk_norm, after build_qkv_f32 ran: each head's q and k RMS-normalized (gain, qk_eps), rotated (the pairs the Node's
// RoPE rotates: rows 2i, 2i + 1 of a head in W_q / W_k order) and quantized; q to the staging area (MXFP8 P8 per
// attention part, for build_q_in); k (hw_kv) to the staging area (MXFP8 P8 [m][KVD], for build_k_append), else k
// and v into the cache (insert_kv). Double-precision arithmetic, rounded once to FP32, so that every platform's host
// computes the same values.
void Llm::qk_host(int l, int m, int pos, bool hw_kv) {
    const int dh = cfg_.dh, half = dh / 2, NKV = cfg_.n_kv, KVD = NKV * dh;
    qk_gains(-1);
    qk_rope_rows(pos, m);
    // a head's dh values x (FP32 bits) -> y: x / rms(x) * gain, then each pair (2k, 2k + 1) rotated by position p
    auto head = [&](const uint32_t* x, const float* gain, int p, uint32_t* y) {
        double t[256];                                  // dh <= 256 (configure_memory)
        double s = 0;
        for (int d = 0; d < dh; ++d) { float f; std::memcpy(&f, &x[d], 4); t[d] = f; s += double(f) * f; }
        const double r = 1.0 / std::sqrt(s / dh + double(cfg_.qk_eps));
        for (int d = 0; d < dh; ++d) t[d] = double(float(t[d] * r * double(gain[d])));
        const std::vector<float>& cs = qk_rope_[size_t(p)];
        for (int k = 0; k < half; ++k) {
            const double a = t[2 * k], b = t[2 * k + 1], c = cs[size_t(2 * k)], sn = cs[size_t(2 * k + 1)];
            const float y0 = float(a * c - b * sn), y1 = float(a * sn + b * c);
            std::memcpy(&y[2 * k], &y0, 4);
            std::memcpy(&y[2 * k + 1], &y1, 4);
        }
    };
    // rows x dh values -> MXFP8 codes and scales at c, s (blocks of 32 within a head: a head quantized alone is the
    // same as in its matrix)
    auto quantize_head = [&](const uint32_t* src, int rows, uint8_t* c, uint8_t* s) {
        thread_local std::vector<uint8_t> hc, hs;
        quantize_fast(src, rows, dh, hc, hs);
        std::memcpy(c, hc.data(), hc.size());
        std::memcpy(s, hs.data(), hs.size());
    };
    // the heads of a tile on up to host_threads() threads; a decode step's run on this one (handing them out gained
    // 0.6% on InternVL, docs/reports/20261024_decode_host)
    auto each_head = [&](int heads, const std::function<void(int)>& f) {
        if (m > 1) parallel_for(heads, f);
        else for (int h = 0; h < heads; ++h) f(h);
    };
    const int G = dh / 32;
    const std::vector<float>& gain = qk_gain_[size_t(l)];
    // q of every part and k, as build_qkv_f32 moved them out: m rows each, one after another from QF_ST
    uint32_t kf = 0;
    for (const AttnPart& a : attn_) kf += uint32_t(m * a.kvs * a.grp * dh / 8);
    std::vector<Word> w(size_t(kf) + size_t(m) * KVD / 8);
    dev_.read(stage_ + uint64_t(map_.QF_ST) * 32, w.data(), w.size() * 32);
    std::vector<uint32_t> x, y;
    std::vector<uint8_t> codes, scales;
    // the MXFP8 images to write: q of every part at Q8_ST + its ub, k (hw_kv) at K8_ST
    const uint32_t q8_words = map_.K8_ST - map_.Q8_ST;
    const uint32_t k8_words = hw_kv ? round8(uint32_t(Layout(m, KVD).words())) : 0;
    std::vector<Word> out(size_t(q8_words) + k8_words);
    // q: per attention part [m][heads dh] -> [heads m][dh] (head-major), MXFP8 P8
    uint32_t qo = 0;
    for (const AttnPart& a : attn_) {
        const int heads = a.kvs * a.grp, QD = heads * dh;
        x.resize(size_t(m) * QD); y.resize(x.size());
        codes.resize(y.size()); scales.resize(y.size() / 32);
        f32_unpack(&w[qo], m, QD, x.data());
        qo += uint32_t(m * QD / 8);
        each_head(heads, [&](int h) {
            const size_t row0 = size_t(h) * m;
            for (int i = 0; i < m; ++i)
                head(&x[size_t(i) * QD + size_t(h) * dh], gain.data(), pos + i, &y[(row0 + i) * dh]);
            quantize_head(&y[row0 * dh], m, &codes[row0 * dh], &scales[row0 * G]);
        });
        Layout(m, dh, false, heads).pack_into(codes.data(), scales.data(), &out[a.ub]);
    }
    // k: [m][NKV dh] in place (the append's A: P8 [m][KVD]), or [NKV m][dh] for insert_kv
    x.resize(size_t(m) * KVD); y.resize(x.size());
    codes.resize(y.size()); scales.resize(y.size() / 32);
    f32_unpack(&w[kf], m, KVD, x.data());
    each_head(NKV, [&](int h) {
        for (int i = 0; i < m; ++i) {
            const size_t to = hw_kv ? size_t(i) * KVD + size_t(h) * dh : (size_t(h) * m + i) * dh;
            head(&x[size_t(i) * KVD + size_t(h) * dh], gain.data() + dh, pos + i, &y[to]);
            if (hw_kv) quantize_head(&y[to], 1, &codes[to], &scales[to / 32]);
        }
        if (!hw_kv) quantize_head(&y[size_t(h) * m * dh], m, &codes[size_t(h) * m * dh], &scales[size_t(h) * m * G]);
    });
    if (hw_kv) {
        Layout(m, KVD).pack_into(codes.data(), scales.data(), &out[q8_words]);
        dev_.write(stage_ + uint64_t(map_.Q8_ST) * 32, out.data(), out.size() * 32);
        return;
    }
    dev_.write(stage_ + uint64_t(map_.Q8_ST) * 32, out.data(), out.size() * 32);
    std::vector<uint32_t> v(static_cast<size_t>(m) * KVD);
    w.resize(size_t(m) * KVD / 8);
    dev_.read(stage_ + uint64_t(map_.VR_ST) * 32, w.data(), w.size() * 32);
    f32_unpack(w.data(), m, KVD, v.data());
    insert_kv(l, m, pos, codes.data(), scales.data(), v.data());
}

// qk_norm, hw_kv, two halves of build_qkv's append. The V^T cell of position pos comes in (a cell that starts:
// zeros), v appended by its GEMM (it needs only X g0), the cell goes back; it follows build_qkv_f32 in one program, so
// the Node runs it while the host normalizes q and k.
void Llm::build_v_append(int l, int m, int pos) {
    Program& p = prog_;
    const LayerImages& L = layers_[size_t(l)];
    const int E = cfg_.n_embd, NKV = cfg_.n_kv, dh = cfg_.dh, KVD = NKV * dh;
    const uint32_t band = uint32_t(Layout(32, dh, true).words()), tail = round8(band);
    const int j = pos / 32, r = pos % 32;
    p.role("kv_in");
    Bound VT = p.bind({map_.I_VT, 0, map_.VTB, uint32_t(NKV) * tail, Order::VT, Kind::Static, true, NKV});
    for (int h = 0; h < NKV; ++h)
        p.move_in(VT.b(h), 0, band, r == 0 ? zero_ : L.vimg + (uint64_t(h) * kv_words_ + uint64_t(j) * band) * 32, true);
    p.role("v");
    BindArgs va{map_.I_VA, 0, map_.VTB, uint32_t(NKV) * tail, Order::VT, Kind::Append, false, NKV};
    va.rows = uint32_t(r);
    Bound VA = p.bind(va);
    GemmArgs gv;
    gv.a = binding(uint8_t(I_X3 + 1), p.gen_of(uint8_t(I_X3 + 1))); gv.b = stream(span(L.wv, "attn_v", 0, 0, KVD / 32)); gv.d = VA.b();
    gv.m = m; gv.n = KVD; gv.k = E; gv.d_heads = NKV; gv.fold = true; gv.eps = cfg_.eps_profile;
    p.gemm(gv);
    p.role("kv_out");
    for (int h = 0; h < NKV; ++h)
        p.move_out(VA.b(h), 0, band, L.vimg + (uint64_t(h) * kv_words_ + uint64_t(j) * band) * 32);
}

// The K band of position pos comes in, the host's k (P8 [m][KVD] at K8_ST) appended by the identity GEMM (exact: the
// products are the MXFP8 values, re-quantized to the same values), the band goes back; a FENCE orders the cache (and
// build_v_append's cells)
void Llm::build_k_append(int l, int m, int pos) {
    Program& p = prog_;
    const LayerImages& L = layers_[size_t(l)];
    const int NKV = cfg_.n_kv, dh = cfg_.dh, KVD = NKV * dh;
    const uint32_t band = uint32_t(Layout(32, dh, true).words()), tail = round8(band);
    const int j = pos / 32, r = pos % 32;
    p.role("kv_in");
    Bound KT = p.bind({map_.I_KT, 0, map_.KTB, uint32_t(NKV) * tail, Order::P32, Kind::Static, true, NKV});
    for (int h = 0; h < NKV; ++h)
        p.move_in(KT.b(h), 0, band, r == 0 ? zero_ : L.kimg + (uint64_t(h) * kv_words_ + uint64_t(j) * band) * 32, true);
    const uint32_t k8 = round8(uint32_t(Layout(m, KVD).words()));
    Bound KN = p.bind({I_KF, 0, map_.KN, k8, Order::P8, Kind::Static, true});
    p.move_in(KN.b(), 0, k8, stage_ + uint64_t(map_.K8_ST) * 32, true);
    p.role("k");
    BindArgs ka{map_.I_KA, 0, map_.KTB, uint32_t(NKV) * tail, Order::P32, Kind::Append, false, NKV};
    ka.rows = uint32_t(r);
    Bound KA = p.bind(ka);
    GemmArgs g;
    g.a = KN.b(); g.b = stream(span(ident_, "ident", 0, 0, KVD / 32)); g.d = KA.b(); g.m = m; g.n = KVD; g.k = KVD; g.d_heads = NKV;
    p.gemm(g);
    p.role("kv_out");
    for (int h = 0; h < NKV; ++h)
        p.move_out(KA.b(h), 0, band, L.kimg + (uint64_t(h) * kv_words_ + uint64_t(j) * band) * 32);
    p.fence();
}

// qk_norm: the host's MXFP8 q of every attention part into its Q (the ATTN of build_rest reads it)
void Llm::build_q_in(int m) {
    Program& p = prog_;
    p.role("q_in");
    for (size_t c = 0; c < attn_.size(); ++c) {
        const AttnPart& a = attn_[c];
        const uint32_t words = round8(uint32_t(Layout(m, cfg_.dh, false, a.kvs * a.grp).words()));
        Bound Q = p.bind({q_index(c), 0, map_.QB + a.ub, words, Order::P8, Kind::Static, true});
        p.move_in(Q.b(), 0, words, stage_ + uint64_t(map_.Q8_ST + a.ub) * 32, true);
    }
}

// attention, o, gate / up, down of a layer (m tokens at pos..; the cache holds pos + m keys)
void Llm::build_rest(int l, int m, int pos) {
    Program& p = prog_;
    const LayerImages& L = layers_[size_t(l)];
    const int E = cfg_.n_embd, dh = cfg_.dh;
    const uint32_t xw = f32_extent(m, E), xgw = round8(uint32_t(Layout(m, E).words()));
    p.role("attn");
    for (size_t c = 0; c < attn_.size(); ++c) {         // one ATTN per attention part, K / V streamed from the cache
        const AttnPart& a = attn_[c];
        const int heads = a.kvs * a.grp;
        Bound O = p.bind({o_index(c), 0, map_.OB + a.ub, round8(uint32_t(Layout(m, dh, false, heads).words()))});
        BindArgs ks{I_KS, 0, uint32_t(map_.STREAMB + 8 * (I_KS - 0xF8)), 8, Order::P32, Kind::Stream};
        ks.src = L.kimg + uint64_t(a.kv0) * kv_words_ * 32; ks.length = kv_words_;
        BindArgs vs{I_VS, 0, uint32_t(map_.STREAMB + 8 * (I_VS - 0xF8)), 8, Order::VT, Kind::Stream};
        vs.src = L.vimg + uint64_t(a.kv0) * kv_words_ * 32; vs.length = kv_words_;
        Bound KS = p.bind(ks), VS = p.bind(vs);
        AttnArgs at;
        at.q = binding(q_index(c), p.gen_of(q_index(c))); at.k = KS.b(); at.v = VS.b(); at.o = O.b(); at.tokens = m;
        at.group = a.grp; at.q_heads = heads; at.keys = pos + m; at.delta = pos; at.dh = dh;
        p.attn(at);
    }
    // A chain of GEMMs y = c + sum_i A_i B_i over K slices: every link but the last writes an FP32 partial sum into
    // the X region its C is not in; the last one writes `family` (D, D + 1 = MXFP8(D g), D + 2 = 1/rms: a fold
    // producer with g at D + 3, bound by the caller).
    struct Link { uint16_t a; Span w; int k, a_heads; };
    // link i of a chain (c: its C, then its D)
    auto link = [&](uint16_t& c, const Link& lk, size_t i, bool last, uint8_t family, uint8_t gen, uint32_t xg_base,
                    uint32_t r_base) {
        const uint32_t base = x_region_ ? map_.XA : map_.XB;
        Bound D;
        if (last) {
            D = p.bind({family, gen, base, xw, Order::F32});
            p.bind({uint8_t(family + 1), gen, xg_base, xgw});
            p.bind({uint8_t(family + 2), gen, r_base, 8});
        } else D = p.bind({uint8_t(I_P + (i & 1)), 0, base, xw, Order::F32});
        GemmArgs gm;
        gm.a = lk.a; gm.b = stream(lk.w); gm.c = c; gm.d = D.b(); gm.m = m; gm.n = E;
        gm.k = lk.k; gm.epi = Epi::Add; gm.a_heads = lk.a_heads; gm.fold = last;
        gm.eps = cfg_.eps_profile;
        p.gemm(gm);
        c = D.b();
        x_region_ ^= 1;
    };
    auto chain = [&](uint16_t c, const std::vector<Link>& links, uint8_t family, uint8_t gen, uint32_t xg_base,
                     uint32_t r_base) {
        for (size_t i = 0; i < links.size(); ++i)
            link(c, links[i], i, i + 1 == links.size(), family, gen, xg_base, r_base);
    };
    // o projection: X2 = O W_o + X (a K slice per attention part); X2 g1, 1/rms (producer; g1 at X2 + 3)
    p.role("o");
    const uint8_t g2 = p.next_gen(I_X2);
    Bound G1 = p.bind({uint8_t(I_X2 + 3), g2, map_.GA, round8(uint32_t(E / 8)), Order::F32, Kind::Static, true});
    p.move_in(G1.b(), 0, uint32_t(E / 8), L.g_ffn.addr, true);
    std::vector<Link> links;
    for (size_t c = 0; c < attn_.size(); ++c) {
        const AttnPart& a = attn_[c];
        const int heads = a.kvs * a.grp;
        links.push_back({binding(o_index(c), p.gen_of(o_index(c))), span(L.wo, "attn_output", int(c), 0, E / 32),
                         heads * dh, heads});
    }
    chain(binding(I_X3, p.gen_of(I_X3)), links, I_X2, g2, map_.XGB, map_.RB);
    // gate / up of FFN part i (a part of whole bands per GEMM) into its H; its link of the down chain
    const uint16_t x2g = binding(uint8_t(I_X2 + 1), g2);
    auto gate_up = [&](size_t i) {
        const FfnPart& f = ffn_[i];
        p.role("gate_up");
        Bound H = p.bind({h_index(i), 0, map_.HB + (map_.ffn_one_h ? 0 : f.ub),
                          round8(uint32_t(Layout(m, 32 * f.bands).words()))});
        GemmArgs gu;
        gu.a = x2g; gu.b = stream(span(L.wgu, "ffn_gate_up", int(i), 0, 2 * f.bands));
        gu.d = H.b(); gu.m = m; gu.n = 32 * f.bands; gu.k = E; gu.epi = Epi::SiluMul; gu.fold = true;
        gu.eps = cfg_.eps_profile;
        p.gemm(gu);
        return Link{H.b(), span(L.wd, "ffn_down", int(i), 0, E / 32), 32 * f.bands, 1};
    };
    // down: X3 = H W_d + X2 (a K slice per FFN part); X3 g0' (the next layer's g0, or the output norm), 1/rms
    const uint8_t g3 = p.next_gen(I_X3);
    const WeightImage& gn = l + 1 < cfg_.n_layer ? layers_[size_t(l + 1)].g_attn : g_out_;
    auto g0_in = [&] {
        p.role("down");
        Bound G0 = p.bind({uint8_t(I_X3 + 3), g3, map_.GB, round8(uint32_t(E / 8)), Order::F32, Kind::Static, true});
        p.move_in(G0.b(), 0, uint32_t(E / 8), gn.addr, true);
    };
    if (!map_.ffn_one_h) {
        links.clear();
        for (size_t i = 0; i < ffn_.size(); ++i) links.push_back(gate_up(i));
        g0_in();
        chain(binding(I_X2, g2), links, I_X3, g3, map_.XGA, map_.RA);
        return;
    }
    // the parts in turn, one H: gate / up of a part, then its link (its BIND waits for the link before to read H)
    g0_in();
    uint16_t c = binding(I_X2, g2);
    for (size_t i = 0; i < ffn_.size(); ++i) {
        const Link lk = gate_up(i);
        p.role("down");
        link(c, lk, i, i + 1 == ffn_.size(), I_X3, g3, map_.XGA, map_.RA);
    }
}

// lm_head: fold consumer of X3 g_out with an FP32 D, lm_cols(m) columns per GEMM; each of `rows` (a row of the tile,
// the device address of its logits) moved out (the last MOVE_OUT reports)
void Llm::build_lm_head(int m, const std::vector<std::pair<int, uint64_t>>& rows) {
    Program& p = prog_;
    const int E = cfg_.n_embd;
    const uint16_t xg = binding(uint8_t(I_X3 + 1), p.gen_of(uint8_t(I_X3 + 1)));
    const int chunk = lm_cols(m), seg = lm_cols(1) / 32;   // columns per GEMM; bands per segment of lm_head's image
    p.role("lm_head");
    for (int c0 = 0, i = 0; c0 < cfg_.vocab; c0 += chunk, ++i) {
        const int n = std::min(chunk, cfg_.vocab - c0);
        Bound LOG = p.bind({uint8_t(I_LOG + (i & 1)), 0, (i & 1) ? map_.LOGB : map_.LOGA, f32_extent(m, chunk), Order::F32});
        GemmArgs h;
        h.a = xg; h.b = stream(span(lm_, "output", c0 / 32 / seg, c0 / 32 % seg, n / 32)); h.d = LOG.b(); h.m = m;
        h.n = n; h.k = E; h.epi = Epi::None; h.fold = true; h.eps = cfg_.eps_profile;
        p.gemm(h);
        for (size_t r = 0; r < rows.size(); ++r)
            p.move_out(LOG.b(), uint32_t(rows[r].first * n / 8), uint32_t(n / 8), rows[r].second + uint64_t(c0) * 4,
                       c0 + chunk >= cfg_.vocab && r + 1 == rows.size());
    }
}

// lm_head columns per GEMM for m rows: a decode step (and any m that fits as many) takes the largest multiple of
// lm_chunk that fits the two regions (f32_extent(tile_, lm_chunk) words each) and a GEMM's MAX_N: 3,072 at tile 32,
// 2,048 at 16; other m take lm_chunk. A decode chunk is a segment of lm_head's image (segments), the others aligned
// parts of one.
int Llm::lm_cols(int m) const {
    const auto fit = [&](int rows) {
        return std::min(MAX_N, int(f32_extent(tile_, cfg_.lm_chunk) / round8(uint32_t(rows)) * 8));
    };
    const int decode = cfg_.lm_chunk * std::max(1, fit(1) / cfg_.lm_chunk);
    return decode <= fit(m) ? decode : cfg_.lm_chunk;
}

void Llm::run(bool final_sync) {
    const uint32_t t0 = profile_ ? dev_.cycles() : 0;
    uint32_t ports[8] = {};             // read-port word counters before the run, then after (profile)
    if (profile_) for (int p = 0; p < 4; ++p) ports[p] = dev_.port_words(p);
    if (host_timing_) timing_.commands += prog_.cmds.size();
    { HostTimer timer(host_timing_ ? &timing_.submit_ms : nullptr); dev_.submit(prog_.cmds); }
    if (!final_sync && !profile_) { prog_.clear(); return; }
    { HostTimer timer(host_timing_ ? &timing_.sync_ms : nullptr); dev_.sync(); }
    const std::vector<g3::Completion> cpl = dev_.take_completions();
    if (profile_) {
        for (int p = 0; p < 4; ++p) ports[4 + p] = dev_.port_words(p);
        profile_->add(prog_.cmds, prog_.roles, t0, cpl, ports);
    }
    prog_.clear();
    for (const g3::Completion& c : cpl)
        if (c.status != g3::ST_OK)
            throw std::runtime_error(std::string("Node command failed: tag ") + std::to_string(c.tag) + " " +
                                     g3::status_name(c.status));
}

// the program submitted, waiting only for the completions of `tags` (the Node runs on with the rest; the next run's
// sync collects it). A profile waits for all, as run(true).
void Llm::run_until(const std::vector<uint8_t>& tags) {
    if (profile_) { run(true); return; }
    if (host_timing_) timing_.commands += prog_.cmds.size();
    { HostTimer timer(host_timing_ ? &timing_.submit_ms : nullptr); dev_.submit(prog_.cmds); }
    { HostTimer timer(host_timing_ ? &timing_.sync_ms : nullptr); for (uint8_t t : tags) dev_.wait(t); }
    prog_.clear();
    for (const g3::Completion& c : dev_.take_completions())
        if (c.status != g3::ST_OK)
            throw std::runtime_error(std::string("Node command failed: tag ") + std::to_string(c.tag) + " " +
                                     g3::status_name(c.status));
}

// tokens of the tile at pos (of `left`): up to the end of pos's 32-token band, so later tiles start bands
int Llm::tile_len(int pos, int left) const {
    const int r = pos % 32;
    return std::min({left,32-r,prefill_tile_});
}

void Llm::forward(const float* x, int m, int pos, float* logits) {
    if (m < 1 || m > tile_ || pos % 32 + m > 32) throw std::invalid_argument("tile of 1..tile_ (the UB map's) tokens inside one band");
    if (pos + m > cfg_.ctx) throw std::out_of_range("positions beyond the KV capacity");
    if (host_timing_) ++timing_.tiles;
    { HostTimer timer(host_timing_ ? &timing_.input_ms : nullptr); host_inputs(x, m, pos); }
    prog_.clear();
    { HostTimer timer(host_timing_ ? &timing_.build_ms : nullptr); build_inputs(m); }
    const bool hw_kv = m == 1 || pos % 32 == 0;
    for (int l = 0; l < cfg_.n_layer; ++l) {
        if (cfg_.qk_norm && !qk_node_) {               // the host's q / k RMSNorm (qk_host)
            std::vector<uint8_t> moved;
            {
                HostTimer timer(host_timing_ ? &timing_.build_ms : nullptr);
                moved = build_qkv_f32(l, m, hw_kv);
                if (hw_kv) build_v_append(l, m, pos);
            }
            if (hw_kv) run_until(moved); else run(true);
            { HostTimer timer(host_timing_ ? &timing_.input_ms : nullptr); qk_host(l, m, pos, hw_kv); }
            HostTimer timer(host_timing_ ? &timing_.build_ms : nullptr);
            if (hw_kv) build_k_append(l, m, pos);
            build_q_in(m);
            build_rest(l, m, pos);
            continue;
        }
        { HostTimer timer(host_timing_ ? &timing_.build_ms : nullptr); build_qkv(l, m, pos, hw_kv); }
        if (!hw_kv) {
            run(true);
            merge_kv(l, m, pos);
        }
        { HostTimer timer(host_timing_ ? &timing_.build_ms : nullptr); build_rest(l, m, pos); }
        if (l == 0 && hw_kv) run(false);    // the first layer runs while the host builds the others
    }
    if (logits) { HostTimer timer(host_timing_ ? &timing_.build_ms : nullptr); build_lm_head(m, {{m - 1, logits_}}); }
    run(true);
    if (logits) { HostTimer timer(host_timing_ ? &timing_.read_ms : nullptr); dev_.read(logits_, logits, size_t(cfg_.vocab) * 4); }
}

void Llm::prefill_rows(const float* x, int n, int pos, const std::vector<int>& out_rows, float* logits) {
    // tiles of up to 32 inside a band; the lm_head of a tile (one GEMM chain) when requested rows fall in it
    const size_t V = size_t(cfg_.vocab);
    if (logits_rows_cap_ < out_rows.size()) {
        logits_rows_ = dev_.alloc(out_rows.size() * V * 4 + 4096);
        logits_rows_cap_ = out_rows.size();
    }
    for (int i = 0, m = 0; i < n; i += m) {
        m = tile_len(pos + i, n - i);
        host_inputs(x + size_t(i) * cfg_.n_embd, m, pos + i);
        prog_.clear();
        build_inputs(m);
        const bool hw_kv = m == 1 || (pos + i) % 32 == 0;
        for (int l = 0; l < cfg_.n_layer; ++l) {
            if (cfg_.qk_norm && !qk_node_) {
                const std::vector<uint8_t> moved = build_qkv_f32(l, m, hw_kv);
                if (hw_kv) { build_v_append(l, m, pos + i); run_until(moved); } else run(true);
                qk_host(l, m, pos + i, hw_kv);
                if (hw_kv) build_k_append(l, m, pos + i);
                build_q_in(m);
                build_rest(l, m, pos + i);
                continue;
            }
            build_qkv(l, m, pos + i, hw_kv);
            if (!hw_kv) {
                run(true);
                merge_kv(l, m, pos + i);
            }
            build_rest(l, m, pos + i);
            if (l == 0 && hw_kv) run(false);    // the first layer runs while the host builds the others
        }
        std::vector<std::pair<int, uint64_t>> rows;
        for (size_t k = 0; k < out_rows.size(); ++k)
            if (out_rows[k] >= i && out_rows[k] < i + m) rows.push_back({out_rows[k] - i, logits_rows_ + k * V * 4});
        if (!rows.empty()) build_lm_head(m, rows);
        run(true);
    }
    if (!out_rows.empty()) dev_.read(logits_rows_, logits, out_rows.size() * V * 4);
}

void Llm::decode(const float* x, int pos, float* logits) { forward(x, 1, pos, logits); }

void Llm::prefill(const float* x, int n, int pos, float* logits) {
    for (int i = 0, m = 0; i < n; i += m) {
        m = tile_len(pos + i, n - i);
        forward(x + size_t(i) * cfg_.n_embd, m, pos + i, i + m >= n ? logits : nullptr);
    }
}

}  // namespace g3rt
