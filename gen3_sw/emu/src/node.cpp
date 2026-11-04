// Gen3 Node v3 command execution: binding table, admission checks (gen3_fpga/node/rtl/g3_ctrl.sv), GEMM
// (g3_gemm_seq.sv + g3f_column.sv output placement), ATTN (g3_attn_seq.sv), VECTOR RMSNORM (gen3_rtl g3_vru.v),
// MOVE_OUT / MOVE_IN / host pushes (g3_ni.sv).
#include "fp_strict.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <thread>

#include "g3emu.h"
#include "g3emu_layout.h"

namespace g3 {

using namespace layout;

// ---------------------------------------------------------------------------------------------------------------
// Basics
// ---------------------------------------------------------------------------------------------------------------
uint64_t Cmd128::field(int lsb, int width) const {
    uint64_t v;
    if (lsb >= 64) v = hi >> (lsb - 64);
    else if (lsb == 0) v = lo;
    else v = (lo >> lsb) | (hi << (64 - lsb));
    return width >= 64 ? v : v & ((uint64_t(1) << width) - 1);
}

const char* status_name(int status) {
    static const char* names[] = {"OK", "UNSUPPORTED", "RESERVED_NONZERO", "RANGE", "BINDING", "ORDER", "CAPACITY",
                                  "POISONED", "ABORTED", "EXECUTION", "QUIESCED", "TIMEOUT", "CONTEXT"};
    return status >= 0 && status < 13 ? names[status] : "?";
}

const Word* ExtMemory::find(uint64_t word_addr) const {
    auto it = pages_.find(word_addr >> PAGE_BITS);
    if (it == pages_.end()) return nullptr;
    const uint64_t i = word_addr & ((1 << PAGE_BITS) - 1);
    return (it->second->present[i >> 6] >> (i & 63)) & 1 ? &it->second->w[i] : nullptr;
}

Word ExtMemory::read(uint64_t word_addr) const {
    const Word* w = find(word_addr);
    return w ? *w : Word{};
}

void ExtMemory::write_words(uint64_t word_addr, const Word* w, size_t n) {
    for (size_t i = 0; i < n; ++i) write(word_addr + i, w[i]);
}

void ExtMemory::read_words(uint64_t word_addr, Word* out, size_t n) const {
    for (size_t i = 0; i < n; ++i) out[i] = read(word_addr + i);
}

void ExtMemory::write(uint64_t word_addr, const Word& w) {
    auto& page = pages_[word_addr >> PAGE_BITS];
    if (!page) page.reset(new Page());
    const uint64_t i = word_addr & ((1 << PAGE_BITS) - 1);
    page->w[i] = w;
    page->present[i >> 6] |= uint64_t(1) << (i & 63);
}

std::vector<uint64_t> ExtMemory::addresses() const {
    std::vector<uint64_t> out;
    for (const auto& kv : pages_)
        for (int i = 0; i < (1 << PAGE_BITS); ++i)
            if ((kv.second->present[i >> 6] >> (i & 63)) & 1) out.push_back((kv.first << PAGE_BITS) | uint64_t(i));
    std::sort(out.begin(), out.end());
    return out;
}

size_t ExtMemory::size() const {
    size_t n = 0;
    for (const auto& kv : pages_)
        for (uint64_t p : kv.second->present) {
            for (uint64_t x = p; x; x &= x - 1) ++n;
        }
    return n;
}

Node::Node(NodeOptions opt) : opt_(opt), ub_(UB_WORDS) {}

void Node::warn(const std::string& s) {
    if (warnings_.size() < 1000) warnings_.push_back(s);
}

void Node::ub_write(uint32_t addr, const Word& w, uint32_t strobe) {
    Word& d = ub_[addr & (UB_WORDS - 1)];
    if (strobe == 0xFFFFFFFFu) { d = w; return; }
    for (int i = 0; i < 32; ++i)
        if ((strobe >> i) & 1) d.b[i] = w.b[i];
}

// a commit event (g3_ctrl event apply): watermark up, final ends the writer
void Node::event(uint8_t idx, uint8_t gen, uint32_t wm, bool final_) {
    Binding& b = urf_[idx];
    if (!b.valid || b.gen != gen) return;
    wm &= 0x7FFF;
    if (wm > b.wm) b.wm = uint16_t(wm);
    if (final_) b.writer = W_NONE;
}

void Node::check_waiters() {
    for (size_t i = 0; i < waiters_.size();) {
        const Waiter& w = waiters_[i];
        const Binding& b = urf_[w.idx];
        int st = -1;
        uint32_t data = 0;
        if (!b.valid || b.gen != w.gen) st = ST_BINDING;
        else if (b.poison) { st = ST_POISONED; data = b.wm; }
        else if (w.mode ? b.writer == W_NONE : (b.writer == W_NONE && b.wm >= b.ext)) { st = ST_OK; data = b.wm; }
        if (st >= 0) {
            complete(w.tag, uint8_t(st), data);
            waiters_.erase(waiters_.begin() + long(i));
        } else {
            ++i;
        }
    }
}

// a source the command reads before its writer finished: valid in hardware (watermark streaming) only when the writer
// is still running; in program order the data is then missing
bool Node::source_pending(uint8_t idx) const {
    const Binding& b = urf_[idx];
    return b.valid && b.writer != W_NONE;
}

namespace {

bool order_src(const Binding& u) { return u.writer == W_NONE && u.wm < u.ext && !u.poison; }

// Words of a UB region (wrapping at 16384) or of external memory, by word offset.
struct WordSource {
    const std::vector<Word>* ub = nullptr;
    uint32_t ub_base = 0;
    const ExtMemory* ext = nullptr;
    uint64_t ext_base = 0;
    Word zero;
    const Word& at(uint64_t off) const {
        if (ub) return (*ub)[(ub_base + off) & (Node::UB_WORDS - 1)];
        const Word* w = ext->find(ext_base + off);
        return w ? *w : zero;
    }
};

// 8 rows of block nb: codes [8][32], scales [8]
void read_block(const WordSource& s, uint32_t nb, uint8_t codes[8][32], uint8_t scales[8]) {
    const Word& sw = s.at(scale_word(nb));
    for (int l = 0; l < 8; ++l) scales[l] = sw.b[scale_byte(nb, l)];
    for (int w = 0; w < 8; ++w) {
        const Word& wd = s.at(block_word(nb) + w);
        for (int l = 0; l < 8; ++l)
            for (int kk = 0; kk < 4; ++kk) codes[l][4 * w + kk] = wd.b[code_byte(kk, l)];
    }
}

// 32 rows of the quad at word offset q: codes [32][32], scales [32]
void read_quad(const WordSource& s, uint64_t q, uint8_t codes[32][32], uint8_t scales[32]) {
    const Word& sw = s.at(q);
    for (int r = 0; r < 32; ++r) scales[r] = sw.b[r];
    for (int p = 0; p < 4; ++p)
        for (int w = 0; w < 8; ++w) {
            const Word& wd = s.at(q + 1 + 8 * p + w);
            for (int l = 0; l < 8; ++l)
                for (int kk = 0; kk < 4; ++kk) codes[8 * p + l][4 * w + kk] = wd.b[code_byte(kk, l)];
        }
}

// f(begin, end) over [0, n), split into contiguous chunks on `threads` threads
template <class F>
void parallel_for(int n, int threads, const F& f) {
    if (threads <= 1 || n < 2) { f(0, n); return; }
    const int chunk = (n + threads - 1) / threads;
    std::vector<std::thread> pool;
    for (int b = chunk; b < n; b += chunk) pool.emplace_back(f, b, std::min(n, b + chunk));
    f(0, std::min(n, chunk));
    for (std::thread& t : pool) t.join();
}

// The 8 payload words and the scale word of a block holding rows codes[l] (l = 0..7)
void pack_block(const uint8_t codes[8][32], Word words[8]) {
    for (int w = 0; w < 8; ++w) {
        words[w] = Word{};
        for (int l = 0; l < 8; ++l)
            for (int kk = 0; kk < 4; ++kk) words[w].b[code_byte(kk, l)] = codes[l][4 * w + kk];
    }
}

}  // namespace

// ---------------------------------------------------------------------------------------------------------------
// Host operations
// ---------------------------------------------------------------------------------------------------------------
void Node::preload(uint64_t byte_addr, const Word* words, size_t n) {
    ext_.write_words(byte_addr >> 5, words, n);
}

void Node::push(uint64_t window_addr, const Word* words, size_t n) {
    for (size_t done = 0; done < n;) {
        const size_t burst = std::min<size_t>(16, n - done);
        const uint64_t off = window_addr + 32 * done - NODE_BASE;
        const uint8_t idx = uint8_t(off >> 19), gen = uint8_t(off >> 27);
        const uint32_t word = uint32_t(off >> 5) & 0x3FFF;
        const Binding& b = urf_[idx];
        if (!(b.valid && b.gen == gen && b.writer == W_EXT)) {
            char msg[160];
            std::snprintf(msg, sizeof msg, "push to idx 0x%02x gen %u word %u: binding not open for external writes "
                          "(the hardware would stall); dropped", idx, gen, word);
            warn(msg);
            return;
        }
        for (size_t i = 0; i < burst; ++i) ub_write(b.base + word + uint32_t(i), words[done + i]);
        const uint32_t wm = (word + uint32_t(burst)) & 0x7FFF;
        event(idx, gen, wm, wm == b.ext);
        done += burst;
        check_waiters();
    }
}

Completion Node::execute(const Cmd128& c) {
    const uint8_t op = uint8_t(c.field(0, 8));
    const uint8_t tag = uint8_t(c.field(8, 8));
    uint8_t st = ST_OK;
    switch (op) {
    case OP_BIND: st = exec_bind(c); break;
    case OP_UNBIND: st = exec_unbind(c); break;
    case OP_MOVE_OUT: st = exec_move_out(c); break;
    case OP_MOVE_IN: st = exec_move_in(c); break;
    case OP_GEMM:
        st = check_gemm(c);
        if (st == ST_OK) run_gemm(c);
        break;
    case OP_ATTN:
        st = check_attn(c);
        if (st == ST_OK) run_attn(c);
        break;
    case OP_VECTOR:
        st = check_vector(c);
        if (st == ST_OK && opt_.vector_stub) {
            event(uint8_t(c.field(64, 8)), uint8_t(c.field(72, 8)), 0, true);
            st = ST_UNSUPPORTED;
        } else if (st == ST_OK) {
            run_vector(c);
        }
        break;
    case OP_FENCE: st = ST_OK; break;
    case OP_ABORT: st = ST_OK; break;   // the target (program order) has finished
    case OP_WAIT: {
        if (opt_.trace)
            std::fprintf(stderr, "cmd WAIT     tag %3u -> placed (idx 0x%02x gen %u mode %d)\n", unsigned(tag),
                         unsigned(c.field(16, 8)), unsigned(c.field(24, 8)), int(c.bit(32)));
        waiters_.push_back({tag, uint8_t(c.field(16, 8)), uint8_t(c.field(24, 8)), c.bit(32)});
        const size_t before = completions_.size();
        check_waiters();
        for (size_t i = before; i < completions_.size(); ++i)
            if (completions_[i].tag == tag) return completions_[i];
        Completion p;
        p.tag = tag;
        p.pending = true;
        return p;
    }
    default: st = ST_UNSUPPORTED; break;
    }
    if (opt_.trace) {
        static const char* ops[] = {"?", "BIND", "UNBIND", "?", "?", "?", "?", "?", "MOVE_OUT", "MOVE_IN"};
        const char* name = op < 10 ? ops[op] : op == OP_GEMM ? "GEMM" : op == OP_ATTN ? "ATTN" : op == OP_VECTOR ?
                           "VECTOR" : op == OP_FENCE ? "FENCE" : op == OP_ABORT ? "ABORT" : "?";
        std::fprintf(stderr, "cmd %-8s tag %3u -> %s   %016llx%016llx\n", name, unsigned(tag), status_name(st),
                     (unsigned long long)c.hi, (unsigned long long)c.lo);
    }
    complete(tag, st);
    const Completion done = completions_.back();
    check_waiters();
    return done;
}

// ---------------------------------------------------------------------------------------------------------------
// BIND / UNBIND / MOVE_OUT / MOVE_IN
// ---------------------------------------------------------------------------------------------------------------
uint8_t Node::exec_bind(const Cmd128& c) {
    const uint32_t idx = uint32_t(c.field(16, 8)), gen = uint32_t(c.field(24, 8));
    // v3: base and extent in units of 8 words
    const uint32_t base = uint32_t(c.field(32, 12)) << 3, ext = (uint32_t(c.field(44, 12)) + 1) << 3;
    const bool order = c.bit(56), ext_w = c.bit(59), tr = c.bit(127);
    const uint32_t kind = uint32_t(c.field(57, 2)), count = uint32_t(c.field(60, 4)) + 1;
    const bool err = gen == 0 || (base & 7) != 0 || base + ext > uint32_t(UB_WORDS) || idx + count > 256 ||
                     (kind == KIND_STREAM && (idx < 0xF8 || count != 1)) || kind == 3 ||
                     (ext_w && kind != KIND_STATIC) ||
                     (c.bit(119) && kind == KIND_APPEND) || (c.bit(119) && kind == KIND_STATIC && ext_w);
    if (err || ext % (8 * count) != 0) return ST_RANGE;
    const uint32_t sub_ext = ext / count;
    // the target entries and the owners of the range are quiescent in program order (R5): warn on an open writer
    for (uint32_t e = 0; e < count; ++e) {
        const Binding& t = urf_[idx + e];
        if (t.valid && t.writer != W_NONE) warn("BIND over a binding whose writer did not finish (hardware waits)");
    }
    // evict the live owners of the granules, claim them
    const uint32_t g0 = base >> 3, g1 = (base + ext) >> 3;
    for (uint32_t g = g0; g < g1; ++g) {
        const uint32_t ow = owner_[g];
        if ((ow >> 16) & 1) {
            Binding& o = urf_[(ow >> 8) & 0xFF];
            if (o.valid && o.gen == (ow & 0xFF)) {
                if (o.writer != W_NONE) warn("BIND over a region whose binding's writer did not finish (hardware waits)");
                o.valid = false;
            }
        }
        const uint32_t sub_e = (g - g0) / (sub_ext >> 3);
        owner_[g] = (1u << 16) | (((idx + sub_e) & 0xFF) << 8) | gen;
    }
    if (kind == KIND_STREAM) {
        sd_src_[idx & 7] = c.field(64, 31);
        sd_qs_[idx & 7] = uint32_t(c.field(95, 12));
        sd_len_[idx & 7] = uint32_t(c.field(107, 20)) + 1;
    }
    for (uint32_t e = 0; e < count; ++e) {
        Binding& b = urf_[idx + e];
        b.valid = true;
        b.gen = uint8_t(gen);
        b.base = uint16_t((base + e * sub_ext) & (UB_WORDS - 1));
        b.ext = uint16_t(sub_ext);
        b.order = order;
        b.tr = tr;
        b.kind = uint8_t(kind);
        b.alen = uint8_t(e == 0 ? count : 1);
        // APPEND: initial rows [118:107]; STATIC with bit 119 (present): the data are there, watermark = extent
        b.wm = uint16_t(kind == KIND_APPEND ? c.field(107, 12) : (kind == KIND_STATIC && c.bit(119)) ? sub_ext : 0);
        b.writer = ext_w ? W_EXT : W_NONE;
        b.poison = false;
    }
    return ST_OK;
}

uint8_t Node::exec_unbind(const Cmd128& c) {
    const uint8_t idx = uint8_t(c.field(16, 8)), gen = uint8_t(c.field(24, 8));
    const Binding& b = urf_[idx];
    if (!b.valid || b.gen != gen) return ST_BINDING;
    const int n = b.alen;
    for (int i = 0; i < n; ++i) urf_[uint8_t(idx + i)].valid = false;
    return ST_OK;
}

uint8_t Node::exec_move_out(const Cmd128& c) {
    const uint8_t idx = uint8_t(c.field(16, 8)), gen = uint8_t(c.field(24, 8));
    const uint32_t off = uint32_t(c.field(32, 16)), words = uint32_t(c.field(48, 16)) + 1;
    const uint64_t dst = c.field(64, 40);
    const Binding& b = urf_[idx];
    if (!b.valid || b.gen != gen) return ST_BINDING;
    if (off + words > b.ext || (dst & 31) != 0) return ST_RANGE;
    const bool append = b.kind == KIND_APPEND;
    if (!append && b.writer == W_NONE && b.wm < b.ext) return ST_ORDER;
    if (b.writer != W_NONE) warn("MOVE_OUT of a binding whose writer did not finish in program order");
    for (uint32_t i = 0; i < words; ++i) ext_.write((dst >> 5) + i, ub_[(b.base + off + i) & (UB_WORDS - 1)]);
    return ST_OK;
}

uint8_t Node::exec_move_in(const Cmd128& c) {
    const uint8_t idx = uint8_t(c.field(16, 8)), gen = uint8_t(c.field(24, 8));
    const uint32_t off = uint32_t(c.field(32, 16)), words = uint32_t(c.field(48, 16)) + 1;
    const uint64_t src = c.field(64, 40);
    const bool last = c.bit(104);
    const Binding& b = urf_[idx];
    if (!b.valid || b.gen != gen) return ST_BINDING;
    if (off + words > b.ext || (src & 31) != 0) return ST_RANGE;
    if (b.kind != KIND_STATIC || b.writer != W_EXT || off != b.wm) return ST_ORDER;
    for (uint32_t i = 0; i < words; ++i) ub_write(b.base + off + i, ext_.read((src >> 5) + i));
    // last: the binding is complete (watermark = extent) and its writer ends
    const uint32_t wm = last ? uint32_t(b.ext) : off + words;
    event(idx, gen, wm, last || wm == b.ext);
    return ST_OK;
}

// ---------------------------------------------------------------------------------------------------------------
// GEMM
// ---------------------------------------------------------------------------------------------------------------
namespace {
struct GemmFields {
    uint8_t ga_i, ga_g, gb_i, gb_g, gc_i, gc_g, gd_i, gd_g;
    uint32_t m, n, g, epi, eps, ah, dh;
    bool fold, fp, fc, has_c, qk;
    explicit GemmFields(const Cmd128& c) {
        ga_i = uint8_t(c.field(16, 8)); ga_g = uint8_t(c.field(24, 8));
        gb_i = uint8_t(c.field(32, 8)); gb_g = uint8_t(c.field(40, 8));
        gc_i = uint8_t(c.field(48, 8)); gc_g = uint8_t(c.field(56, 8));
        gd_i = uint8_t(c.field(64, 8)); gd_g = uint8_t(c.field(72, 8));
        m = uint32_t(c.field(80, 12)) + 1; n = uint32_t(c.field(92, 12)) + 1; g = uint32_t(c.field(104, 8)) + 1;
        epi = uint32_t(c.field(112, 4)); eps = uint32_t(c.field(116, 2));
        ah = uint32_t(c.field(118, 4)) + 1; dh = uint32_t(c.field(122, 4)) + 1;
        qk = c.bit(126); fold = c.bit(127);   // v3: [126] q / k RMSNorm per head (Gen2's B K-major flag)
        fp = fold && epi == EPI_ADD; fc = fold && epi != EPI_ADD;
        has_c = epi == EPI_ADD || epi == EPI_MUL || epi == EPI_ROPE;
    }
};
}  // namespace

uint8_t Node::check_gemm(const Cmd128& c) const {
    const GemmFields f(c);
    const Binding &s0 = urf_[f.ga_i], &s1 = urf_[f.gb_i], &s2 = urf_[f.gc_i], &s3 = urf_[f.gd_i];
    const Binding &s4 = urf_[uint8_t(f.gd_i + 1)], &s5 = urf_[uint8_t(f.gd_i + 2)], &s6 = urf_[uint8_t(f.gd_i + 3)];
    const Binding& s7 = urf_[uint8_t(f.ga_i + 1)];
    const bool b_stream = s1.kind == KIND_STREAM;
    const bool d_array = s3.alen > 1, d_append = s3.kind == KIND_APPEND, d_tr = s3.tr;
    const bool d_vt = d_tr && s3.order, d_f32 = d_tr && !s3.order;
    const bool c_f32 = f.has_c && s2.tr && !s2.order;
    const uint32_t d_arow = s3.wm;
    const uint32_t nt = (f.n + 31) >> 5, rpn = (f.m + 31) >> 5;
    const bool krange = f.g % f.ah != 0 || nt % f.dh != 0;
    if (f.epi > 6 || f.m > 4095 || f.n > 4095) return ST_UNSUPPORTED;   // the Node's 12-bit M, N counts
    if (d_vt && (!d_append || f.epi != 0 || (f.m != 1 && (d_arow & 31) != 0))) return ST_UNSUPPORTED;
    if (!d_tr && d_append && f.m != 1 && (d_arow & 7) != 0) return ST_UNSUPPORTED;
    if (d_f32 && (d_append || d_array || f.dh != 1 || f.epi == EPI_SILU_MUL || (f.n & 63) != 0)) return ST_UNSUPPORTED;
    if ((c_f32 && f.epi != EPI_ADD && f.epi != EPI_ROPE) || (f.has_c && s2.tr && s2.order)) return ST_UNSUPPORTED;
    if (f.has_c != (c.field(48, 16) != 0)) return ST_RESERVED;
    if (krange) return ST_RANGE;
    if (!s0.valid || s0.gen != f.ga_g || !s1.valid || s1.gen != f.gb_g || !s3.valid || s3.gen != f.gd_g ||
        (f.has_c && (!s2.valid || s2.gen != f.gc_g)))
        return ST_BINDING;
    if (s0.kind != KIND_STATIC || s0.order || (!b_stream && (s1.kind != KIND_STATIC || !s1.order)) ||
        (f.has_c && (s2.kind != KIND_STATIC || s2.order)))
        return ST_UNSUPPORTED;
    if (order_src(s0) || (!b_stream && order_src(s1)) || (f.has_c && order_src(s2))) return ST_ORDER;
    if (!d_append && (s3.writer != W_NONE || s3.wm != 0 || s3.kind != KIND_STATIC)) return ST_ORDER;
    if (d_array && s3.alen != f.dh) return ST_RANGE;
    if (f.fp && (d_array || d_append || d_vt || f.dh != 1 || (f.n & 63) != 0)) return ST_UNSUPPORTED;
    if ((f.fp && (!s4.valid || s4.gen != f.gd_g || !s5.valid || s5.gen != f.gd_g || !s6.valid || s6.gen != f.gd_g)) ||
        (f.fc && (!s7.valid || s7.gen != f.ga_g)))
        return ST_BINDING;
    if ((f.fp && (s4.kind != KIND_STATIC || s4.order || s4.tr || (!d_f32 && s4.ext != s3.ext) ||
                  s5.kind != KIND_STATIC || s5.ext < rpn * 4 || s6.kind != KIND_STATIC || s6.order)) ||
        (f.fc && s7.kind != KIND_STATIC))
        return ST_UNSUPPORTED;
    if ((f.fp && (s4.writer != W_NONE || s4.wm != 0 || s5.writer != W_NONE || s5.wm != 0 || order_src(s6))) ||
        (f.fc && order_src(s7)))
        return ST_ORDER;
    // q / k RMSNorm per head (R56, qk): a fold consumer with RoPE; heads of 64 or 128 columns (D heads: n / dh); a
    // head of 128 columns is one two-window tile (m <= 16, B streamed); the gain row: FP32 at C idx + 1, C's
    // generation; one that nobody writes must be complete (the Node waits for one still being written)
    const Binding& s8 = urf_[uint8_t(f.gc_i + 1)];
    const uint32_t qk_w = 32 * (nt / f.dh);
    if (f.qk) {
        if (!f.fc || f.epi != EPI_ROPE || d_vt || d_f32 || (qk_w != 64 && qk_w != 128) ||
            (qk_w == 128 && (f.m > 16 || !b_stream)))
            return ST_UNSUPPORTED;
        if (!s8.valid || s8.gen != f.gc_g) return ST_BINDING;
        if (s8.kind != KIND_STATIC || !s8.tr || s8.order || s8.ext < qk_w / 8) return ST_UNSUPPORTED;
        if (s8.writer == W_NONE && s8.wm < s8.ext) return ST_ORDER;
    }
    if (s0.poison || s1.poison || (f.has_c && s2.poison) || s3.poison || (f.fp && s6.poison) || (f.fc && s7.poison) ||
        (f.qk && s8.poison))
        return ST_POISONED;
    return ST_OK;
}

void Node::run_gemm(const Cmd128& c) {
    const GemmFields f(c);
    const Binding A = urf_[f.ga_i], B = urf_[f.gb_i], C = urf_[f.gc_i], D = urf_[f.gd_i];
    const Binding XG = urf_[uint8_t(f.gd_i + 1)], RB = urf_[uint8_t(f.gd_i + 2)], GB = urf_[uint8_t(f.gd_i + 3)];
    const Binding RQ = urf_[uint8_t(f.ga_i + 1)];
    const int m = int(f.m), n = int(f.n), g = int(f.g), K = 32 * g;
    const int epi = int(f.epi);
    const bool gated = epi == EPI_SILU_MUL, rope = epi == EPI_ROPE;
    const bool b_stream = B.kind == KIND_STREAM;
    const bool d_array = D.alen > 1, d_append = D.kind == KIND_APPEND;
    const bool d_vt = D.tr && D.order, d_f32 = D.tr && !D.order, d_p32 = D.order;
    const bool c_f32 = f.has_c && C.tr && !C.order;
    const bool g_f32 = f.fp && GB.tr && !GB.order;
    const int arow = D.wm;
    const int nt = (n + 31) / 32, npad = 32 * nt;
    const int nbands = gated ? 2 * nt : nt;
    const int rpn = (m + 31) / 32, panels = (m + 7) / 8;
    const int a_gh = g / int(f.ah), d_g = nt / int(f.dh), c_g = rope ? d_g : nt;
    const int dhd = int(f.dh);
    const bool app1 = d_append && !D.tr && m == 1;
    const bool simple_d = !d_array && !d_append && dhd == 1 && !d_p32 && !D.tr;

    if (source_pending(f.ga_i) || (!b_stream && source_pending(f.gb_i)) || (f.has_c && source_pending(f.gc_i)) ||
        (f.fp && source_pending(uint8_t(f.gd_i + 3))) || (f.fc && source_pending(uint8_t(f.ga_i + 1))))
        warn("GEMM reads a binding whose writer did not finish in program order");
    // writer marks (g3_ctrl W_WRITER walk): D, the array elements, a producer's x * g and 1/rms
    const int marks = d_array ? D.alen : f.fp ? 3 : 1;
    for (int e = 0; e < marks; ++e) urf_[uint8_t(f.gd_i + e)].writer = W_PARENT;

    // ---- B rows (output columns): bands of 32 rows; masked columns read as zero codes with scale 127. A streamed B
    // in quarters (BIND [106:95] != 0): band b at quarter b % 4 (a quarter stride of 512 words per unit apart),
    // position b / 4
    const int brows = nbands * 32;
    std::vector<uint8_t> bc(size_t(brows) * K), bs(size_t(brows) * g);
    WordSource bsrc;
    if (b_stream) { bsrc.ext = &ext_; bsrc.ext_base = sd_src_[f.gb_i & 7]; }
    else { bsrc.ub = &ub_; bsrc.ub_base = B.base; }
    const uint64_t quarter = b_stream ? uint64_t(sd_qs_[f.gb_i & 7]) * 512 : 0;
    {
        uint8_t qc[32][32], qs[32];
        for (int b = 0; b < nbands; ++b)
            for (int kg = 0; kg < g; ++kg) {
                const uint64_t off = quarter ? quarter * uint64_t(b % 4) + uint64_t(33) * uint64_t((b / 4) * g + kg)
                                             : uint64_t(33) * uint64_t(b * g + kg);
                read_quad(bsrc, off, qc, qs);
                for (int r = 0; r < 32; ++r) {
                    const int col = gated ? 32 * (b / 2) + r : 32 * b + r;
                    const int row = 32 * b + r;
                    const bool ok = col < n;
                    std::memcpy(&bc[size_t(row) * K + 32 * kg], qc[r], 32);
                    if (!ok) std::memset(&bc[size_t(row) * K + 32 * kg], 0, 32);
                    bs[size_t(row) * g + kg] = ok ? qs[r] : 127;
                }
            }
    }
    std::vector<int32_t> bq(size_t(brows) * K);
    std::vector<uint8_t> bn(size_t(brows) * g);
    mx_decode_rows(brows, g, bc.data(), bs.data(), bq.data(), bn.data());
    const MxRowsView bv{bq.data(), bs.data(), bn.data(), g};

    WordSource asrc;
    asrc.ub = &ub_; asrc.ub_base = A.base;
    WordSource csrc;
    csrc.ub = &ub_; csrc.ub_base = C.base;
    // q / k RMSNorm: the gain row (FP32, C idx + 1) and the factors k of the pass's rows and heads
    std::vector<uint32_t> qk_gain, qk_k;
    if (f.qk) {
        const Binding GN = urf_[uint8_t(f.gc_i + 1)];
        for (int d = 0; d < int(32 * (uint32_t(nt) / f.dh)); ++d)
            qk_gain.push_back(ub_[(GN.base + d / 8) & (UB_WORDS - 1)].u32(d % 8));
    }
    // producer's g row (decoded)
    std::vector<uint32_t> gd;
    if (f.fp) {
        gd.assign(npad, 0);
        for (int col = 0; col < npad; ++col) {
            if (g_f32) {
                gd[col] = ub_[(GB.base + col / 8) & (UB_WORDS - 1)].u32(col % 8);
            } else {
                const int go = col / 32, k = col % 32;
                const Word& sw = ub_[(GB.base + scale_word(go)) & (UB_WORDS - 1)];
                const Word& wd = ub_[(GB.base + block_word(go) + code_word(k)) & (UB_WORDS - 1)];
                gd[col] = decode_e4m3(wd.b[code_byte(k, 0)], sw.b[scale_byte(go, 0)]);
            }
        }
    }

    std::vector<uint8_t> ac(size_t(32) * K), as(size_t(32) * g);
    std::vector<int32_t> aq(size_t(32) * K);
    std::vector<uint8_t> an(size_t(32) * g);
    std::vector<uint32_t> acc(size_t(32) * brows);
    std::vector<uint32_t> y(size_t(32) * npad);
    std::vector<uint8_t> xc(size_t(32) * npad), xs(size_t(32) * nt);
    for (int rp = 0; rp < rpn; ++rp) {
        const int np = std::min(4, panels - 4 * rp);
        const int rows_n = std::min(32, m - 32 * rp);
        const int rows_calc = d_vt ? 32 : 8 * np;           // rows of the pass's output jobs
        // A rows of the pass (rows beyond m: zero codes, scale 127)
        std::fill(ac.begin(), ac.end(), 0);
        std::fill(as.begin(), as.end(), 127);
        uint8_t blk[8][32], bsc[8];
        for (int p = 0; p < np; ++p) {
            const int P = 4 * rp + p;
            for (int kg = 0; kg < g; ++kg) {
                const int head = kg / a_gh, gl = kg % a_gh;
                read_block(asrc, uint32_t(head * panels * a_gh + P * a_gh + gl), blk, bsc);
                for (int l = 0; l < 8; ++l) {
                    const int r = 8 * p + l;
                    if (r >= rows_n) continue;
                    std::memcpy(&ac[size_t(r) * K + 32 * kg], blk[l], 32);
                    as[size_t(r) * g + kg] = bsc[l];
                }
            }
        }
        mx_decode_rows(32, g, ac.data(), as.data(), aq.data(), an.data());
        const MxRowsView av{aq.data(), as.data(), an.data(), g};
        const int64_t work = int64_t(std::min(rows_calc, rows_n)) * brows * K;
        parallel_for(brows, work >= (int64_t(1) << 22) ? opt_.threads : 1, [&](int j0, int j1) {
            for (int r = 0; r < rows_calc; ++r)
                for (int j = j0; j < j1; ++j)
                    acc[size_t(r) * brows + j] = r < rows_n ? gemm_bf48_dot(av, r, bv, j) : 0u;
        });
        // fold consumer: 1/rms of the pass's rows (A idx+1)
        uint32_t rr[32] = {0};
        if (f.fc)
            for (int r = 0; r < 32; ++r) rr[r] = ub_[(RQ.base + 4 * rp + r / 8) & (UB_WORDS - 1)].u32(r % 8);
        // q / k RMSNorm per head (qk, R56): z = sum x 1/rms of the row; the head's 1/rms r_h of MXFP8(z) (the fold's
        // sum of squares and 1/rms over the head's columns, 1/H = 1/head width); each value of the head then
        // (sum x gain) x k, k = r_h x 1/rms of the row, before the RoPE
        if (f.qk) {
            const int hw = int(32 * (uint32_t(nt) / f.dh)), heads = npad / hw;
            qk_k.assign(size_t(32) * heads, 0);
            for (int r = 0; r < rows_calc; ++r)
                for (int h = 0; h < heads; ++h) {
                    uint32_t z[128], zd[128];
                    for (int d = 0; d < hw; ++d) z[d] = f32_mul(acc[size_t(r) * brows + h * hw + d], rr[r]);
                    for (int b = 0; b < hw / 32; ++b) {
                        uint8_t cc[32], sc;
                        quantize_block32(&z[32 * b], cc, &sc);
                        for (int j = 0; j < 32; ++j) zd[32 * b + j] = decode_e4m3(cc[j], sc);
                    }
                    qk_k[size_t(r) * heads + h] = f32_mul(inv_rms_row(zd, hw, int(f.eps)), rr[r]);
                }
        }
        // epilogue
        uint32_t cv[32], v[32];
        uint8_t cblk[8][32], cbs[8];
        for (int go = 0; go < nt; ++go) {
            int cached_p = -1;
            for (int r = 0; r < rows_calc; ++r) {
                const uint32_t* arow_acc = &acc[size_t(r) * brows];
                if (gated) {
                    for (int j = 0; j < 32; ++j) {
                        uint32_t gate = arow_acc[32 * (2 * go) + j], up = arow_acc[32 * (2 * go + 1) + j];
                        if (f.fc) { gate = f32_mul(gate, rr[r]); up = f32_mul(up, rr[r]); }
                        v[j] = f32_mul(silu_pwl(gate), up);
                    }
                } else {
                    if (f.qk) {
                        const int hw = int(32 * (uint32_t(nt) / f.dh)), heads = npad / hw;
                        const uint32_t kf = qk_k[size_t(r) * heads + (32 * go) / hw];
                        for (int j = 0; j < 32; ++j)
                            v[j] = f32_mul(f32_mul(arow_acc[32 * go + j], qk_gain[(32 * go + j) % hw]), kf);
                    } else
                        for (int j = 0; j < 32; ++j) {
                            v[j] = arow_acc[32 * go + j];
                            if (f.fc) v[j] = f32_mul(v[j], rr[r]);
                        }
                    if (f.has_c) {
                        if (c_f32) {
                            // FP32 rows: row r at 4 n rp + r (n / 8), the output group's 4 words; an FP32 RoPE table
                            // (R56): rows of the head width (c_g groups), the head-local group's words
                            const uint32_t nw = uint32_t(n) >> 3;
                            const uint32_t w0 = rope ? C.base + uint32_t(128 * c_g * rp) + uint32_t(r * 4 * c_g) +
                                                           uint32_t(4 * (go % d_g))
                                                     : C.base + uint32_t(4 * n * rp) + uint32_t(r) * nw + uint32_t(4 * go);
                            for (int j = 0; j < 32; ++j) cv[j] = ub_[(w0 + j / 8) & (UB_WORDS - 1)].u32(j % 8);
                        } else {
                            const int p = r / 8;
                            if (p != cached_p) {
                                const int cg = rope ? go % d_g : go;
                                read_block(csrc, uint32_t((4 * rp + p) * c_g + cg), cblk, cbs);
                                cached_p = p;
                            }
                            for (int j = 0; j < 32; ++j) cv[j] = decode_e4m3(cblk[r % 8][j], cbs[r % 8]);
                        }
                        if (epi == EPI_ADD) for (int j = 0; j < 32; ++j) v[j] = f32_add(v[j], cv[j]);
                        else if (epi == EPI_MUL) for (int j = 0; j < 32; ++j) v[j] = f32_mul(v[j], cv[j]);
                        else { uint32_t t[32]; rope_pairs(v, cv, t, 32); std::memcpy(v, t, sizeof t); }
                    } else if (epi == EPI_SILU) {
                        for (int j = 0; j < 32; ++j) v[j] = silu_pwl(v[j]);
                    } else if (epi == EPI_GELU) {
                        for (int j = 0; j < 32; ++j) v[j] = gelu_pwl(v[j]);
                    }
                }
                std::memcpy(&y[size_t(r) * npad + 32 * go], v, sizeof v);
            }
        }
        // ---- outputs
        auto place = [&](int go, uint32_t pl_base, uint32_t& base, uint32_t& pan, uint32_t& gl) {
            const int q = go / d_g, rem = go % d_g;
            gl = uint32_t(dhd > 1 ? rem : go);
            base = (d_array && dhd > 1) ? ((pl_base + uint32_t(q) * D.ext) & (UB_WORDS - 1)) : pl_base;
            if (d_vt) pan = uint32_t(4 * ((arow >> 5) + rp));
            else if (d_append) pan = uint32_t((arow >> 3) + (m == 1 ? 0 : 4 * rp));
            else if (d_array) pan = uint32_t(4 * rp);
            else pan = uint32_t((dhd > 1 ? q * panels : 0) + 4 * rp);
            pan &= 0xFFF;
        };
        auto block_of = [&](uint32_t pan, int p, uint32_t gl, bool rows_layout) -> uint32_t {
            const uint32_t P = pan + uint32_t(p);
            return rows_layout ? p8_block(P, gl, uint32_t(d_g)) : p32_block(P, gl, uint32_t(d_g));
        };
        // MXFP8 codes of y (x for a producer), per row and output group
        if (!d_vt) quantize_matrix(y.data(), rows_calc, npad, npad, xc.data(), xs.data());
        if (d_f32) {
            // FP32 rows: row r at word base + 4 n rp + r (n / 8), columns 8 per word
            for (int r = 0; r < rows_calc; ++r)
                for (int col = 0; col < n; col += 8) {
                    Word w;
                    for (int l = 0; l < 8; ++l) w.set_u32(l, y[size_t(r) * npad + col + l]);
                    ub_write(D.base + uint32_t(4 * n * rp) + uint32_t(r * 4 * nt) + uint32_t(col / 8), w);
                }
        } else if (d_vt) {
            const bool tail = m == 1 && (arow & 31) != 0;
            for (int go = 0; go < nt; ++go) {
                uint32_t base, pan, gl;
                place(go, D.base, base, pan, gl);
                const uint32_t cell = base + 33 * ((pan >> 2) * uint32_t(d_g) + gl);
                uint8_t codes[32][32];   // [dim][token]
                uint8_t scales[32];
                if (!tail) {
                    for (int j = 0; j < 32; ++j) {
                        uint32_t col[32];
                        for (int t = 0; t < 32; ++t) col[t] = y[size_t(t) * npad + 32 * go + j];
                        scales[j] = choose_scale(row_max_bits(col, 32));
                        for (int t = 0; t < 32; ++t) codes[j][t] = quantize_e4m3(col[t], scales[j]);
                    }
                } else {
                    // rewrite of the partial cell: old codes re-rounded to the new scales, the new token inserted
                    WordSource vsrc;
                    vsrc.ub = &ub_; vsrc.ub_base = cell;
                    uint8_t old[32][32], sold[32];
                    read_quad(vsrc, 0, old, sold);
                    const int kt = arow & 31;
                    for (int j = 0; j < 32; ++j) {
                        bool zero = true;
                        for (int t = 0; t < 32; ++t) zero &= (old[j][t] & 0x7F) == 0;
                        const uint32_t x = y[size_t(0) * npad + 32 * go + j];
                        scales[j] = vt_tail_scale(sold[j], zero, x);
                        for (int t = 0; t < 32; ++t)
                            codes[j][t] = t == kt ? quantize_e4m3(x, scales[j]) : vt_requant_code(old[j][t], sold[j], scales[j]);
                    }
                }
                Word sw;
                std::memcpy(sw.b, scales, 32);
                for (int p = 0; p < 4; ++p) {
                    Word words[8];
                    uint8_t rows8[8][32];
                    for (int l = 0; l < 8; ++l) std::memcpy(rows8[l], codes[8 * p + l], 32);
                    pack_block(rows8, words);
                    for (int w = 0; w < 8; ++w) ub_write(cell + 1 + 8 * p + w, words[w]);
                }
                ub_write(cell, sw);
            }
        } else {
            // rows layouts (P8, P32): 8 words and the 8 scale bytes of each (panel, output group) block. A P8 pass
            // with one row (not APPEND, not a fold producer) writes its panel's other rows as zero codes with scale
            // 127, as the RTL's one-row output pass does (R56)
            const bool rows_layout = !d_p32;
            const bool row1 = rows_layout && !app1 && !f.fp && m - 32 * rp == 1;
            for (int go = 0; go < nt; ++go) {
                uint32_t base, pan, gl;
                place(go, D.base, base, pan, gl);
                for (int p = 0; p < (app1 ? 1 : np); ++p) {
                    const uint32_t nb = block_of(pan, p, gl, rows_layout);
                    uint8_t rows8[8][32], sc8[8];
                    for (int l = 0; l < 8; ++l) {
                        const int r = app1 ? 0 : 8 * p + l;
                        if (row1 && l != 0) { std::memset(rows8[l], 0, 32); sc8[l] = 127; continue; }
                        std::memcpy(rows8[l], &xc[size_t(r) * npad + 32 * go], 32);
                        sc8[l] = xs[size_t(r) * nt + go];
                    }
                    Word words[8];
                    pack_block(rows8, words);
                    Word sw;
                    for (int l = 0; l < 8; ++l) sw.b[scale_byte(nb, l)] = sc8[l];
                    if (app1) {
                        const int q = arow & 7;
                        const uint32_t strobe = (3u << (2 * q)) | (3u << (16 + 2 * q));
                        for (int w = 0; w < 8; ++w) ub_write(base + block_word(nb) + w, words[w], strobe);
                        ub_write(base + scale_word(nb), sw, 1u << scale_byte(nb, uint32_t(q)));
                    } else {
                        for (int w = 0; w < 8; ++w) ub_write(base + block_word(nb) + w, words[w]);
                        ub_write(base + scale_word(nb), sw, 0xFFu << (8 * (nb & 3)));
                    }
                }
            }
        }
        if (f.fp) {
            // x * g (P8 blocks at the x * g base) and the rows' 1/rms of MXFP8(x)
            std::vector<uint32_t> xg(size_t(rows_calc) * npad);
            for (int r = 0; r < rows_calc; ++r)
                for (int col = 0; col < npad; ++col)
                    xg[size_t(r) * npad + col] = f32_mul(y[size_t(r) * npad + col], gd[col]);
            std::vector<uint8_t> gc(size_t(rows_calc) * npad), gs(size_t(rows_calc) * nt);
            quantize_matrix(xg.data(), rows_calc, npad, npad, gc.data(), gs.data());
            for (int go = 0; go < nt; ++go)
                for (int p = 0; p < np; ++p) {
                    const uint32_t nb = p8_block(uint32_t(4 * rp + p), uint32_t(go), uint32_t(nt));
                    uint8_t rows8[8][32];
                    Word sw, words[8];
                    for (int l = 0; l < 8; ++l) {
                        std::memcpy(rows8[l], &gc[size_t(8 * p + l) * npad + 32 * go], 32);
                        sw.b[scale_byte(nb, l)] = gs[size_t(8 * p + l) * nt + go];
                    }
                    pack_block(rows8, words);
                    for (int w = 0; w < 8; ++w) ub_write(XG.base + block_word(nb) + w, words[w]);
                    ub_write(XG.base + scale_word(nb), sw, 0xFFu << (8 * (nb & 3)));
                }
            std::vector<uint32_t> xd(npad);
            for (int r = 0; r < rows_calc; ++r) {
                for (int col = 0; col < npad; ++col)
                    xd[col] = decode_e4m3(xc[size_t(r) * npad + col], xs[size_t(r) * nt + col / 32]);
                rms_mem_[r] = inv_rms_row(xd.data(), npad, int(f.eps));
            }
            for (int i = 0; i < 4; ++i) {
                Word w;
                for (int l = 0; l < 8; ++l) w.set_u32(l, rms_mem_[8 * i + l]);
                ub_write(RB.base + uint32_t(4 * rp + i), w);
            }
        }
    }
    // final commits (g3_gemm_seq G_FINAL)
    const bool fp_f32 = f.fp && d_f32;
    if (simple_d) {
        event(f.gd_i, f.gd_g, D.ext, true);
        if (f.fp) {
            event(uint8_t(f.gd_i + 1), f.gd_g, D.ext, true);
            event(uint8_t(f.gd_i + 2), f.gd_g, RB.ext, true);
        }
    } else if (fp_f32) {
        event(f.gd_i, f.gd_g, D.ext, true);
        event(uint8_t(f.gd_i + 1), f.gd_g, XG.ext, true);
        event(uint8_t(f.gd_i + 2), f.gd_g, RB.ext, true);
    } else {
        const int elems = d_array ? D.alen : 1;
        for (int e = 0; e < elems; ++e)
            event(uint8_t(f.gd_i + e), f.gd_g, d_append ? uint32_t(arow + m) : D.ext, true);
    }
}

// ---------------------------------------------------------------------------------------------------------------
// ATTN
// ---------------------------------------------------------------------------------------------------------------
namespace {
struct AttnFields {
    uint8_t aq_i, aq_g, ak_i, ak_g, av_i, av_g, ao_i, ao_g;
    uint32_t tokens, qh, grp, keys, G, ctx, kvh, hp;
    int32_t delta;
    explicit AttnFields(const Cmd128& c) {
        aq_i = uint8_t(c.field(16, 8)); aq_g = uint8_t(c.field(24, 8));
        ak_i = uint8_t(c.field(32, 8)); ak_g = uint8_t(c.field(40, 8));
        av_i = uint8_t(c.field(48, 8)); av_g = uint8_t(c.field(56, 8));
        ao_i = uint8_t(c.field(64, 8)); ao_g = uint8_t(c.field(72, 8));
        tokens = uint32_t(c.field(80, 11)) + 1; qh = uint32_t(c.field(91, 4)) + 1; grp = uint32_t(c.field(95, 2)) + 1;
        keys = uint32_t(c.field(97, 11)) + 1;
        const uint32_t d = uint32_t(c.field(108, 14));
        delta = (d & 0x2000) ? int32_t(d) - 0x4000 : int32_t(d);
        G = uint32_t(c.field(122, 3)) + 1; ctx = uint32_t(c.field(126, 2));
        kvh = qh / grp; hp = (tokens + 7) >> 3;
    }
};
}  // namespace

uint8_t Node::check_attn(const Cmd128& c) const {
    const AttnFields f(c);
    const Binding& q = urf_[f.aq_i];
    const Binding& o = urf_[f.ao_i];
    const Binding* k[4];
    const Binding* v[4];
    for (int x = 0; x < 4; ++x) { k[x] = &urf_[uint8_t(f.ak_i + x)]; v[x] = &urf_[uint8_t(f.av_i + x)]; }
    const bool stream = k[0]->kind == KIND_STREAM && v[0]->kind == KIND_STREAM;
    const uint32_t qwords = 33 * ((f.qh * f.hp * f.G + 3) >> 2);
    const uint32_t kvwords = 33 * f.G * ((f.keys + 31) >> 5);
    const int kvh = int(f.kvh);
    // streamed K / V (v3): one STREAM binding each; KV head h's image at src + h * length
    if (f.ctx != 0 || f.kvh > 4) return ST_UNSUPPORTED;
    if (f.qh % f.grp != 0) return ST_RANGE;
    uint8_t s = ST_OK;
    if (!q.valid || q.gen != f.aq_g || !o.valid || o.gen != f.ao_g) s = ST_BINDING;
    for (int x = 0; x < 4; ++x)
        if (x < kvh && (x == 0 || !stream) &&
            (!k[x]->valid || k[x]->gen != f.ak_g || !v[x]->valid || v[x]->gen != f.av_g)) s = ST_BINDING;
    if (s != ST_OK) return s;
    if (q.kind != KIND_STATIC || q.order || o.kind != KIND_STATIC || o.order ||
        (k[0]->kind == KIND_STREAM) != (v[0]->kind == KIND_STREAM) || !k[0]->order || k[0]->tr || !v[0]->tr ||
        !v[0]->order || (kvh > 1 && !stream && (k[0]->alen != kvh || v[0]->alen != kvh)))
        return ST_UNSUPPORTED;
    if (q.ext < qwords || o.ext < qwords) return ST_RANGE;
    for (int x = 0; x < kvh; ++x)
        if (!stream && (k[x]->ext < kvwords || v[x]->ext < kvwords)) return ST_RANGE;
    if (order_src(q)) s = ST_ORDER;
    for (int x = 0; x < kvh; ++x)
        if (!stream) {
            for (const Binding* b : {k[x], v[x]})
                if (b->writer == W_NONE && !b->poison && (b->kind == KIND_APPEND ? b->wm < f.keys : b->wm < b->ext))
                    s = ST_ORDER;
        }
    if (o.writer != W_NONE || o.wm != 0) s = ST_ORDER;
    if (s != ST_OK) return s;
    if (q.poison || o.poison) return ST_POISONED;
    for (int x = 0; x < kvh; ++x)
        if ((x == 0 || !stream) && (k[x]->poison || v[x]->poison)) return ST_POISONED;
    return ST_OK;
}

void Node::run_attn(const Cmd128& c) {
    const AttnFields f(c);
    const Binding Q = urf_[f.aq_i], O = urf_[f.ao_i];
    const bool stream = urf_[f.ak_i].kind == KIND_STREAM && urf_[f.av_i].kind == KIND_STREAM;
    const int G = int(f.G), dh = 32 * G, keys = int(f.keys);
    const int cap = (keys + 31) / 32 * 32, kg = cap / 32;
    const int hp = int(f.hp), tokens = int(f.tokens), grp = int(f.grp);
    if (source_pending(f.aq_i)) warn("ATTN reads Q before its writer finished in program order");
    urf_[f.ao_i].writer = W_PARENT;
    std::vector<uint8_t> kc(size_t(cap) * dh), ks(size_t(cap) * G), vc(size_t(dh) * cap), vs(size_t(dh) * kg);
    std::vector<int32_t> kq9(kc.size()), vq9(vc.size());
    std::vector<uint8_t> qc(dh), qs(G), oc(size_t(hp) * 8 * dh), os(size_t(hp) * 8);
    WordSource qsrc;
    qsrc.ub = &ub_; qsrc.ub_base = Q.base;
    uint8_t quad[32][32], qsc[32];
    for (int kv = 0; kv < int(f.kvh); ++kv) {
        WordSource ksrc, vsrc;
        if (stream) {
            ksrc.ext = &ext_; ksrc.ext_base = sd_src_[f.ak_i & 7] + uint64_t(kv) * sd_len_[f.ak_i & 7];
            vsrc.ext = &ext_; vsrc.ext_base = sd_src_[f.av_i & 7] + uint64_t(kv) * sd_len_[f.av_i & 7];
        } else {
            const Binding& kb = urf_[uint8_t(f.ak_i + kv)];
            const Binding& vb = urf_[uint8_t(f.av_i + kv)];
            if (source_pending(uint8_t(f.ak_i + kv)) || source_pending(uint8_t(f.av_i + kv)))
                warn("ATTN reads K / V^T before their writer finished in program order");
            ksrc.ub = &ub_; ksrc.ub_base = kb.base;
            vsrc.ub = &ub_; vsrc.ub_base = vb.base;
        }
        // K rows (P32: band j, dim chunk c at quad j G + c), V^T cells (key group j, dim chunk c at quad j G + c)
        for (int j = 0; j < kg; ++j)
            for (int cc = 0; cc < G; ++cc) {
                read_quad(ksrc, uint64_t(33) * uint64_t(j * G + cc), quad, qsc);
                for (int r = 0; r < 32; ++r) {
                    std::memcpy(&kc[size_t(32 * j + r) * dh + 32 * cc], quad[r], 32);
                    ks[size_t(32 * j + r) * G + cc] = qsc[r];
                }
                read_quad(vsrc, uint64_t(33) * uint64_t(j * G + cc), quad, qsc);
                for (int r = 0; r < 32; ++r) {
                    std::memcpy(&vc[size_t(32 * cc + r) * cap + 32 * j], quad[r], 32);
                    vs[size_t(32 * cc + r) * kg + j] = qsc[r];
                }
            }
        if (opt_.attn_requantize && (opt_.attn_requantize_mask & 2)) requantize_blocks(kc.data(), ks.data(), cap, dh);
        if (opt_.attn_requantize && (opt_.attn_requantize_mask & 4)) requantize_vt(vc.data(), vs.data(), dh, cap, keys);
        for (size_t i = 0; i < kc.size(); ++i) kq9[i] = e4m3_q9(kc[i]);
        for (size_t i = 0; i < vc.size(); ++i) vq9[i] = e4m3_q9(vc[i]);
        AttnKV akv;
        akv.dh = dh; akv.cap = cap;
        akv.k_codes = kc.data(); akv.k_scales = ks.data(); akv.vt_codes = vc.data(); akv.vt_scales = vs.data();
        akv.k_q9 = kq9.data(); akv.vt_q9 = vq9.data();
        for (int h = kv * grp; h < (kv + 1) * grp; ++h) {
            for (int t = 0; t < hp * 8; ++t) {
                const int P = h * hp + t / 8, lane = t % 8;
                int visible = 0;
                if (t < tokens) {
                    const int64_t lim = int64_t(f.delta) + t + 1;
                    visible = int(lim < 0 ? 0 : lim > keys ? keys : lim);
                    for (int cc = 0; cc < G; ++cc) {
                        uint8_t blk[8][32], sc8[8];
                        read_block(qsrc, p8_block(uint32_t(P), uint32_t(cc), uint32_t(G)), blk, sc8);
                        std::memcpy(&qc[32 * cc], blk[lane], 32);
                        qs[cc] = sc8[lane];
                    }
                    if (opt_.attn_requantize && (opt_.attn_requantize_mask & 1)) requantize_blocks(qc.data(), qs.data(), 1, dh);
                }
                attention_row(akv, qc.data(), qs.data(), keys, visible, &oc[size_t(t) * dh], &os[t]);
            }
            // O blocks of the head's panels: 8 words + the 8 row scales
            for (int p = 0; p < hp; ++p)
                for (int cc = 0; cc < G; ++cc) {
                    const uint32_t nb = p8_block(uint32_t(h * hp + p), uint32_t(cc), uint32_t(G));
                    uint8_t rows8[8][32];
                    Word sw, words[8];
                    for (int l = 0; l < 8; ++l) {
                        std::memcpy(rows8[l], &oc[size_t(8 * p + l) * dh + 32 * cc], 32);
                        sw.b[scale_byte(nb, l)] = os[8 * p + l];
                    }
                    pack_block(rows8, words);
                    for (int w = 0; w < 8; ++w) ub_write(O.base + block_word(nb) + w, words[w]);
                    ub_write(O.base + scale_word(nb), sw, 0xFFu << (8 * (nb & 3)));
                }
        }
    }
    event(f.ao_i, f.ao_g, O.ext, true);
}

// ---------------------------------------------------------------------------------------------------------------
// VECTOR (RMSNORM only; the other operations are UNSUPPORTED)
// ---------------------------------------------------------------------------------------------------------------
uint8_t Node::check_vector(const Cmd128& c) const {
    const uint8_t vm_i = uint8_t(c.field(16, 8)), vm_g = uint8_t(c.field(24, 8));
    const uint8_t va_i = uint8_t(c.field(48, 8)), va_g = uint8_t(c.field(56, 8));
    const uint8_t vr_i = uint8_t(c.field(64, 8)), vr_g = uint8_t(c.field(72, 8));
    const uint32_t rhs = uint32_t(c.field(32, 16)), res2 = uint32_t(c.field(80, 16));
    const uint32_t op = uint32_t(c.field(116, 4));
    const bool whole = c.bit(122);
    const Binding &s0 = urf_[vm_i], &s1 = urf_[va_i], &s2 = urf_[vr_i];
    if (op != 8 || rhs != 0 || res2 != 0 || whole) return ST_UNSUPPORTED;
    if (!s0.valid || s0.gen != vm_g || !s1.valid || s1.gen != va_g || !s2.valid || s2.gen != vr_g) return ST_BINDING;
    if (s0.kind != KIND_STATIC || s0.order || s1.kind != KIND_STATIC || s1.order || s2.kind != KIND_STATIC || s2.order)
        return ST_UNSUPPORTED;
    if (order_src(s0) || order_src(s1) || s2.writer != W_NONE || s2.wm != 0) return ST_ORDER;
    if (s0.poison || s1.poison) return ST_POISONED;
    return ST_OK;
}

void Node::run_vector(const Cmd128& c) {
    const uint8_t vm_i = uint8_t(c.field(16, 8)), va_i = uint8_t(c.field(48, 8));
    const uint8_t vr_i = uint8_t(c.field(64, 8)), vr_g = uint8_t(c.field(72, 8));
    const int rows = int(c.field(96, 12)) + 1, groups = int(c.field(108, 8)) + 1, eps = int(c.field(120, 2));
    const Binding X = urf_[vm_i], GA = urf_[va_i], R = urf_[vr_i];
    const int w = 32 * groups, panels = (rows + 7) / 8;
    if (source_pending(vm_i) || source_pending(va_i)) warn("VECTOR reads a binding before its writer finished");
    urf_[vr_i].writer = W_PARENT;
    WordSource xsrc, gsrc;
    xsrc.ub = &ub_; xsrc.ub_base = X.base;
    gsrc.ub = &ub_; gsrc.ub_base = GA.base;
    std::vector<uint32_t> gamma(w), x(size_t(8) * w), out(size_t(8) * w);
    uint8_t blk[8][32], sc8[8];
    for (int gi = 0; gi < groups; ++gi) {
        read_block(gsrc, uint32_t(gi), blk, sc8);
        for (int k = 0; k < 32; ++k) gamma[32 * gi + k] = decode_e4m3(blk[0][k], sc8[0]);
    }
    std::vector<uint8_t> codes(size_t(8) * w), scales(size_t(8) * groups);
    for (int P = 0; P < panels; ++P) {
        for (int gi = 0; gi < groups; ++gi) {
            read_block(xsrc, p8_block(uint32_t(P), uint32_t(gi), uint32_t(groups)), blk, sc8);
            for (int l = 0; l < 8; ++l)
                for (int k = 0; k < 32; ++k) x[size_t(l) * w + 32 * gi + k] = decode_e4m3(blk[l][k], sc8[l]);
        }
        for (int l = 0; l < 8; ++l) rmsnorm_row(&x[size_t(l) * w], gamma.data(), w, eps, &out[size_t(l) * w]);
        quantize_matrix(out.data(), 8, w, w, codes.data(), scales.data());
        for (int gi = 0; gi < groups; ++gi) {
            const uint32_t nb = p8_block(uint32_t(P), uint32_t(gi), uint32_t(groups));
            uint8_t rows8[8][32];
            Word sw, words[8];
            for (int l = 0; l < 8; ++l) {
                std::memcpy(rows8[l], &codes[size_t(l) * w + 32 * gi], 32);
                sw.b[scale_byte(nb, l)] = scales[size_t(l) * groups + gi];
            }
            pack_block(rows8, words);
            for (int ww = 0; ww < 8; ++ww) ub_write(R.base + block_word(nb) + ww, words[ww]);
            ub_write(R.base + scale_word(nb), sw, 0xFFu << (8 * (nb & 3)));
        }
    }
    event(vr_i, vr_g, R.ext, true);
}

}  // namespace g3
