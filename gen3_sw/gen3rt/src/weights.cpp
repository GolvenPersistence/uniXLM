#include "gen3rt/layout.h"
#include "gen3rt/llm.h"
#include "gen3rt/sha256.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace g3rt {
std::vector<WeightImage *> Llm::weight_slots() {
    std::vector<WeightImage *> out;
    for (auto &l : layers_)
        for (auto *w : {&l.wq, &l.wk, &l.wv, &l.wo, &l.wgu, &l.wd, &l.g_attn, &l.g_ffn})
            out.push_back(w);
    if (cfg_.qk_norm)
        for (auto &l : layers_)
            for (auto *w : {&l.g_q, &l.g_k})
                out.push_back(w);
    out.push_back(&g_out_);
    out.push_back(&lm_);
    return out;
}
// weight_slots' roles, for their placement in device memory ("" for the gains)
std::vector<std::string> Llm::weight_roles() const {
    std::vector<std::string> out;
    for (size_t l = 0; l < layers_.size(); ++l)
        for (const char *r : {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_up", "ffn_down", "", ""})
            out.push_back(r);
    if (cfg_.qk_norm)
        out.insert(out.end(), 2 * layers_.size(), std::string());
    out.push_back("");
    out.push_back("output");
    return out;
}
std::vector<uint32_t> Llm::weight_sizes() const {
    std::vector<uint32_t> out;
    const int E = cfg_.n_embd, F = cfg_.n_ff, Q = cfg_.n_head * cfg_.dh, K = cfg_.n_kv * cfg_.dh;
    for (int l = 0; l < cfg_.n_layer; ++l) {
        for (auto shape : {std::pair<int, int>{Q, E}, {K, E}, {K, E}, {E, Q}, {2 * F, E}, {E, F}})
            out.push_back(uint32_t(Layout(shape.first, shape.second, true).words()));
        out.push_back(uint32_t(E / 8));
        out.push_back(uint32_t(E / 8));
    }
    if (cfg_.qk_norm)
        for (int l = 0; l < 2 * cfg_.n_layer; ++l)
            out.push_back(uint32_t(cfg_.dh / 8));
    out.push_back(uint32_t(E / 8));
    out.push_back(uint32_t(Layout(cfg_.vocab, E, true).words()));
    return out;
}
namespace {
std::string header(const LlmConfig &c, const std::string &model, const std::string &layout) {
    if (model.size() != 64 || model.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::invalid_argument("model identity must be lowercase SHA256");
    // Version 1 fixes MXFP8 quantization, P32 layout, 32-row gate/up interleave and byte order. Version 2: version 1
    // with W_q / W_k / W_o / W_d arranged for the program (Llm::arrange), described by the layout tag.
    std::string s = layout.empty() ? "G3WGT001" : "G3WGT002";
    s += model;
    for (int n : {c.n_layer, c.n_embd, c.n_ff, c.n_head, c.n_kv, c.dh, c.vocab})
        for (int j = 0; j < 4; ++j)
            s += char(uint32_t(n) >> (j * 8));
    if (!layout.empty()) {
        for (int j = 0; j < 4; ++j)
            s += char(uint32_t(layout.size()) >> (j * 8));
        s += layout;
    }
    if (c.qk_norm)          // the q / k norm gains follow the other weights
        s += "QKNORM";
    return s;
}
void word_count(uint32_t n, char b[4]) {
    for (int i = 0; i < 4; ++i)
        b[i] = char(n >> (8 * i));
}
} // namespace
void Llm::save_weights(const std::string &path, const std::string &model) {
    if (!complete())
        throw std::runtime_error("cannot cache incomplete weights");
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error("create weight cache: " + tmp);
    Sha256 hash;
    auto put = [&](const void *p, size_t n) {
        f.write(static_cast<const char *>(p), std::streamsize(n));
        hash.update(p, n);
    };
    auto h = header(cfg_, model, weight_layout());
    put(h.data(), h.size());
    const auto slots = weight_slots();
    const auto sizes = weight_sizes();
    std::vector<Word> buf(32768);
    for (size_t i = 0; i < slots.size(); ++i) {
        const auto &w = *slots[i];
        if (w.words != sizes[i])
            throw std::runtime_error("cache weight shape mismatch");
        char b[4];
        word_count(w.words, b);
        put(b, 4);
        const auto it = placed_.find(w.addr);
        if (it != placed_.end()) {
            // in quarters: each segment's bands back in order
            std::vector<Word> q[4];
            for (const auto &s : it->second) {
                for (int r = 0; r < 4; ++r) {
                    q[r].resize(size_t(std::max(0, (s.bands - r + 3) / 4)) * s.band_words);
                    if (!q[r].empty())
                        dev_.read(s.addr + uint64_t(r) * s.stride * 16384, q[r].data(), q[r].size() * 32);
                }
                for (int b = 0; b < s.bands; ++b)
                    put(&q[b % 4][size_t(b / 4) * s.band_words], size_t(s.band_words) * 32);
            }
            continue;
        }
        for (size_t off = 0; off < w.words;) {
            const size_t n = std::min(buf.size(), size_t(w.words) - off);
            dev_.read(w.addr + off * 32, buf.data(), n * 32);
            put(buf.data(), n * 32);
            off += n;
        }
    }
    const auto digest = hash.finish();
    f.write(digest.data(), 64);
    f.close();
    if (!f)
        throw std::runtime_error("write weight cache failed: " + tmp);
    std::filesystem::rename(tmp, path);
}
void Llm::load_weights(const std::string &path, const std::string &model) {
    for (auto *w : weight_slots())
        if (w->words)
            throw std::runtime_error("load cache requires fresh weights");
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("open weight cache: " + path);
    Sha256 hash;
    auto get = [&](void *p, size_t n) {
        if (!f.read(static_cast<char *>(p), std::streamsize(n)))
            throw std::runtime_error("truncated weight cache");
        hash.update(p, n);
    };
    const auto expected = header(cfg_, model, weight_layout());
    std::string h(expected.size(), '\0');
    get(h.data(), h.size());
    if (h != expected)
        throw std::runtime_error("weight cache model/config/version mismatch");
    const auto slots = weight_slots();
    const auto sizes = weight_sizes();
    const auto roles = weight_roles();
    std::vector<Word> buf(32768);
    struct Rollback {
        const std::vector<WeightImage *> &slots;
        std::map<uint64_t, std::vector<Segment>> &placed;
        std::vector<uint64_t> added;
        bool done = false;
        ~Rollback() {
            if (!done) {
                for (auto *w : slots)
                    *w = {};
                for (uint64_t a : added)
                    placed.erase(a);
            }
        }
    } rollback{slots, placed_, {}};
    for (size_t i = 0; i < slots.size(); ++i) {
        char b[4], want[4];
        get(b, 4);
        word_count(sizes[i], want);
        if (std::memcmp(b, want, 4))
            throw std::runtime_error("weight cache shape mismatch");
        auto &w = *slots[i];
        w.words = sizes[i];
        size_t bytes = 0;
        auto segs = quarter_layout(roles[i], bytes);
        if (!segs.empty()) {
            // in quarters (Llm::place), a segment at a time
            w.addr = dev_.alloc(bytes);
            size_t words = 0;
            for (auto &s : segs) {
                s.addr += w.addr;
                buf.resize(size_t(s.bands) * s.band_words);
                get(buf.data(), buf.size() * 32);
                write_quarters(s, buf.data());
                words += buf.size();
            }
            if (words != w.words)
                throw std::runtime_error("weight cache: segments of " + roles[i]);
            placed_[w.addr] = segs;
            rollback.added.push_back(w.addr);
            dev_bytes_ += bytes;
            continue;
        }
        buf.resize(32768);
        w.addr = dev_.alloc(size_t(w.words) * 32);
        for (size_t off = 0; off < w.words;) {
            const size_t n = std::min(buf.size(), size_t(w.words) - off);
            get(buf.data(), n * 32);
            dev_.write(w.addr + off * 32, buf.data(), n * 32);
            off += n;
        }
        dev_bytes_ += size_t(w.words) * 32;
    }
    std::string digest(64, '\0');
    if (!f.read(digest.data(), 64) || digest != hash.finish() || f.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("weight cache checksum/trailer mismatch");
    norm0_.clear();
    rollback.done = true;
}
} // namespace g3rt
