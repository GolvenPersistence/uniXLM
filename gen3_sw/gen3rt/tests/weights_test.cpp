#include "gen3rt/llm.h"
#include "gen3rt/sha256.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <iterator>
using namespace g3rt;
struct Memory : Device {
    std::vector<unsigned char> bytes;
    uint64_t alloc(size_t n) override {
        auto p = bytes.size();
        bytes.resize(p + ((n + 4095) / 4096) * 4096);
        return p;
    }
    void write(uint64_t a, const void *p, size_t n) override {
        assert(a + n <= bytes.size());
        std::memcpy(bytes.data() + a, p, n);
    }
    void read(uint64_t a, void *p, size_t n) override {
        assert(a + n <= bytes.size());
        std::memcpy(p, bytes.data() + a, n);
    }
    void submit(const std::vector<Cmd128> &) override { throw std::runtime_error("unexpected execute"); }
    void sync() override {}
    std::string name() const override { return "memory"; }
};
std::string contents(const std::string &p) {
    std::ifstream f(p, std::ios::binary);
    return {std::istreambuf_iterator<char>(f), {}};
}
void put(const std::string &p, const std::string &b) {
    std::ofstream f(p, std::ios::binary);
    f.write(b.data(), b.size());
}
int main(int argc, char **argv) {
    assert(argc == 2);
    const auto scratch =
        std::filesystem::path(argv[1]) / std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
    std::filesystem::create_directories(scratch);
    std::filesystem::current_path(scratch);
    for (auto pair :
         {std::pair<std::string, std::string>{"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
          {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
          {std::string(1000000, 'a'), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"}}) {
        Sha256 sha;
        for (char c : pair.first)
            sha.update(&c, 1);
        assert(sha.finish() == pair.second);
    }
    LlmConfig cfg;
    cfg.n_layer = 2;
    cfg.n_embd = 64;
    cfg.n_ff = 64;
    cfg.n_head = 2;
    cfg.n_kv = 1;
    cfg.dh = 32;
    cfg.vocab = 128;
    cfg.ctx = 32;
    std::vector<float> w(128 * 64), g(64);
    for (size_t i = 0; i < w.size(); ++i)
        w[i] = float(std::sin(double(i) * 0.17) * 0.5);
    for (int i = 0; i < 64; ++i)
        g[i] = 1.f + float(i) * 0.001f;
    const std::string id(64, 'a');
    Memory a, b, c;
    Llm direct(a, cfg), old(b, cfg), loaded(c, cfg);
    for (int l = 0; l < 2; ++l) {
        direct.upload_layer(
            l, {w.data(), w.data(), w.data(), w.data(), w.data(), w.data() + 17, w.data(), g.data(), g.data()});
        for (auto role : {"attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate", "ffn_up", "ffn_down"})
            assert(old.upload_tensor(l, role, std::string(role) == "ffn_up" ? w.data() + 17 : w.data(),
                                     std::string(role) == "attn_k" || std::string(role) == "attn_v" ? 32 : 64, 64));
        for (auto role : {"attn_norm", "ffn_norm"})
            assert(old.upload_tensor(l, role, g.data(), 1, 64));
    }
    direct.upload_output(g.data(), w.data());
    old.upload_output(g.data(), w.data());
    direct.save_weights("direct.bin", id);
    old.save_weights("legacy.bin", id);
    const auto good = contents("direct.bin");
    assert(good == contents("legacy.bin"));
    loaded.load_weights("direct.bin", id);
    assert(loaded.complete());
    loaded.save_weights("loaded.bin", id);
    assert(good == contents("loaded.bin"));
    int rejected = 0;
    auto reject = [&](std::string data, std::string identity = std::string(64, 'a')) {
        put("bad.bin", data);
        Memory m;
        Llm l(m, cfg);
        try {
            l.load_weights("bad.bin", identity);
            assert(false);
        } catch (const std::runtime_error &) {
            assert(!l.complete());
            ++rejected;
        }
    };
    for (size_t off : {size_t(0), size_t(8), size_t(72), size_t(100), size_t(104), good.size() - 1}) {
        auto x = good;
        x[off] ^= 1;
        reject(x);
    }
    reject(good.substr(0, good.size() - 1));
    reject(good + "x");
    reject(good, std::string(64, 'b'));
    std::cout << "PASS SHA256 vectors, host interleave vs legacy, cache roundtrip, " << rejected
              << " corruption/identity rejections\n";
}
