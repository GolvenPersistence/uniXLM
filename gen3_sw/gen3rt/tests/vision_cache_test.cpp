// Real projector weights + fixed patch input: validates cache transparency, misses and upload invalidation.
#include "gen3rt/board.h"
#include "gen3rt/profile.h"
#include "gen3rt/sha256.h"
#include "gen3rt/vit.h"
#include "ggml.h"
#include "gguf.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
using namespace g3rt;
int main(int argc, char **argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "usage: vision_cache_test mmproj.gguf patches.f32 [timed uncached encodes]\n";
        return 2;
    }
    ggml_context *gc = nullptr;
    gguf_init_params gp{false, &gc};
    auto *gf = gguf_init_from_file(argv[1], gp);
    assert(gf);
    std::map<std::string, std::vector<float>> keep;
    auto tensor = [&](const std::string &n) {
        auto *t = ggml_get_tensor(gc, n.c_str());
        if (!t)
            throw std::runtime_error(n);
        return t;
    };
    auto W = [&](const std::string &n) {
        auto *t = tensor(n);
        auto &v = keep[n];
        v.resize(size_t(ggml_nelements(t)));
        if (t->type == GGML_TYPE_F32)
            std::memcpy(v.data(), t->data, v.size() * 4);
        else
            ggml_get_type_traits(t->type)->to_float(t->data, v.data(), v.size());
        return v.data();
    };
    VitConfig c;
    c.n_patch = int(tensor("v.position_embd.weight")->ne[1]);
    c.proj_out = int(tensor("mm.model.fc.weight")->ne[1]);
    c.n_layer = 0;
    while (ggml_get_tensor(gc, ("v.blk." + std::to_string(c.n_layer) + ".attn_q.weight").c_str()))
        ++c.n_layer;
    const bool swapped = tensor("v.blk.0.ffn_down.weight")->ne[1] > tensor("v.blk.0.ffn_down.weight")->ne[0];
    const std::string f1 = swapped ? "ffn_down" : "ffn_up", f2 = swapped ? "ffn_up" : "ffn_down";
    VitWeights w;
    w.patch_w = W("v.patch_embd.weight");
    w.patch_b = W("v.patch_embd.bias");
    w.pos = W("v.position_embd.weight");
    w.post_w = W("v.post_ln.weight");
    w.post_b = W("v.post_ln.bias");
    w.proj_w = W("mm.model.fc.weight");
    for (int l = 0; l < c.n_layer; ++l) {
        auto b = "v.blk." + std::to_string(l) + ".";
        VitLayerWeights q;
        q.ln1_w = W(b + "ln1.weight");
        q.ln1_b = W(b + "ln1.bias");
        q.ln2_w = W(b + "ln2.weight");
        q.ln2_b = W(b + "ln2.bias");
        q.q_w = W(b + "attn_q.weight");
        q.q_b = W(b + "attn_q.bias");
        q.k_w = W(b + "attn_k.weight");
        q.k_b = W(b + "attn_k.bias");
        q.v_w = W(b + "attn_v.weight");
        q.v_b = W(b + "attn_v.bias");
        q.o_w = W(b + "attn_out.weight");
        q.o_b = W(b + "attn_out.bias");
        q.fc1_w = W(b + f1 + ".weight");
        q.fc1_b = W(b + f1 + ".bias");
        q.fc2_w = W(b + f2 + ".weight");
        q.fc2_b = W(b + f2 + ".bias");
        w.layers.push_back(q);
    }
    std::vector<float> patches(size_t(c.n_patch) * c.patch_k),
        out(size_t(c.n_patch / (c.shuffle * c.shuffle)) * c.proj_out), ref(out.size());
    std::ifstream f(argv[2], std::ios::binary);
    assert(f.read(reinterpret_cast<char *>(patches.data()), patches.size() * 4));
    assert(f.peek() == EOF);
    BoardDevice dev;
    Vit vit(dev, c);
    vit.upload(w);
    // per encode: its time, the host's wait for the Node and the host stages (Vit::host_seconds) of this encode, and
    // its CLOCK_MONOTONIC window (t0, t1: the rail sampler's clock)
    double node0 = 0, host0[4] = {};
    auto mono = [](std::chrono::steady_clock::time_point t) {
        return std::chrono::duration<double>(t.time_since_epoch()).count();
    };
    auto run = [&](const char *label) {
        auto t = std::chrono::steady_clock::now();
        vit.encode(patches.data(), out.data());
        const auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t).count();
        const double *h = vit.host_seconds();
        std::cout << label << " ms=" << ms << " node_s=" << vit.node_seconds() - node0 << " host_s="
                  << h[0] - host0[0] << "," << h[1] - host0[1] << "," << h[2] - host0[2] << "," << h[3] - host0[3]
                  << " hits=" << vit.cache_hits() << " misses=" << vit.cache_misses() << std::fixed
                  << std::setprecision(4) << " t0=" << mono(t) << " t1=" << mono(t1) << std::defaultfloat
                  << std::setprecision(6) << std::endl;
        node0 = vit.node_seconds();
        for (int i = 0; i < 4; ++i)
            host0[i] = h[i];
        for (auto x : out)
            assert(std::isfinite(x));
    };
    vit.set_encode_cache(false);
    run("uncached");
    ref = out;
    vit.set_encode_cache(true);
    run("cache_miss");
    assert(!std::memcmp(out.data(), ref.data(), out.size() * 4));
    for (int i = 0; i < 5; ++i) {
        run("cache_hit");
        assert(!std::memcmp(out.data(), ref.data(), out.size() * 4));
    }
    const auto original = ref;
    patches[0] += 1.f;
    run("changed_patch_miss");
    assert(vit.cache_hits() == 5);
    ref = out;
    vit.set_encode_cache(false);
    run("changed_patch_uncached");
    assert(!std::memcmp(out.data(), ref.data(), out.size() * 4));
    patches[0] -= 1.f; // restore exactly from the original file below
    f.clear();
    f.seekg(0);
    f.read(reinterpret_cast<char *>(patches.data()), patches.size() * 4);
    vit.set_encode_cache(true);
    run("restore_miss");
    vit.upload(w);
    const auto misses = vit.cache_misses();
    run("after_upload_miss");
    assert(vit.cache_misses() == misses + 1);
    assert(!std::memcmp(out.data(), original.data(), out.size() * 4));
    vit.set_encode_cache(false);
    for (int i = 0, n = argc == 4 ? std::atoi(argv[3]) : 0; i < n; ++i) {
        run("timed");
        assert(!std::memcmp(out.data(), original.data(), out.size() * 4));
    }
    if (const char *pv = std::getenv("VISION_PROFILE"); pv && *pv == '1') {
        // one encode with every command reporting: the Node's cycles per command role (no overlap with the host)
        Profile prof;
        const char *dump = std::getenv("VISION_PROFILE_DUMP");   // every command's completion cycle (Profile)
        if (dump && !prof.open_dump(dump))
            return 1;
        vit.set_profile(&prof);
        run("profiled");
        vit.set_profile(nullptr);
        assert(!std::memcmp(out.data(), original.data(), out.size() * 4));
        prof.print(stdout, dev.core_mhz());
    }
    Sha256 sha;
    sha.update(out.data(), out.size() * 4);
    std::cout << "VISION_CACHE_PASS output_sha256=" << sha.finish() << std::endl;
    gguf_free(gf);
    ggml_free(gc);
}
