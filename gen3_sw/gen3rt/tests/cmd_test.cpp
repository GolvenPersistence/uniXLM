#include "gen3rt/cmd.h"
#include <cassert>
#include <iostream>
#include <random>
namespace g3rt {
Cmd128 reference_bind(uint8_t, const BindArgs &);
Cmd128 reference_gemm(uint8_t, const GemmArgs &);
Cmd128 reference_attn(uint8_t, const AttnArgs &);
Cmd128 reference_move_out(uint8_t, uint16_t, uint32_t, uint32_t, uint64_t);
Cmd128 reference_move_in(uint8_t, uint16_t, uint32_t, uint32_t, uint64_t, bool);
Cmd128 reference_fence(uint8_t);
Cmd128 reference_wait(uint8_t, uint16_t, bool);
Cmd128 reference_unbind(uint8_t, uint16_t);
} // namespace g3rt
using namespace g3rt;
void eq(Cmd128 a, Cmd128 b) { assert(a.lo == b.lo && a.hi == b.hi); }
int main() {
    std::mt19937_64 rng(20261021);
    for (int i = 0; i < 100000; ++i) {
        uint8_t tag = rng();
        BindArgs b;
        b.idx = rng();
        b.gen = rng();
        b.base = (rng() % 4096) * 8;
        b.extent = (rng() % 4096 + 1) * 8;
        b.order = Order(rng() % 4);
        b.kind = Kind(rng() % 4);
        b.ext = rng() % 2;
        b.count = rng() % 16 + 1;
        b.src = (rng() & ((1ull << 43) - 1)) << 5;
        b.rows = rng() % 4096;
        b.length = rng() % (1 << 20) + 1;
        b.present = rng() % 2;
        eq(cmd_bind(tag, b), reference_bind(tag, b));
        GemmArgs g;
        g.a = rng();
        g.b = rng();
        g.c = rng();
        g.d = rng();
        g.m = rng() % 4096 + 1;
        g.n = rng() % 4096 + 1;
        g.k = (rng() % 256 + 1) * 32;
        g.epi = Epi(rng() % 8);
        g.fold = rng() % 2;
        g.eps = rng() % 4;
        g.width = 1 << (rng() % 3);
        g.a_heads = rng() % 16 + 1;
        g.d_heads = rng() % 16 + 1;
        eq(cmd_gemm(tag, g), reference_gemm(tag, g));
        AttnArgs a;
        a.q = rng();
        a.k = rng();
        a.v = rng();
        a.o = rng();
        a.tokens = rng() % 2048 + 1;
        a.q_heads = rng() % 16 + 1;
        a.group = rng() % 4 + 1;
        a.keys = rng() % 2048 + 1;
        a.delta = int(rng() % 16384) - 8192;
        a.dh = (rng() % 8 + 1) * 32;
        eq(cmd_attn(tag, a), reference_attn(tag, a));
        const uint16_t binding = rng();
        const uint32_t off = rng() % 65536, words = rng() % 65536 + 1;
        const uint64_t addr = rng() & ((1ull << 40) - 1);
        const bool last = rng() % 2;
        eq(cmd_move_in(tag, binding, off, words, addr, last), reference_move_in(tag, binding, off, words, addr, last));
        eq(cmd_move_out(tag, binding, off, words, addr), reference_move_out(tag, binding, off, words, addr));
        eq(cmd_fence(tag), reference_fence(tag));
        eq(cmd_wait(tag, binding, last), reference_wait(tag, binding, last));
        eq(cmd_unbind(tag, binding), reference_unbind(tag, binding));
    }
    for (int n : {0, 4097}) {
        GemmArgs g;
        g.m = n;
        bool x = false, y = false;
        try {
            cmd_gemm(0, g);
        } catch (const std::exception &) {
            x = true;
        }
        try {
            reference_gemm(0, g);
        } catch (const std::exception &) {
            y = true;
        }
        assert(x && y);
    }
    std::cout << "PASS 800000 command encodings against frozen bitwise encoder; invalid dimensions rejected\n";
}
