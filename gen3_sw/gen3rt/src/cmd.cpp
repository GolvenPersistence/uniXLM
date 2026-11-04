#include "gen3rt/cmd.h"

#include <stdexcept>
#include <string>

namespace g3rt {
namespace {

[[noreturn]] void field_range_error(int lsb, int width, uint64_t value) {
    throw std::out_of_range("command field at " + std::to_string(lsb) + " width " + std::to_string(width) +
                            " out of range: " + std::to_string(value));
}

struct Fields {
    Cmd128 c;
    void set(int lsb, int width, uint64_t value) {
        if (width < 64 && (value >> width) != 0)
            field_range_error(lsb, width, value);
        if (lsb >= 64) c.hi |= value << (lsb - 64);
        else {
            c.lo |= value << lsb;
            if (width > 64 - lsb) c.hi |= value >> (64 - lsb);
        }
    }
};

}  // namespace

Cmd128 cmd_bind(uint8_t tag, const BindArgs& a) {
    Fields f;
    f.set(0, 8, g3::OP_BIND); f.set(8, 8, tag); f.set(16, 8, a.idx); f.set(24, 8, a.gen);
    // v3: base and extent in units of 8 words
    if ((a.base & 7) != 0 || (a.extent & 7) != 0 || a.extent == 0)
        throw std::invalid_argument("BIND base / extent not a multiple of 8 words: " + std::to_string(a.base) + " / " +
                                    std::to_string(a.extent));
    f.set(32, 12, a.base >> 3); f.set(44, 12, a.extent / 8 - 1);
    f.set(56, 1, a.order == Order::P32 || a.order == Order::VT);
    f.set(57, 2, uint8_t(a.kind)); f.set(59, 1, a.ext); f.set(60, 4, uint32_t(a.count - 1));
    if (a.quarters && (a.kind != Kind::Stream || (a.src >> 36) != 0 || (a.quarters >> 12) != 0))
        throw std::invalid_argument("BIND quarters " + std::to_string(a.quarters) + " (STREAM, source below 64 GiB)");
    f.set(64, 43, (a.src >> 5) | (uint64_t(a.quarters) << 31));
    if (a.kind == Kind::Append) f.set(107, 12, a.rows);
    else if (a.kind == Kind::Stream) f.set(107, 20, a.length ? a.length - 1 : 0);
    else f.set(119, 1, a.present);
    f.set(127, 1, a.order == Order::F32 || a.order == Order::VT);
    return f.c;
}

Cmd128 cmd_gemm(uint8_t tag, const GemmArgs& g) {
    static const int width_code[5] = {0, 1, 2, -1, 3};
    Fields f;
    f.set(0, 8, g3::OP_GEMM); f.set(8, 8, tag); f.set(16, 16, g.a); f.set(32, 16, g.b); f.set(48, 16, g.c);
    f.set(64, 16, g.d); f.set(80, 12, uint32_t(g.m - 1)); f.set(92, 12, uint32_t(g.n - 1));
    f.set(104, 8, uint32_t(g.k / 32 - 1)); f.set(112, 4, uint8_t(g.epi));
    f.set(116, 2, uint32_t(g.fold ? g.eps : width_code[g.width]));
    f.set(118, 4, uint32_t(g.a_heads - 1)); f.set(122, 4, uint32_t(g.d_heads - 1));
    f.set(126, 1, g.qk_norm);
    f.set(127, 1, g.fold);
    return f.c;
}

Cmd128 cmd_attn(uint8_t tag, const AttnArgs& a) {
    Fields f;
    f.set(0, 8, g3::OP_ATTN); f.set(8, 8, tag); f.set(16, 16, a.q); f.set(32, 16, a.k); f.set(48, 16, a.v);
    f.set(64, 16, a.o); f.set(80, 11, uint32_t(a.tokens - 1)); f.set(91, 4, uint32_t(a.q_heads - 1));
    f.set(95, 2, uint32_t(a.group - 1)); f.set(97, 11, uint32_t(a.keys - 1));
    f.set(108, 14, uint32_t(a.delta) & 0x3FFF); f.set(122, 3, uint32_t(a.dh / 32 - 1));
    return f.c;
}

static Cmd128 move(uint8_t op, uint8_t tag, uint16_t b, uint32_t offset, uint32_t words, uint64_t addr, bool last) {
    Fields f;
    f.set(0, 8, op); f.set(8, 8, tag); f.set(16, 16, b); f.set(32, 16, offset); f.set(48, 16, words - 1);
    f.set(64, 40, addr); f.set(104, 1, last);
    return f.c;
}

Cmd128 cmd_move_out(uint8_t tag, uint16_t src, uint32_t offset, uint32_t words, uint64_t addr) {
    return move(g3::OP_MOVE_OUT, tag, src, offset, words, addr, false);
}

Cmd128 cmd_move_in(uint8_t tag, uint16_t dst, uint32_t offset, uint32_t words, uint64_t addr, bool last) {
    return move(g3::OP_MOVE_IN, tag, dst, offset, words, addr, last);
}

Cmd128 cmd_fence(uint8_t tag) {
    Fields f;
    f.set(0, 8, g3::OP_FENCE); f.set(8, 8, tag);
    return f.c;
}

Cmd128 cmd_wait(uint8_t tag, uint16_t b, bool mode) {
    Fields f;
    f.set(0, 8, g3::OP_WAIT); f.set(8, 8, tag); f.set(16, 16, b); f.set(32, 1, mode);
    return f.c;
}

Cmd128 cmd_unbind(uint8_t tag, uint16_t b) {
    Fields f;
    f.set(0, 8, g3::OP_UNBIND); f.set(8, 8, tag); f.set(16, 16, b);
    return f.c;
}

}  // namespace g3rt
