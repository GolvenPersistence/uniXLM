// Command encoders of the Gen3 Node v3 (docs/gen3/node_v3.md s3; the current RTL fields of s3.1), bit-exact with
// tb/gen2/g2lib.py (cmd_bind, cmd_gemm, cmd_attn, cmd_move_out, cmd_simple).
#pragma once

#include <cstdint>

#include "g3emu.h"

namespace g3rt {

using g3::Cmd128;

enum class Order : uint8_t { P8, P32, F32, VT };
enum class Kind : uint8_t { Static = 0, Append = 1, Stream = 2 };
enum class Epi : uint8_t { None = 0, Add = 1, Mul = 2, Silu = 3, Gelu = 4, SiluMul = 5, Rope = 6 };

// binding operand of a command: {gen, idx}
inline uint16_t binding(uint8_t idx, uint8_t gen) { return uint16_t(uint16_t(gen) << 8 | idx); }

struct BindArgs {
    uint8_t idx = 0, gen = 0;
    uint32_t base = 0, extent = 0;      // UB words
    Order order = Order::P8;
    Kind kind = Kind::Static;
    bool ext = false;                   // external writer (MOVE_IN / host push)
    int count = 1;                      // array elements
    uint64_t src = 0;                   // STREAM: byte address (32-byte aligned)
    uint32_t length = 0;                // STREAM: words
    uint32_t quarters = 0;              // STREAM, a GEMM's B: band b at quarter b % 4, position b / 4, the quarter
                                        // stride in 16 KiB units (0: bands in order; the source below 64 GiB)
    uint32_t rows = 0;                  // APPEND: initial rows
    bool present = false;               // STATIC: the data are already in the range
};
Cmd128 cmd_bind(uint8_t tag, const BindArgs& a);

struct GemmArgs {
    uint16_t a = 0, b = 0, c = 0, d = 0;
    int m = 1, n = 32, k = 32;
    Epi epi = Epi::None;
    int width = 0;
    int a_heads = 1, d_heads = 1;
    bool fold = false;                  // RMSNorm fold: ADD = producer, else consumer
    int eps = 2;                        // fold eps profile (g3::epsilon_bits)
    // q / k RMSNorm per head (R56, [126]): a fold consumer with ROPE; heads of 64 or 128 columns (n / d_heads; 128:
    // m <= 16); each head's 1/rms (fold eps profile) and the FP32 gain row at C idx + 1 (C's generation) before RoPE
    bool qk_norm = false;
};
Cmd128 cmd_gemm(uint8_t tag, const GemmArgs& g);

struct AttnArgs {
    uint16_t q = 0, k = 0, v = 0, o = 0;
    int tokens = 1, q_heads = 1, group = 1, keys = 1, delta = 0, dh = 64;
};
Cmd128 cmd_attn(uint8_t tag, const AttnArgs& a);

Cmd128 cmd_move_out(uint8_t tag, uint16_t src, uint32_t offset, uint32_t words, uint64_t addr);
Cmd128 cmd_move_in(uint8_t tag, uint16_t dst, uint32_t offset, uint32_t words, uint64_t addr, bool last);
Cmd128 cmd_fence(uint8_t tag);
Cmd128 cmd_wait(uint8_t tag, uint16_t b, bool mode = false);
Cmd128 cmd_unbind(uint8_t tag, uint16_t b);

}  // namespace g3rt
