// GEMM epilogues (tb/gen2/core_vectors.py): 64-segment PWL SiLU / GELU, RoPE pair rotation.
#include "fp_strict.h"
#include "g3emu_kernels.h"
#include "tables.h"

namespace g3 {

uint32_t silu_pwl(uint32_t x) {
    const bool negative = (x >> 31) != 0;
    const uint32_t magnitude = x & 0x7FFFFFFFu;
    uint32_t slope, bias;
    if (magnitude >= 0x41000000u) {
        slope = negative ? 0u : 0x3F800000u;
        bias = 0;
    } else {
        const double v = double(f32_value(magnitude));
        const int index = v < 2 ? int(v * 16) : v < 4 ? 32 + int((v - 2) * 8) : 48 + int((v - 4) * 4);
        slope = tables::SILU[3 * index + (negative ? 1 : 0)];
        bias = tables::SILU[3 * index + 2];
    }
    return f32_add(f32_mul(x, slope), bias);
}

uint32_t gelu_pwl(uint32_t x) {
    const bool negative = (x >> 31) != 0;
    const uint32_t magnitude = x & 0x7FFFFFFFu;
    uint32_t slope, bias;
    if (magnitude >= 0x40800000u) {
        slope = negative ? 0u : 0x3F800000u;
        bias = 0;
    } else {
        const double v = double(f32_value(magnitude));
        int index = int(v * 16);
        if (index > 63) index = 63;
        slope = tables::GELU[3 * index + (negative ? 1 : 0)];
        bias = tables::GELU[3 * index + 2];
    }
    return f32_add(f32_mul(x, slope), bias);
}

void rope_pairs(const uint32_t* x, const uint32_t* cs, uint32_t* out, int n) {
    for (int i = 0; i + 1 < n; i += 2) {
        const uint32_t x0 = x[i], x1 = x[i + 1];
        const uint32_t cos_ = cs[i], sin_ = cs[i + 1];
        const uint32_t neg_sin = sin_ ^ 0x80000000u;
        out[i] = f32_add(f32_mul(x0, cos_), f32_mul(x1, neg_sin));
        out[i + 1] = f32_add(f32_mul(x1, cos_), f32_mul(x0, sin_));
    }
}

}  // namespace g3
