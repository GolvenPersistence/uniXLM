// RMSNorm numerics: the VRU sequence of g2lib.rmsnorm_rows, 1/rms of the RMSNorm fold (g3lib.inv_rms_rows).
#include "fp_strict.h"
#include "g3emu_kernels.h"
#include "tables.h"

namespace g3 {

uint32_t inverse_count(int n) { return f32_from_double(1.0 / double(n)); }

uint32_t epsilon_bits(int profile) { return tables::EPSILON[profile & 3]; }

uint32_t rsqrt_bits(uint32_t v) {
    uint32_t y = 0x5F375A86u - (v >> 1);
    for (int i = 0; i < 2; ++i) {
        const uint32_t square = f32_mul(y, y);
        const uint32_t xy2 = f32_mul(v, square);
        const uint32_t half = f32_mul(xy2, 0x3F000000u);
        const uint32_t corr = f32_add(f32_mul(half, 0xBF800000u), 0x3FC00000u);
        y = f32_mul(y, corr);
    }
    return y;
}

uint32_t inv_rms_row(const uint32_t* x, int w, int eps_profile) {
    uint32_t lanes[16] = {0};
    for (int j = 0; j < w; ++j) lanes[j % 16] = f32_add(lanes[j % 16], f32_mul(x[j], x[j]));
    for (int width = 16; width > 1; width /= 2)
        for (int i = 0; i < width / 2; ++i) lanes[i] = f32_add(lanes[2 * i], lanes[2 * i + 1]);
    const uint32_t var = f32_mul(lanes[0], inverse_count(w));
    return rsqrt_bits(f32_add(var, epsilon_bits(eps_profile)));
}

void rmsnorm_row(const uint32_t* x, const uint32_t* gamma, int w, int eps_profile, uint32_t* out) {
    const uint32_t s = inv_rms_row(x, w, eps_profile);
    for (int j = 0; j < w; ++j) {
        const uint32_t y = f32_add(f32_mul(x[j], s), 0);
        out[j] = f32_add(f32_mul(y, gamma[j]), 0);
    }
}

}  // namespace g3
