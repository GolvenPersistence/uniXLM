// FP32 helpers, MXFP8 quantise / decode (Gen1 reference model), bf48 rounding.
#include "fp_strict.h"
#include <cmath>
#include <cstring>

#include "bits.h"
#include "g3emu_kernels.h"

namespace g3 {

// ---------------------------------------------------------------------------------------------------------------
// FP32
// ---------------------------------------------------------------------------------------------------------------
float f32_value(uint32_t bits) {
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

uint32_t f32_bits(float v) {
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
}

bool f32_is_nan(uint32_t x) { return (x & 0x7F800000u) == 0x7F800000u && (x & 0x7FFFFFu) != 0; }

uint32_t f32_add(uint32_t x, uint32_t y) {
    const uint32_t r = f32_bits(f32_value(x) + f32_value(y));
    if (f32_is_nan(r) || f32_is_nan(x) || f32_is_nan(y)) return NAN32;
    return r;
}

uint32_t f32_mul(uint32_t x, uint32_t y) {
    const uint32_t r = f32_bits(f32_value(x) * f32_value(y));
    return f32_is_nan(r) ? NAN32 : r;
}

uint32_t f32_sub(uint32_t x, uint32_t y) { return f32_bits(f32_value(x) - f32_value(y)); }

uint32_t f32_from_double(double v) { return f32_bits(static_cast<float>(v)); }

// ---------------------------------------------------------------------------------------------------------------
// E4M3 / E8M0
// ---------------------------------------------------------------------------------------------------------------
namespace {
struct E4m3Table {
    int32_t q[256];
    E4m3Table() {
        for (int v = 0; v < 256; ++v) {
            if ((v & 0x7F) == 0x7F) { q[v] = 0; continue; }
            const int exponent = (v >> 3) & 0xF;
            const int mantissa = v & 7;
            const int magnitude = exponent == 0 ? mantissa : (8 + mantissa) << (exponent - 1);
            q[v] = (v & 0x80) ? -magnitude : magnitude;
        }
    }
};
const E4m3Table kE4m3;

// _round_right_even(value, shift) for shift > 0
uint64_t round_right_even(uint64_t value, int shift) {
    if (shift <= 0) return value << (-shift);
    if (shift >= 63) return 0;   // value < 2^24 here: quotient 0 and remainder below half
    const uint64_t quotient = value >> shift;
    const uint64_t remainder = value & ((uint64_t(1) << shift) - 1);
    const uint64_t half = uint64_t(1) << (shift - 1);
    if (remainder > half || (remainder == half && (quotient & 1))) return quotient + 1;
    return quotient;
}
}  // namespace

int32_t e4m3_q9(uint8_t code) { return kE4m3.q[code]; }

bool e4m3_is_nan(uint8_t code) { return (code & 0x7F) == 0x7F; }

uint8_t choose_scale(uint32_t max_bits) {
    const uint32_t maximum = max_bits & 0x7FFFFFFFu;
    const uint32_t exponent = (maximum >> 23) & 0xFF;
    const uint32_t fraction = maximum & 0x7FFFFF;
    if (exponent == 0xFF && fraction) return 0xFF;
    if (maximum == 0) return 127;
    if (exponent == 0xFF) return 254;
    int scale = int(exponent) - 8 + (fraction > 0x600000 ? 1 : 0);
    if (scale < 0) scale = 0;
    if (scale > 254) scale = 254;
    return uint8_t(scale);
}

uint32_t row_max_bits(const uint32_t* v, int n) {
    uint32_t maximum = 0;
    for (int i = 0; i < n; ++i) {
        if (f32_is_nan(v[i])) return NAN32;
        const uint32_t magnitude = v[i] & 0x7FFFFFFFu;
        if (magnitude > maximum) maximum = magnitude;
    }
    return maximum;
}

uint8_t quantize_e4m3(uint32_t value, uint8_t scale) {
    const uint32_t sign = value >> 31;
    const uint32_t exponent_field = (value >> 23) & 0xFF;
    const uint32_t fraction = value & 0x7FFFFF;
    if (exponent_field == 0xFF && fraction) return 0x7F;
    if ((value & 0x7FFFFFFFu) == 0) return uint8_t(sign << 7);
    if (scale == 0xFF) return 0x7F;
    if (exponent_field == 0xFF) return uint8_t((sign << 7) | 0x7E);
    int source_exponent;
    uint64_t significand;
    if (exponent_field == 0) {
        const int msb = bit_length32(fraction) - 1;
        source_exponent = msb - 149;
        significand = uint64_t(fraction) << (23 - msb);
    } else {
        source_exponent = int(exponent_field) - 127;
        significand = (uint64_t(1) << 23) | fraction;
    }
    int target_exponent = source_exponent - int(scale) + 127;
    uint32_t magnitude_code;
    if (target_exponent > 8) {
        magnitude_code = 0x7E;
    } else if (target_exponent >= -6) {
        uint64_t rounded = round_right_even(significand, 20);
        if (rounded >= 16) {
            rounded = 8;
            target_exponent += 1;
        }
        if (target_exponent > 8 || (target_exponent == 8 && rounded > 14))
            magnitude_code = 0x7E;
        else
            magnitude_code = uint32_t((target_exponent + 7) * 8 + int(rounded) - 8);
    } else {
        const uint64_t rounded = round_right_even(significand, 14 - target_exponent);
        magnitude_code = rounded >= 8 ? 0x08 : uint32_t(rounded);
    }
    return uint8_t((sign << 7) | magnitude_code);
}

uint32_t decode_e4m3(uint8_t code, uint8_t scale) {
    if (scale == 0xFF || (code & 0x7F) == 0x7F) return NAN32;
    const int32_t q = kE4m3.q[code];
    if (q == 0) return uint32_t(code & 0x80) << 24;
    return f32_from_double(std::ldexp(double(q), int(scale) - 136));
}

void quantize_block32(const uint32_t* v, uint8_t* codes, uint8_t* scale) {
    const uint8_t s = choose_scale(row_max_bits(v, 32));
    *scale = s;
    for (int k = 0; k < 32; ++k) codes[k] = quantize_e4m3(v[k], s);
}

void quantize_matrix(const uint32_t* v, int rows, int cols, int ld, uint8_t* codes, uint8_t* scales) {
    const int groups = cols / 32;
    for (int i = 0; i < rows; ++i)
        for (int g = 0; g < groups; ++g)
            quantize_block32(v + size_t(i) * ld + 32 * g, codes + size_t(i) * cols + 32 * g,
                             scales + size_t(i) * groups + g);
}

// ---------------------------------------------------------------------------------------------------------------
// bf48
// ---------------------------------------------------------------------------------------------------------------
uint32_t bf48_to_fp32(const Bf48& a) {
    if (!a.have || a.m == 0) return 0;
    const uint64_t magnitude = a.m < 0 ? uint64_t(-a.m) : uint64_t(a.m);
    const int64_t lead = int64_t(a.e) + bit_length64(magnitude) - 1;   // position of the leading one
    if (lead < -126) return 0;
    return f32_from_double(std::ldexp(double(a.m), a.e));
}

}  // namespace g3
