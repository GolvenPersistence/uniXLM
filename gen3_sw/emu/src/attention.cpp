// Gen3 attention numerics, one query row at a time: 64-key block-max softmax, bf48 QK and PV
// (tb/gen3/attention_numerics.py Gen3Attention.run with G3_ACC=bf48, tb/gen2/block_max_attention.py helpers,
// tb/node/native_attention_numpy.py reciprocal / output quantisation).
#include "fp_strict.h"
#include <cmath>
#include <vector>

#include "g3emu_kernels.h"
#include "tables.h"

namespace g3 {

namespace {

constexpr int P_SCALE = 119;
constexpr int FRAC = 16;
constexpr int CFRAC = 12;
constexpr int64_t K_LIMIT = 1 << 14;
constexpr uint32_t LOG2E_BITS = 0x3FB8AA3Bu;   // np.float32(1.4426950408889634)

// floor(x * 2^frac) for finite FP32 x (exact in FP64)
int64_t floor_fixed(uint32_t x, int frac) {
    return int64_t(std::floor(std::ldexp(double(f32_value(x)), frac)));
}

bool f32_finite(uint32_t x) { return (x & 0x7F800000u) != 0x7F800000u; }

// E4M3 code (scale 119) of 2^d for FP32 d <= 0 (-inf allowed): block_max_attention.p_codes
int p_code(uint32_t d) {
    const bool live = f32_finite(d) && f32_value(d) >= -18.0f;
    if (!live) return 0;
    const int64_t u = floor_fixed(d, FRAC) + (int64_t(8) << FRAC);
    int64_t n = u >> FRAC;
    const int64_t f = u & ((int64_t(1) << FRAC) - 1);
    int64_t m = 0;
    for (int i = 0; i < 8; ++i) m += f >= tables::T_NORMAL[i] ? 1 : 0;
    if (m == 8) { n += 1; m = 0; }
    if (n >= -6) return int((n + 7) * 8 + m);
    const int64_t w = u + (int64_t(9) << FRAC);
    int64_t j = (w > tables::T_SUB[0]) ? 1 : 0;
    for (int i = 1; i < 8; ++i) j += w >= tables::T_SUB[i] ? 1 : 0;
    return int(j == 8 ? 8 : j);
}

// k = ceil(m), c in [128, 256]: block_max_attention.correction (m finite FP32)
void correction(uint32_t m0, int64_t* k_out, int64_t* c_out) {
    const float mv = f32_value(m0);
    const bool sat = std::fabs(mv) >= float(K_LIMIT);
    const uint32_t m = sat ? 0u : m0;
    const double md = double(f32_value(m));
    const int64_t mf = floor_fixed(m, CFRAC);
    const double whole = std::floor(md);
    const int64_t k = int64_t(whole) + (md != whole ? 1 : 0);
    const int64_t g = mf - (k - 1) * (int64_t(1) << CFRAC);
    const int64_t i = (g >> 8) < 15 ? (g >> 8) : 15;
    const int64_t r = g >= 4096 ? 256 : (g & 255);
    const int64_t val = tables::C_TABLE[i] + (((tables::C_TABLE[i + 1] - tables::C_TABLE[i]) * r + 128) >> 8);
    const int64_t c = (val + 128) >> 8;
    if (sat) {
        *k_out = mv < 0 ? -K_LIMIT : K_LIMIT;
        *c_out = 256;
    } else {
        *k_out = k;
        *c_out = c;
    }
}

// FP32 x * 2^-delta by exponent subtraction (delta >= 0)
uint32_t scale_down(uint32_t bits, int64_t delta) {
    const int64_t e = (bits >> 23) & 255;
    if (delta == 0 || e == 0 || e == 255) return bits;
    if (e <= delta) return bits & 0x80000000u;
    return uint32_t(int64_t(bits) - (delta << 23));
}

// FP32 c/256 * 2^shift by exponent arithmetic
uint32_t scaled_correction(int64_t c, int64_t shift) {
    const int64_t exponent = (c == 256 ? 127 : 126) + shift;
    const int64_t mantissa = c == 256 ? 0 : (c - 128) << 16;
    if (exponent >= 255) return 0x7F800000u;
    if (exponent <= 0) return 0;
    return uint32_t((exponent << 23) | mantissa);
}

// Gen1 reciprocal (native_attention_numpy.reciprocal), x > 0 finite
uint32_t reciprocal(uint32_t bits) {
    const int exponent = int((bits >> 23) & 255) - 127;
    const uint32_t frac = bits & 0x7FFFFF;
    const int index = int(frac >> 18);
    const int64_t part = frac & ((1u << 18) - 1);
    const int64_t base = tables::RECIP_LUT[index], following = tables::RECIP_LUT[index + 1];
    const int64_t v = (base - following) * part;   // rint(v / 2^18), half to even
    int64_t q = v >> 18;
    const int64_t rem = v & ((int64_t(1) << 18) - 1);
    if (rem > (int64_t(1) << 17) || (rem == (int64_t(1) << 17) && (q & 1))) q += 1;
    const int64_t mantissa = base - q;
    return f32_from_double(std::ldexp(double(mantissa), -23 - exponent));
}

// E8M0 row scale of the O output (native_attention_numpy.scale): 127 for 0
uint8_t output_scale(uint32_t maximum) {
    if (maximum == 0) return 127;   // also -0 cannot occur: maximum is an |x|
    int s = int((maximum >> 23) & 255) - 8 + ((maximum & 0x7FFFFF) > 0x600000 ? 1 : 0);
    if (s < 0) s = 0;
    if (s > 254) s = 254;
    return uint8_t(s);
}

}  // namespace

uint32_t attn_qk_scale(int groups) {
    // np.float32(1 / math.sqrt(dh)) (the RTL table qk_scale), times np.float32(log2 e) in FP32
    const uint32_t s = f32_from_double(1.0 / std::sqrt(double(32 * groups)));
    return f32_bits(f32_value(s) * f32_value(LOG2E_BITS));
}

void attention_row(const AttnKV& kv, const uint8_t* q_codes, const uint8_t* q_scales, int keys, int visible,
                   uint8_t* o_codes, uint8_t* o_scale) {
    const int dh = kv.dh;
    const int G = dh / 32;
    const int cap = kv.cap;
    const int kgroups = cap / 32;
    const uint32_t big_s = attn_qk_scale(G);
    int32_t qq[256];
    for (int t = 0; t < dh; ++t) qq[t] = e4m3_q9(q_codes[t]);

    int64_t ref = 0;
    bool have = false;
    uint32_t den = 0;
    std::vector<Bf48> o(dh);
    uint32_t scores[64];
    int32_t pi[64];
    bool allowed[64];
    const int blocks = (keys + 63) / 64;
    for (int jb = 0; jb < blocks; ++jb) {
        const int k0 = jb * 64;
        const int width = keys - k0 < 64 ? keys - k0 : 64;
        bool present = false;
        for (int kk = 0; kk < 64; ++kk) {
            scores[kk] = 0;
            allowed[kk] = kk < width && k0 + kk < visible;
            present |= allowed[kk];
        }
        for (int kk = 0; kk < width; ++kk) {
            const uint8_t* kc = kv.k_codes + size_t(k0 + kk) * dh;
            const uint8_t* ks = kv.k_scales + size_t(k0 + kk) * G;
            Bf48 s;
            for (int g = 0; g < G; ++g) {
                int64_t x = 0;
                if (kv.k_q9) {
                    const int32_t* kq = kv.k_q9 + size_t(k0 + kk) * dh + 32 * g;
                    for (int t = 0; t < 32; ++t) x += int64_t(qq[32 * g + t]) * int64_t(kq[t]);
                } else {
                    for (int t = 0; t < 32; ++t) x += int64_t(qq[32 * g + t]) * int64_t(e4m3_q9(kc[32 * g + t]));
                }
                if (x != 0) bf48_add(s, x, int32_t(q_scales[g]) + int32_t(ks[g]) - 272);
            }
            scores[kk] = bf48_to_fp32(s);
        }
        // block maximum over the visible keys, k_j and c_j
        uint32_t raw_max = 0;
        if (present) {
            float mx = -INFINITY;
            for (int kk = 0; kk < 64; ++kk)
                if (allowed[kk] && f32_value(scores[kk]) > mx) mx = f32_value(scores[kk]);
            raw_max = f32_bits(mx);
            // np.max returns the first of equal values; +0/-0 cannot both occur (scores are never -0)
        }
        const uint32_t mj = f32_bits(f32_value(raw_max) * f32_value(big_s));
        int64_t kj, c;
        correction(mj, &kj, &c);
        const int64_t new_ref = have ? (ref > kj ? ref : kj) : kj;
        int64_t delta = 0;
        if (present && have) delta = new_ref - ref < 255 ? new_ref - ref : 255;
        if (present) ref = new_ref;
        have |= present;
        const int64_t shift = present ? kj - ref : 0;
        int64_t pscale = 127;
        if (present) {
            pscale = P_SCALE + shift;
            if (pscale < 0) pscale = 0;
            if (pscale > 254) pscale = 254;
        }
        // rescale O (exponent) and l (FP32) by 2^-delta
        for (int d = 0; d < dh; ++d) {
            o[d].e -= int32_t(delta);
            if (o[d].e < -512) o[d].have = false;
        }
        den = scale_down(den, delta);
        // P codes
        int64_t psum = 0;
        for (int kk = 0; kk < 64; ++kk) {
            int code = 0;
            if (allowed[kk] && present) {
                const uint32_t t = f32_bits(f32_value(scores[kk]) * f32_value(big_s));
                const uint32_t dd = f32_bits(f32_value(t) - f32_value(mj));
                code = p_code(dd);
            }
            pi[kk] = e4m3_q9(uint8_t(code));
            psum += pi[kk];
        }
        if (present) {
            const uint32_t total = f32_from_double(std::ldexp(double(psum), P_SCALE - 136));
            const uint32_t c_shift = scaled_correction(c, shift);
            den = f32_bits(f32_value(den) + f32_value(f32_bits(f32_value(total) * f32_value(c_shift))));
            // PV: one K32 group of keys at a time
            const int subs = (width + 31) / 32;
            for (int sub = 0; sub < subs; ++sub) {
                const int kg = k0 / 32 + sub;
                if (kg >= kgroups) break;
                const int32_t* pp = pi + 32 * sub;
                for (int d = 0; d < dh; ++d) {
                    const uint8_t* vc = kv.vt_codes + size_t(d) * cap + 32 * kg;
                    int64_t pdot = 0;
                    if (kv.vt_q9) {
                        const int32_t* vq = kv.vt_q9 + size_t(d) * cap + 32 * kg;
                        for (int t = 0; t < 32; ++t) pdot += int64_t(pp[t]) * int64_t(vq[t]);
                    } else {
                        for (int t = 0; t < 32; ++t) pdot += int64_t(pp[t]) * int64_t(e4m3_q9(vc[t]));
                    }
                    const int64_t x = (pdot * c) >> 2;
                    if (x != 0)
                        bf48_add(o[d], x, int32_t(pscale) + int32_t(kv.vt_scales[size_t(d) * kgroups + kg]) - 278);
                }
            }
        }
    }
    // O = o / l, row scale over Dh, quantised
    const uint32_t inv = reciprocal(f32_value(den) > 0 ? den : 0x3F800000u);
    uint32_t normalized[256];
    uint32_t maximum = 0;
    for (int d = 0; d < dh; ++d) {
        normalized[d] = f32_bits(f32_value(bf48_to_fp32(o[d])) * f32_value(inv));
        const uint32_t a = normalized[d] & 0x7FFFFFFFu;
        if (a > maximum) maximum = a;
    }
    const uint8_t s = output_scale(maximum);
    *o_scale = s;
    for (int d = 0; d < dh; ++d) o_codes[d] = quantize_e4m3(normalized[d], s);
}

void requantize_blocks(uint8_t* codes, uint8_t* scales, int rows, int cols) {
    const int groups = cols / 32;
    uint32_t v[32];
    for (int i = 0; i < rows; ++i)
        for (int g = 0; g < groups; ++g) {
            uint8_t* c = codes + size_t(i) * cols + 32 * g;
            const uint8_t s = scales[size_t(i) * groups + g];
            for (int t = 0; t < 32; ++t) v[t] = decode_e4m3(c[t], s);
            quantize_block32(v, c, scales + size_t(i) * groups + g);
        }
}

void requantize_vt(uint8_t* vt_codes, uint8_t* vt_scales, int dh, int cap, int keys) {
    for (int d = 0; d < dh; ++d)
        for (int k = keys; k < cap; ++k) vt_codes[size_t(d) * cap + k] = 0;
    requantize_blocks(vt_codes, vt_scales, dh, cap);
}

// ---------------------------------------------------------------------------------------------------------------
// V^T decode tail
// ---------------------------------------------------------------------------------------------------------------
uint8_t vt_requant_code(uint8_t code, uint8_t so, uint8_t sn) {
    if (!(sn != so && (code & 0x7F) != 0)) return code;
    if (so == 0xFF || sn == 0xFF || (code & 0x7F) == 0x7F) return 0x7F;
    const int d = uint8_t(sn - so);
    const int exponent = (code >> 3) & 0xF;
    const int s = exponent == 0 ? (code & 7) : (8 | (code & 7));
    const int en = (exponent == 0 ? 1 : exponent) - d;
    if (en > 0) return (s & 8) ? uint8_t((code & 0x80) | (en << 3) | (code & 7)) : code;
    int kq = 1 - en;
    if (kq > 5) kq = 5;
    int q = s >> kq;
    const int rem = s & ((1 << kq) - 1);
    const int half = 1 << (kq - 1);
    if (kq == 5) q = 0;
    else if (rem > half || (rem == half && (q & 1))) q += 1;
    return uint8_t((code & 0x80) | q);
}

uint8_t vt_tail_scale(uint8_t old, bool zero, uint32_t x) {
    const uint8_t s_tok = choose_scale(x & 0x7FFFFFFFu);
    if (zero) return s_tok;
    if ((x & 0x7FFFFFFFu) == 0) return old;
    return s_tok > old ? s_tok : old;
}

}  // namespace g3
