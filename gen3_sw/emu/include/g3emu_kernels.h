// Gen3 Node v3 numeric kernels (docs/gen3/node_v3.md s1.4-1.5): bit-exact C++ versions of the Python references
// tb/gen2/g2lib.py, tb/gen2/core_vectors.py, tb/gen3/g3lib.py and tb/gen3/attention_numerics.py.
//
// All kernels are plain functions over plain arrays (no global state), so the command emulator (g3emu.h) and a
// future ggml backend can share them. FP32 values travel as their IEEE bits (uint32_t).
//
// Build requirement: IEEE binary32 arithmetic without FMA contraction (-ffp-contract=off) and without
// flush-to-zero (no -ffast-math). The kernels never rely on x87 or long double.
#pragma once

#include <cstddef>
#include <cstdint>

namespace g3 {

constexpr uint32_t NAN32 = 0x7FC00000u;      // canonical NaN of every FP32 result

// ---------------------------------------------------------------------------------------------------------------
// FP32 helpers (g2lib.fp32_add_bits / fp32_mul_bits: round to nearest even, subnormals kept, NaN canonical)
// ---------------------------------------------------------------------------------------------------------------
bool f32_is_nan(uint32_t x);
uint32_t f32_add(uint32_t x, uint32_t y);
uint32_t f32_mul(uint32_t x, uint32_t y);
uint32_t f32_sub(uint32_t x, uint32_t y);    // numpy float32 subtract (NaN not canonicalised: used inside softmax)
uint32_t f32_from_double(double v);          // one RNE rounding (numpy .astype(float32))
float f32_value(uint32_t bits);
uint32_t f32_bits(float v);

// ---------------------------------------------------------------------------------------------------------------
// MXFP8: E4M3 elements, one E8M0 scale per 32 elements (Gen1 reference model gemm_core_mxfp8_reference_model.py)
// ---------------------------------------------------------------------------------------------------------------
int32_t e4m3_q9(uint8_t code);                        // value = q * 2^-9; NaN codes (0x7F, 0xFF) give 0
bool e4m3_is_nan(uint8_t code);                       // (code & 0x7F) == 0x7F
uint8_t choose_scale(uint32_t max_bits);              // choose_scale_e8m0 (NaN -> 0xFF, 0 -> 127)
uint32_t row_max_bits(const uint32_t* v, int n);      // fp32_row_maximum: |max| bits, NaN -> NAN32
uint8_t quantize_e4m3(uint32_t value, uint8_t scale); // quantize_fp32_to_e4m3 (RNE, saturating to 448)
uint32_t decode_e4m3(uint8_t code, uint8_t scale);    // g2lib.decode_e4m3_fp32 (exact; signed zero kept)
// 32 FP32 values -> 32 codes with their own scale (g2lib.quantize_row)
void quantize_block32(const uint32_t* v, uint8_t* codes, uint8_t* scale);
// FP32 [rows][cols] (cols % 32 == 0, row stride ld) -> codes [rows][cols], scales [rows][cols / 32]
void quantize_matrix(const uint32_t* v, int rows, int cols, int ld, uint8_t* codes, uint8_t* scales);

// ---------------------------------------------------------------------------------------------------------------
// bf48 block-floating accumulation (g2lib.bf_add / bf_to_fp32_bits, node_v3.md s1.5)
// ---------------------------------------------------------------------------------------------------------------
struct Bf48 {
    int64_t m = 0;      // 48-bit signed mantissa
    int32_t e = 0;      // exponent
    bool have = false;  // a sum has started
};
// One accumulation step with a live value x != 0 (the caller skips x == 0): first value starts the sum; else the
// smaller exponent is shifted right arithmetically (at most 63) and added; a sum leaving [-2^46, 2^46) is shifted
// right by one.
inline void bf48_add(Bf48& a, int64_t x, int32_t xe) {
    if (!a.have) {
        a.m = x; a.e = xe; a.have = true;
        return;
    }
    const bool up = xe > a.e;
    int64_t d = up ? int64_t(xe) - a.e : int64_t(a.e) - xe;
    if (d > 63) d = 63;
    const int64_t big = up ? x : a.m;
    const int64_t sml = up ? a.m : x;
    int64_t t = big + (sml >> d);
    int32_t et = up ? xe : a.e;
    if (t >= (int64_t(1) << 46) || t < -(int64_t(1) << 46)) {
        t >>= 1;
        et += 1;
    }
    a.m = t; a.e = et;
}
// RNE(m * 2^e) as FP32 bits; a leading one below 2^-126 gives +0; no sum (or m == 0) gives +0.
uint32_t bf48_to_fp32(const Bf48& a);

// ---------------------------------------------------------------------------------------------------------------
// GEMM: K32 blocks with exact integer products, bf48 accumulation over the K groups in order (g2lib.gemm_bf48).
// A: codes [m][groups*32], scales [m][groups]; B: codes [n][groups*32], scales [n][groups] (rows of B are output
// columns). out [m][ldo] FP32 bits. A NaN element or a 0xFF scale in a group makes the output NaN.
// ---------------------------------------------------------------------------------------------------------------
void gemm_bf48(int m, int n, int groups, const uint8_t* a_codes, const uint8_t* a_scales, const uint8_t* b_codes,
               const uint8_t* b_scales, uint32_t* out, int ldo);

// Decoded operand rows for repeated use (the q9 integers, scales and per-group NaN flags of each row).
struct MxRowsView {
    const int32_t* q;       // [rows][groups*32]
    const uint8_t* scale;   // [rows][groups]
    const uint8_t* nan;     // [rows][groups]: a NaN element or scale 0xFF
    int groups;
};
// Decode codes/scales of `rows` rows into q9 / NaN arrays (q [rows*groups*32], nan [rows*groups]).
void mx_decode_rows(int rows, int groups, const uint8_t* codes, const uint8_t* scales, int32_t* q, uint8_t* nan);
// One output value: bf48 sum over the groups of row i of A and row j of B.
uint32_t gemm_bf48_dot(const MxRowsView& a, int i, const MxRowsView& b, int j);

// ---------------------------------------------------------------------------------------------------------------
// Epilogues (core_vectors.py): 64-segment PWL activations, RoPE pair rotation
// ---------------------------------------------------------------------------------------------------------------
uint32_t silu_pwl(uint32_t x);    // x * slope + bias, silu_pwl64.coefficients
uint32_t gelu_pwl(uint32_t x);    // gelu_pwl64.coefficients
// RoPE on adjacent columns: out[2i] = x[2i] cos - x[2i+1] sin, out[2i+1] = x[2i+1] cos + x[2i] sin; cs holds the
// interleaved {cos, sin} table. n even.
void rope_pairs(const uint32_t* x, const uint32_t* cs, uint32_t* out, int n);

// ---------------------------------------------------------------------------------------------------------------
// RMSNorm (g2lib.rmsnorm_rows, g3lib.inv_rms_rows)
// ---------------------------------------------------------------------------------------------------------------
uint32_t inverse_count(int n);             // RNE(1/n) as FP32 bits
uint32_t rsqrt_bits(uint32_t v);           // magic constant and two Newton steps
uint32_t epsilon_bits(int profile);        // g2lib.EPSILON
// 1/rms of one row of FP32 bits: squares into 16 partial sums by column % 16, pairwise sum, x RNE(1/w), + eps, rsqrt
uint32_t inv_rms_row(const uint32_t* x, int w, int eps_profile);
// VECTOR RMSNORM row: out = ((x * s) + 0) * gamma + 0 with s = inv_rms_row(x)
void rmsnorm_row(const uint32_t* x, const uint32_t* gamma, int w, int eps_profile, uint32_t* out);

// ---------------------------------------------------------------------------------------------------------------
// Attention: 64-key block-max softmax with bf48 QK / PV (attention_numerics.Gen3Attention.run, G3_ACC=bf48)
// ---------------------------------------------------------------------------------------------------------------
struct AttnKV {
    int dh = 0;                       // multiple of 32
    int cap = 0;                      // key capacity (multiple of 32): rows of K, columns of V^T
    const uint8_t* k_codes = nullptr; // [cap][dh]
    const uint8_t* k_scales = nullptr;// [cap][dh/32]
    const uint8_t* vt_codes = nullptr;// [dh][cap]
    const uint8_t* vt_scales = nullptr;// [dh][cap/32]
    // optional: the codes decoded once (e4m3_q9 of k_codes / vt_codes), to save the lookups of every query row
    const int32_t* k_q9 = nullptr;    // [cap][dh]
    const int32_t* vt_q9 = nullptr;   // [dh][cap]
};
// QK scale S = FP32(FP32(1/sqrt(Dh)) * FP32(log2 e)), Dh = 32 * groups.
uint32_t attn_qk_scale(int groups);
// One query row: q codes [dh], scales [dh/32]; keys: number of keys; visible: keys 0..visible-1 are visible
// (visible = clamp(delta + token + 1, 0, keys)). Output: dh codes and the row scale (all groups).
void attention_row(const AttnKV& kv, const uint8_t* q_codes, const uint8_t* q_scales, int keys, int visible,
                   uint8_t* o_codes, uint8_t* o_scale);
// The oracle's input requantisation (g3lib.attention decodes Q, K, V^T and Gen3Attention.run quantises them again):
// each 32-element block is decoded and re-quantised with the scale of its decoded maximum. For V^T, keys >= keys
// are zeroed first (the oracle pads V with zeros). In place.
void requantize_blocks(uint8_t* codes, uint8_t* scales, int rows, int cols);
void requantize_vt(uint8_t* vt_codes, uint8_t* vt_scales, int dh, int cap, int keys);

// ---------------------------------------------------------------------------------------------------------------
// V^T append (decode tail, g3lib.VtCache / the RTL rewrite of a partial cell)
// ---------------------------------------------------------------------------------------------------------------
// Re-round an E4M3 code from scale so to a larger scale sn (the RTL requant_down; equal to decode + quantize).
uint8_t vt_requant_code(uint8_t code, uint8_t so, uint8_t sn);
// New scale of one dim when token value x (FP32 bits) is appended to a partial cell with old scale `old` whose stored
// codes are all zero (`zero`): zero -> scale of x; x == 0 -> old; else max(old, scale of x).
uint8_t vt_tail_scale(uint8_t old, bool zero, uint32_t x);

}  // namespace g3
