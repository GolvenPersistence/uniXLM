// K32 GEMM with exact integer block products and bf48 accumulation (g2lib.gemm_bf48).
#include "fp_strict.h"
#include <vector>

#include "g3emu_kernels.h"

namespace g3 {

void mx_decode_rows(int rows, int groups, const uint8_t* codes, const uint8_t* scales, int32_t* q, uint8_t* nan) {
    const int k = groups * 32;
    for (int i = 0; i < rows; ++i) {
        for (int g = 0; g < groups; ++g) {
            bool bad = scales[size_t(i) * groups + g] == 0xFF;
            const uint8_t* c = codes + size_t(i) * k + 32 * g;
            int32_t* dst = q + size_t(i) * k + 32 * g;
            for (int t = 0; t < 32; ++t) {
                dst[t] = e4m3_q9(c[t]);
                bad |= e4m3_is_nan(c[t]);
            }
            nan[size_t(i) * groups + g] = bad ? 1 : 0;
        }
    }
}

uint32_t gemm_bf48_dot(const MxRowsView& a, int i, const MxRowsView& b, int j) {
    const int groups = a.groups;
    const int k = groups * 32;
    const int32_t* aq = a.q + size_t(i) * k;
    const int32_t* bq = b.q + size_t(j) * k;
    const uint8_t* as = a.scale + size_t(i) * groups;
    const uint8_t* bs = b.scale + size_t(j) * groups;
    const uint8_t* an = a.nan + size_t(i) * groups;
    const uint8_t* bn = b.nan + size_t(j) * groups;
    Bf48 acc;
    bool nan = false;
    for (int g = 0; g < groups; ++g) {
        nan |= (an[g] | bn[g]) != 0;
        // exact: |q| <= 229376, so every product is below 2^36 and every partial sum below 2^41 (FP64 is exact)
        const int32_t* ap = aq + 32 * g;
        const int32_t* bp = bq + 32 * g;
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int t = 0; t < 32; t += 4) {
            s0 += double(ap[t]) * double(bp[t]);
            s1 += double(ap[t + 1]) * double(bp[t + 1]);
            s2 += double(ap[t + 2]) * double(bp[t + 2]);
            s3 += double(ap[t + 3]) * double(bp[t + 3]);
        }
        const int64_t x = int64_t((s0 + s1) + (s2 + s3));
        if (x != 0) bf48_add(acc, x, int32_t(as[g]) + int32_t(bs[g]) - 272);
    }
    return nan ? NAN32 : bf48_to_fp32(acc);
}

void gemm_bf48(int m, int n, int groups, const uint8_t* a_codes, const uint8_t* a_scales, const uint8_t* b_codes,
               const uint8_t* b_scales, uint32_t* out, int ldo) {
    const size_t k = size_t(groups) * 32;
    std::vector<int32_t> aq(size_t(m) * k), bq(size_t(n) * k);
    std::vector<uint8_t> an(size_t(m) * groups), bn(size_t(n) * groups);
    mx_decode_rows(m, groups, a_codes, a_scales, aq.data(), an.data());
    mx_decode_rows(n, groups, b_codes, b_scales, bq.data(), bn.data());
    const MxRowsView av{aq.data(), a_scales, an.data(), groups};
    const MxRowsView bv{bq.data(), b_scales, bn.data(), groups};
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) out[size_t(i) * ldo + j] = gemm_bf48_dot(av, i, bv, j);
}

}  // namespace g3
