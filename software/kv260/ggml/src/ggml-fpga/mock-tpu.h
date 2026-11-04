#pragma once

#include "tpu-dispatch.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int64_t total_hw_calls;
    int64_t calls_32x128x128;
    int64_t calls_16x128x128;
    int64_t calls_1x128x128;
    int64_t calls_32x64x64;
    int64_t calls_16x64x64;
    int64_t calls_1x64x64;
    int64_t calls_1x32x32;
    int64_t calls_1x16x32;
    int64_t calls_32x32x32;
    int64_t calls_32x32x16;
    int64_t calls_16x32x32;
    int64_t calls_32x16x16;
    int64_t calls_16x16x16;
    int64_t total_padded_elements;
    int64_t total_useful_elements;
} mock_tpu_tiling_stats_t;

void mock_tpu_gemm_int8_int32(
    const int8_t  * a,
    const int8_t  * b,
    const int32_t * c,
    int32_t       * d,
    uint32_t m,
    uint32_t n,
    uint32_t k);

void mock_tpu_gemm_int8_int32_view(
    const int8_t    * a,
    const int8_t    * b,
    const int32_t   * c,
    tpu_int32_view_t * out_view,
    uint32_t m,
    uint32_t n,
    uint32_t k);

void mock_tpu_gemm_int8_int32_prepacked_b(
    const int8_t  * a,
    const int8_t  * b_kn,
    const int32_t * c,
    int32_t       * d,
    uint32_t m,
    uint32_t n,
    uint32_t k);

void mock_tpu_gemm_int8_int32_prepacked_b_view(
    const int8_t    * a,
    const int8_t    * b_kn,
    const int32_t   * c,
    tpu_int32_view_t * out_view,
    uint32_t m,
    uint32_t n,
    uint32_t k);

// Tiled wrapper: handles arbitrary M/N/K by splitting into hardware-supported
// tile configs and zero-padding tile boundaries as needed.
void mock_tpu_gemm_int8_int32_tiled(
    const int8_t  * a,
    const int8_t  * b,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    mock_tpu_tiling_stats_t * stats);

void mock_tpu_gemm_int8_int32_tiled_prepacked_b(
    const int8_t  * a,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    mock_tpu_tiling_stats_t * stats);

// Future: dedicated vector fast path beyond the current M=1 tile configs
// void mock_tpu_gemv_int8_int32(
//     const int8_t  * a,
//     const int8_t  * b,
//     const int32_t * c,
//     int32_t       * d,
//     uint32_t n,
//     uint32_t k);

#ifdef __cplusplus
}
#endif
