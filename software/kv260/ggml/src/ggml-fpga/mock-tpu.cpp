#include "mock-tpu.h"
#include "tpu-dispatch.h"
#include "tpu-tiling.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

static thread_local std::vector<int32_t> g_mock_tpu_view_scratch;
static thread_local std::vector<int32_t> g_mock_tpu_psum_ping;
static thread_local std::vector<int32_t> g_mock_tpu_psum_pong;
static thread_local tpu_accum_key_t g_mock_tpu_psum_key;
static thread_local int g_mock_tpu_psum_valid = 0;
static thread_local int g_mock_tpu_psum_active = 0;

static int mock_tpu_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static int mock_tpu_legacy_direct_enabled(void) {
    return mock_tpu_env_enabled("GGML_FPGA_MOCK_LEGACY_DIRECT");
}

static int mock_tpu_accum_key_equal(const tpu_accum_key_t * a, const tpu_accum_key_t * b) {
    return a->op_id == b->op_id &&
           a->m0 == b->m0 &&
           a->n0 == b->n0 &&
           a->m == b->m &&
           a->n == b->n;
}

static int32_t mock_tpu_desc_c_value(
        const tpu_tile_execute_desc_t * desc,
        const std::vector<int32_t> & accum,
        uint32_t i,
        uint32_t j) {
    if (desc->c_source == TPU_C_SOURCE_ZERO) {
        return 0;
    }
    if (desc->c_source == TPU_C_SOURCE_HOST) {
        return desc->c != NULL ? desc->c[i*desc->n + j] : 0;
    }
    if (i < desc->accum_key.m && j < desc->accum_key.n) {
        return accum[(size_t) i * desc->accum_key.n + j];
    }
    return 0;
}

static int8_t mock_tpu_desc_b_value(const tpu_tile_execute_desc_t * desc, uint32_t p, uint32_t j) {
    if (desc->b_layout == TPU_B_LAYOUT_KN) {
        return desc->b[p*desc->n + j];
    }
    return desc->b[j*desc->k + p];
}

static int mock_tpu_tile_execute_cached(
        const tpu_tile_execute_desc_t * desc,
        tpu_tile_execute_result_t * result) {
    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }
    if (desc == NULL || desc->a == NULL || desc->b == NULL ||
        (desc->d_dest == TPU_D_DEST_HOST_READBACK && desc->d == NULL)) {
        errno = EINVAL;
        return -1;
    }

    const std::vector<int32_t> * c_accum = NULL;
    if (desc->c_source == TPU_C_SOURCE_DEVICE_ACCUM) {
        if (!g_mock_tpu_psum_valid ||
            !mock_tpu_accum_key_equal(&g_mock_tpu_psum_key, &desc->accum_key)) {
            errno = ENOENT;
            return -1;
        }
        c_accum = g_mock_tpu_psum_active == 0 ? &g_mock_tpu_psum_ping : &g_mock_tpu_psum_pong;
        if (result != NULL) {
            result->accum_input_hit = 1;
        }
    }

    std::vector<int32_t> * out_accum = NULL;
    if (desc->d_dest == TPU_D_DEST_DEVICE_ACCUM) {
        const int inactive = g_mock_tpu_psum_active == 0 ? 1 : 0;
        out_accum = inactive == 0 ? &g_mock_tpu_psum_ping : &g_mock_tpu_psum_pong;
        out_accum->assign((size_t) desc->accum_key.m * desc->accum_key.n, 0);
    } else {
        memset(desc->d, 0, (size_t) desc->m * desc->n * sizeof(int32_t));
    }

    for (uint32_t i = 0; i < desc->m; ++i) {
        for (uint32_t j = 0; j < desc->n; ++j) {
            int32_t acc = mock_tpu_desc_c_value(desc, c_accum != NULL ? *c_accum : g_mock_tpu_psum_ping, i, j);
            for (uint32_t p = 0; p < desc->k; ++p) {
                acc += (int32_t) desc->a[i*desc->k + p] * (int32_t) mock_tpu_desc_b_value(desc, p, j);
            }

            if (desc->d_dest == TPU_D_DEST_DEVICE_ACCUM) {
                if (i < desc->accum_key.m && j < desc->accum_key.n) {
                    (*out_accum)[(size_t) i * desc->accum_key.n + j] = acc;
                }
            } else {
                desc->d[i*desc->n + j] = acc;
            }
        }
    }

    if (desc->d_dest == TPU_D_DEST_DEVICE_ACCUM) {
        g_mock_tpu_psum_active = g_mock_tpu_psum_active == 0 ? 1 : 0;
        g_mock_tpu_psum_key = desc->accum_key;
        g_mock_tpu_psum_valid = 1;
        if (result != NULL) {
            result->accum_output_valid = 1;
        }
    } else {
        g_mock_tpu_psum_valid = 0;
        if (result != NULL) {
            result->readback_done = 1;
            result->readback_direct_final = 0;
            result->bytes_device_to_host_d = (uint64_t) desc->accum_key.m * desc->accum_key.n * sizeof(int32_t);
        }
    }
    return 0;
}

void mock_tpu_gemm_int8_int32(
    const int8_t  * a,
    const int8_t  * b,
    const int32_t * c,
    int32_t       * d,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            int32_t acc = c ? c[i*n + j] : 0;
            for (uint32_t p = 0; p < k; ++p) {
                acc += (int32_t) a[i*k + p] * (int32_t) b[j*k + p];
            }
            d[i*n + j] = acc;
        }
    }
}

void tpu_gemm_int8_int32(
    const int8_t  * a,
    const int8_t  * b,
    const int32_t * c,
    int32_t       * d,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
    mock_tpu_gemm_int8_int32(a, b, c, d, m, n, k);
}

void mock_tpu_gemm_int8_int32_view(
    const int8_t    * a,
    const int8_t    * b,
    const int32_t   * c,
    tpu_int32_view_t * out_view,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
    if (out_view == NULL) {
        return;
    }

    g_mock_tpu_view_scratch.resize((size_t) m * (size_t) n);
    mock_tpu_gemm_int8_int32(a, b, c, g_mock_tpu_view_scratch.data(), m, n, k);
    out_view->data = g_mock_tpu_view_scratch.data();
    out_view->m = m;
    out_view->n = n;
    out_view->stride = n;
    out_view->elem_count = m * n;
}

void tpu_gemm_int8_int32_view(
    const int8_t    * a,
    const int8_t    * b,
    const int32_t   * c,
    tpu_int32_view_t * out_view,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
    mock_tpu_gemm_int8_int32_view(a, b, c, out_view, m, n, k);
}

void mock_tpu_gemm_int8_int32_prepacked_b(
    const int8_t  * a,
    const int8_t  * b_kn,
    const int32_t * c,
    int32_t       * d,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
    for (uint32_t i = 0; i < m; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            int32_t acc = c ? c[i*n + j] : 0;
            for (uint32_t p = 0; p < k; ++p) {
                acc += (int32_t) a[i*k + p] * (int32_t) b_kn[p*n + j];
            }
            d[i*n + j] = acc;
        }
    }
}

void tpu_gemm_int8_int32_prepacked_b(
    const int8_t  * a,
    const int8_t  * b_kn,
    const int32_t * c,
    int32_t       * d,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
    mock_tpu_gemm_int8_int32_prepacked_b(a, b_kn, c, d, m, n, k);
}

void mock_tpu_gemm_int8_int32_prepacked_b_view(
    const int8_t    * a,
    const int8_t    * b_kn,
    const int32_t   * c,
    tpu_int32_view_t * out_view,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
    if (out_view == NULL) {
        return;
    }

    g_mock_tpu_view_scratch.resize((size_t) m * (size_t) n);
    mock_tpu_gemm_int8_int32_prepacked_b(a, b_kn, c, g_mock_tpu_view_scratch.data(), m, n, k);
    out_view->data = g_mock_tpu_view_scratch.data();
    out_view->m = m;
    out_view->n = n;
    out_view->stride = n;
    out_view->elem_count = m * n;
}

void tpu_gemm_int8_int32_prepacked_b_view(
    const int8_t    * a,
    const int8_t    * b_kn,
    const int32_t   * c,
    tpu_int32_view_t * out_view,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
    mock_tpu_gemm_int8_int32_prepacked_b_view(a, b_kn, c, out_view, m, n, k);
}

static void mock_tpu_copy_tiling_stats(
        mock_tpu_tiling_stats_t * dst,
        const tpu_tiling_stats_t * src) {
    if (dst == NULL) {
        return;
    }

    memset(dst, 0, sizeof(*dst));
    if (src == NULL) {
        return;
    }

    dst->total_hw_calls = src->total_hw_calls;
    dst->calls_32x128x128 = src->calls_32x128x128;
    dst->calls_16x128x128 = src->calls_16x128x128;
    dst->calls_1x128x128 = src->calls_1x128x128;
    dst->calls_32x64x64 = src->calls_32x64x64;
    dst->calls_16x64x64 = src->calls_16x64x64;
    dst->calls_1x64x64 = src->calls_1x64x64;
    dst->calls_1x32x32 = src->calls_1x32x32;
    dst->calls_1x16x32 = src->calls_1x16x32;
    dst->calls_32x32x32 = src->calls_32x32x32;
    dst->calls_32x32x16 = src->calls_32x32x16;
    dst->calls_16x32x32 = src->calls_16x32x32;
    dst->calls_32x16x16 = src->calls_32x16x16;
    dst->calls_16x16x16 = src->calls_16x16x16;
    dst->total_padded_elements = src->total_padded_elements;
    dst->total_useful_elements = src->total_useful_elements;
}

void tpu_gemm_int8_int32_tiled(
    const int8_t  * a,
    const int8_t  * b,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats) {
    if (!mock_tpu_legacy_direct_enabled()) {
        tpu_gemm_int8_int32_tiled_device_psum_impl(a, b, d, M, N, K, stats, mock_tpu_tile_execute_cached, NULL);
        return;
    }
    tpu_gemm_int8_int32_tiled_direct_impl(a, b, d, M, N, K, stats, tpu_gemm_int8_int32);
}

void tpu_gemm_int8_int32_tiled_prepacked_b_with_cache(
    const int8_t  * a,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    void * packed_b_cache) {
    (void) packed_b_cache;
    if (!mock_tpu_legacy_direct_enabled()) {
        tpu_gemm_int8_int32_tiled_prepacked_b_device_psum_impl(
            a,
            b_kn,
            d,
            M,
            N,
            K,
            stats,
            mock_tpu_tile_execute_cached,
            NULL,
            NULL);
        return;
    }
    tpu_gemm_int8_int32_tiled_prepacked_b_direct_impl(
        a,
        b_kn,
        d,
        M,
        N,
        K,
        stats,
        tpu_gemm_int8_int32_prepacked_b);
}

void tpu_gemm_f32_int8_int32_tiled_prepacked_b_with_cache(
    const float   * a,
    const float   * a_inv_scales,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    void * packed_b_cache) {
    (void) packed_b_cache;
    tpu_gemm_f32_int8_int32_tiled_prepacked_b_device_psum_impl(
        a,
        a_inv_scales,
        b_kn,
        d,
        M,
        N,
        K,
        stats,
        mock_tpu_tile_execute_cached,
        NULL,
        NULL);
}

void tpu_gemm_int8_int32_tiled_prepacked_b(
    const int8_t  * a,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats) {
    tpu_gemm_int8_int32_tiled_prepacked_b_with_cache(a, b_kn, d, M, N, K, stats, NULL);
}

void mock_tpu_gemm_int8_int32_tiled(
    const int8_t  * a,
    const int8_t  * b,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    mock_tpu_tiling_stats_t * stats) {
    tpu_tiling_stats_t dispatch_stats = {};
    if (!mock_tpu_legacy_direct_enabled()) {
        tpu_gemm_int8_int32_tiled_device_psum_impl(a, b, d, M, N, K, &dispatch_stats, mock_tpu_tile_execute_cached, NULL);
    } else {
        tpu_gemm_int8_int32_tiled_direct_impl(a, b, d, M, N, K, &dispatch_stats, tpu_gemm_int8_int32);
    }
    mock_tpu_copy_tiling_stats(stats, &dispatch_stats);
}

void mock_tpu_gemm_int8_int32_tiled_prepacked_b(
    const int8_t  * a,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    mock_tpu_tiling_stats_t * stats) {
    tpu_tiling_stats_t dispatch_stats = {};
    if (!mock_tpu_legacy_direct_enabled()) {
        tpu_gemm_int8_int32_tiled_prepacked_b_device_psum_impl(
            a,
            b_kn,
            d,
            M,
            N,
            K,
            &dispatch_stats,
            mock_tpu_tile_execute_cached,
            NULL,
            NULL);
    } else {
        tpu_gemm_int8_int32_tiled_prepacked_b_direct_impl(
            a,
            b_kn,
            d,
            M,
            N,
            K,
            &dispatch_stats,
            tpu_gemm_int8_int32_prepacked_b);
    }
    mock_tpu_copy_tiling_stats(stats, &dispatch_stats);
}

int tpu_runtime_init(void) {
    return 0;
}

void tpu_runtime_cleanup(void) {
}

void tpu_runtime_get_stats(tpu_runtime_stats_t * stats) {
    if (stats == NULL) {
        return;
    }

    memset(stats, 0, sizeof(*stats));
}
