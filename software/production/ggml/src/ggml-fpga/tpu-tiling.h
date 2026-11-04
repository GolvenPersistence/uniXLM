#pragma once

#include "tpu-dispatch.h"

#include <chrono>
#include <cmath>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <utility>
#include <vector>

typedef void (*tpu_gemm_fn)(
    const int8_t  * a,
    const int8_t  * b,
    const int32_t * c,
    int32_t       * d,
    uint32_t m,
    uint32_t n,
    uint32_t k);

typedef void (*tpu_gemm_view_fn)(
    const int8_t     * a,
    const int8_t     * b,
    const int32_t    * c,
    tpu_int32_view_t * out_view,
    uint32_t m,
    uint32_t n,
    uint32_t k);

typedef void (*tpu_gemm_prepacked_b_fn)(
    const int8_t  * a,
    const int8_t  * b_kn,
    const int32_t * c,
    int32_t       * d,
    uint32_t m,
    uint32_t n,
    uint32_t k);

typedef void (*tpu_gemm_prepacked_b_view_fn)(
    const int8_t     * a,
    const int8_t     * b_kn,
    const int32_t    * c,
    tpu_int32_view_t * out_view,
    uint32_t m,
    uint32_t n,
    uint32_t k);

typedef struct {
    uint32_t m;
    uint32_t n;
    uint32_t k;
} tpu_hw_cfg_t;

static inline uint32_t tpu_parse_requested_max_nk_local(void) {
    const char * value = getenv("GGML_FPGA_MAX_NK");
    char * endptr = NULL;
    unsigned long parsed = 0;

    if (value == NULL || *value == '\0') {
        return 128u;
    }

    parsed = strtoul(value, &endptr, 10);
    if (endptr == value || (endptr != NULL && *endptr != '\0')) {
        return 128u;
    }

    if (parsed < 64u) {
        return 64u;
    }

    return (uint32_t) parsed;
}

static inline uint32_t tpu_get_requested_max_nk(void) {
    return tpu_parse_requested_max_nk_local();
}

static inline uint32_t tpu_get_effective_max_nk(void) {
    const uint32_t requested = tpu_get_requested_max_nk();

    if (requested >= 128u) {
        return 128u;
    }

    if (requested >= 64u) {
        return 64u;
    }

    return 64u;
}

static inline int tpu_use_large_tile_family(void) {
    return tpu_get_effective_max_nk() > 32u;
}

static inline const char * tpu_tile_family_name(void) {
    switch (tpu_get_effective_max_nk()) {
        case 128u:
            return "nk-max-128";
        case 64u:
        default:
            return "nk-max-64";
    }
}

static inline int tpu_tiling_debug_enabled(void) {
    const char * value = getenv("GGML_FPGA_TILING_DEBUG");

    if (value == NULL || *value == '\0') {
        return 0;
    }

    return strcmp(value, "0") != 0;
}

static inline int tpu_tiling_env_truthy(const char * name) {
    const char * value = getenv(name);
    if (value == NULL || *value == '\0') {
        return 0;
    }
    return strcmp(value, "0") != 0 &&
           strcmp(value, "false") != 0 &&
           strcmp(value, "FALSE") != 0 &&
           strcmp(value, "off") != 0 &&
           strcmp(value, "OFF") != 0;
}

static inline int tpu_sequence_b16_bankfriendly_layout_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_tiling_env_truthy("TPU_V2_SEQUENCE_B16_BANKFRIENDLY_LAYOUT") ||
                 tpu_tiling_env_truthy("TPU_V2_PL_OP_B16_BANKFRIENDLY_LAYOUT");
    }
    return cached;
}

static inline int tpu_tiling_should_log_call(uint64_t call_idx) {
    return call_idx <= 8u || (call_idx % 128u) == 0u;
}

static inline int64_t tpu_tiling_now_ns(void) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static inline const tpu_hw_cfg_t * tpu_get_hw_cfg_table(size_t * count) {
    static const tpu_hw_cfg_t cfgs[] = {
        { 1, 64, 64 },
        { 16, 64, 64 },
        { 32, 64, 64 },
        { 1, 128, 128 },
        { 16, 128, 128 },
        { 32, 128, 128 },
    };

    if (count != NULL) {
        *count = sizeof(cfgs) / sizeof(cfgs[0]);
    }

    return cfgs;
}

static inline tpu_hw_cfg_t tpu_select_hw_cfg(uint32_t max_nk, uint32_t m, uint32_t n, uint32_t k) {
    size_t cfg_count = 0;
    const tpu_hw_cfg_t * cfgs = tpu_get_hw_cfg_table(&cfg_count);

    for (size_t i = 0; i < cfg_count; ++i) {
        const tpu_hw_cfg_t cfg = cfgs[i];

        if (cfg.n > max_nk || cfg.k > max_nk) {
            continue;
        }

        if (cfg.m >= m && cfg.n >= n && cfg.k >= k) {
            return cfg;
        }
    }

    return tpu_hw_cfg_t { 0, 0, 0 };
}

static inline void tpu_update_tiling_stats(
        tpu_tiling_stats_t * stats,
        const tpu_hw_cfg_t cfg,
        uint32_t m_actual,
        uint32_t n_actual,
        uint32_t k_actual) {
    if (stats == NULL) {
        return;
    }

    stats->total_hw_calls++;

    if (cfg.m == 32 && cfg.n == 128 && cfg.k == 128) {
        stats->calls_32x128x128++;
    } else if (cfg.m == 16 && cfg.n == 128 && cfg.k == 128) {
        stats->calls_16x128x128++;
    } else if (cfg.m == 1 && cfg.n == 128 && cfg.k == 128) {
        stats->calls_1x128x128++;
    } else if (cfg.m == 32 && cfg.n == 64 && cfg.k == 64) {
        stats->calls_32x64x64++;
    } else if (cfg.m == 16 && cfg.n == 64 && cfg.k == 64) {
        stats->calls_16x64x64++;
    } else if (cfg.m == 1 && cfg.n == 64 && cfg.k == 64) {
        stats->calls_1x64x64++;
    } else if (cfg.m == 1 && cfg.n == 32 && cfg.k == 32) {
        stats->calls_1x32x32++;
    } else if (cfg.m == 1 && cfg.n == 16 && cfg.k == 32) {
        stats->calls_1x16x32++;
    } else if (cfg.m == 32 && cfg.n == 32 && cfg.k == 32) {
        stats->calls_32x32x32++;
    } else if (cfg.m == 32 && cfg.n == 32 && cfg.k == 16) {
        stats->calls_32x32x16++;
    } else if (cfg.m == 16 && cfg.n == 32 && cfg.k == 32) {
        stats->calls_16x32x32++;
    } else if (cfg.m == 32 && cfg.n == 16 && cfg.k == 16) {
        stats->calls_32x16x16++;
    } else if (cfg.m == 16 && cfg.n == 16 && cfg.k == 16) {
        stats->calls_16x16x16++;
    }

    const int64_t useful = (int64_t) m_actual * k_actual + (int64_t) n_actual * k_actual;
    const int64_t padded = (int64_t) cfg.m * cfg.k + (int64_t) cfg.n * cfg.k - useful;

    stats->total_useful_elements += useful;
    stats->total_padded_elements += padded;
}

static inline void tpu_tiling_log_summary(
        uint64_t call_idx,
        const char * mode,
        uint32_t M,
        uint32_t N,
        uint32_t K,
        uint64_t outer_tiles,
        uint64_t hw_tiles,
        size_t partial_bytes,
        size_t a_bytes,
        size_t b_bytes,
        size_t c_bytes,
        size_t d_bytes,
        uint64_t zero_bytes,
        uint64_t copy_bytes,
        int64_t zero_ns,
        int64_t copy_a_ns,
        int64_t copy_b_ns,
        int64_t copy_c_ns,
        int64_t tile_call_ns,
        int64_t copy_out_partial_ns,
        int64_t copy_out_final_ns,
        const tpu_tiling_stats_t * stats) {
    if (!tpu_tiling_debug_enabled() || !tpu_tiling_should_log_call(call_idx)) {
        return;
    }

    fprintf(stderr,
            "[TPU Tiling] call=%" PRIu64 " mode=%s requested_max_nk=%u effective_max_nk=%u M=%u N=%u K=%u outer_tiles=%" PRIu64 " hw_tiles=%" PRIu64 " scratch(partial/a/b/c/d)=%zu/%zu/%zu/%zu/%zu zeroed=%.3f MiB copied=%.3f MiB zero=%.3f ms copy_a=%.3f ms copy_b=%.3f ms copy_c=%.3f ms tile=%.3f ms copy_out_partial=%.3f ms copy_out_final=%.3f ms direct_final=%" PRId64 "/%" PRId64 " cfg(128)=%" PRId64 "/%" PRId64 "/%" PRId64 " cfg(64)=%" PRId64 "/%" PRId64 "/%" PRId64 " cfg(32)=%" PRId64 "/%" PRId64 "/%" PRId64 "/%" PRId64 "/%" PRId64 "/%" PRId64 "/%" PRId64 "\n",
            call_idx,
            mode,
            tpu_get_requested_max_nk(),
            tpu_get_effective_max_nk(),
            M,
            N,
            K,
            outer_tiles,
            hw_tiles,
            partial_bytes,
            a_bytes,
            b_bytes,
            c_bytes,
            d_bytes,
            zero_bytes / (1024.0 * 1024.0),
            copy_bytes / (1024.0 * 1024.0),
            zero_ns / 1e6,
            copy_a_ns / 1e6,
            copy_b_ns / 1e6,
            copy_c_ns / 1e6,
            tile_call_ns / 1e6,
            copy_out_partial_ns / 1e6,
            copy_out_final_ns / 1e6,
            stats ? stats->direct_final_hits : 0,
            stats ? stats->direct_final_misses : 0,
            stats ? stats->calls_32x128x128 : 0,
            stats ? stats->calls_16x128x128 : 0,
            stats ? stats->calls_1x128x128 : 0,
            stats ? stats->calls_32x64x64 : 0,
            stats ? stats->calls_16x64x64 : 0,
            stats ? stats->calls_1x64x64 : 0,
            stats ? stats->calls_32x32x32 : 0,
            stats ? stats->calls_32x32x16 : 0,
            stats ? stats->calls_16x32x32 : 0,
            stats ? stats->calls_32x16x16 : 0,
            stats ? stats->calls_16x16x16 : 0,
            stats ? stats->calls_1x32x32 : 0,
            stats ? stats->calls_1x16x32 : 0);
}

template <uint32_t TILE_NK>
static inline void tpu_gemm_int8_int32_tiled_direct_impl_for_max_nk(
    const int8_t  * a,
    const int8_t  * b,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_gemm_fn single_tile_fn,
    const char * mode) {
    static uint64_t s_tiled_call_counter = 0;
    const uint32_t tile_m = 32;
    const uint32_t tile_n = TILE_NK;
    const uint32_t tile_k = TILE_NK;
    const uint64_t call_idx = ++s_tiled_call_counter;
    uint64_t debug_zero_bytes = 0;
    uint64_t debug_copy_bytes = 0;
    uint64_t debug_outer_tiles = 0;
    uint64_t debug_hw_tiles = 0;
    int64_t zero_ns = 0;
    int64_t copy_a_ns = 0;
    int64_t copy_b_ns = 0;
    int64_t copy_c_ns = 0;
    int64_t tile_call_ns = 0;
    int64_t copy_out_partial_ns = 0;
    int64_t copy_out_final_ns = 0;

    for (uint32_t m0 = 0; m0 < M; m0 += tile_m) {
        const uint32_t m_actual = (M - m0) < tile_m ? (M - m0) : tile_m;

        for (uint32_t n0 = 0; n0 < N; n0 += tile_n) {
            const uint32_t n_actual = (N - n0) < tile_n ? (N - n0) : tile_n;
            const uint32_t partial_stride = n_actual;
            debug_outer_tiles++;

            int32_t partial_sum[32 * TILE_NK];
            const size_t partial_bytes = (size_t) m_actual * partial_stride * sizeof(int32_t);
            const int64_t zero_start = tpu_tiling_now_ns();
            memset(partial_sum, 0, partial_bytes);
            zero_ns += tpu_tiling_now_ns() - zero_start;
            debug_zero_bytes += partial_bytes;

            for (uint32_t k0 = 0; k0 < K; k0 += tile_k) {
                const uint32_t k_actual = (K - k0) < tile_k ? (K - k0) : tile_k;
                const tpu_hw_cfg_t cfg = tpu_select_hw_cfg(TILE_NK, m_actual, n_actual, k_actual);
                debug_hw_tiles++;

                int8_t  a_tile[32 * TILE_NK];
                int8_t  b_tile[TILE_NK * TILE_NK];
                int32_t c_tile[32 * TILE_NK];
                int32_t d_tile[32 * TILE_NK];

                if (cfg.m == 0 || cfg.n == 0 || cfg.k == 0) {
                    return;
                }

                const size_t a_tile_bytes = (size_t) cfg.m * cfg.k * sizeof(int8_t);
                const size_t b_tile_bytes = (size_t) cfg.n * cfg.k * sizeof(int8_t);
                const size_t c_tile_bytes = (size_t) cfg.m * cfg.n * sizeof(int32_t);
                const int64_t tile_zero_start = tpu_tiling_now_ns();
                memset(a_tile, 0, a_tile_bytes);
                memset(b_tile, 0, b_tile_bytes);
                memset(c_tile, 0, c_tile_bytes);
                zero_ns += tpu_tiling_now_ns() - tile_zero_start;
                debug_zero_bytes += a_tile_bytes + b_tile_bytes + c_tile_bytes;

                const int64_t copy_a_start = tpu_tiling_now_ns();
                for (uint32_t mi = 0; mi < m_actual; ++mi) {
                    memcpy(a_tile + mi*cfg.k, a + (m0 + mi)*K + k0, k_actual*sizeof(int8_t));
                    debug_copy_bytes += (uint64_t) k_actual * sizeof(int8_t);
                }
                copy_a_ns += tpu_tiling_now_ns() - copy_a_start;

                const int64_t copy_b_start = tpu_tiling_now_ns();
                for (uint32_t ni = 0; ni < n_actual; ++ni) {
                    memcpy(b_tile + ni*cfg.k, b + (n0 + ni)*K + k0, k_actual*sizeof(int8_t));
                    debug_copy_bytes += (uint64_t) k_actual * sizeof(int8_t);
                }
                copy_b_ns += tpu_tiling_now_ns() - copy_b_start;

                const int64_t copy_c_start = tpu_tiling_now_ns();
                for (uint32_t mi = 0; mi < m_actual; ++mi) {
                    memcpy(c_tile + mi*cfg.n, partial_sum + mi*partial_stride, n_actual*sizeof(int32_t));
                    debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
                }
                copy_c_ns += tpu_tiling_now_ns() - copy_c_start;

                tpu_update_tiling_stats(stats, cfg, m_actual, n_actual, k_actual);
                const int64_t tile_call_start = tpu_tiling_now_ns();
                single_tile_fn(a_tile, b_tile, c_tile, d_tile, cfg.m, cfg.n, cfg.k);
                tile_call_ns += tpu_tiling_now_ns() - tile_call_start;

                const int64_t copy_out_partial_start = tpu_tiling_now_ns();
                for (uint32_t mi = 0; mi < m_actual; ++mi) {
                    memcpy(partial_sum + mi*partial_stride, d_tile + mi*cfg.n, n_actual*sizeof(int32_t));
                    debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
                }
                copy_out_partial_ns += tpu_tiling_now_ns() - copy_out_partial_start;
            }

            const int64_t copy_out_final_start = tpu_tiling_now_ns();
            for (uint32_t mi = 0; mi < m_actual; ++mi) {
                memcpy(d + (m0 + mi)*N + n0, partial_sum + mi*partial_stride, n_actual*sizeof(int32_t));
                debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
            }
            copy_out_final_ns += tpu_tiling_now_ns() - copy_out_final_start;
        }
    }

    if (stats != NULL) {
        stats->total_zero_ns += zero_ns;
        stats->total_copy_a_ns += copy_a_ns;
        stats->total_copy_b_ns += copy_b_ns;
        stats->total_copy_c_ns += copy_c_ns;
        stats->total_tile_call_ns += tile_call_ns;
        stats->total_copy_out_partial_ns += copy_out_partial_ns;
        stats->total_copy_out_final_ns += copy_out_final_ns;
    }

    tpu_tiling_log_summary(
        call_idx,
        mode,
        M,
        N,
        K,
        debug_outer_tiles,
        debug_hw_tiles,
        sizeof(int32_t[32 * TILE_NK]),
        sizeof(int8_t[32 * TILE_NK]),
        sizeof(int8_t[TILE_NK * TILE_NK]),
        sizeof(int32_t[32 * TILE_NK]),
        sizeof(int32_t[32 * TILE_NK]),
        debug_zero_bytes,
        debug_copy_bytes,
        zero_ns,
        copy_a_ns,
        copy_b_ns,
        copy_c_ns,
        tile_call_ns,
        copy_out_partial_ns,
        copy_out_final_ns,
        stats);
}

template <uint32_t TILE_NK>
static inline void tpu_gemm_int8_int32_tiled_view_impl_for_max_nk(
    const int8_t     * a,
    const int8_t     * b,
    int32_t          * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_gemm_view_fn single_tile_fn,
    const char * mode) {
    static uint64_t s_tiled_call_counter = 0;
    const uint32_t tile_m = 32;
    const uint32_t tile_n = TILE_NK;
    const uint32_t tile_k = TILE_NK;
    const uint64_t call_idx = ++s_tiled_call_counter;
    uint64_t debug_zero_bytes = 0;
    uint64_t debug_copy_bytes = 0;
    uint64_t debug_outer_tiles = 0;
    uint64_t debug_hw_tiles = 0;
    int64_t zero_ns = 0;
    int64_t copy_a_ns = 0;
    int64_t copy_b_ns = 0;
    int64_t copy_c_ns = 0;
    int64_t tile_call_ns = 0;
    int64_t copy_out_partial_ns = 0;
    int64_t copy_out_final_ns = 0;

    for (uint32_t m0 = 0; m0 < M; m0 += tile_m) {
        const uint32_t m_actual = (M - m0) < tile_m ? (M - m0) : tile_m;

        for (uint32_t n0 = 0; n0 < N; n0 += tile_n) {
            const uint32_t n_actual = (N - n0) < tile_n ? (N - n0) : tile_n;
            const uint32_t partial_stride = n_actual;
            debug_outer_tiles++;

            int32_t partial_sum[32 * TILE_NK];
            const size_t partial_bytes = (size_t) m_actual * partial_stride * sizeof(int32_t);
            const int64_t zero_start = tpu_tiling_now_ns();
            memset(partial_sum, 0, partial_bytes);
            zero_ns += tpu_tiling_now_ns() - zero_start;
            debug_zero_bytes += partial_bytes;

            for (uint32_t k0 = 0; k0 < K; k0 += tile_k) {
                const uint32_t k_actual = (K - k0) < tile_k ? (K - k0) : tile_k;
                const tpu_hw_cfg_t cfg = tpu_select_hw_cfg(TILE_NK, m_actual, n_actual, k_actual);
                debug_hw_tiles++;

                int8_t  a_tile[32 * TILE_NK];
                int8_t  b_tile[TILE_NK * TILE_NK];
                int32_t c_tile[32 * TILE_NK];
                tpu_int32_view_t d_view = {};

                if (cfg.m == 0 || cfg.n == 0 || cfg.k == 0) {
                    return;
                }

                const size_t a_tile_bytes = (size_t) cfg.m * cfg.k * sizeof(int8_t);
                const size_t b_tile_bytes = (size_t) cfg.n * cfg.k * sizeof(int8_t);
                const size_t c_tile_bytes = (size_t) cfg.m * cfg.n * sizeof(int32_t);
                const int64_t tile_zero_start = tpu_tiling_now_ns();
                memset(a_tile, 0, a_tile_bytes);
                memset(b_tile, 0, b_tile_bytes);
                memset(c_tile, 0, c_tile_bytes);
                zero_ns += tpu_tiling_now_ns() - tile_zero_start;
                debug_zero_bytes += a_tile_bytes + b_tile_bytes + c_tile_bytes;

                const int64_t copy_a_start = tpu_tiling_now_ns();
                for (uint32_t mi = 0; mi < m_actual; ++mi) {
                    memcpy(a_tile + mi*cfg.k, a + (m0 + mi)*K + k0, k_actual*sizeof(int8_t));
                    debug_copy_bytes += (uint64_t) k_actual * sizeof(int8_t);
                }
                copy_a_ns += tpu_tiling_now_ns() - copy_a_start;

                const int64_t copy_b_start = tpu_tiling_now_ns();
                for (uint32_t ni = 0; ni < n_actual; ++ni) {
                    memcpy(b_tile + ni*cfg.k, b + (n0 + ni)*K + k0, k_actual*sizeof(int8_t));
                    debug_copy_bytes += (uint64_t) k_actual * sizeof(int8_t);
                }
                copy_b_ns += tpu_tiling_now_ns() - copy_b_start;

                const int64_t copy_c_start = tpu_tiling_now_ns();
                for (uint32_t mi = 0; mi < m_actual; ++mi) {
                    memcpy(c_tile + mi*cfg.n, partial_sum + mi*partial_stride, n_actual*sizeof(int32_t));
                    debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
                }
                copy_c_ns += tpu_tiling_now_ns() - copy_c_start;

                tpu_update_tiling_stats(stats, cfg, m_actual, n_actual, k_actual);
                const int64_t tile_call_start = tpu_tiling_now_ns();
                single_tile_fn(a_tile, b_tile, c_tile, &d_view, cfg.m, cfg.n, cfg.k);
                tile_call_ns += tpu_tiling_now_ns() - tile_call_start;

                if (d_view.data == NULL || d_view.m < cfg.m || d_view.n < cfg.n ||
                    d_view.elem_count < cfg.m * cfg.n) {
                    abort();
                }

                const int64_t copy_out_partial_start = tpu_tiling_now_ns();
                for (uint32_t mi = 0; mi < m_actual; ++mi) {
                    memcpy(partial_sum + mi*partial_stride, d_view.data + mi*cfg.n, n_actual*sizeof(int32_t));
                    debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
                }
                copy_out_partial_ns += tpu_tiling_now_ns() - copy_out_partial_start;
            }

            const int64_t copy_out_final_start = tpu_tiling_now_ns();
            for (uint32_t mi = 0; mi < m_actual; ++mi) {
                memcpy(d + (m0 + mi)*N + n0, partial_sum + mi*partial_stride, n_actual*sizeof(int32_t));
                debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
            }
            copy_out_final_ns += tpu_tiling_now_ns() - copy_out_final_start;
        }
    }

    if (stats != NULL) {
        stats->total_zero_ns += zero_ns;
        stats->total_copy_a_ns += copy_a_ns;
        stats->total_copy_b_ns += copy_b_ns;
        stats->total_copy_c_ns += copy_c_ns;
        stats->total_tile_call_ns += tile_call_ns;
        stats->total_copy_out_partial_ns += copy_out_partial_ns;
        stats->total_copy_out_final_ns += copy_out_final_ns;
    }

    tpu_tiling_log_summary(
        call_idx,
        mode,
        M,
        N,
        K,
        debug_outer_tiles,
        debug_hw_tiles,
        sizeof(int32_t[32 * TILE_NK]),
        sizeof(int8_t[32 * TILE_NK]),
        sizeof(int8_t[TILE_NK * TILE_NK]),
        sizeof(int32_t[32 * TILE_NK]),
        0u,
        debug_zero_bytes,
        debug_copy_bytes,
        zero_ns,
        copy_a_ns,
        copy_b_ns,
        copy_c_ns,
        tile_call_ns,
        copy_out_partial_ns,
        copy_out_final_ns,
        stats);
}

template <uint32_t TILE_NK>
static inline void tpu_gemm_int8_int32_tiled_prepacked_b_direct_impl_for_max_nk(
    const int8_t  * a,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_gemm_prepacked_b_fn single_tile_fn,
    const char * mode) {
    static uint64_t s_tiled_call_counter = 0;
    const uint32_t tile_m = 32;
    const uint32_t tile_n = TILE_NK;
    const uint32_t tile_k = TILE_NK;
    const uint64_t call_idx = ++s_tiled_call_counter;
    uint64_t debug_zero_bytes = 0;
    uint64_t debug_copy_bytes = 0;
    uint64_t debug_outer_tiles = 0;
    uint64_t debug_hw_tiles = 0;
    int64_t zero_ns = 0;
    int64_t copy_a_ns = 0;
    int64_t copy_b_ns = 0;
    int64_t copy_c_ns = 0;
    int64_t tile_call_ns = 0;
    int64_t copy_out_partial_ns = 0;
    int64_t copy_out_final_ns = 0;

    for (uint32_t m0 = 0; m0 < M; m0 += tile_m) {
        const uint32_t m_actual = (M - m0) < tile_m ? (M - m0) : tile_m;

        for (uint32_t n0 = 0; n0 < N; n0 += tile_n) {
            const uint32_t n_actual = (N - n0) < tile_n ? (N - n0) : tile_n;
            const uint32_t partial_stride = n_actual;
            debug_outer_tiles++;

            int32_t partial_sum[32 * TILE_NK];
            const size_t partial_bytes = (size_t) m_actual * partial_stride * sizeof(int32_t);
            const int64_t zero_start = tpu_tiling_now_ns();
            memset(partial_sum, 0, partial_bytes);
            zero_ns += tpu_tiling_now_ns() - zero_start;
            debug_zero_bytes += partial_bytes;

            for (uint32_t k0 = 0; k0 < K; k0 += tile_k) {
                const uint32_t k_actual = (K - k0) < tile_k ? (K - k0) : tile_k;
                const tpu_hw_cfg_t cfg = tpu_select_hw_cfg(TILE_NK, m_actual, n_actual, k_actual);
                debug_hw_tiles++;

                int8_t  a_tile[32 * TILE_NK];
                int8_t  b_tile_kn[TILE_NK * TILE_NK];
                int32_t c_tile[32 * TILE_NK];
                int32_t d_tile[32 * TILE_NK];

                if (cfg.m == 0 || cfg.n == 0 || cfg.k == 0) {
                    return;
                }

                const size_t a_tile_bytes = (size_t) cfg.m * cfg.k * sizeof(int8_t);
                const size_t b_tile_bytes = (size_t) cfg.k * cfg.n * sizeof(int8_t);
                const size_t c_tile_bytes = (size_t) cfg.m * cfg.n * sizeof(int32_t);
                const int64_t tile_zero_start = tpu_tiling_now_ns();
                memset(a_tile, 0, a_tile_bytes);
                memset(b_tile_kn, 0, b_tile_bytes);
                memset(c_tile, 0, c_tile_bytes);
                zero_ns += tpu_tiling_now_ns() - tile_zero_start;
                debug_zero_bytes += a_tile_bytes + b_tile_bytes + c_tile_bytes;

                const int64_t copy_a_start = tpu_tiling_now_ns();
                for (uint32_t mi = 0; mi < m_actual; ++mi) {
                    memcpy(a_tile + mi*cfg.k, a + (m0 + mi)*K + k0, k_actual*sizeof(int8_t));
                    debug_copy_bytes += (uint64_t) k_actual * sizeof(int8_t);
                }
                copy_a_ns += tpu_tiling_now_ns() - copy_a_start;

                const int64_t copy_b_start = tpu_tiling_now_ns();
                for (uint32_t kk = 0; kk < k_actual; ++kk) {
                    memcpy(b_tile_kn + kk*cfg.n, b_kn + (k0 + kk)*N + n0, n_actual*sizeof(int8_t));
                    debug_copy_bytes += (uint64_t) n_actual * sizeof(int8_t);
                }
                copy_b_ns += tpu_tiling_now_ns() - copy_b_start;

                const int64_t copy_c_start = tpu_tiling_now_ns();
                for (uint32_t mi = 0; mi < m_actual; ++mi) {
                    memcpy(c_tile + mi*cfg.n, partial_sum + mi*partial_stride, n_actual*sizeof(int32_t));
                    debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
                }
                copy_c_ns += tpu_tiling_now_ns() - copy_c_start;

                tpu_update_tiling_stats(stats, cfg, m_actual, n_actual, k_actual);
                const int64_t tile_call_start = tpu_tiling_now_ns();
                single_tile_fn(a_tile, b_tile_kn, c_tile, d_tile, cfg.m, cfg.n, cfg.k);
                tile_call_ns += tpu_tiling_now_ns() - tile_call_start;

                const int64_t copy_out_partial_start = tpu_tiling_now_ns();
                for (uint32_t mi = 0; mi < m_actual; ++mi) {
                    memcpy(partial_sum + mi*partial_stride, d_tile + mi*cfg.n, n_actual*sizeof(int32_t));
                    debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
                }
                copy_out_partial_ns += tpu_tiling_now_ns() - copy_out_partial_start;
            }

            const int64_t copy_out_final_start = tpu_tiling_now_ns();
            for (uint32_t mi = 0; mi < m_actual; ++mi) {
                memcpy(d + (m0 + mi)*N + n0, partial_sum + mi*partial_stride, n_actual*sizeof(int32_t));
                debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
            }
            copy_out_final_ns += tpu_tiling_now_ns() - copy_out_final_start;
        }
    }

    if (stats != NULL) {
        stats->total_zero_ns += zero_ns;
        stats->total_copy_a_ns += copy_a_ns;
        stats->total_copy_b_ns += copy_b_ns;
        stats->total_copy_c_ns += copy_c_ns;
        stats->total_tile_call_ns += tile_call_ns;
        stats->total_copy_out_partial_ns += copy_out_partial_ns;
        stats->total_copy_out_final_ns += copy_out_final_ns;
    }

    tpu_tiling_log_summary(
        call_idx,
        mode,
        M,
        N,
        K,
        debug_outer_tiles,
        debug_hw_tiles,
        sizeof(int32_t[32 * TILE_NK]),
        sizeof(int8_t[32 * TILE_NK]),
        sizeof(int8_t[TILE_NK * TILE_NK]),
        sizeof(int32_t[32 * TILE_NK]),
        sizeof(int32_t[32 * TILE_NK]),
        debug_zero_bytes,
        debug_copy_bytes,
        zero_ns,
        copy_a_ns,
        copy_b_ns,
        copy_c_ns,
        tile_call_ns,
        copy_out_partial_ns,
        copy_out_final_ns,
        stats);
}

template <uint32_t TILE_NK>
static inline void tpu_gemm_int8_int32_tiled_prepacked_b_view_impl_for_max_nk(
    const int8_t     * a,
    const int8_t     * b_kn,
    int32_t          * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_gemm_prepacked_b_view_fn single_tile_fn,
    const char * mode) {
    static uint64_t s_tiled_call_counter = 0;
    const uint32_t tile_m = 32;
    const uint32_t tile_n = TILE_NK;
    const uint32_t tile_k = TILE_NK;
    const uint64_t call_idx = ++s_tiled_call_counter;
    uint64_t debug_zero_bytes = 0;
    uint64_t debug_copy_bytes = 0;
    uint64_t debug_outer_tiles = 0;
    uint64_t debug_hw_tiles = 0;
    int64_t zero_ns = 0;
    int64_t copy_a_ns = 0;
    int64_t copy_b_ns = 0;
    int64_t copy_c_ns = 0;
    int64_t tile_call_ns = 0;
    int64_t copy_out_partial_ns = 0;
    int64_t copy_out_final_ns = 0;

    for (uint32_t m0 = 0; m0 < M; m0 += tile_m) {
        const uint32_t m_actual = (M - m0) < tile_m ? (M - m0) : tile_m;

        for (uint32_t n0 = 0; n0 < N; n0 += tile_n) {
            const uint32_t n_actual = (N - n0) < tile_n ? (N - n0) : tile_n;
            const uint32_t partial_stride = n_actual;
            debug_outer_tiles++;

            int32_t partial_sum[32 * TILE_NK];
            const size_t partial_bytes = (size_t) m_actual * partial_stride * sizeof(int32_t);
            const int64_t zero_start = tpu_tiling_now_ns();
            memset(partial_sum, 0, partial_bytes);
            zero_ns += tpu_tiling_now_ns() - zero_start;
            debug_zero_bytes += partial_bytes;

            for (uint32_t k0 = 0; k0 < K; k0 += tile_k) {
                const uint32_t k_actual = (K - k0) < tile_k ? (K - k0) : tile_k;
                const tpu_hw_cfg_t cfg = tpu_select_hw_cfg(TILE_NK, m_actual, n_actual, k_actual);
                debug_hw_tiles++;

                int8_t  a_tile[32 * TILE_NK];
                int8_t  b_tile_kn[TILE_NK * TILE_NK];
                int32_t c_tile[32 * TILE_NK];
                tpu_int32_view_t d_view = {};

                if (cfg.m == 0 || cfg.n == 0 || cfg.k == 0) {
                    return;
                }

                const size_t a_tile_bytes = (size_t) cfg.m * cfg.k * sizeof(int8_t);
                const size_t b_tile_bytes = (size_t) cfg.k * cfg.n * sizeof(int8_t);
                const size_t c_tile_bytes = (size_t) cfg.m * cfg.n * sizeof(int32_t);
                const int64_t tile_zero_start = tpu_tiling_now_ns();
                memset(a_tile, 0, a_tile_bytes);
                memset(b_tile_kn, 0, b_tile_bytes);
                memset(c_tile, 0, c_tile_bytes);
                zero_ns += tpu_tiling_now_ns() - tile_zero_start;
                debug_zero_bytes += a_tile_bytes + b_tile_bytes + c_tile_bytes;

                const int64_t copy_a_start = tpu_tiling_now_ns();
                for (uint32_t mi = 0; mi < m_actual; ++mi) {
                    memcpy(a_tile + mi*cfg.k, a + (m0 + mi)*K + k0, k_actual*sizeof(int8_t));
                    debug_copy_bytes += (uint64_t) k_actual * sizeof(int8_t);
                }
                copy_a_ns += tpu_tiling_now_ns() - copy_a_start;

                const int64_t copy_b_start = tpu_tiling_now_ns();
                for (uint32_t kk = 0; kk < k_actual; ++kk) {
                    memcpy(b_tile_kn + kk*cfg.n, b_kn + (k0 + kk)*N + n0, n_actual*sizeof(int8_t));
                    debug_copy_bytes += (uint64_t) n_actual * sizeof(int8_t);
                }
                copy_b_ns += tpu_tiling_now_ns() - copy_b_start;

                const int64_t copy_c_start = tpu_tiling_now_ns();
                for (uint32_t mi = 0; mi < m_actual; ++mi) {
                    memcpy(c_tile + mi*cfg.n, partial_sum + mi*partial_stride, n_actual*sizeof(int32_t));
                    debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
                }
                copy_c_ns += tpu_tiling_now_ns() - copy_c_start;

                tpu_update_tiling_stats(stats, cfg, m_actual, n_actual, k_actual);
                const int64_t tile_call_start = tpu_tiling_now_ns();
                single_tile_fn(a_tile, b_tile_kn, c_tile, &d_view, cfg.m, cfg.n, cfg.k);
                tile_call_ns += tpu_tiling_now_ns() - tile_call_start;

                if (d_view.data == NULL || d_view.m < cfg.m || d_view.n < cfg.n ||
                    d_view.elem_count < cfg.m * cfg.n) {
                    abort();
                }

                const int64_t copy_out_partial_start = tpu_tiling_now_ns();
                for (uint32_t mi = 0; mi < m_actual; ++mi) {
                    memcpy(partial_sum + mi*partial_stride, d_view.data + mi*cfg.n, n_actual*sizeof(int32_t));
                    debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
                }
                copy_out_partial_ns += tpu_tiling_now_ns() - copy_out_partial_start;
            }

            const int64_t copy_out_final_start = tpu_tiling_now_ns();
            for (uint32_t mi = 0; mi < m_actual; ++mi) {
                memcpy(d + (m0 + mi)*N + n0, partial_sum + mi*partial_stride, n_actual*sizeof(int32_t));
                debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
            }
            copy_out_final_ns += tpu_tiling_now_ns() - copy_out_final_start;
        }
    }

    if (stats != NULL) {
        stats->total_zero_ns += zero_ns;
        stats->total_copy_a_ns += copy_a_ns;
        stats->total_copy_b_ns += copy_b_ns;
        stats->total_copy_c_ns += copy_c_ns;
        stats->total_tile_call_ns += tile_call_ns;
        stats->total_copy_out_partial_ns += copy_out_partial_ns;
        stats->total_copy_out_final_ns += copy_out_final_ns;
    }

    tpu_tiling_log_summary(
        call_idx,
        mode,
        M,
        N,
        K,
        debug_outer_tiles,
        debug_hw_tiles,
        sizeof(int32_t[32 * TILE_NK]),
        sizeof(int8_t[32 * TILE_NK]),
        sizeof(int8_t[TILE_NK * TILE_NK]),
        sizeof(int32_t[32 * TILE_NK]),
        0u,
        debug_zero_bytes,
        debug_copy_bytes,
        zero_ns,
        copy_a_ns,
        copy_b_ns,
        copy_c_ns,
        tile_call_ns,
        copy_out_partial_ns,
        copy_out_final_ns,
        stats);
}

static inline void tpu_gemm_int8_int32_tiled_direct_impl(
    const int8_t  * a,
    const int8_t  * b,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_gemm_fn single_tile_fn) {
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    switch (tpu_get_effective_max_nk()) {
        case 128u:
            tpu_gemm_int8_int32_tiled_direct_impl_for_max_nk<128>(a, b, d, M, N, K, stats, single_tile_fn, "plain");
            return;
        case 64u:
        default:
            tpu_gemm_int8_int32_tiled_direct_impl_for_max_nk<64>(a, b, d, M, N, K, stats, single_tile_fn, "plain");
            return;
    }
}

static inline void tpu_gemm_int8_int32_tiled_view_impl(
    const int8_t     * a,
    const int8_t     * b,
    int32_t          * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_gemm_view_fn single_tile_fn) {
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    switch (tpu_get_effective_max_nk()) {
        case 128u:
            tpu_gemm_int8_int32_tiled_view_impl_for_max_nk<128>(a, b, d, M, N, K, stats, single_tile_fn, "plain");
            return;
        case 64u:
        default:
            tpu_gemm_int8_int32_tiled_view_impl_for_max_nk<64>(a, b, d, M, N, K, stats, single_tile_fn, "plain");
            return;
    }
}

static inline void tpu_gemm_int8_int32_tiled_prepacked_b_direct_impl(
    const int8_t  * a,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_gemm_prepacked_b_fn single_tile_fn) {
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    switch (tpu_get_effective_max_nk()) {
        case 128u:
            tpu_gemm_int8_int32_tiled_prepacked_b_direct_impl_for_max_nk<128>(a, b_kn, d, M, N, K, stats, single_tile_fn, "prepacked_b");
            return;
        case 64u:
        default:
            tpu_gemm_int8_int32_tiled_prepacked_b_direct_impl_for_max_nk<64>(a, b_kn, d, M, N, K, stats, single_tile_fn, "prepacked_b");
            return;
    }
}

static inline void tpu_gemm_int8_int32_tiled_prepacked_b_view_impl(
    const int8_t     * a,
    const int8_t     * b_kn,
    int32_t          * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_gemm_prepacked_b_view_fn single_tile_fn) {
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    switch (tpu_get_effective_max_nk()) {
        case 128u:
            tpu_gemm_int8_int32_tiled_prepacked_b_view_impl_for_max_nk<128>(a, b_kn, d, M, N, K, stats, single_tile_fn, "prepacked_b");
            return;
        case 64u:
        default:
            tpu_gemm_int8_int32_tiled_prepacked_b_view_impl_for_max_nk<64>(a, b_kn, d, M, N, K, stats, single_tile_fn, "prepacked_b");
            return;
    }
}

static inline uint32_t tpu_tiling_int8_word_count(uint32_t rows, uint32_t cols) {
    return (rows * cols + 7u) / 8u;
}

static inline int8_t tpu_tiling_quantize_f32_to_i8(float value, float inv_scale) {
    const float scaled = std::round(value * inv_scale);
    const float clamped = scaled < -127.0f ? -127.0f : (scaled > 127.0f ? 127.0f : scaled);
    return (int8_t) clamped;
}

struct tpu_raw_a_cache_key_t {
    const float * a;
    const float * inv_scales;
    uint32_t source_k;
    uint32_t m0;
    uint32_t k0;
    uint32_t m_actual;
    uint32_t k_actual;
    uint32_t m;
    uint32_t k;

    bool operator==(const tpu_raw_a_cache_key_t & other) const {
        return a == other.a &&
               inv_scales == other.inv_scales &&
               source_k == other.source_k &&
               m0 == other.m0 &&
               k0 == other.k0 &&
               m_actual == other.m_actual &&
               k_actual == other.k_actual &&
               m == other.m &&
               k == other.k;
    }
};

struct tpu_raw_a_cache_key_hash_t {
    size_t operator()(const tpu_raw_a_cache_key_t & key) const {
        uint64_t h = (uint64_t) (uintptr_t) key.a;
        h = h * 1315423911u + (uint64_t) (uintptr_t) key.inv_scales;
        h = h * 1315423911u + key.source_k;
        h = h * 1315423911u + key.m0;
        h = h * 1315423911u + key.k0;
        h = h * 1315423911u + key.m_actual;
        h = h * 1315423911u + key.k_actual;
        h = h * 1315423911u + key.m;
        h = h * 1315423911u + key.k;
        return (size_t) h;
    }
};

typedef std::unordered_map<tpu_raw_a_cache_key_t, std::vector<int8_t>, tpu_raw_a_cache_key_hash_t> tpu_raw_a_cache_t;

static inline void tpu_build_raw_a_tile_from_int8(
        const int8_t * src_mk,
        uint32_t source_k,
        uint32_t m0,
        uint32_t k0,
        uint32_t m_actual,
        uint32_t k_actual,
        uint32_t m,
        uint32_t k,
        int8_t * dst) {
    memset(dst, 0, (size_t) m * k * sizeof(int8_t));
    for (uint32_t mi = 0; mi < m_actual; ++mi) {
        memcpy(dst + mi*k, src_mk + (m0 + mi)*source_k + k0, k_actual*sizeof(int8_t));
    }
}

static inline void tpu_build_raw_a_tile_from_f32(
        const float * src_mk,
        const float * row_inv_scales,
        uint32_t source_k,
        uint32_t m0,
        uint32_t k0,
        uint32_t m_actual,
        uint32_t k_actual,
        uint32_t m,
        uint32_t k,
        int8_t * dst) {
    memset(dst, 0, (size_t) m * k * sizeof(int8_t));
    for (uint32_t mi = 0; mi < m_actual; ++mi) {
        const float inv_scale = row_inv_scales[m0 + mi];
        const float * src_row = src_mk + (m0 + mi)*source_k + k0;
        int8_t * dst_row = dst + mi*k;
        if (inv_scale == 0.0f) {
            continue;
        }
        for (uint32_t kk = 0; kk < k_actual; ++kk) {
            dst_row[kk] = tpu_tiling_quantize_f32_to_i8(src_row[kk], inv_scale);
        }
    }
}

static inline const int8_t * tpu_get_or_build_raw_a_tile_from_f32(
        tpu_raw_a_cache_t & cache,
        const tpu_raw_a_cache_key_t & key,
        const float * source_mk,
        const float * row_inv_scales,
        uint32_t source_k,
        uint32_t m_actual,
        uint32_t k_actual,
        uint32_t m,
        uint32_t k) {
    auto it = cache.find(key);
    if (it == cache.end()) {
        std::vector<int8_t> raw((size_t) m * k);
        tpu_build_raw_a_tile_from_f32(source_mk, row_inv_scales, source_k, key.m0, key.k0, m_actual, k_actual, m, k, raw.data());
        it = cache.emplace(key, std::move(raw)).first;
    }
    return it->second.data();
}

struct tpu_packed_b_cache_key_t {
    const int8_t * b;
    uint32_t source_n;
    uint32_t n0;
    uint32_t k0;
    uint32_t n_actual;
    uint32_t k_actual;
    uint32_t n;
    uint32_t k;
    uint32_t pack_layout;

    bool operator==(const tpu_packed_b_cache_key_t & other) const {
        return b == other.b &&
               source_n == other.source_n &&
               n0 == other.n0 &&
               k0 == other.k0 &&
               n_actual == other.n_actual &&
               k_actual == other.k_actual &&
               n == other.n &&
               k == other.k &&
               pack_layout == other.pack_layout;
    }
};

struct tpu_packed_b_cache_key_hash_t {
    size_t operator()(const tpu_packed_b_cache_key_t & key) const {
        uint64_t h = (uint64_t) (uintptr_t) key.b;
        h = h * 1315423911u + key.source_n;
        h = h * 1315423911u + key.n0;
        h = h * 1315423911u + key.k0;
        h = h * 1315423911u + key.n_actual;
        h = h * 1315423911u + key.k_actual;
        h = h * 1315423911u + key.n;
        h = h * 1315423911u + key.k;
        h = h * 1315423911u + key.pack_layout;
        return (size_t) h;
    }
};

static inline void tpu_pack_b_int8_kn_source_to_words(
        const int8_t * src_kn,
        uint32_t src_n,
        uint32_t n0,
        uint32_t k0,
        uint32_t n_actual,
        uint32_t k_actual,
        uint32_t n,
        uint32_t k,
        std::vector<uint64_t> & dst_words) {
    const uint32_t word_count = tpu_tiling_int8_word_count(k, n);
    dst_words.assign((size_t) word_count, 0);

    if (tpu_sequence_b16_bankfriendly_layout_enabled() && (n % 8u) == 0u && (k % 16u) == 0u) {
        const uint32_t kg_count = k / 16u;
        const uint32_t nb_count = n / 8u;
        const uint32_t entries_per_lane = kg_count * nb_count * 2u;
        const size_t required_words = (size_t) 4u * entries_per_lane * 2u;
        if (required_words > dst_words.size()) {
            abort();
        }
        for (uint32_t kg = 0; kg < kg_count; ++kg) {
            const uint32_t k16 = kg * 16u;
            const uint32_t valid_k = (k16 < k_actual) ?
                ((k_actual - k16) < 16u ? (k_actual - k16) : 16u) : 0u;
            for (uint32_t nb = 0; nb < nb_count; ++nb) {
                const uint32_t n_block = nb * 8u;
                uint64_t words[4][2][2] = {};
                for (uint32_t kk = 0; kk < valid_k; ++kk) {
                    const int8_t * row = src_kn + (k0 + k16 + kk)*src_n + n0 + n_block;
                    for (uint32_t n_inner = 0; n_inner < 8u; ++n_inner) {
                        const uint32_t nn = n_block + n_inner;
                        if (nn >= n_actual) {
                            continue;
                        }
                        const uint32_t axi_lane = n_inner & 3u;
                        const uint32_t bank_half = n_inner >> 2;
                        const uint8_t value = (uint8_t) row[n_inner];
                        if (kk < 8u) {
                            words[axi_lane][bank_half][0] |= ((uint64_t) value) << (kk * 8u);
                        } else {
                            words[axi_lane][bank_half][1] |= ((uint64_t) value) << ((kk - 8u) * 8u);
                        }
                    }
                }
                for (uint32_t axi_lane = 0; axi_lane < 4u; ++axi_lane) {
                    for (uint32_t bank_half = 0; bank_half < 2u; ++bank_half) {
                        const uint32_t entry_off = (kg * nb_count + nb) * 2u + bank_half;
                        const size_t word_idx =
                            ((size_t) axi_lane * entries_per_lane + entry_off) * 2u;
                        dst_words[word_idx] = words[axi_lane][bank_half][0];
                        dst_words[word_idx + 1u] = words[axi_lane][bank_half][1];
                    }
                }
            }
        }
        return;
    }

    for (uint32_t kk = 0; kk < k_actual; ++kk) {
        for (uint32_t nn = 0; nn < n_actual; ++nn) {
            const uint32_t linear_idx = kk * n + nn;
            const uint32_t word_idx = linear_idx / 8u;
            const uint32_t lane = linear_idx % 8u;
            dst_words[word_idx] |= ((uint64_t) (uint8_t) src_kn[(k0 + kk)*src_n + n0 + nn]) << (lane * 8u);
        }
    }
}

typedef std::unordered_map<tpu_packed_b_cache_key_t, std::vector<uint64_t>, tpu_packed_b_cache_key_hash_t> tpu_packed_b_cache_t;

static inline const int8_t * tpu_get_or_build_packed_b_tile(
        tpu_packed_b_cache_t & cache,
        const tpu_packed_b_cache_key_t & key,
        const int8_t * source_kn,
        uint32_t source_n,
        uint32_t n_actual,
        uint32_t k_actual,
        uint32_t n,
        uint32_t k) {
    auto it = cache.find(key);
    if (it == cache.end()) {
        std::vector<uint64_t> packed;
        tpu_pack_b_int8_kn_source_to_words(source_kn, source_n, key.n0, key.k0, n_actual, k_actual, n, k, packed);
        it = cache.emplace(key, std::move(packed)).first;
    }

    return (const int8_t *) it->second.data();
}

static inline tpu_packed_b_cache_t * tpu_resolve_packed_b_cache(void * cache) {
    return cache == NULL ? NULL : (tpu_packed_b_cache_t *) cache;
}

template <uint32_t TILE_NK>
static inline void tpu_gemm_int8_int32_tiled_device_psum_impl_for_max_nk(
    const int8_t  * a,
    const int8_t  * b,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_tile_execute_cached_fn tile_fn,
    tpu_tile_sequence_execute_cached_fn sequence_fn,
    tpu_b_layout_t b_layout,
    const char * mode,
    void * packed_b_cache_ptr) {
    static uint64_t s_tiled_call_counter = 0;
    const uint32_t tile_m = 32;
    const uint32_t tile_n = TILE_NK;
    const uint32_t tile_k = TILE_NK;
    const uint64_t call_idx = ++s_tiled_call_counter;
    uint64_t debug_zero_bytes = 0;
    uint64_t debug_copy_bytes = 0;
    uint64_t debug_outer_tiles = 0;
    uint64_t debug_hw_tiles = 0;
    int64_t zero_ns = 0;
    int64_t copy_a_ns = 0;
    int64_t copy_b_ns = 0;
    int64_t copy_c_ns = 0;
    int64_t tile_call_ns = 0;
    int64_t copy_out_partial_ns = 0;
    int64_t copy_out_final_ns = 0;
    tpu_packed_b_cache_t local_packed_b_cache;
    tpu_packed_b_cache_t * packed_b_cache = tpu_resolve_packed_b_cache(packed_b_cache_ptr);
    const bool persistent_packed_b_cache = packed_b_cache != NULL;
    if (packed_b_cache == NULL) {
        packed_b_cache = &local_packed_b_cache;
    }
    const bool use_packed_b_cache = b_layout == TPU_B_LAYOUT_KN && sequence_fn != NULL;

    for (uint32_t m0 = 0; m0 < M; m0 += tile_m) {
        const uint32_t m_actual = (M - m0) < tile_m ? (M - m0) : tile_m;

        for (uint32_t n0 = 0; n0 < N; n0 += tile_n) {
            const uint32_t n_actual = (N - n0) < tile_n ? (N - n0) : tile_n;
            const tpu_accum_key_t accum_key = { call_idx, m0, n0, m_actual, n_actual };
            debug_outer_tiles++;

            const uint32_t first_k_actual = K < tile_k ? K : tile_k;
            const tpu_hw_cfg_t chain_cfg = tpu_select_hw_cfg(TILE_NK, m_actual, n_actual, first_k_actual);
            if (chain_cfg.m == 0 || chain_cfg.n == 0 || chain_cfg.k == 0) {
                return;
            }

            for (uint32_t k0 = 0; k0 < K;) {
                const uint32_t remaining_k_tiles = (K - k0 + tile_k - 1u) / tile_k;
                uint32_t sequence_count = remaining_k_tiles < 8u ? remaining_k_tiles : 8u;
                if (sequence_fn != NULL && remaining_k_tiles > sequence_count &&
                    (remaining_k_tiles - sequence_count) == 1u && sequence_count > 2u) {
                    sequence_count--;
                }
                const bool try_sequence = sequence_fn != NULL && sequence_count > 1u;
                tpu_tile_execute_desc_t sequence_descs[8] = {};
                int8_t sequence_a_tiles[8][32 * TILE_NK];
                int8_t sequence_b_tiles[8][TILE_NK * TILE_NK];
                int32_t d_tile[32 * TILE_NK];
                bool can_sequence = try_sequence;

                for (uint32_t si = 0; si < sequence_count; ++si) {
                    const uint32_t tile_k0 = k0 + si * tile_k;
                    const uint32_t k_actual = (K - tile_k0) < tile_k ? (K - tile_k0) : tile_k;
                    const tpu_hw_cfg_t cfg = chain_cfg;
                    const bool first_k = tile_k0 == 0;
                    const bool last_k = (tile_k0 + tile_k) >= K;
                    debug_hw_tiles++;

                    if (cfg.m == 0 || cfg.n == 0 || cfg.k == 0) {
                        return;
                    }

                    const size_t a_tile_bytes = (size_t) cfg.m * cfg.k * sizeof(int8_t);
                    const size_t b_tile_bytes = (size_t) cfg.n * cfg.k * sizeof(int8_t);
                    const int64_t tile_zero_start = tpu_tiling_now_ns();
                    memset(sequence_a_tiles[si], 0, a_tile_bytes);
                    debug_zero_bytes += a_tile_bytes;
                    if (!use_packed_b_cache) {
                        memset(sequence_b_tiles[si], 0, b_tile_bytes);
                        debug_zero_bytes += b_tile_bytes;
                    }
                    zero_ns += tpu_tiling_now_ns() - tile_zero_start;

                    const int64_t copy_a_start = tpu_tiling_now_ns();
                    for (uint32_t mi = 0; mi < m_actual; ++mi) {
                        memcpy(sequence_a_tiles[si] + mi*cfg.k, a + (m0 + mi)*K + tile_k0, k_actual*sizeof(int8_t));
                        debug_copy_bytes += (uint64_t) k_actual * sizeof(int8_t);
                    }
                    copy_a_ns += tpu_tiling_now_ns() - copy_a_start;

                    const int64_t copy_b_start = tpu_tiling_now_ns();
                    tpu_b_layout_t desc_b_layout = b_layout;
                    const int8_t * desc_b = sequence_b_tiles[si];
                    if (b_layout == TPU_B_LAYOUT_KN) {
                        if (use_packed_b_cache) {
                            const tpu_packed_b_cache_key_t cache_key = {
                                b, N, n0, tile_k0, n_actual, k_actual, cfg.n, cfg.k,
                                (uint32_t) tpu_sequence_b16_bankfriendly_layout_enabled()
                            };
                            desc_b = tpu_get_or_build_packed_b_tile(
                                    *packed_b_cache,
                                    cache_key,
                                    b,
                                    N,
                                    n_actual,
                                    k_actual,
                                    cfg.n,
                                    cfg.k);
                            desc_b_layout = TPU_B_LAYOUT_DEVICE_PACKED;
                        } else {
                            for (uint32_t kk = 0; kk < k_actual; ++kk) {
                                memcpy(sequence_b_tiles[si] + kk*cfg.n, b + (tile_k0 + kk)*N + n0, n_actual*sizeof(int8_t));
                                debug_copy_bytes += (uint64_t) n_actual * sizeof(int8_t);
                            }
                        }
                    } else {
                        for (uint32_t ni = 0; ni < n_actual; ++ni) {
                            memcpy(sequence_b_tiles[si] + ni*cfg.k, b + (n0 + ni)*K + tile_k0, k_actual*sizeof(int8_t));
                            debug_copy_bytes += (uint64_t) k_actual * sizeof(int8_t);
                        }
                    }
                    copy_b_ns += tpu_tiling_now_ns() - copy_b_start;

                    tpu_update_tiling_stats(stats, cfg, m_actual, n_actual, k_actual);

                    sequence_descs[si].a = sequence_a_tiles[si];
                    sequence_descs[si].b = desc_b;
                    sequence_descs[si].d = last_k ? d_tile : NULL;
                    sequence_descs[si].d_final = last_k ? (d + m0*N + n0) : NULL;
                    sequence_descs[si].d_final_stride = N;
                    sequence_descs[si].d_actual_m = m_actual;
                    sequence_descs[si].d_actual_n = n_actual;
                    sequence_descs[si].m = cfg.m;
                    sequence_descs[si].n = cfg.n;
                    sequence_descs[si].k = cfg.k;
                    sequence_descs[si].a_layout = sequence_fn != NULL ? TPU_A_LAYOUT_RAW_MK : TPU_A_LAYOUT_MK;
                    sequence_descs[si].a_bytes = sequence_fn != NULL ? a_tile_bytes : 0;
                    sequence_descs[si].b_layout = desc_b_layout;
                    if (desc_b_layout == TPU_B_LAYOUT_DEVICE_PACKED && persistent_packed_b_cache) {
                        sequence_descs[si].b_cache_owner = packed_b_cache;
                        sequence_descs[si].b_cache_source = b;
                        sequence_descs[si].b_cache_source_n = N;
                        sequence_descs[si].b_cache_n0 = n0;
                        sequence_descs[si].b_cache_k0 = tile_k0;
                        sequence_descs[si].b_cache_n_actual = n_actual;
                        sequence_descs[si].b_cache_k_actual = k_actual;
                    }
                    sequence_descs[si].c_source = first_k ? TPU_C_SOURCE_ZERO : TPU_C_SOURCE_DEVICE_ACCUM;
                    sequence_descs[si].d_dest = last_k ? TPU_D_DEST_HOST_READBACK : TPU_D_DEST_DEVICE_ACCUM;
                    sequence_descs[si].accum_key = accum_key;
                    sequence_descs[si].next_desc = NULL;
                }

                tpu_tile_execute_result_t result = {};
                const int64_t tile_call_start = tpu_tiling_now_ns();
                if (can_sequence) {
                    if (sequence_fn(sequence_descs, sequence_count, &result) != 0) {
                        abort();
                    }
                    tile_call_ns += tpu_tiling_now_ns() - tile_call_start;
                    k0 += sequence_count * tile_k;
                } else {
                    for (uint32_t si = 0; si < sequence_count; ++si) {
                        if (si + 1u < sequence_count) {
                            sequence_descs[si].next_desc = &sequence_descs[si + 1u];
                        }
                        tpu_tile_execute_result_t single_result = {};
                        if (tile_fn(&sequence_descs[si], &single_result) != 0) {
                            abort();
                        }
                        result = single_result;
                    }
                    tile_call_ns += tpu_tiling_now_ns() - tile_call_start;
                    k0 += sequence_count * tile_k;
                }

                const tpu_tile_execute_desc_t * last_desc = &sequence_descs[sequence_count - 1u];
                if (last_desc->d_dest == TPU_D_DEST_HOST_READBACK && !result.readback_direct_final) {
                    const int64_t copy_out_final_start = tpu_tiling_now_ns();
                    for (uint32_t mi = 0; mi < m_actual; ++mi) {
                        memcpy(d + (m0 + mi)*N + n0, d_tile + mi*last_desc->n, n_actual*sizeof(int32_t));
                        debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
                    }
                    copy_out_final_ns += tpu_tiling_now_ns() - copy_out_final_start;
                    if (stats != NULL) {
                        stats->direct_final_misses++;
                    }
                } else if (last_desc->d_dest == TPU_D_DEST_HOST_READBACK && stats != NULL) {
                    stats->direct_final_hits++;
                }
            }
        }
    }

    if (stats != NULL) {
        stats->total_zero_ns += zero_ns;
        stats->total_copy_a_ns += copy_a_ns;
        stats->total_copy_b_ns += copy_b_ns;
        stats->total_copy_c_ns += copy_c_ns;
        stats->total_tile_call_ns += tile_call_ns;
        stats->total_copy_out_partial_ns += copy_out_partial_ns;
        stats->total_copy_out_final_ns += copy_out_final_ns;
    }

    tpu_tiling_log_summary(
        call_idx,
        mode,
        M,
        N,
        K,
        debug_outer_tiles,
        debug_hw_tiles,
        0u,
        sizeof(int8_t[32 * TILE_NK]),
        sizeof(int8_t[TILE_NK * TILE_NK]),
        0u,
        sizeof(int32_t[32 * TILE_NK]),
        debug_zero_bytes,
        debug_copy_bytes,
        zero_ns,
        copy_a_ns,
        copy_b_ns,
        copy_c_ns,
        tile_call_ns,
        copy_out_partial_ns,
        copy_out_final_ns,
        stats);
}

static inline void tpu_gemm_int8_int32_tiled_device_psum_impl(
    const int8_t  * a,
    const int8_t  * b,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_tile_execute_cached_fn tile_fn,
    tpu_tile_sequence_execute_cached_fn sequence_fn) {
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    switch (tpu_get_effective_max_nk()) {
        case 128u:
            tpu_gemm_int8_int32_tiled_device_psum_impl_for_max_nk<128>(
                a, b, d, M, N, K, stats, tile_fn, sequence_fn, TPU_B_LAYOUT_NK, "plain_device_psum", NULL);
            return;
        case 64u:
        default:
            tpu_gemm_int8_int32_tiled_device_psum_impl_for_max_nk<64>(
                a, b, d, M, N, K, stats, tile_fn, sequence_fn, TPU_B_LAYOUT_NK, "plain_device_psum", NULL);
            return;
    }
}

template <uint32_t TILE_NK>
static inline void tpu_gemm_f32_int8_int32_tiled_device_psum_impl_for_max_nk(
    const float   * a,
    const float   * a_inv_scales,
    const int8_t  * b,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_tile_execute_cached_fn tile_fn,
    tpu_tile_sequence_execute_cached_fn sequence_fn,
    tpu_b_layout_t b_layout,
    const char * mode,
    void * packed_b_cache_ptr) {
    static uint64_t s_tiled_call_counter = 0;
    const uint32_t tile_m = 32;
    const uint32_t tile_n = TILE_NK;
    const uint32_t tile_k = TILE_NK;
    const uint64_t call_idx = ++s_tiled_call_counter;
    uint64_t debug_zero_bytes = 0;
    uint64_t debug_copy_bytes = 0;
    uint64_t debug_outer_tiles = 0;
    uint64_t debug_hw_tiles = 0;
    int64_t zero_ns = 0;
    int64_t copy_a_ns = 0;
    int64_t copy_b_ns = 0;
    int64_t copy_c_ns = 0;
    int64_t tile_call_ns = 0;
    int64_t copy_out_partial_ns = 0;
    int64_t copy_out_final_ns = 0;
    tpu_raw_a_cache_t raw_a_cache;
    tpu_packed_b_cache_t local_packed_b_cache;
    tpu_packed_b_cache_t * packed_b_cache = tpu_resolve_packed_b_cache(packed_b_cache_ptr);
    const bool persistent_packed_b_cache = packed_b_cache != NULL;
    if (packed_b_cache == NULL) {
        packed_b_cache = &local_packed_b_cache;
    }
    const bool use_packed_b_cache = b_layout == TPU_B_LAYOUT_KN && sequence_fn != NULL;

    for (uint32_t m0 = 0; m0 < M; m0 += tile_m) {
        const uint32_t m_actual = (M - m0) < tile_m ? (M - m0) : tile_m;

        for (uint32_t n0 = 0; n0 < N; n0 += tile_n) {
            const uint32_t n_actual = (N - n0) < tile_n ? (N - n0) : tile_n;
            const tpu_accum_key_t accum_key = { call_idx, m0, n0, m_actual, n_actual };
            debug_outer_tiles++;

            const uint32_t first_k_actual = K < tile_k ? K : tile_k;
            const tpu_hw_cfg_t chain_cfg = tpu_select_hw_cfg(TILE_NK, m_actual, n_actual, first_k_actual);
            if (chain_cfg.m == 0 || chain_cfg.n == 0 || chain_cfg.k == 0) {
                return;
            }

            for (uint32_t k0 = 0; k0 < K;) {
                const uint32_t remaining_k_tiles = (K - k0 + tile_k - 1u) / tile_k;
                uint32_t sequence_count = remaining_k_tiles < 8u ? remaining_k_tiles : 8u;
                if (sequence_fn != NULL && remaining_k_tiles > sequence_count &&
                    (remaining_k_tiles - sequence_count) == 1u && sequence_count > 2u) {
                    sequence_count--;
                }
                const bool try_sequence = sequence_fn != NULL && sequence_count > 1u;
                tpu_tile_execute_desc_t sequence_descs[8] = {};
                int8_t sequence_b_tiles[8][TILE_NK * TILE_NK];
                int32_t d_tile[32 * TILE_NK];
                bool can_sequence = try_sequence;

                for (uint32_t si = 0; si < sequence_count; ++si) {
                    const uint32_t tile_k0 = k0 + si * tile_k;
                    const uint32_t k_actual = (K - tile_k0) < tile_k ? (K - tile_k0) : tile_k;
                    const tpu_hw_cfg_t cfg = chain_cfg;
                    const bool first_k = tile_k0 == 0;
                    const bool last_k = (tile_k0 + tile_k) >= K;
                    debug_hw_tiles++;

                    if (cfg.m == 0 || cfg.n == 0 || cfg.k == 0) {
                        return;
                    }

                    const size_t a_tile_bytes = (size_t) cfg.m * cfg.k * sizeof(int8_t);
                    const size_t b_tile_bytes = (size_t) cfg.n * cfg.k * sizeof(int8_t);
                    const int64_t copy_a_start = tpu_tiling_now_ns();
                    const tpu_raw_a_cache_key_t a_cache_key = { a, a_inv_scales, K, m0, tile_k0, m_actual, k_actual, cfg.m, cfg.k };
                    const int8_t * desc_a = tpu_get_or_build_raw_a_tile_from_f32(
                            raw_a_cache,
                            a_cache_key,
                            a,
                            a_inv_scales,
                            K,
                            m_actual,
                            k_actual,
                            cfg.m,
                            cfg.k);
                    copy_a_ns += tpu_tiling_now_ns() - copy_a_start;
                    debug_zero_bytes += a_tile_bytes;
                    debug_copy_bytes += (uint64_t) m_actual * k_actual * sizeof(int8_t);

                    const int64_t tile_zero_start = tpu_tiling_now_ns();
                    if (!use_packed_b_cache) {
                        memset(sequence_b_tiles[si], 0, b_tile_bytes);
                        debug_zero_bytes += b_tile_bytes;
                    }
                    zero_ns += tpu_tiling_now_ns() - tile_zero_start;

                    const int64_t copy_b_start = tpu_tiling_now_ns();
                    tpu_b_layout_t desc_b_layout = b_layout;
                    const int8_t * desc_b = sequence_b_tiles[si];
                    if (b_layout == TPU_B_LAYOUT_KN) {
                        if (use_packed_b_cache) {
                            const tpu_packed_b_cache_key_t cache_key = {
                                b, N, n0, tile_k0, n_actual, k_actual, cfg.n, cfg.k,
                                (uint32_t) tpu_sequence_b16_bankfriendly_layout_enabled()
                            };
                            desc_b = tpu_get_or_build_packed_b_tile(
                                    *packed_b_cache,
                                    cache_key,
                                    b,
                                    N,
                                    n_actual,
                                    k_actual,
                                    cfg.n,
                                    cfg.k);
                            desc_b_layout = TPU_B_LAYOUT_DEVICE_PACKED;
                        } else {
                            for (uint32_t kk = 0; kk < k_actual; ++kk) {
                                memcpy(sequence_b_tiles[si] + kk*cfg.n, b + (tile_k0 + kk)*N + n0, n_actual*sizeof(int8_t));
                                debug_copy_bytes += (uint64_t) n_actual * sizeof(int8_t);
                            }
                        }
                    } else {
                        for (uint32_t ni = 0; ni < n_actual; ++ni) {
                            memcpy(sequence_b_tiles[si] + ni*cfg.k, b + (n0 + ni)*K + tile_k0, k_actual*sizeof(int8_t));
                            debug_copy_bytes += (uint64_t) k_actual * sizeof(int8_t);
                        }
                    }
                    copy_b_ns += tpu_tiling_now_ns() - copy_b_start;

                    tpu_update_tiling_stats(stats, cfg, m_actual, n_actual, k_actual);

                    sequence_descs[si].a = desc_a;
                    sequence_descs[si].b = desc_b;
                    sequence_descs[si].d = last_k ? d_tile : NULL;
                    sequence_descs[si].d_final = last_k ? (d + m0*N + n0) : NULL;
                    sequence_descs[si].d_final_stride = N;
                    sequence_descs[si].d_actual_m = m_actual;
                    sequence_descs[si].d_actual_n = n_actual;
                    sequence_descs[si].m = cfg.m;
                    sequence_descs[si].n = cfg.n;
                    sequence_descs[si].k = cfg.k;
                    sequence_descs[si].a_layout = TPU_A_LAYOUT_RAW_MK;
                    sequence_descs[si].a_bytes = a_tile_bytes;
                    sequence_descs[si].b_layout = desc_b_layout;
                    if (desc_b_layout == TPU_B_LAYOUT_DEVICE_PACKED && persistent_packed_b_cache) {
                        sequence_descs[si].b_cache_owner = packed_b_cache;
                        sequence_descs[si].b_cache_source = b;
                        sequence_descs[si].b_cache_source_n = N;
                        sequence_descs[si].b_cache_n0 = n0;
                        sequence_descs[si].b_cache_k0 = tile_k0;
                        sequence_descs[si].b_cache_n_actual = n_actual;
                        sequence_descs[si].b_cache_k_actual = k_actual;
                    }
                    sequence_descs[si].c_source = first_k ? TPU_C_SOURCE_ZERO : TPU_C_SOURCE_DEVICE_ACCUM;
                    sequence_descs[si].d_dest = last_k ? TPU_D_DEST_HOST_READBACK : TPU_D_DEST_DEVICE_ACCUM;
                    sequence_descs[si].accum_key = accum_key;
                    sequence_descs[si].next_desc = NULL;
                }

                tpu_tile_execute_result_t result = {};
                const int64_t tile_call_start = tpu_tiling_now_ns();
                if (can_sequence) {
                    if (sequence_fn(sequence_descs, sequence_count, &result) != 0) {
                        abort();
                    }
                    tile_call_ns += tpu_tiling_now_ns() - tile_call_start;
                    k0 += sequence_count * tile_k;
                } else {
                    for (uint32_t si = 0; si < sequence_count; ++si) {
                        if (si + 1u < sequence_count) {
                            sequence_descs[si].next_desc = &sequence_descs[si + 1u];
                        }
                        tpu_tile_execute_result_t single_result = {};
                        if (tile_fn(&sequence_descs[si], &single_result) != 0) {
                            abort();
                        }
                        result = single_result;
                    }
                    tile_call_ns += tpu_tiling_now_ns() - tile_call_start;
                    k0 += sequence_count * tile_k;
                }

                const tpu_tile_execute_desc_t * last_desc = &sequence_descs[sequence_count - 1u];
                if (last_desc->d_dest == TPU_D_DEST_HOST_READBACK && !result.readback_direct_final) {
                    const int64_t copy_out_final_start = tpu_tiling_now_ns();
                    for (uint32_t mi = 0; mi < m_actual; ++mi) {
                        memcpy(d + (m0 + mi)*N + n0, d_tile + mi*last_desc->n, n_actual*sizeof(int32_t));
                        debug_copy_bytes += (uint64_t) n_actual * sizeof(int32_t);
                    }
                    copy_out_final_ns += tpu_tiling_now_ns() - copy_out_final_start;
                    if (stats != NULL) {
                        stats->direct_final_misses++;
                    }
                } else if (last_desc->d_dest == TPU_D_DEST_HOST_READBACK && stats != NULL) {
                    stats->direct_final_hits++;
                }
            }
        }
    }

    if (stats != NULL) {
        stats->total_zero_ns += zero_ns;
        stats->total_copy_a_ns += copy_a_ns;
        stats->total_copy_b_ns += copy_b_ns;
        stats->total_copy_c_ns += copy_c_ns;
        stats->total_tile_call_ns += tile_call_ns;
        stats->total_copy_out_partial_ns += copy_out_partial_ns;
        stats->total_copy_out_final_ns += copy_out_final_ns;
    }

    tpu_tiling_log_summary(
        call_idx,
        mode,
        M,
        N,
        K,
        debug_outer_tiles,
        debug_hw_tiles,
        0u,
        sizeof(int8_t[32 * TILE_NK]),
        sizeof(int8_t[TILE_NK * TILE_NK]),
        0u,
        sizeof(int32_t[32 * TILE_NK]),
        debug_zero_bytes,
        debug_copy_bytes,
        zero_ns,
        copy_a_ns,
        copy_b_ns,
        copy_c_ns,
        tile_call_ns,
        copy_out_partial_ns,
        copy_out_final_ns,
        stats);
}

static inline void tpu_gemm_f32_int8_int32_tiled_prepacked_b_device_psum_impl(
    const float   * a,
    const float   * a_inv_scales,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_tile_execute_cached_fn tile_fn,
    tpu_tile_sequence_execute_cached_fn sequence_fn,
    void * packed_b_cache) {
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    switch (tpu_get_effective_max_nk()) {
        case 128u:
            tpu_gemm_f32_int8_int32_tiled_device_psum_impl_for_max_nk<128>(
                a, a_inv_scales, b_kn, d, M, N, K, stats, tile_fn, sequence_fn, TPU_B_LAYOUT_KN, "fused_raw_a_prepacked_b_device_psum", packed_b_cache);
            return;
        case 64u:
        default:
            tpu_gemm_f32_int8_int32_tiled_device_psum_impl_for_max_nk<64>(
                a, a_inv_scales, b_kn, d, M, N, K, stats, tile_fn, sequence_fn, TPU_B_LAYOUT_KN, "fused_raw_a_prepacked_b_device_psum", packed_b_cache);
            return;
    }
}
static inline void tpu_gemm_int8_int32_tiled_prepacked_b_device_psum_impl(
    const int8_t  * a,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    tpu_tile_execute_cached_fn tile_fn,
    tpu_tile_sequence_execute_cached_fn sequence_fn,
    void * packed_b_cache) {
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
    }
    if (M == 0 || N == 0 || K == 0) {
        return;
    }

    switch (tpu_get_effective_max_nk()) {
        case 128u:
            tpu_gemm_int8_int32_tiled_device_psum_impl_for_max_nk<128>(
                a, b_kn, d, M, N, K, stats, tile_fn, sequence_fn, TPU_B_LAYOUT_KN, "prepacked_b_device_psum", packed_b_cache);
            return;
        case 64u:
        default:
            tpu_gemm_int8_int32_tiled_device_psum_impl_for_max_nk<64>(
                a, b_kn, d, M, N, K, stats, tile_fn, sequence_fn, TPU_B_LAYOUT_KN, "prepacked_b_device_psum", packed_b_cache);
            return;
    }
}
