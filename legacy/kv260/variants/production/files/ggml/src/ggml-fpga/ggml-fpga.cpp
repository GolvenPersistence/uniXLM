#include "ggml-impl.h"
#include "ggml-fpga.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include "tpu-dispatch.h"
#include "tpu-tiling.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define GGML_FPGA_HAVE_ACT_QUANT_NEON 1
#else
#define GGML_FPGA_HAVE_ACT_QUANT_NEON 0
#endif

struct tpu_packed_b_cache_owner {
    tpu_packed_b_cache_t cache;
    bool matrix_task_prepack_attempted = false;
    bool matrix_task_prepack_ok = false;
};

struct fpga_quant_params {
    float scale = 1.0f;
};

static_assert(sizeof(fpga_quant_params) == sizeof(float),
        "fpga_quant_params must stay scale-only for NEON scale loads");

struct weight_cache_entry {
    std::vector<int8_t> int8_data;
    std::vector<int8_t> int8_data_kn;
    std::vector<fpga_quant_params> qp;
    tpu_packed_b_cache_owner packed_b_cache;
    int64_t N = 0;
    int64_t K = 0;
    bool matrix_task_prepack_allowed = true;
    bool matrix_task_decode_nk_allowed = false;
};

struct dynamic_src0_cache_key {
    const ggml_tensor * tensor = nullptr;
    const void * base = nullptr;
    int64_t n = 0;
    int64_t k = 0;

    bool operator==(const dynamic_src0_cache_key & other) const {
        return tensor == other.tensor &&
               base == other.base &&
               n == other.n &&
               k == other.k;
    }
};

struct dynamic_src0_cache_key_hash {
    size_t operator()(const dynamic_src0_cache_key & key) const {
        size_t h = std::hash<const void *>()(key.tensor);
        h ^= std::hash<const void *>()(key.base) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>()(key.n) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        h ^= std::hash<int64_t>()(key.k) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        return h;
    }
};

struct ggml_backend_fpga_graph_wall_stats {
    uint64_t graphs = 0;
    uint64_t nodes = 0;
    uint64_t mul_mat_nodes = 0;
    uint64_t add_nodes = 0;
    uint64_t layout_nodes = 0;
    int64_t wall_ns = 0;
    int64_t mul_mat_wall_ns = 0;
    int64_t add_wall_ns = 0;
    int64_t layout_wall_ns = 0;
    int64_t weight_cache_ns = 0;
    int64_t dynamic_src0_ns = 0;
    int64_t activation_quant_ns = 0;
    int64_t gemm_call_ns = 0;
    int64_t dequant_ns = 0;
    int64_t mul_mat_setup_ns = 0;
    int64_t weight_cache_lookup_ns = 0;
    int64_t matrix_task_prepack_ns = 0;
    int64_t mul_mat_gate_ns = 0;
    int64_t activation_source_lookup_ns = 0;
    int64_t tiling_stats_accum_ns = 0;
    int64_t ffn_consume_ns = 0;
    int64_t mul_mat_instance_admin_ns = 0;
    int64_t mul_mat_residual_ns = 0;
    int64_t mul_mat_nested_overrun_ns = 0;
    tpu_runtime_stats_t runtime = {};
};

struct ggml_backend_fpga_dequant_path_stats {
    int64_t rows_total = 0;
    int64_t rows_neon = 0;
    int64_t rows_scalar = 0;
    int64_t rows_scratch_copy = 0;
    int64_t cols_total = 0;
};

struct ggml_backend_fpga_dynamic_src0_stats {
    int64_t total = 0;
    int64_t op_non_none = 0;
    int64_t view_src = 0;
    int64_t input_flag = 0;
    std::map<std::tuple<int64_t, int64_t, int64_t>, int64_t> shape_counts;
    std::map<std::string, int64_t> op_counts;
    std::map<std::string, int64_t> name_counts;
};

struct ggml_backend_fpga_matrix_task_shape_stats {
    int64_t attempts = 0;
    int64_t used = 0;
    int64_t fallback = 0;
    uint64_t task_count = 0;
    uint64_t tile_count = 0;
    uint64_t b_hits = 0;
    uint64_t b_misses = 0;
    uint64_t prepare_b_us = 0;
    uint64_t submit_us = 0;
    uint64_t output_sync_us = 0;
    uint64_t total_us = 0;
};

struct ggml_backend_fpga_activation_reuse_entry {
    const ggml_tensor * tensor = nullptr;
    const void * base = nullptr;
    int64_t rows = 0;
    int64_t cols = 0;
    int64_t row_stride = 0;
    std::vector<int8_t> data_owner;
    std::vector<float> scales_owner;
    std::vector<float> inv_scales_owner;
    const int8_t * data = nullptr;
    const float * scales = nullptr;
    const float * inv_scales = nullptr;
    bool valid = false;
};

struct ggml_backend_fpga_context {
    int64_t n_mul_mat_calls = 0;
    int64_t n_gemm_instances = 0;
    int64_t n_empty_mul_mat_skipped = 0;
    int64_t n_empty_gemm_instances_skipped = 0;
    int64_t total_quantize_ns = 0;
    int64_t activation_quantize_ns = 0;
    int64_t weight_cache_build_ns = 0;
    int64_t total_gemm_ns = 0;
    int64_t total_dequant_ns = 0;
    ggml_backend_fpga_dequant_path_stats dequant_path = {};
    int64_t ffn_stream_cache_hits = 0;
    int64_t ffn_stream_cache_misses = 0;

    int64_t weight_cache_hits = 0;
    int64_t weight_cache_misses = 0;
    int64_t weight_cache_bypasses = 0;
    int64_t dynamic_src0_build_ns = 0;
    ggml_backend_fpga_dynamic_src0_stats dynamic_src0_stats = {};
    int64_t dynamic_src0_cache_hits = 0;
    int64_t dynamic_src0_cache_misses = 0;
    int64_t dynamic_src0_cache_stores = 0;
    int64_t dynamic_src0_cache_clears = 0;
    int64_t dynamic_src0_cache_store_bytes = 0;
    int64_t dynamic_src0_prepared_hits = 0;
    int64_t dynamic_src0_prepared_misses = 0;
    int64_t dynamic_src0_prepared_import_ns = 0;
    int64_t dynamic_src0_prepared_check_entries = 0;
    int64_t dynamic_src0_prepared_check_int8_bad_entries = 0;
    int64_t dynamic_src0_prepared_check_int8_bad_values = 0;
    int64_t dynamic_src0_prepared_check_scale_bad_entries = 0;
    int64_t dynamic_src0_prepared_check_scale_bad_rows = 0;
    int64_t dynamic_src0_prepared_check_max_i8_abs_diff = 0;
    double dynamic_src0_prepared_check_max_scale_abs_diff = 0.0;
    int64_t mul_mat_setup_ns = 0;
    int64_t weight_cache_lookup_ns = 0;
    int64_t matrix_task_prepack_ns = 0;
    int64_t mul_mat_gate_ns = 0;
    int64_t activation_source_lookup_ns = 0;
    int64_t tiling_stats_accum_ns = 0;
    int64_t ffn_consume_ns = 0;
    int64_t mul_mat_instance_admin_ns = 0;
    ggml_backend_fpga_activation_reuse_entry activation_reuse = {};
    int64_t activation_reuse_attempts = 0;
    int64_t activation_reuse_hits = 0;
    int64_t activation_reuse_misses = 0;
    int64_t activation_reuse_stores = 0;
    int64_t activation_reuse_store_ns = 0;
    int64_t activation_reuse_store_bytes = 0;
    int64_t activation_reuse_clears = 0;
    int64_t ffn_dual_attempts = 0;
    int64_t ffn_dual_hits = 0;
    int64_t ffn_dual_misses = 0;

    int64_t min_k = std::numeric_limits<int64_t>::max();
    int64_t min_n = std::numeric_limits<int64_t>::max();
    int64_t min_m = std::numeric_limits<int64_t>::max();
    int64_t max_k = 0;
    int64_t max_n = 0;
    int64_t max_m = 0;

    std::map<int64_t, int64_t> k_value_counts;
    std::map<int64_t, int64_t> n_value_counts;
    std::map<int64_t, int64_t> m_value_counts;
    std::map<std::tuple<int64_t, int64_t, int64_t>, int64_t> gemm_shape_counts;
    std::map<std::tuple<int64_t, int64_t, int64_t>, int64_t> empty_gemm_shape_counts;
    std::map<std::string, ggml_backend_fpga_matrix_task_shape_stats> matrix_task_shape_stats;

    int64_t tiling_total_hw_calls = 0;
    int64_t tiling_calls_32x128x128 = 0;
    int64_t tiling_calls_16x128x128 = 0;
    int64_t tiling_calls_1x128x128 = 0;
    int64_t tiling_calls_32x64x64 = 0;
    int64_t tiling_calls_16x64x64 = 0;
    int64_t tiling_calls_1x64x64 = 0;
    int64_t tiling_calls_1x32x32 = 0;
    int64_t tiling_calls_1x16x32 = 0;
    int64_t tiling_calls_32x32x32 = 0;
    int64_t tiling_calls_32x32x16 = 0;
    int64_t tiling_calls_16x32x32 = 0;
    int64_t tiling_calls_32x16x16 = 0;
    int64_t tiling_calls_16x16x16 = 0;
    int64_t tiling_total_padded_elements = 0;
    int64_t tiling_total_useful_elements = 0;
    int64_t tiling_zero_ns = 0;
    int64_t tiling_copy_a_ns = 0;
    int64_t tiling_copy_b_ns = 0;
    int64_t tiling_copy_c_ns = 0;
    int64_t tiling_tile_call_ns = 0;
    int64_t tiling_copy_out_partial_ns = 0;
    int64_t tiling_copy_out_final_ns = 0;

    tpu_runtime_stats_t runtime_stats_accumulated = {};
    ggml_backend_fpga_graph_wall_stats graph_wall_stats[GGML_BACKEND_FPGA_PHASE_COUNT] = {};

    std::unordered_map<const void *, weight_cache_entry> weight_cache;
    std::unordered_map<dynamic_src0_cache_key, weight_cache_entry, dynamic_src0_cache_key_hash> dynamic_src0_cache;

    std::vector<float> scratch_w_float;
    std::vector<float> scratch_a_inv_scales;
    std::vector<int8_t> scratch_a_int8;
    std::vector<fpga_quant_params> scratch_a_qp;
    std::vector<int32_t> scratch_d_int32;
    std::vector<int32_t> scratch_d_int32_dequant;
};

struct ggml_backend_fpga_rejection_stats {
    std::mutex mutex;
    int64_t total_rejected_mul_mat = 0;
    std::map<std::string, int64_t> rejected_mul_mat;
};

static bool ggml_backend_fpga_is_vit_attn_strided_target(const ggml_tensor * op);

static int64_t ggml_backend_fpga_now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool ggml_backend_fpga_log_summary_enabled() {
    const char * value = getenv("GGML_FPGA_LOG_SUMMARY");
    return value != NULL && *value != '\0' && strcmp(value, "0") != 0;
}

static bool ggml_backend_fpga_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != NULL && *value != '\0' && strcmp(value, "0") != 0;
}

static bool ggml_backend_fpga_env_enabled_by_default(const char * name) {
    const char * value = getenv(name);
    return value == NULL || *value == '\0' || strcmp(value, "0") != 0;
}

static int64_t ggml_backend_fpga_env_i64_or_default(const char * name, int64_t default_value) {
    const char * value = getenv(name);
    if (value == NULL || *value == '\0') {
        return default_value;
    }

    char * end = NULL;
    errno = 0;
    const long long parsed = strtoll(value, &end, 0);
    if (errno != 0 || end == value || (end != NULL && *end != '\0')) {
        return default_value;
    }
    return (int64_t) parsed;
}

static bool ggml_backend_fpga_fused_raw_a_enabled() {
#ifdef GGML_FPGA_USE_MOCK
    return false;
#else
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_FUSED_RAW_A") ? 1 : 0;
    }
    return cached != 0;
#endif
}

static bool ggml_backend_fpga_act_quant_fused_rows_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_ACT_QUANT_FUSED_ROWS") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_act_quant_neon_enabled() {
#if GGML_FPGA_HAVE_ACT_QUANT_NEON
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_ACT_QUANT_NEON") ? 1 : 0;
    }
    return cached != 0;
#else
    return false;
#endif
}

static bool ggml_backend_fpga_strided_mul_mat_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_STRIDED_MUL_MAT") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_src1_strided_mul_mat_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_SRC1_STRIDED_MUL_MAT") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_src1_strided_materialize_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_SRC1_STRIDED_MATERIALIZE") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_vit_attn_strided_mul_mat_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_VIT_ATTN_STRIDED_MUL_MAT") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_vit_attn_pl_op_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_VIT_ATTN_PL_OP") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_mul_mat_submit_trace_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_MUL_MAT_SUBMIT_TRACE") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_add_op_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_ADD_OP") ? 1 : 0;
    }
    return cached != 0;
}

struct ggml_backend_fpga_r5_matrix_task_decode_config {
    bool enabled = false;
    bool r5_wait = false;
    bool multi_instance = false;
    int64_t max_m = 1;
    int64_t min_k = 64;
    int64_t min_n = 64;
};

static const ggml_backend_fpga_r5_matrix_task_decode_config &
ggml_backend_fpga_r5_matrix_task_decode_config_get() {
    static bool initialized = false;
    static ggml_backend_fpga_r5_matrix_task_decode_config cfg;
    if (!initialized) {
        cfg.enabled = ggml_backend_fpga_env_enabled("TPU_V2_R5_MATRIX_TASK_DECODE");
        cfg.r5_wait = ggml_backend_fpga_env_enabled("TPU_R5_WAIT_STATUS");
        cfg.multi_instance = ggml_backend_fpga_env_enabled("TPU_V2_R5_MATRIX_TASK_MULTI_INSTANCE");
        cfg.max_m = ggml_backend_fpga_env_i64_or_default("TPU_V2_R5_MATRIX_TASK_DECODE_MAX_M", 1);
        cfg.min_k = ggml_backend_fpga_env_i64_or_default("TPU_V2_R5_MATRIX_TASK_DECODE_MIN_K", 64);
        cfg.min_n = ggml_backend_fpga_env_i64_or_default("TPU_V2_R5_MATRIX_TASK_DECODE_MIN_N", 64);
        initialized = true;
    }
    return cfg;
}

static bool ggml_backend_fpga_r5_matrix_task_decode_nk_allowed(
        int64_t n,
        int64_t k) {
#ifdef GGML_FPGA_USE_MOCK
    (void) n;
    (void) k;
    return false;
#else
    const auto & cfg = ggml_backend_fpga_r5_matrix_task_decode_config_get();
    return cfg.enabled && cfg.r5_wait && n >= cfg.min_n && k >= cfg.min_k;
#endif
}

static bool ggml_backend_fpga_matrix_task_nk_shape_allowed(
        int64_t n,
        int64_t k) {
#ifdef GGML_FPGA_USE_MOCK
    (void) n;
    (void) k;
    return false;
#else
    const auto & cfg = ggml_backend_fpga_r5_matrix_task_decode_config_get();
    return n >= cfg.min_n && k >= cfg.min_k;
#endif
}

static bool ggml_backend_fpga_pl_op_descriptor_m_allowed(int64_t m) {
#ifdef GGML_FPGA_USE_MOCK
    (void) m;
    return false;
#else
    return ggml_backend_fpga_env_enabled("TPU_V2_PL_OP_DESCRIPTOR") &&
           m > 0 &&
           m <= ggml_backend_fpga_env_i64_or_default("TPU_V2_PL_OP_DESCRIPTOR_MAX_M", 1);
#endif
}

static bool ggml_backend_fpga_vit_attn_pl_op_allowed(
        const ggml_tensor * op,
        int64_t m,
        int64_t n,
        int64_t k) {
    return ggml_backend_fpga_vit_attn_pl_op_enabled() &&
           ggml_backend_fpga_is_vit_attn_strided_target(op) &&
           ggml_backend_fpga_pl_op_descriptor_m_allowed(m) &&
           ggml_backend_fpga_matrix_task_nk_shape_allowed(n, k);
}

static bool ggml_backend_fpga_r5_matrix_task_decode_enabled(
        int64_t m,
        int64_t n,
        int64_t k,
        int64_t gemm_instances) {
#ifdef GGML_FPGA_USE_MOCK
    (void) m;
    (void) n;
    (void) k;
    (void) gemm_instances;
    return false;
#else
    const auto & cfg = ggml_backend_fpga_r5_matrix_task_decode_config_get();
    return (gemm_instances == 1 || cfg.multi_instance) &&
           m > 0 && m <= cfg.max_m &&
           ggml_backend_fpga_r5_matrix_task_decode_nk_allowed(n, k);
#endif
}

static bool ggml_backend_fpga_r5_matrix_task_dynamic_multi_instance_enabled() {
#ifdef GGML_FPGA_USE_MOCK
    return false;
#else
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("TPU_V2_R5_MATRIX_TASK_DYNAMIC_MULTI_INSTANCE") ? 1 : 0;
    }
    return cached != 0;
#endif
}

static bool ggml_backend_fpga_r5_matrix_task_strict() {
#ifdef GGML_FPGA_USE_MOCK
    return false;
#else
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("TPU_V2_R5_MATRIX_TASK_STRICT") ? 1 : 0;
    }
    return cached != 0;
#endif
}

static bool ggml_backend_fpga_r5_matrix_task_prepack_enabled() {
#ifdef GGML_FPGA_USE_MOCK
    return false;
#else
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("TPU_V2_R5_MATRIX_TASK_PREPACK") ? 1 : 0;
    }
    return cached != 0;
#endif
}

static bool ggml_backend_fpga_matrix_task_direct_view_enabled() {
#ifdef GGML_FPGA_USE_MOCK
    return false;
#else
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("TPU_V2_MATRIX_TASK_DIRECT_VIEW") ? 1 : 0;
    }
    return cached != 0;
#endif
}

static bool ggml_backend_fpga_matrix_task_shape_stats_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("TPU_V2_MATMUL_TASK_SHAPE_STATS") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_r5_matrix_task_prepack_allowed_name(const ggml_tensor * src0) {
#ifdef GGML_FPGA_USE_MOCK
    (void) src0;
    return false;
#else
    static int prepack_all = -1;
    if (prepack_all < 0) {
        prepack_all = ggml_backend_fpga_env_enabled("TPU_V2_R5_MATRIX_TASK_PREPACK_ALL") ? 1 : 0;
    }
    if (prepack_all != 0) {
        return true;
    }
    if (src0 == NULL || src0->name[0] == '\0') {
        return true;
    }

    // The vision/mmproj weights are consumed in the prompt path only. Prepacking
    // them creates matrix-task B arenas that are never used by the decode R5
    // gate, and sampled100 showed this cost leaking into prefill.
    return strncmp(src0->name, "v.", 2) != 0 &&
           strncmp(src0->name, "mm.", 3) != 0;
#endif
}

static bool ggml_backend_fpga_act_quant_parallel_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_ACT_QUANT_PARALLEL") ? 1 : 0;
    }
    return cached != 0;
}

static int64_t ggml_backend_fpga_act_quant_parallel_threads() {
    static int64_t cached = 0;
    if (cached == 0) {
        int64_t threads = ggml_backend_fpga_env_i64_or_default("GGML_FPGA_ACT_QUANT_THREADS", 2);
        if (threads < 1) {
            threads = 1;
        }
        if (threads > 4) {
            threads = 4;
        }
        cached = threads;
    }
    return cached;
}

static int64_t ggml_backend_fpga_act_quant_parallel_min_elems() {
    static int64_t cached = -1;
    if (cached < 0) {
        const int64_t min_elems = ggml_backend_fpga_env_i64_or_default(
                "GGML_FPGA_ACT_QUANT_PARALLEL_MIN_ELEMS",
                262144);
        cached = min_elems < 0 ? 0 : min_elems;
    }
    return cached;
}

static bool ggml_backend_fpga_dequant_scratch_copy_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("TPU_V2_DEQUANT_SCRATCH_COPY") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_dynamic_src0_cache_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_DYNAMIC_SRC0_CACHE") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_dynamic_src0_prepared_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_DYNAMIC_SRC0_PREPARED") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_dynamic_src0_prepared_tensor_allowed(const ggml_tensor * tensor) {
    if (!ggml_backend_fpga_dynamic_src0_prepared_enabled() || tensor == nullptr) {
        return false;
    }
    const bool allow_k = ggml_backend_fpga_env_enabled_by_default("GGML_FPGA_DYNAMIC_SRC0_PREPARED_K");
    const bool allow_v = ggml_backend_fpga_env_enabled_by_default("GGML_FPGA_DYNAMIC_SRC0_PREPARED_V");
    if (std::strncmp(tensor->name, "Kcur_", 5) == 0) {
        return allow_k;
    }
    if (std::strncmp(tensor->name, "Vcur_", 5) == 0) {
        return allow_v;
    }
    return false;
}

static bool ggml_backend_fpga_dynamic_src0_prepared_check_only_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_DYNAMIC_SRC0_PREPARED_CHECK_ONLY") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_activation_reuse_sibling_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_ACT_REUSE_SIBLING") ? 1 : 0;
    }
    return cached != 0;
}

static bool ggml_backend_fpga_ffn_sibling_dual_enabled() {
    static int cached = -1;
    if (cached < 0) {
        cached = ggml_backend_fpga_env_enabled("GGML_FPGA_FUSE_FFN_UP_GATE_DUAL") ? 1 : 0;
    }
    return cached != 0;
}

static int64_t ggml_backend_fpga_activation_reuse_min_elems() {
    static int64_t cached = -1;
    if (cached < 0) {
        const int64_t min_elems = ggml_backend_fpga_env_i64_or_default(
                "GGML_FPGA_ACT_REUSE_SIBLING_MIN_ELEMS",
                512);
        cached = min_elems < 0 ? 0 : min_elems;
    }
    return cached;
}

static int64_t ggml_backend_fpga_activation_reuse_max_elems() {
    static int64_t cached = -1;
    if (cached < 0) {
        const int64_t max_elems = ggml_backend_fpga_env_i64_or_default(
                "GGML_FPGA_ACT_REUSE_SIBLING_MAX_ELEMS",
                4096);
        cached = max_elems <= 0 ? std::numeric_limits<int64_t>::max() : max_elems;
    }
    return cached;
}

static uint64_t ggml_backend_fpga_counter_delta(uint64_t current, uint64_t baseline) {
    return current >= baseline ? current - baseline : 0;
}

static void ggml_backend_fpga_setenv_default(const char * name, const char * value) {
    if (getenv(name) == NULL) {
        (void) setenv(name, value, 0);
    }
}

static tpu_runtime_stats_t ggml_backend_fpga_runtime_stats_delta(
        const tpu_runtime_stats_t & current,
        const tpu_runtime_stats_t & baseline) {
    tpu_runtime_stats_t delta = {};
    delta.tile_call_count = ggml_backend_fpga_counter_delta(current.tile_call_count, baseline.tile_call_count);
    delta.prevd_chain_runs = ggml_backend_fpga_counter_delta(current.prevd_chain_runs, baseline.prevd_chain_runs);
    delta.prevd_over_signoff_runs = ggml_backend_fpga_counter_delta(current.prevd_over_signoff_runs, baseline.prevd_over_signoff_runs);
    delta.prevd_ab_preload_count = ggml_backend_fpga_counter_delta(current.prevd_ab_preload_count, baseline.prevd_ab_preload_count);
    delta.prevd_host_c_upload_count = ggml_backend_fpga_counter_delta(current.prevd_host_c_upload_count, baseline.prevd_host_c_upload_count);
    delta.prevd_host_d_readback_count = ggml_backend_fpga_counter_delta(current.prevd_host_d_readback_count, baseline.prevd_host_d_readback_count);
    delta.prevd_direct_final_readback_count = ggml_backend_fpga_counter_delta(current.prevd_direct_final_readback_count, baseline.prevd_direct_final_readback_count);
    delta.total_wait_done_us = ggml_backend_fpga_counter_delta(current.total_wait_done_us, baseline.total_wait_done_us);
    delta.total_read_us = ggml_backend_fpga_counter_delta(current.total_read_us, baseline.total_read_us);
    delta.total_read_sync_us = ggml_backend_fpga_counter_delta(current.total_read_sync_us, baseline.total_read_sync_us);
    delta.total_read_copy_us = ggml_backend_fpga_counter_delta(current.total_read_copy_us, baseline.total_read_copy_us);
    delta.total_read_unpack_us = ggml_backend_fpga_counter_delta(current.total_read_unpack_us, baseline.total_read_unpack_us);
    delta.total_read_mmio_us = ggml_backend_fpga_counter_delta(current.total_read_mmio_us, baseline.total_read_mmio_us);
    delta.total_read_debug_dump_us = ggml_backend_fpga_counter_delta(current.total_read_debug_dump_us, baseline.total_read_debug_dump_us);
    delta.total_read_bytes = ggml_backend_fpga_counter_delta(current.total_read_bytes, baseline.total_read_bytes);
    delta.total_read_output_buffer_count = ggml_backend_fpga_counter_delta(current.total_read_output_buffer_count, baseline.total_read_output_buffer_count);
    delta.total_read_memcpy_count = ggml_backend_fpga_counter_delta(current.total_read_memcpy_count, baseline.total_read_memcpy_count);
    delta.total_read_unpack_count = ggml_backend_fpga_counter_delta(current.total_read_unpack_count, baseline.total_read_unpack_count);
    delta.total_read_mmio_count = ggml_backend_fpga_counter_delta(current.total_read_mmio_count, baseline.total_read_mmio_count);
    delta.read_op_batch_count = ggml_backend_fpga_counter_delta(current.read_op_batch_count, baseline.read_op_batch_count);
    delta.read_op_batch_total_us = ggml_backend_fpga_counter_delta(current.read_op_batch_total_us, baseline.read_op_batch_total_us);
    delta.read_op_batch_sync_us = ggml_backend_fpga_counter_delta(current.read_op_batch_sync_us, baseline.read_op_batch_sync_us);
    delta.read_op_batch_copy_us = ggml_backend_fpga_counter_delta(current.read_op_batch_copy_us, baseline.read_op_batch_copy_us);
    delta.read_op_batch_bytes = ggml_backend_fpga_counter_delta(current.read_op_batch_bytes, baseline.read_op_batch_bytes);
    delta.read_async_count = ggml_backend_fpga_counter_delta(current.read_async_count, baseline.read_async_count);
    delta.read_async_total_us = ggml_backend_fpga_counter_delta(current.read_async_total_us, baseline.read_async_total_us);
    delta.read_async_sync_us = ggml_backend_fpga_counter_delta(current.read_async_sync_us, baseline.read_async_sync_us);
    delta.read_async_copy_us = ggml_backend_fpga_counter_delta(current.read_async_copy_us, baseline.read_async_copy_us);
    delta.read_async_bytes = ggml_backend_fpga_counter_delta(current.read_async_bytes, baseline.read_async_bytes);
    delta.read_async_batch_begin_wait_us = ggml_backend_fpga_counter_delta(current.read_async_batch_begin_wait_us, baseline.read_async_batch_begin_wait_us);
    delta.read_async_batch_begin_wait_count = ggml_backend_fpga_counter_delta(current.read_async_batch_begin_wait_count, baseline.read_async_batch_begin_wait_count);
    delta.read_async_stage_wait_us = ggml_backend_fpga_counter_delta(current.read_async_stage_wait_us, baseline.read_async_stage_wait_us);
    delta.read_async_stage_wait_count = ggml_backend_fpga_counter_delta(current.read_async_stage_wait_count, baseline.read_async_stage_wait_count);
    delta.read_async_flush_wait_us = ggml_backend_fpga_counter_delta(current.read_async_flush_wait_us, baseline.read_async_flush_wait_us);
    delta.read_async_flush_wait_count = ggml_backend_fpga_counter_delta(current.read_async_flush_wait_count, baseline.read_async_flush_wait_count);
    delta.read_final_desc_count = ggml_backend_fpga_counter_delta(current.read_final_desc_count, baseline.read_final_desc_count);
    delta.read_final_desc_total_us = ggml_backend_fpga_counter_delta(current.read_final_desc_total_us, baseline.read_final_desc_total_us);
    delta.read_final_desc_sync_us = ggml_backend_fpga_counter_delta(current.read_final_desc_sync_us, baseline.read_final_desc_sync_us);
    delta.read_final_desc_copy_us = ggml_backend_fpga_counter_delta(current.read_final_desc_copy_us, baseline.read_final_desc_copy_us);
    delta.read_final_desc_unpack_us = ggml_backend_fpga_counter_delta(current.read_final_desc_unpack_us, baseline.read_final_desc_unpack_us);
    delta.read_final_desc_mmio_us = ggml_backend_fpga_counter_delta(current.read_final_desc_mmio_us, baseline.read_final_desc_mmio_us);
    delta.read_final_desc_bytes = ggml_backend_fpga_counter_delta(current.read_final_desc_bytes, baseline.read_final_desc_bytes);
    delta.read_copy_job_count = ggml_backend_fpga_counter_delta(current.read_copy_job_count, baseline.read_copy_job_count);
    delta.read_copy_row_count = ggml_backend_fpga_counter_delta(current.read_copy_row_count, baseline.read_copy_row_count);
    delta.read_copy_elem_count = ggml_backend_fpga_counter_delta(current.read_copy_elem_count, baseline.read_copy_elem_count);
    delta.read_copy_useful_bytes = ggml_backend_fpga_counter_delta(current.read_copy_useful_bytes, baseline.read_copy_useful_bytes);
    delta.read_copy_transfer_bytes = ggml_backend_fpga_counter_delta(current.read_copy_transfer_bytes, baseline.read_copy_transfer_bytes);
    delta.read_copy_padded_jobs = ggml_backend_fpga_counter_delta(current.read_copy_padded_jobs, baseline.read_copy_padded_jobs);
    delta.read_copy_strided_jobs = ggml_backend_fpga_counter_delta(current.read_copy_strided_jobs, baseline.read_copy_strided_jobs);
    delta.read_copy_bulk_candidate_jobs = ggml_backend_fpga_counter_delta(current.read_copy_bulk_candidate_jobs, baseline.read_copy_bulk_candidate_jobs);
    delta.read_copy_bulk_candidate_bytes = ggml_backend_fpga_counter_delta(current.read_copy_bulk_candidate_bytes, baseline.read_copy_bulk_candidate_bytes);
    delta.read_copy_bulk_memcpy_jobs = ggml_backend_fpga_counter_delta(current.read_copy_bulk_memcpy_jobs, baseline.read_copy_bulk_memcpy_jobs);
    delta.read_copy_bulk_memcpy_bytes = ggml_backend_fpga_counter_delta(current.read_copy_bulk_memcpy_bytes, baseline.read_copy_bulk_memcpy_bytes);
    delta.read_copy_row_memcpy_rows = ggml_backend_fpga_counter_delta(current.read_copy_row_memcpy_rows, baseline.read_copy_row_memcpy_rows);
    delta.read_copy_rows_1 = ggml_backend_fpga_counter_delta(current.read_copy_rows_1, baseline.read_copy_rows_1);
    delta.read_copy_rows_2_15 = ggml_backend_fpga_counter_delta(current.read_copy_rows_2_15, baseline.read_copy_rows_2_15);
    delta.read_copy_rows_16 = ggml_backend_fpga_counter_delta(current.read_copy_rows_16, baseline.read_copy_rows_16);
    delta.read_copy_rows_17_31 = ggml_backend_fpga_counter_delta(current.read_copy_rows_17_31, baseline.read_copy_rows_17_31);
    delta.read_copy_rows_32 = ggml_backend_fpga_counter_delta(current.read_copy_rows_32, baseline.read_copy_rows_32);
    delta.read_copy_rows_gt32 = ggml_backend_fpga_counter_delta(current.read_copy_rows_gt32, baseline.read_copy_rows_gt32);
    delta.read_copy_cols_le16 = ggml_backend_fpga_counter_delta(current.read_copy_cols_le16, baseline.read_copy_cols_le16);
    delta.read_copy_cols_le32 = ggml_backend_fpga_counter_delta(current.read_copy_cols_le32, baseline.read_copy_cols_le32);
    delta.read_copy_cols_le64 = ggml_backend_fpga_counter_delta(current.read_copy_cols_le64, baseline.read_copy_cols_le64);
    delta.read_copy_cols_65_127 = ggml_backend_fpga_counter_delta(current.read_copy_cols_65_127, baseline.read_copy_cols_65_127);
    delta.read_copy_cols_128 = ggml_backend_fpga_counter_delta(current.read_copy_cols_128, baseline.read_copy_cols_128);
    delta.read_copy_cols_gt128 = ggml_backend_fpga_counter_delta(current.read_copy_cols_gt128, baseline.read_copy_cols_gt128);
    delta.total_prepare_us = ggml_backend_fpga_counter_delta(current.total_prepare_us, baseline.total_prepare_us);
    delta.total_pack_us = ggml_backend_fpga_counter_delta(current.total_pack_us, baseline.total_pack_us);
    delta.total_pack_a_us = ggml_backend_fpga_counter_delta(current.total_pack_a_us, baseline.total_pack_a_us);
    delta.total_pack_b_us = ggml_backend_fpga_counter_delta(current.total_pack_b_us, baseline.total_pack_b_us);
    delta.total_copy_us = ggml_backend_fpga_counter_delta(current.total_copy_us, baseline.total_copy_us);
    delta.total_copy_a_us = ggml_backend_fpga_counter_delta(current.total_copy_a_us, baseline.total_copy_a_us);
    delta.total_copy_b_us = ggml_backend_fpga_counter_delta(current.total_copy_b_us, baseline.total_copy_b_us);
    delta.total_control_us = ggml_backend_fpga_counter_delta(current.total_control_us, baseline.total_control_us);
    delta.total_preload_us = ggml_backend_fpga_counter_delta(current.total_preload_us, baseline.total_preload_us);
    delta.total_preload_sync_us = ggml_backend_fpga_counter_delta(current.total_preload_sync_us, baseline.total_preload_sync_us);
    delta.total_preload_wait_us = ggml_backend_fpga_counter_delta(current.total_preload_wait_us, baseline.total_preload_wait_us);
    delta.total_preload_setup_us = ggml_backend_fpga_counter_delta(current.total_preload_setup_us, baseline.total_preload_setup_us);
    delta.total_preload_xfer_us = ggml_backend_fpga_counter_delta(current.total_preload_xfer_us, baseline.total_preload_xfer_us);
    delta.total_start_us = ggml_backend_fpga_counter_delta(current.total_start_us, baseline.total_start_us);
    delta.total_start_sync_us = ggml_backend_fpga_counter_delta(current.total_start_sync_us, baseline.total_start_sync_us);
    delta.total_start_ready_us = ggml_backend_fpga_counter_delta(current.total_start_ready_us, baseline.total_start_ready_us);
    delta.total_start_setup_us = ggml_backend_fpga_counter_delta(current.total_start_setup_us, baseline.total_start_setup_us);
    delta.total_start_mmio_us = ggml_backend_fpga_counter_delta(current.total_start_mmio_us, baseline.total_start_mmio_us);
    delta.total_start_xfer_us = ggml_backend_fpga_counter_delta(current.total_start_xfer_us, baseline.total_start_xfer_us);
    delta.total_wait_prev_us = ggml_backend_fpga_counter_delta(current.total_wait_prev_us, baseline.total_wait_prev_us);
    delta.total_wait_final_us = ggml_backend_fpga_counter_delta(current.total_wait_final_us, baseline.total_wait_final_us);
    delta.total_cleanup_us = ggml_backend_fpga_counter_delta(current.total_cleanup_us, baseline.total_cleanup_us);
    delta.total_log_us = ggml_backend_fpga_counter_delta(current.total_log_us, baseline.total_log_us);
    delta.total_unlock_us = ggml_backend_fpga_counter_delta(current.total_unlock_us, baseline.total_unlock_us);
    delta.total_other_us = ggml_backend_fpga_counter_delta(current.total_other_us, baseline.total_other_us);
    delta.prevd_sequence_program_count = ggml_backend_fpga_counter_delta(current.prevd_sequence_program_count, baseline.prevd_sequence_program_count);
    delta.prevd_sequence_fallback_count = ggml_backend_fpga_counter_delta(current.prevd_sequence_fallback_count, baseline.prevd_sequence_fallback_count);
    delta.total_sequence_us = ggml_backend_fpga_counter_delta(current.total_sequence_us, baseline.total_sequence_us);
    delta.total_sequence_sync_us = ggml_backend_fpga_counter_delta(current.total_sequence_sync_us, baseline.total_sequence_sync_us);
    delta.total_sequence_ready_us = ggml_backend_fpga_counter_delta(current.total_sequence_ready_us, baseline.total_sequence_ready_us);
    delta.total_sequence_program_us = ggml_backend_fpga_counter_delta(current.total_sequence_program_us, baseline.total_sequence_program_us);
    delta.total_sequence_wait_done_us = ggml_backend_fpga_counter_delta(current.total_sequence_wait_done_us, baseline.total_sequence_wait_done_us);
    delta.auto_next_program_count = ggml_backend_fpga_counter_delta(current.auto_next_program_count, baseline.auto_next_program_count);
    delta.auto_next_fallback_count = ggml_backend_fpga_counter_delta(current.auto_next_fallback_count, baseline.auto_next_fallback_count);
    delta.matmul_task_count = ggml_backend_fpga_counter_delta(current.matmul_task_count, baseline.matmul_task_count);
    delta.matmul_task_tile_count = ggml_backend_fpga_counter_delta(current.matmul_task_tile_count, baseline.matmul_task_tile_count);
    delta.matmul_task_fallback_count = ggml_backend_fpga_counter_delta(current.matmul_task_fallback_count, baseline.matmul_task_fallback_count);
    delta.matmul_task_r5_cycles = ggml_backend_fpga_counter_delta(current.matmul_task_r5_cycles, baseline.matmul_task_r5_cycles);
    delta.matmul_task_fpga_wait_cycles = ggml_backend_fpga_counter_delta(current.matmul_task_fpga_wait_cycles, baseline.matmul_task_fpga_wait_cycles);
    delta.matmul_task_total_us = ggml_backend_fpga_counter_delta(current.matmul_task_total_us, baseline.matmul_task_total_us);
    delta.matmul_task_entry_setup_us = ggml_backend_fpga_counter_delta(current.matmul_task_entry_setup_us, baseline.matmul_task_entry_setup_us);
    delta.matmul_task_lock_us = ggml_backend_fpga_counter_delta(current.matmul_task_lock_us, baseline.matmul_task_lock_us);
    delta.matmul_task_reserve_us = ggml_backend_fpga_counter_delta(current.matmul_task_reserve_us, baseline.matmul_task_reserve_us);
    delta.matmul_task_prepare_b_wall_us = ggml_backend_fpga_counter_delta(current.matmul_task_prepare_b_wall_us, baseline.matmul_task_prepare_b_wall_us);
    delta.matmul_task_copy_a_wall_us = ggml_backend_fpga_counter_delta(current.matmul_task_copy_a_wall_us, baseline.matmul_task_copy_a_wall_us);
    delta.matmul_task_sync_a_us = ggml_backend_fpga_counter_delta(current.matmul_task_sync_a_us, baseline.matmul_task_sync_a_us);
    delta.matmul_task_addr_check_us = ggml_backend_fpga_counter_delta(current.matmul_task_addr_check_us, baseline.matmul_task_addr_check_us);
    delta.matmul_task_submit_us = ggml_backend_fpga_counter_delta(current.matmul_task_submit_us, baseline.matmul_task_submit_us);
    delta.matmul_task_output_sync_us = ggml_backend_fpga_counter_delta(current.matmul_task_output_sync_us, baseline.matmul_task_output_sync_us);
    delta.matmul_task_output_view_us = ggml_backend_fpga_counter_delta(current.matmul_task_output_view_us, baseline.matmul_task_output_view_us);
    delta.matmul_task_output_copy_us = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_us, baseline.matmul_task_output_copy_us);
    delta.matmul_task_output_copy_jobs = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_jobs, baseline.matmul_task_output_copy_jobs);
    delta.matmul_task_output_copy_bytes = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_bytes, baseline.matmul_task_output_copy_bytes);
    delta.matmul_task_output_copy_direct_jobs = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_direct_jobs, baseline.matmul_task_output_copy_direct_jobs);
    delta.matmul_task_output_copy_direct_us = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_direct_us, baseline.matmul_task_output_copy_direct_us);
    delta.matmul_task_output_copy_direct_bytes = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_direct_bytes, baseline.matmul_task_output_copy_direct_bytes);
    delta.matmul_task_output_copy_row_jobs = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_row_jobs, baseline.matmul_task_output_copy_row_jobs);
    delta.matmul_task_output_copy_row_us = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_row_us, baseline.matmul_task_output_copy_row_us);
    delta.matmul_task_output_copy_row_bytes = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_row_bytes, baseline.matmul_task_output_copy_row_bytes);
    delta.matmul_task_output_copy_row_rows = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_row_rows, baseline.matmul_task_output_copy_row_rows);
    delta.matmul_task_output_copy_padded_jobs = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_padded_jobs, baseline.matmul_task_output_copy_padded_jobs);
    delta.matmul_task_output_copy_padded_us = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_padded_us, baseline.matmul_task_output_copy_padded_us);
    delta.matmul_task_output_copy_m1_jobs = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_m1_jobs, baseline.matmul_task_output_copy_m1_jobs);
    delta.matmul_task_output_copy_m1_us = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_m1_us, baseline.matmul_task_output_copy_m1_us);
    delta.matmul_task_output_copy_m32_jobs = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_m32_jobs, baseline.matmul_task_output_copy_m32_jobs);
    delta.matmul_task_output_copy_m32_us = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_m32_us, baseline.matmul_task_output_copy_m32_us);
    delta.matmul_task_output_copy_n64_jobs = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_n64_jobs, baseline.matmul_task_output_copy_n64_jobs);
    delta.matmul_task_output_copy_n64_us = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_n64_us, baseline.matmul_task_output_copy_n64_us);
    delta.matmul_task_output_copy_n128_jobs = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_n128_jobs, baseline.matmul_task_output_copy_n128_jobs);
    delta.matmul_task_output_copy_n128_us = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_n128_us, baseline.matmul_task_output_copy_n128_us);
    delta.matmul_task_output_copy_n_other_jobs = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_n_other_jobs, baseline.matmul_task_output_copy_n_other_jobs);
    delta.matmul_task_output_copy_n_other_us = ggml_backend_fpga_counter_delta(current.matmul_task_output_copy_n_other_us, baseline.matmul_task_output_copy_n_other_us);
    delta.matmul_task_post_us = ggml_backend_fpga_counter_delta(current.matmul_task_post_us, baseline.matmul_task_post_us);
    delta.matmul_task_post_stats_us = ggml_backend_fpga_counter_delta(current.matmul_task_post_stats_us, baseline.matmul_task_post_stats_us);
    delta.matmul_task_free_us = ggml_backend_fpga_counter_delta(current.matmul_task_free_us, baseline.matmul_task_free_us);
    delta.matmul_task_residual_us = ggml_backend_fpga_counter_delta(current.matmul_task_residual_us, baseline.matmul_task_residual_us);
    delta.pl_op_desc_count = ggml_backend_fpga_counter_delta(current.pl_op_desc_count, baseline.pl_op_desc_count);
    delta.pl_op_desc_tile_count = ggml_backend_fpga_counter_delta(current.pl_op_desc_tile_count, baseline.pl_op_desc_tile_count);
    delta.pl_op_desc_program_us = ggml_backend_fpga_counter_delta(current.pl_op_desc_program_us, baseline.pl_op_desc_program_us);
    delta.pl_op_desc_wait_us = ggml_backend_fpga_counter_delta(current.pl_op_desc_wait_us, baseline.pl_op_desc_wait_us);
    delta.pl_op_desc_status_polls = ggml_backend_fpga_counter_delta(current.pl_op_desc_status_polls, baseline.pl_op_desc_status_polls);
    delta.b_packed_arena_hits = ggml_backend_fpga_counter_delta(current.b_packed_arena_hits, baseline.b_packed_arena_hits);
    delta.b_packed_arena_misses = ggml_backend_fpga_counter_delta(current.b_packed_arena_misses, baseline.b_packed_arena_misses);
    return delta;
}

static void ggml_backend_fpga_runtime_stats_add(
        tpu_runtime_stats_t & total,
        const tpu_runtime_stats_t & delta) {
    total.tile_call_count += delta.tile_call_count;
    total.prevd_chain_runs += delta.prevd_chain_runs;
    total.prevd_over_signoff_runs += delta.prevd_over_signoff_runs;
    total.prevd_ab_preload_count += delta.prevd_ab_preload_count;
    total.prevd_host_c_upload_count += delta.prevd_host_c_upload_count;
    total.prevd_host_d_readback_count += delta.prevd_host_d_readback_count;
    total.prevd_direct_final_readback_count += delta.prevd_direct_final_readback_count;
    total.total_wait_done_us += delta.total_wait_done_us;
    total.total_read_us += delta.total_read_us;
    total.total_read_sync_us += delta.total_read_sync_us;
    total.total_read_copy_us += delta.total_read_copy_us;
    total.total_read_unpack_us += delta.total_read_unpack_us;
    total.total_read_mmio_us += delta.total_read_mmio_us;
    total.total_read_debug_dump_us += delta.total_read_debug_dump_us;
    total.total_read_bytes += delta.total_read_bytes;
    total.total_read_output_buffer_count += delta.total_read_output_buffer_count;
    total.total_read_memcpy_count += delta.total_read_memcpy_count;
    total.total_read_unpack_count += delta.total_read_unpack_count;
    total.total_read_mmio_count += delta.total_read_mmio_count;
    total.read_op_batch_count += delta.read_op_batch_count;
    total.read_op_batch_total_us += delta.read_op_batch_total_us;
    total.read_op_batch_sync_us += delta.read_op_batch_sync_us;
    total.read_op_batch_copy_us += delta.read_op_batch_copy_us;
    total.read_op_batch_bytes += delta.read_op_batch_bytes;
    total.read_async_count += delta.read_async_count;
    total.read_async_total_us += delta.read_async_total_us;
    total.read_async_sync_us += delta.read_async_sync_us;
    total.read_async_copy_us += delta.read_async_copy_us;
    total.read_async_bytes += delta.read_async_bytes;
    total.read_async_batch_begin_wait_us += delta.read_async_batch_begin_wait_us;
    total.read_async_batch_begin_wait_count += delta.read_async_batch_begin_wait_count;
    total.read_async_stage_wait_us += delta.read_async_stage_wait_us;
    total.read_async_stage_wait_count += delta.read_async_stage_wait_count;
    total.read_async_flush_wait_us += delta.read_async_flush_wait_us;
    total.read_async_flush_wait_count += delta.read_async_flush_wait_count;
    total.read_final_desc_count += delta.read_final_desc_count;
    total.read_final_desc_total_us += delta.read_final_desc_total_us;
    total.read_final_desc_sync_us += delta.read_final_desc_sync_us;
    total.read_final_desc_copy_us += delta.read_final_desc_copy_us;
    total.read_final_desc_unpack_us += delta.read_final_desc_unpack_us;
    total.read_final_desc_mmio_us += delta.read_final_desc_mmio_us;
    total.read_final_desc_bytes += delta.read_final_desc_bytes;
    total.read_copy_job_count += delta.read_copy_job_count;
    total.read_copy_row_count += delta.read_copy_row_count;
    total.read_copy_elem_count += delta.read_copy_elem_count;
    total.read_copy_useful_bytes += delta.read_copy_useful_bytes;
    total.read_copy_transfer_bytes += delta.read_copy_transfer_bytes;
    total.read_copy_padded_jobs += delta.read_copy_padded_jobs;
    total.read_copy_strided_jobs += delta.read_copy_strided_jobs;
    total.read_copy_bulk_candidate_jobs += delta.read_copy_bulk_candidate_jobs;
    total.read_copy_bulk_candidate_bytes += delta.read_copy_bulk_candidate_bytes;
    total.read_copy_bulk_memcpy_jobs += delta.read_copy_bulk_memcpy_jobs;
    total.read_copy_bulk_memcpy_bytes += delta.read_copy_bulk_memcpy_bytes;
    total.read_copy_row_memcpy_rows += delta.read_copy_row_memcpy_rows;
    total.read_copy_rows_1 += delta.read_copy_rows_1;
    total.read_copy_rows_2_15 += delta.read_copy_rows_2_15;
    total.read_copy_rows_16 += delta.read_copy_rows_16;
    total.read_copy_rows_17_31 += delta.read_copy_rows_17_31;
    total.read_copy_rows_32 += delta.read_copy_rows_32;
    total.read_copy_rows_gt32 += delta.read_copy_rows_gt32;
    total.read_copy_cols_le16 += delta.read_copy_cols_le16;
    total.read_copy_cols_le32 += delta.read_copy_cols_le32;
    total.read_copy_cols_le64 += delta.read_copy_cols_le64;
    total.read_copy_cols_65_127 += delta.read_copy_cols_65_127;
    total.read_copy_cols_128 += delta.read_copy_cols_128;
    total.read_copy_cols_gt128 += delta.read_copy_cols_gt128;
    total.total_prepare_us += delta.total_prepare_us;
    total.total_pack_us += delta.total_pack_us;
    total.total_pack_a_us += delta.total_pack_a_us;
    total.total_pack_b_us += delta.total_pack_b_us;
    total.total_copy_us += delta.total_copy_us;
    total.total_copy_a_us += delta.total_copy_a_us;
    total.total_copy_b_us += delta.total_copy_b_us;
    total.total_control_us += delta.total_control_us;
    total.total_preload_us += delta.total_preload_us;
    total.total_preload_sync_us += delta.total_preload_sync_us;
    total.total_preload_wait_us += delta.total_preload_wait_us;
    total.total_preload_setup_us += delta.total_preload_setup_us;
    total.total_preload_xfer_us += delta.total_preload_xfer_us;
    total.total_start_us += delta.total_start_us;
    total.total_start_sync_us += delta.total_start_sync_us;
    total.total_start_ready_us += delta.total_start_ready_us;
    total.total_start_setup_us += delta.total_start_setup_us;
    total.total_start_mmio_us += delta.total_start_mmio_us;
    total.total_start_xfer_us += delta.total_start_xfer_us;
    total.total_wait_prev_us += delta.total_wait_prev_us;
    total.total_wait_final_us += delta.total_wait_final_us;
    total.total_cleanup_us += delta.total_cleanup_us;
    total.total_log_us += delta.total_log_us;
    total.total_unlock_us += delta.total_unlock_us;
    total.total_other_us += delta.total_other_us;
    total.prevd_sequence_program_count += delta.prevd_sequence_program_count;
    total.prevd_sequence_fallback_count += delta.prevd_sequence_fallback_count;
    total.total_sequence_us += delta.total_sequence_us;
    total.total_sequence_sync_us += delta.total_sequence_sync_us;
    total.total_sequence_ready_us += delta.total_sequence_ready_us;
    total.total_sequence_program_us += delta.total_sequence_program_us;
    total.total_sequence_wait_done_us += delta.total_sequence_wait_done_us;
    total.auto_next_program_count += delta.auto_next_program_count;
    total.auto_next_fallback_count += delta.auto_next_fallback_count;
    total.matmul_task_count += delta.matmul_task_count;
    total.matmul_task_tile_count += delta.matmul_task_tile_count;
    total.matmul_task_fallback_count += delta.matmul_task_fallback_count;
    total.matmul_task_r5_cycles += delta.matmul_task_r5_cycles;
    total.matmul_task_fpga_wait_cycles += delta.matmul_task_fpga_wait_cycles;
    total.matmul_task_total_us += delta.matmul_task_total_us;
    total.matmul_task_entry_setup_us += delta.matmul_task_entry_setup_us;
    total.matmul_task_lock_us += delta.matmul_task_lock_us;
    total.matmul_task_reserve_us += delta.matmul_task_reserve_us;
    total.matmul_task_prepare_b_wall_us += delta.matmul_task_prepare_b_wall_us;
    total.matmul_task_copy_a_wall_us += delta.matmul_task_copy_a_wall_us;
    total.matmul_task_sync_a_us += delta.matmul_task_sync_a_us;
    total.matmul_task_addr_check_us += delta.matmul_task_addr_check_us;
    total.matmul_task_submit_us += delta.matmul_task_submit_us;
    total.matmul_task_output_sync_us += delta.matmul_task_output_sync_us;
    total.matmul_task_output_view_us += delta.matmul_task_output_view_us;
    total.matmul_task_output_copy_us += delta.matmul_task_output_copy_us;
    total.matmul_task_output_copy_jobs += delta.matmul_task_output_copy_jobs;
    total.matmul_task_output_copy_bytes += delta.matmul_task_output_copy_bytes;
    total.matmul_task_output_copy_direct_jobs += delta.matmul_task_output_copy_direct_jobs;
    total.matmul_task_output_copy_direct_us += delta.matmul_task_output_copy_direct_us;
    total.matmul_task_output_copy_direct_bytes += delta.matmul_task_output_copy_direct_bytes;
    total.matmul_task_output_copy_row_jobs += delta.matmul_task_output_copy_row_jobs;
    total.matmul_task_output_copy_row_us += delta.matmul_task_output_copy_row_us;
    total.matmul_task_output_copy_row_bytes += delta.matmul_task_output_copy_row_bytes;
    total.matmul_task_output_copy_row_rows += delta.matmul_task_output_copy_row_rows;
    total.matmul_task_output_copy_padded_jobs += delta.matmul_task_output_copy_padded_jobs;
    total.matmul_task_output_copy_padded_us += delta.matmul_task_output_copy_padded_us;
    total.matmul_task_output_copy_m1_jobs += delta.matmul_task_output_copy_m1_jobs;
    total.matmul_task_output_copy_m1_us += delta.matmul_task_output_copy_m1_us;
    total.matmul_task_output_copy_m32_jobs += delta.matmul_task_output_copy_m32_jobs;
    total.matmul_task_output_copy_m32_us += delta.matmul_task_output_copy_m32_us;
    total.matmul_task_output_copy_n64_jobs += delta.matmul_task_output_copy_n64_jobs;
    total.matmul_task_output_copy_n64_us += delta.matmul_task_output_copy_n64_us;
    total.matmul_task_output_copy_n128_jobs += delta.matmul_task_output_copy_n128_jobs;
    total.matmul_task_output_copy_n128_us += delta.matmul_task_output_copy_n128_us;
    total.matmul_task_output_copy_n_other_jobs += delta.matmul_task_output_copy_n_other_jobs;
    total.matmul_task_output_copy_n_other_us += delta.matmul_task_output_copy_n_other_us;
    total.matmul_task_post_us += delta.matmul_task_post_us;
    total.matmul_task_post_stats_us += delta.matmul_task_post_stats_us;
    total.matmul_task_free_us += delta.matmul_task_free_us;
    total.matmul_task_residual_us += delta.matmul_task_residual_us;
    total.pl_op_desc_count += delta.pl_op_desc_count;
    total.pl_op_desc_tile_count += delta.pl_op_desc_tile_count;
    total.pl_op_desc_program_us += delta.pl_op_desc_program_us;
    total.pl_op_desc_wait_us += delta.pl_op_desc_wait_us;
    total.pl_op_desc_status_polls += delta.pl_op_desc_status_polls;
    total.b_packed_arena_hits += delta.b_packed_arena_hits;
    total.b_packed_arena_misses += delta.b_packed_arena_misses;
}

static const char * ggml_backend_fpga_log_prefix() {
#ifdef GGML_FPGA_USE_MOCK
    return "[FPGA Mock]";
#else
    return "[FPGA]";
#endif
}

static int64_t ggml_backend_fpga_ns_delta_nonnegative(int64_t after, int64_t before) {
    return after >= before ? after - before : 0;
}

static ggml_backend_fpga_rejection_stats & ggml_backend_fpga_get_rejection_stats() {
    static ggml_backend_fpga_rejection_stats stats;
    return stats;
}

static bool ggml_backend_fpga_is_contiguous_2d(const ggml_tensor * t) {
    return ggml_is_contiguous_2(t) && t->nb[0] == ggml_type_size(t->type);
}

static bool ggml_backend_fpga_is_dense_2d_plane(const ggml_tensor * t) {
    return t->nb[0] == ggml_type_size(t->type) &&
           t->nb[1] == ggml_row_size(t->type, t->ne[0]);
}

static bool ggml_backend_fpga_is_row_dense_2d_plane(const ggml_tensor * t) {
    return t->nb[0] == ggml_type_size(t->type) &&
           t->nb[1] >= ggml_row_size(t->type, t->ne[0]);
}

static bool ggml_backend_fpga_is_vit_attn_strided_target(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    return src0->type == GGML_TYPE_F16 &&
           src1->type == GGML_TYPE_F32 &&
           op->type == GGML_TYPE_F32 &&
           src0->ne[0] == 64 &&
           src0->ne[1] == 1024 &&
           src1->ne[1] == 1024 &&
           src0->ne[2] == 12 &&
           src0->ne[3] == 1 &&
           src1->ne[2] == 12 &&
           src1->ne[3] == 1 &&
           ggml_backend_fpga_is_dense_2d_plane(src0) &&
           ggml_backend_fpga_is_row_dense_2d_plane(src1);
}

static bool ggml_backend_fpga_supports_src0_type(enum ggml_type type) {
    return type == GGML_TYPE_F32 || ggml_get_type_traits(type)->to_float != NULL;
}

static bool ggml_backend_fpga_can_cache_src0(const ggml_tensor * src0) {
    return src0->op == GGML_OP_NONE &&
           src0->view_src == nullptr &&
           !(src0->flags & GGML_TENSOR_FLAG_INPUT);
}

static const char * ggml_backend_fpga_tensor_name_or_anon(const ggml_tensor * t) {
    return t != nullptr && t->name[0] != '\0' ? t->name : "<anon>";
}

static void ggml_backend_fpga_record_dynamic_src0(
        ggml_backend_fpga_context * ctx,
        const ggml_tensor * src0,
        int64_t m,
        int64_t n,
        int64_t k) {
    if (ctx == nullptr || src0 == nullptr) {
        return;
    }
    ggml_backend_fpga_dynamic_src0_stats & stats = ctx->dynamic_src0_stats;
    stats.total++;
    if (src0->op != GGML_OP_NONE) {
        stats.op_non_none++;
        stats.op_counts[ggml_op_name(src0->op)]++;
    }
    if (src0->view_src != nullptr) {
        stats.view_src++;
    }
    if (src0->flags & GGML_TENSOR_FLAG_INPUT) {
        stats.input_flag++;
    }
    stats.shape_counts[std::make_tuple(m, n, k)]++;
    stats.name_counts[ggml_backend_fpga_tensor_name_or_anon(src0)]++;
}

static bool ggml_backend_fpga_supports_mul_mat(const ggml_tensor * op, std::string * reason) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    bool supported = true;
    const bool allow_strided = ggml_backend_fpga_strided_mul_mat_enabled();
    const bool allow_vit_attn_strided =
        ggml_backend_fpga_vit_attn_strided_mul_mat_enabled() &&
        ggml_backend_fpga_is_vit_attn_strided_target(op);
    const bool allow_vit_attn_pl_op =
        ggml_backend_fpga_vit_attn_pl_op_enabled() &&
        ggml_backend_fpga_is_vit_attn_strided_target(op) &&
        ggml_backend_fpga_pl_op_descriptor_m_allowed(src1->ne[1]) &&
        ggml_backend_fpga_matrix_task_nk_shape_allowed(src0->ne[1], src0->ne[0]);
    const bool allow_src1_strided =
        allow_strided ||
        ggml_backend_fpga_src1_strided_mul_mat_enabled() ||
        allow_vit_attn_strided ||
        allow_vit_attn_pl_op;

    auto add_reason = [&](const char * text) {
        supported = false;
        if (reason == nullptr) {
            return;
        }
        if (!reason->empty()) {
            reason->append(",");
        }
        reason->append(text);
    };

    if (!ggml_backend_fpga_supports_src0_type(src0->type)) {
        add_reason("src0_type_no_to_float");
    }
    if (op->type != GGML_TYPE_F32) {
        add_reason("dst_not_f32");
    }
    if (src1->type != GGML_TYPE_F32) {
        add_reason("src1_not_f32");
    }
    const bool src0_dense = ggml_backend_fpga_is_contiguous_2d(src0);
    const bool src1_dense = ggml_backend_fpga_is_contiguous_2d(src1);
    const bool src0_row_dense = allow_strided && ggml_backend_fpga_is_row_dense_2d_plane(src0);
    const bool src1_row_dense = allow_src1_strided && ggml_backend_fpga_is_row_dense_2d_plane(src1);

    if (!src0_dense && !src0_row_dense) {
        add_reason("src0_not_contiguous_2d");
    }
    if (!src1_dense && !src1_row_dense) {
        add_reason("src1_not_contiguous_2d");
    }
    if (!ggml_backend_fpga_is_dense_2d_plane(src0) && !src0_row_dense) {
        add_reason("src0_not_dense_2d_plane");
    }
    if (!ggml_backend_fpga_is_dense_2d_plane(src1) && !src1_row_dense) {
        add_reason("src1_not_dense_2d_plane");
    }
    if (src0->ne[2] != 0 && src1->ne[2] % src0->ne[2] != 0) {
        add_reason("batch2_not_divisible");
    }
    if (src0->ne[3] != 0 && src1->ne[3] % src0->ne[3] != 0) {
        add_reason("batch3_not_divisible");
    }
    if (src0->ne[0] < 64) {
        add_reason("k_lt_64");
    }
    if (src0->ne[1] < 64) {
        add_reason("n_lt_64");
    }

    return supported;
}

static bool ggml_backend_fpga_supports_add(const ggml_tensor * op) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];
    if (op->type != GGML_TYPE_F32 ||
            src0->type != GGML_TYPE_F32 ||
            src1->type != GGML_TYPE_F32 ||
            !ggml_can_repeat(src1, src0) ||
            !ggml_is_contiguous(src0) ||
            !ggml_is_contiguous(op) ||
            !ggml_is_contiguous(src1)) {
        return false;
    }

    const bool same_shape =
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        src1->ne[2] == src0->ne[2] &&
        src1->ne[3] == src0->ne[3];
    const bool bias_1d =
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == 1 &&
        src1->ne[2] == 1 &&
        src1->ne[3] == 1;
    const bool scalar = ggml_nelements(src1) == 1;

    if (!ggml_backend_fpga_add_op_enabled()) {
        return false;
    }

    return same_shape || bias_1d || scalar;
}

static void ggml_backend_fpga_record_rejected_mul_mat(const ggml_tensor * op, const std::string & reason) {
    const ggml_tensor * src0 = op->src[0];
    const ggml_tensor * src1 = op->src[1];

    std::string key = std::string("reason=") + reason +
            " | types(src0/src1/dst)=" +
            ggml_type_name(src0->type) + "/" +
            ggml_type_name(src1->type) + "/" +
            ggml_type_name(op->type) +
            " | M=" + std::to_string(src1->ne[1]) +
            " N=" + std::to_string(src0->ne[1]) +
            " K=" + std::to_string(src0->ne[0]) +
            " | batch(src0)=" + std::to_string(src0->ne[2]) + "x" + std::to_string(src0->ne[3]) +
            " batch(src1)=" + std::to_string(src1->ne[2]) + "x" + std::to_string(src1->ne[3]);

    auto & stats = ggml_backend_fpga_get_rejection_stats();
    std::lock_guard<std::mutex> lock(stats.mutex);
    stats.total_rejected_mul_mat++;
    stats.rejected_mul_mat[key]++;
}

static void ggml_backend_fpga_record_shape(
        ggml_backend_fpga_context * ctx,
        int64_t m,
        int64_t n,
        int64_t k,
        int64_t instances) {
    ctx->n_gemm_instances += instances;
    ctx->m_value_counts[m] += instances;
    ctx->n_value_counts[n] += instances;
    ctx->k_value_counts[k] += instances;
    ctx->gemm_shape_counts[std::make_tuple(m, n, k)] += instances;
}

static void ggml_backend_fpga_record_empty_shape(
        ggml_backend_fpga_context * ctx,
        int64_t m,
        int64_t n,
        int64_t k,
        int64_t instances) {
    ctx->n_empty_mul_mat_skipped++;
    ctx->n_empty_gemm_instances_skipped += instances;
    ctx->empty_gemm_shape_counts[std::make_tuple(m, n, k)] += instances;
}

static void ggml_backend_fpga_log_value_counts(const char * label, const std::map<int64_t, int64_t> & counts) {
    GGML_LOG_INFO("  %s values: %zu unique\n", label, counts.size());
    for (const auto & entry : counts) {
        GGML_LOG_INFO("    %s=%" PRId64 " [count=%" PRId64 "]\n", label, entry.first, entry.second);
    }
}

static void ggml_backend_fpga_log_string_counts(const char * label, const std::map<std::string, int64_t> & counts) {
    GGML_LOG_INFO("  %s: %zu unique\n", label, counts.size());
    for (const auto & entry : counts) {
        GGML_LOG_INFO("    [count=%" PRId64 "] %s\n", entry.second, entry.first.c_str());
    }
}

static void ggml_backend_fpga_log_shape_counts(
        const char * label,
        const std::map<std::tuple<int64_t, int64_t, int64_t>, int64_t> & counts) {
    GGML_LOG_INFO("  %s: %zu unique\n", label, counts.size());
    for (const auto & entry : counts) {
        const auto & shape = entry.first;
        GGML_LOG_INFO("    [count=%" PRId64 "] M=%" PRId64 " N=%" PRId64 " K=%" PRId64 "\n",
                entry.second,
                std::get<0>(shape),
                std::get<1>(shape),
                std::get<2>(shape));
    }
}

static std::string ggml_backend_fpga_tensor_name_bucket(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->name[0] == '\0') {
        return "unnamed";
    }
    if (std::strncmp(tensor->name, "Kcur_", 5) == 0) {
        return "Kcur";
    }
    if (std::strncmp(tensor->name, "Vcur_", 5) == 0) {
        return "Vcur";
    }
    std::string name(tensor->name);
    if (name.size() > 48) {
        name.resize(48);
    }
    return name;
}

static void ggml_backend_fpga_record_matrix_task_shape(
        ggml_backend_fpga_context * ctx,
        const ggml_tensor * src0,
        int64_t m,
        int64_t n,
        int64_t k,
        int64_t instances,
        bool cacheable_src0,
        bool dynamic_multi_instance,
        bool persistent_owner,
        bool have_prepared_activation,
        bool direct_view,
        bool try_vit_attn_pl_op,
        bool used_r5_matrix_task,
        const tpu_runtime_stats_t & delta) {
    if (!ggml_backend_fpga_matrix_task_shape_stats_enabled()) {
        return;
    }

    std::string key =
        "M=" + std::to_string(m) +
        " N=" + std::to_string(n) +
        " K=" + std::to_string(k) +
        " inst=" + std::to_string(instances) +
        " src0=" + ggml_backend_fpga_tensor_name_bucket(src0) +
        " op=" + (src0 != nullptr ? ggml_op_name(src0->op) : "NULL") +
        " cacheable=" + std::to_string(cacheable_src0 ? 1 : 0) +
        " dyn_multi=" + std::to_string(dynamic_multi_instance ? 1 : 0) +
        " owner=" + std::to_string(persistent_owner ? 1 : 0) +
        " prepared_a=" + std::to_string(have_prepared_activation ? 1 : 0) +
        " direct_view=" + std::to_string(direct_view ? 1 : 0) +
        " vit_pl_op=" + std::to_string(try_vit_attn_pl_op ? 1 : 0);

    ggml_backend_fpga_matrix_task_shape_stats & stats = ctx->matrix_task_shape_stats[key];
    stats.attempts++;
    if (used_r5_matrix_task) {
        stats.used++;
    } else {
        stats.fallback++;
    }
    stats.task_count += delta.matmul_task_count;
    stats.tile_count += delta.matmul_task_tile_count;
    stats.b_hits += delta.b_packed_arena_hits;
    stats.b_misses += delta.b_packed_arena_misses;
    stats.prepare_b_us += delta.matmul_task_prepare_b_wall_us;
    stats.submit_us += delta.matmul_task_submit_us;
    stats.output_sync_us += delta.matmul_task_output_sync_us;
    stats.total_us += delta.matmul_task_total_us;
}

static void ggml_backend_fpga_log_matrix_task_shape_stats(
        const std::map<std::string, ggml_backend_fpga_matrix_task_shape_stats> & counts) {
    GGML_LOG_INFO("  matrix_task shape stats: %zu unique\n", counts.size());
    for (const auto & entry : counts) {
        const ggml_backend_fpga_matrix_task_shape_stats & stats = entry.second;
        GGML_LOG_INFO("    attempts=%" PRId64 " used=%" PRId64 " fallback=%" PRId64
                " tasks=%" PRIu64 " tiles=%" PRIu64
                " b_hit=%" PRIu64 " b_miss=%" PRIu64
                " prepare_b=%.3f ms submit=%.3f ms output_sync=%.3f ms total=%.3f ms | %s\n",
                stats.attempts,
                stats.used,
                stats.fallback,
                stats.task_count,
                stats.tile_count,
                stats.b_hits,
                stats.b_misses,
                stats.prepare_b_us / 1000.0,
                stats.submit_us / 1000.0,
                stats.output_sync_us / 1000.0,
                stats.total_us / 1000.0,
                entry.first.c_str());
    }
}

static void ggml_backend_fpga_accumulate_tiling_stats(
        ggml_backend_fpga_context * ctx,
        const tpu_tiling_stats_t & stats) {
    ctx->tiling_total_hw_calls += stats.total_hw_calls;
    ctx->tiling_calls_32x128x128 += stats.calls_32x128x128;
    ctx->tiling_calls_16x128x128 += stats.calls_16x128x128;
    ctx->tiling_calls_1x128x128 += stats.calls_1x128x128;
    ctx->tiling_calls_32x64x64 += stats.calls_32x64x64;
    ctx->tiling_calls_16x64x64 += stats.calls_16x64x64;
    ctx->tiling_calls_1x64x64 += stats.calls_1x64x64;
    ctx->tiling_calls_1x32x32 += stats.calls_1x32x32;
    ctx->tiling_calls_1x16x32 += stats.calls_1x16x32;
    ctx->tiling_calls_32x32x32 += stats.calls_32x32x32;
    ctx->tiling_calls_32x32x16 += stats.calls_32x32x16;
    ctx->tiling_calls_16x32x32 += stats.calls_16x32x32;
    ctx->tiling_calls_32x16x16 += stats.calls_32x16x16;
    ctx->tiling_calls_16x16x16 += stats.calls_16x16x16;
    ctx->tiling_total_padded_elements += stats.total_padded_elements;
    ctx->tiling_total_useful_elements += stats.total_useful_elements;
    ctx->tiling_zero_ns += stats.total_zero_ns;
    ctx->tiling_copy_a_ns += stats.total_copy_a_ns;
    ctx->tiling_copy_b_ns += stats.total_copy_b_ns;
    ctx->tiling_copy_c_ns += stats.total_copy_c_ns;
    ctx->tiling_tile_call_ns += stats.total_tile_call_ns;
    ctx->tiling_copy_out_partial_ns += stats.total_copy_out_partial_ns;
    ctx->tiling_copy_out_final_ns += stats.total_copy_out_final_ns;
}

static int8_t ggml_backend_fpga_quantize_value(float value, float inv_scale) {
    const float scaled = std::round(value * inv_scale);
    const float clamped = std::max(-127.0f, std::min(127.0f, scaled));
    return (int8_t) clamped;
}

static float ggml_backend_fpga_row_max_abs_scalar(const float * src, int64_t cols) {
    float max_abs = 0.0f;
    for (int64_t col = 0; col < cols; ++col) {
        const float abs_value = std::fabs(src[col]);
        if (abs_value > max_abs) {
            max_abs = abs_value;
        }
    }
    return max_abs;
}

#if GGML_FPGA_HAVE_ACT_QUANT_NEON
static float ggml_backend_fpga_row_max_abs_neon(const float * src, int64_t cols) {
    float32x4_t max_v = vdupq_n_f32(0.0f);
    int64_t col = 0;

    for (; col + 4 <= cols; col += 4) {
        const float32x4_t abs_v = vabsq_f32(vld1q_f32(src + col));
        max_v = vmaxnmq_f32(max_v, abs_v);
    }

    float lanes[4];
    vst1q_f32(lanes, max_v);
    float max_abs = 0.0f;
    for (int lane = 0; lane < 4; ++lane) {
        if (lanes[lane] > max_abs) {
            max_abs = lanes[lane];
        }
    }
    for (; col < cols; ++col) {
        const float abs_value = std::fabs(src[col]);
        if (abs_value > max_abs) {
            max_abs = abs_value;
        }
    }
    return max_abs;
}
#endif

static float ggml_backend_fpga_row_max_abs(
        const float * src,
        int64_t cols,
        bool use_neon) {
#if GGML_FPGA_HAVE_ACT_QUANT_NEON
    if (use_neon) {
        return ggml_backend_fpga_row_max_abs_neon(src, cols);
    }
#else
    (void) use_neon;
#endif
    return ggml_backend_fpga_row_max_abs_scalar(src, cols);
}

static void ggml_backend_fpga_quantize_row_values_scalar(
        const float * src,
        int64_t cols,
        float inv_scale,
        int8_t * dst) {
    for (int64_t col = 0; col < cols; ++col) {
        dst[col] = ggml_backend_fpga_quantize_value(src[col], inv_scale);
    }
}

#if GGML_FPGA_HAVE_ACT_QUANT_NEON
static void ggml_backend_fpga_quantize_row_values_neon(
        const float * src,
        int64_t cols,
        float inv_scale,
        int8_t * dst) {
    const float32x4_t scale_v = vdupq_n_f32(inv_scale);
    const int32x4_t min_v = vdupq_n_s32(-127);
    const int32x4_t max_v = vdupq_n_s32(127);
    int64_t col = 0;

    for (; col + 8 <= cols; col += 8) {
        float32x4_t f0 = vmulq_f32(vld1q_f32(src + col), scale_v);
        float32x4_t f1 = vmulq_f32(vld1q_f32(src + col + 4), scale_v);
        int32x4_t q0 = vcvtaq_s32_f32(f0);
        int32x4_t q1 = vcvtaq_s32_f32(f1);
        q0 = vmaxq_s32(min_v, vminq_s32(max_v, q0));
        q1 = vmaxq_s32(min_v, vminq_s32(max_v, q1));

        const int16x8_t q16 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        const int8x8_t q8 = vqmovn_s16(q16);
        vst1_s8(dst + col, q8);
    }

    ggml_backend_fpga_quantize_row_values_scalar(src + col, cols - col, inv_scale, dst + col);
}
#endif

static void ggml_backend_fpga_quantize_row_values(
        const float * src,
        int64_t cols,
        float inv_scale,
        bool use_neon,
        int8_t * dst) {
#if GGML_FPGA_HAVE_ACT_QUANT_NEON
    if (use_neon) {
        ggml_backend_fpga_quantize_row_values_neon(src, cols, inv_scale, dst);
        return;
    }
#else
    (void) use_neon;
#endif
    ggml_backend_fpga_quantize_row_values_scalar(src, cols, inv_scale, dst);
}

static void ggml_backend_fpga_quantize_row_scales(
        const float * src,
        int64_t rows,
        int64_t cols,
        int64_t row_stride,
        std::vector<fpga_quant_params> & qp,
        std::vector<float> & inv_scales) {
    qp.resize((size_t) rows);
    inv_scales.resize((size_t) rows);
    const bool use_neon = ggml_backend_fpga_act_quant_neon_enabled();

    for (int64_t row = 0; row < rows; ++row) {
        const float * row_src = src + row*row_stride;
        const float max_abs = ggml_backend_fpga_row_max_abs(row_src, cols, use_neon);
        qp[(size_t) row].scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        inv_scales[(size_t) row] = max_abs > 0.0f ? 127.0f / max_abs : 0.0f;
    }
}

static void ggml_backend_fpga_quantize_row_scales(
        const float * src,
        int64_t rows,
        int64_t cols,
        std::vector<fpga_quant_params> & qp,
        std::vector<float> & inv_scales) {
    ggml_backend_fpga_quantize_row_scales(src, rows, cols, cols, qp, inv_scales);
}

static void ggml_backend_fpga_quantize_rows_with_scales(
        const float * src,
        int64_t rows,
        int64_t cols,
        int64_t row_stride,
        const std::vector<float> & inv_scales,
        std::vector<int8_t> & dst) {
    dst.resize((size_t) (rows * cols));
    const bool use_neon = ggml_backend_fpga_act_quant_neon_enabled();

    for (int64_t row = 0; row < rows; ++row) {
        const float * row_src = src + row*row_stride;
        int8_t * row_dst = dst.data() + row*cols;
        const float inv_scale = inv_scales[(size_t) row];
        if (inv_scale == 0.0f) {
            std::memset(row_dst, 0, (size_t) cols);
            continue;
        }

        ggml_backend_fpga_quantize_row_values(row_src, cols, inv_scale, use_neon, row_dst);
    }
}

static void ggml_backend_fpga_quantize_rows_with_scales(
        const float * src,
        int64_t rows,
        int64_t cols,
        const std::vector<float> & inv_scales,
        std::vector<int8_t> & dst) {
    ggml_backend_fpga_quantize_rows_with_scales(src, rows, cols, cols, inv_scales, dst);
}

static void ggml_backend_fpga_quantize_rows_and_scales(
        const float * src,
        int64_t rows,
        int64_t cols,
        int64_t row_stride,
        std::vector<fpga_quant_params> & qp,
        std::vector<float> & inv_scales,
        std::vector<int8_t> & dst) {
    qp.resize((size_t) rows);
    inv_scales.resize((size_t) rows);
    dst.resize((size_t) (rows * cols));
    const bool use_neon = ggml_backend_fpga_act_quant_neon_enabled();

    const int64_t requested_threads = ggml_backend_fpga_act_quant_parallel_threads();
    const int64_t total_elems = rows * cols;
    if (ggml_backend_fpga_act_quant_parallel_enabled() &&
            requested_threads > 1 &&
            rows > 1 &&
            total_elems >= ggml_backend_fpga_act_quant_parallel_min_elems()) {
        const int64_t worker_count = std::min<int64_t>(requested_threads, rows);
        std::vector<std::thread> workers;
        workers.reserve((size_t) worker_count);

        for (int64_t worker = 0; worker < worker_count; ++worker) {
            const int64_t row_begin = (rows * worker) / worker_count;
            const int64_t row_end = (rows * (worker + 1)) / worker_count;

            workers.emplace_back([=, &qp, &inv_scales, &dst]() {
                for (int64_t row = row_begin; row < row_end; ++row) {
                    const float * row_src = src + row*row_stride;
                    int8_t * row_dst = dst.data() + row*cols;
                    const float max_abs = ggml_backend_fpga_row_max_abs(row_src, cols, use_neon);

                    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
                    const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 0.0f;
                    qp[(size_t) row].scale = scale;
                    inv_scales[(size_t) row] = inv_scale;

                    if (inv_scale == 0.0f) {
                        std::memset(row_dst, 0, (size_t) cols);
                        continue;
                    }

                    ggml_backend_fpga_quantize_row_values(row_src, cols, inv_scale, use_neon, row_dst);
                }
            });
        }

        for (std::thread & worker : workers) {
            worker.join();
        }
        return;
    }

    for (int64_t row = 0; row < rows; ++row) {
        const float * row_src = src + row*row_stride;
        int8_t * row_dst = dst.data() + row*cols;
        const float max_abs = ggml_backend_fpga_row_max_abs(row_src, cols, use_neon);

        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 0.0f;
        qp[(size_t) row].scale = scale;
        inv_scales[(size_t) row] = inv_scale;

        if (inv_scale == 0.0f) {
            std::memset(row_dst, 0, (size_t) cols);
            continue;
        }

        ggml_backend_fpga_quantize_row_values(row_src, cols, inv_scale, use_neon, row_dst);
    }
}

static void ggml_backend_fpga_quantize_rows_and_scales(
        const float * src,
        int64_t rows,
        int64_t cols,
        std::vector<fpga_quant_params> & qp,
        std::vector<float> & inv_scales,
        std::vector<int8_t> & dst) {
    ggml_backend_fpga_quantize_rows_and_scales(src, rows, cols, cols, qp, inv_scales, dst);
}

static void ggml_backend_fpga_quantize_rows(
        const float * src,
        int64_t rows,
        int64_t cols,
        std::vector<int8_t> & dst,
        std::vector<fpga_quant_params> & qp) {
    std::vector<float> inv_scales;
    ggml_backend_fpga_quantize_row_scales(src, rows, cols, qp, inv_scales);
    ggml_backend_fpga_quantize_rows_with_scales(src, rows, cols, inv_scales, dst);
}

static void ggml_backend_fpga_activation_reuse_clear(ggml_backend_fpga_context * ctx) {
    if (ctx == nullptr) {
        return;
    }
    if (ctx->activation_reuse.valid) {
        ctx->activation_reuse_clears++;
    }
    ctx->activation_reuse.tensor = nullptr;
    ctx->activation_reuse.base = nullptr;
    ctx->activation_reuse.rows = 0;
    ctx->activation_reuse.cols = 0;
    ctx->activation_reuse.row_stride = 0;
    ctx->activation_reuse.data = nullptr;
    ctx->activation_reuse.scales = nullptr;
    ctx->activation_reuse.inv_scales = nullptr;
    ctx->activation_reuse.valid = false;
}

static bool ggml_backend_fpga_activation_reuse_allowed(
        const ggml_tensor * tensor,
        int64_t rows,
        int64_t cols,
        int64_t row_stride,
        int64_t gemm_instances,
        bool use_fused_raw_a) {
    if (!ggml_backend_fpga_activation_reuse_sibling_enabled()) {
        return false;
    }
    if (ggml_backend_fpga_get_phase() != GGML_BACKEND_FPGA_PHASE_DECODE) {
        return false;
    }
    if (tensor == nullptr || use_fused_raw_a || gemm_instances != 1 || rows != 1 || cols <= 0) {
        return false;
    }
    if (row_stride < cols) {
        return false;
    }
    const int64_t elems = rows * cols;
    return elems >= ggml_backend_fpga_activation_reuse_min_elems() &&
           elems <= ggml_backend_fpga_activation_reuse_max_elems();
}

static bool ggml_backend_fpga_activation_reuse_get(
        ggml_backend_fpga_context * ctx,
        const ggml_tensor * tensor,
        const void * base,
        int64_t rows,
        int64_t cols,
        int64_t row_stride,
        ggml_fpga_prepared_activation * out) {
    ctx->activation_reuse_attempts++;
    const ggml_backend_fpga_activation_reuse_entry & entry = ctx->activation_reuse;
    if (!entry.valid ||
            entry.tensor != tensor ||
            entry.base != base ||
            entry.rows != rows ||
            entry.cols != cols ||
            entry.row_stride != row_stride) {
        ctx->activation_reuse_misses++;
        if (entry.valid) {
            ggml_backend_fpga_activation_reuse_clear(ctx);
        }
        return false;
    }
    out->data = entry.data;
    out->scales = entry.scales;
    out->inv_scales = entry.inv_scales;
    out->rows = entry.rows;
    out->cols = entry.cols;
    ctx->activation_reuse_hits++;
    return true;
}

static void ggml_backend_fpga_activation_reuse_store(
        ggml_backend_fpga_context * ctx,
        const ggml_tensor * tensor,
        const void * base,
        int64_t rows,
        int64_t cols,
        int64_t row_stride,
        std::vector<int8_t> & data,
        std::vector<fpga_quant_params> & qp,
        std::vector<float> & inv_scales) {
    const int64_t start_ns = ggml_backend_fpga_now_ns();
    ggml_backend_fpga_activation_reuse_entry & entry = ctx->activation_reuse;
    entry.tensor = tensor;
    entry.base = base;
    entry.rows = rows;
    entry.cols = cols;
    entry.row_stride = row_stride;
    entry.data_owner = std::move(data);
    entry.scales_owner.resize(qp.size());
    for (size_t i = 0; i < qp.size(); ++i) {
        entry.scales_owner[i] = qp[i].scale;
    }
    entry.inv_scales_owner = std::move(inv_scales);
    entry.data = entry.data_owner.data();
    entry.scales = entry.scales_owner.data();
    entry.inv_scales = entry.inv_scales_owner.data();
    entry.valid = true;
    ctx->activation_reuse_stores++;
    ctx->activation_reuse_store_bytes +=
        (int64_t) entry.data_owner.size() +
        (int64_t) entry.scales_owner.size() * (int64_t) sizeof(float) +
        (int64_t) entry.inv_scales_owner.size() * (int64_t) sizeof(float);
    ctx->activation_reuse_store_ns += ggml_backend_fpga_now_ns() - start_ns;
}

static void ggml_backend_fpga_dequantize_int32_row_scalar(
        const int32_t * src,
        int64_t cols,
        float a_scale,
        const fpga_quant_params * w_qp,
        char * dst,
        int64_t dst_col_stride) {
    for (int64_t col = 0; col < cols; ++col) {
        *(float *) (dst + col * dst_col_stride) =
            (float) src[(size_t) col] * a_scale * w_qp[(size_t) col].scale;
    }
}

#if GGML_FPGA_HAVE_ACT_QUANT_NEON
static void ggml_backend_fpga_dequantize_int32_row_neon_contiguous(
        const int32_t * src,
        int64_t cols,
        float a_scale,
        const fpga_quant_params * w_qp,
        float * dst) {
    const float32x4_t a_scale_v = vdupq_n_f32(a_scale);
    const float * w_scale = &w_qp[0].scale;
    int64_t col = 0;

    for (; col + 8 <= cols; col += 8) {
        const float32x4_t acc0 = vcvtq_f32_s32(vld1q_s32(src + col));
        const float32x4_t acc1 = vcvtq_f32_s32(vld1q_s32(src + col + 4));
        const float32x4_t ws0 = vld1q_f32(w_scale + col);
        const float32x4_t ws1 = vld1q_f32(w_scale + col + 4);
        vst1q_f32(dst + col, vmulq_f32(vmulq_f32(acc0, a_scale_v), ws0));
        vst1q_f32(dst + col + 4, vmulq_f32(vmulq_f32(acc1, a_scale_v), ws1));
    }

    for (; col + 4 <= cols; col += 4) {
        const float32x4_t acc = vcvtq_f32_s32(vld1q_s32(src + col));
        const float32x4_t ws = vld1q_f32(w_scale + col);
        vst1q_f32(dst + col, vmulq_f32(vmulq_f32(acc, a_scale_v), ws));
    }

    for (; col < cols; ++col) {
        dst[col] = (float) src[(size_t) col] * a_scale * w_qp[(size_t) col].scale;
    }
}
#endif

static void ggml_backend_fpga_dequantize_int32_row(
        const int32_t * src,
        int64_t cols,
        float a_scale,
        const fpga_quant_params * w_qp,
        char * dst,
        int64_t dst_col_stride) {
#if GGML_FPGA_HAVE_ACT_QUANT_NEON
    if (dst_col_stride == (int64_t) sizeof(float)) {
        ggml_backend_fpga_dequantize_int32_row_neon_contiguous(
                src, cols, a_scale, w_qp, (float *) dst);
        return;
    }
#endif
    ggml_backend_fpga_dequantize_int32_row_scalar(src, cols, a_scale, w_qp, dst, dst_col_stride);
}

static void ggml_backend_fpga_transpose_int8_nk_to_kn(
        const std::vector<int8_t> & src_nk,
        int64_t n,
        int64_t k,
        std::vector<int8_t> & dst_kn) {
    dst_kn.resize((size_t) (n * k));

    for (int64_t row_n = 0; row_n < n; ++row_n) {
        const int8_t * src_row = src_nk.data() + row_n*k;
        for (int64_t col_k = 0; col_k < k; ++col_k) {
            dst_kn[(size_t) (col_k*n + row_n)] = src_row[col_k];
        }
    }
}

static void ggml_backend_fpga_dequantize_src0(
        const ggml_tensor * src0,
        const char * base,
        std::vector<float> & dst) {
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    dst.resize((size_t) (k * n));

    if (src0->type == GGML_TYPE_F32) {
        for (int64_t row = 0; row < n; ++row) {
            const char * row_src = base + row*src0->nb[1];
            std::memcpy(dst.data() + row*k, row_src, (size_t) k * sizeof(float));
        }
        return;
    }

    ggml_to_float_t to_float = ggml_get_type_traits(src0->type)->to_float;
    GGML_ASSERT(to_float != NULL);

    for (int64_t row = 0; row < n; ++row) {
        to_float(base + row*src0->nb[1], dst.data() + row*k, k);
    }
}

static void ggml_backend_fpga_build_src0_entry(
        const ggml_tensor * src0,
        const char * base,
        int64_t n,
        int64_t k,
        std::vector<float> & src0_float,
        weight_cache_entry & entry) {
    ggml_backend_fpga_dequantize_src0(src0, base, src0_float);

    entry.N = n;
    entry.K = k;
    entry.matrix_task_prepack_allowed = ggml_backend_fpga_r5_matrix_task_prepack_allowed_name(src0);
    entry.matrix_task_decode_nk_allowed = ggml_backend_fpga_r5_matrix_task_decode_nk_allowed(n, k);
    ggml_backend_fpga_quantize_rows(src0_float.data(), n, k, entry.int8_data, entry.qp);
#ifndef GGML_FPGA_USE_MOCK
    ggml_backend_fpga_transpose_int8_nk_to_kn(entry.int8_data, n, k, entry.int8_data_kn);
#endif
}

static void ggml_backend_fpga_build_src0_entry_from_prepared(
        const ggml_tensor * src0,
        int64_t n,
        int64_t k,
        const ggml_fpga_prepared_activation & prepared,
        weight_cache_entry & entry,
        bool keep_nk_copy) {
    entry.N = n;
    entry.K = k;
    entry.matrix_task_prepack_allowed = ggml_backend_fpga_r5_matrix_task_prepack_allowed_name(src0);
    entry.matrix_task_decode_nk_allowed = ggml_backend_fpga_r5_matrix_task_decode_nk_allowed(n, k);
    entry.qp.resize((size_t) n);
    for (int64_t row = 0; row < n; ++row) {
        entry.qp[(size_t) row].scale = prepared.scales[(size_t) row];
    }
#ifndef GGML_FPGA_USE_MOCK
    if (keep_nk_copy) {
        entry.int8_data.assign(prepared.data, prepared.data + (size_t) n * (size_t) k);
        ggml_backend_fpga_transpose_int8_nk_to_kn(entry.int8_data, n, k, entry.int8_data_kn);
    } else {
        entry.int8_data.clear();
        entry.int8_data_kn.resize((size_t) n * (size_t) k);
        for (int64_t row_n = 0; row_n < n; ++row_n) {
            const int8_t * src_row = prepared.data + (size_t) row_n * (size_t) k;
            for (int64_t col_k = 0; col_k < k; ++col_k) {
                entry.int8_data_kn[(size_t) (col_k * n + row_n)] = src_row[col_k];
            }
        }
    }
#else
    GGML_UNUSED(keep_nk_copy);
    entry.int8_data.assign(prepared.data, prepared.data + (size_t) n * (size_t) k);
#endif
}

static bool ggml_backend_fpga_node_name_is(const ggml_tensor * node, const char * tag) {
    if (node == nullptr || tag == nullptr) {
        return false;
    }
    const size_t tag_len = std::strlen(tag);
    if (std::strncmp(node->name, tag, tag_len) != 0) {
        return false;
    }
    const char next = node->name[tag_len];
    return next == '\0' || next == '-' || next == '#';
}

static weight_cache_entry * ggml_backend_fpga_get_or_build_weight_entry(
        ggml_backend_fpga_context * ctx,
        const ggml_tensor * src0,
        const char * w_base,
        int64_t n,
        int64_t k,
        int64_t & cache_lookup_ns,
        int64_t & cache_build_ns) {
    const int64_t cache_lookup_start_ns = ggml_backend_fpga_now_ns();
    auto cache_it = ctx->weight_cache.find(w_base);
    const int64_t lookup_delta_ns = ggml_backend_fpga_now_ns() - cache_lookup_start_ns;
    cache_lookup_ns += lookup_delta_ns;
    ctx->weight_cache_lookup_ns += lookup_delta_ns;

    if (cache_it != ctx->weight_cache.end()) {
        ctx->weight_cache_hits++;
        return &cache_it->second;
    }

    ctx->weight_cache_misses++;
    const int64_t cache_build_start_ns = ggml_backend_fpga_now_ns();

    weight_cache_entry entry;
    ggml_backend_fpga_build_src0_entry(src0, w_base, n, k, ctx->scratch_w_float, entry);

    const int64_t build_delta_ns = ggml_backend_fpga_now_ns() - cache_build_start_ns;
    cache_build_ns += build_delta_ns;
    ctx->weight_cache_build_ns += build_delta_ns;
    ctx->total_quantize_ns += build_delta_ns;

    cache_it = ctx->weight_cache.emplace(w_base, std::move(entry)).first;
    return &cache_it->second;
}

static void ggml_backend_fpga_check_prepared_src0_entry(
        ggml_backend_fpga_context * ctx,
        const weight_cache_entry & reference,
        const weight_cache_entry & prepared) {
    ctx->dynamic_src0_prepared_check_entries++;
    int64_t bad_values = 0;
    int64_t max_i8_abs_diff = 0;
    const size_t values = std::min(reference.int8_data.size(), prepared.int8_data.size());
    for (size_t i = 0; i < values; ++i) {
        const int64_t diff = std::llabs((int64_t) reference.int8_data[i] - (int64_t) prepared.int8_data[i]);
        if (diff != 0) {
            bad_values++;
            if (diff > max_i8_abs_diff) {
                max_i8_abs_diff = diff;
            }
        }
    }
    if (reference.int8_data.size() != prepared.int8_data.size()) {
        bad_values += (int64_t) std::max(reference.int8_data.size(), prepared.int8_data.size()) - (int64_t) values;
    }
    if (bad_values != 0) {
        ctx->dynamic_src0_prepared_check_int8_bad_entries++;
        ctx->dynamic_src0_prepared_check_int8_bad_values += bad_values;
        ctx->dynamic_src0_prepared_check_max_i8_abs_diff =
            std::max(ctx->dynamic_src0_prepared_check_max_i8_abs_diff, max_i8_abs_diff);
    }

    int64_t bad_scale_rows = 0;
    double max_scale_abs_diff = 0.0;
    const size_t rows = std::min(reference.qp.size(), prepared.qp.size());
    for (size_t row = 0; row < rows; ++row) {
        const double diff = std::fabs((double) reference.qp[row].scale - (double) prepared.qp[row].scale);
        if (diff > 1e-8) {
            bad_scale_rows++;
            if (diff > max_scale_abs_diff) {
                max_scale_abs_diff = diff;
            }
        }
    }
    if (reference.qp.size() != prepared.qp.size()) {
        bad_scale_rows += (int64_t) std::max(reference.qp.size(), prepared.qp.size()) - (int64_t) rows;
    }
    if (bad_scale_rows != 0) {
        ctx->dynamic_src0_prepared_check_scale_bad_entries++;
        ctx->dynamic_src0_prepared_check_scale_bad_rows += bad_scale_rows;
        ctx->dynamic_src0_prepared_check_max_scale_abs_diff =
            std::max(ctx->dynamic_src0_prepared_check_max_scale_abs_diff, max_scale_abs_diff);
    }
}

static bool ggml_backend_fpga_activation_reuse_same_sibling(
        const ggml_tensor * a,
        const ggml_tensor * b) {
    if (a == nullptr || b == nullptr ||
            a->op != GGML_OP_MUL_MAT ||
            b->op != GGML_OP_MUL_MAT ||
            a->src[0] == nullptr ||
            a->src[1] == nullptr ||
            b->src[0] == nullptr ||
            b->src[1] == nullptr) {
        return false;
    }
    return a->src[1] == b->src[1] &&
           a->src[0]->ne[0] == b->src[0]->ne[0] &&
           a->src[1]->ne[1] == b->src[1]->ne[1] &&
           a->src[1]->nb[1] == b->src[1]->nb[1];
}

static bool ggml_backend_fpga_activation_reuse_future_sibling(
        ggml_cgraph * cgraph,
        int node_idx,
        const ggml_tensor * node) {
    if (!ggml_backend_fpga_activation_reuse_sibling_enabled() ||
            cgraph == nullptr ||
            node == nullptr ||
            node->op != GGML_OP_MUL_MAT ||
            node->src[0] == nullptr ||
            node->src[1] == nullptr) {
        return false;
    }

    const int64_t rows = node->src[1]->ne[1];
    const int64_t cols = node->src[0]->ne[0];
    const int64_t row_stride = node->src[1]->nb[1] / (int64_t) sizeof(float);
    if (!ggml_backend_fpga_activation_reuse_allowed(
                node->src[1],
                rows,
                cols,
                row_stride,
                node->ne[2] * node->ne[3],
                ggml_backend_fpga_fused_raw_a_enabled())) {
        return false;
    }

    for (int j = node_idx + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * cand = cgraph->nodes[j];
        if (cand == nullptr || cand->op != GGML_OP_MUL_MAT ||
                cand->src[0] == nullptr || cand->src[1] == nullptr) {
            continue;
        }
        const int64_t cand_rows = cand->src[1]->ne[1];
        const int64_t cand_cols = cand->src[0]->ne[0];
        const int64_t cand_row_stride = cand->src[1]->nb[1] / (int64_t) sizeof(float);
        const bool cand_allowed = ggml_backend_fpga_activation_reuse_allowed(
                cand->src[1],
                cand_rows,
                cand_cols,
                cand_row_stride,
                cand->ne[2] * cand->ne[3],
                ggml_backend_fpga_fused_raw_a_enabled());
        if (!cand_allowed) {
            continue;
        }
        return ggml_backend_fpga_activation_reuse_same_sibling(node, cand);
    }
    return false;
}

static int64_t ggml_backend_fpga_prepack_weight_entry_for_matrix_task(
        weight_cache_entry & cached,
        int64_t n,
        int64_t k) {
#ifdef GGML_FPGA_USE_MOCK
    (void) cached;
    (void) n;
    (void) k;
    return 0;
#else
    const int64_t prepack_start_ns = ggml_backend_fpga_now_ns();
    if (!cached.packed_b_cache.matrix_task_prepack_attempted &&
            ggml_backend_fpga_r5_matrix_task_prepack_enabled() &&
            cached.matrix_task_prepack_allowed &&
            cached.matrix_task_decode_nk_allowed &&
            ggml_backend_fpga_r5_matrix_task_decode_enabled(1, n, k, 1)) {
        cached.packed_b_cache.matrix_task_prepack_attempted = true;
        const int prepack_rc = tpu_matrix_task_prepack_b_with_cache(
                cached.int8_data_kn.data(),
                (uint32_t) n,
                (uint32_t) k,
                &cached.packed_b_cache.cache);
        cached.packed_b_cache.matrix_task_prepack_ok = prepack_rc == 0;
    }
    return ggml_backend_fpga_now_ns() - prepack_start_ns;
#endif
}

static bool ggml_backend_fpga_run_m1_prequantized_matmul_to_node(
        ggml_backend_fpga_context * ctx,
        ggml_tensor * dst,
        weight_cache_entry & cached,
        const int8_t * a_int8,
        const float * a_scales,
        int64_t m,
        int64_t n,
        int64_t k,
        int64_t & gemm_ns,
        int64_t & tiling_stats_ns,
        int64_t & dequant_ns) {
    GGML_ASSERT(m == 1);
    GGML_ASSERT(cached.N == n && cached.K == k);

    std::vector<int32_t> & d_int32 = ctx->scratch_d_int32;
    std::vector<int32_t> & d_int32_dequant_scratch = ctx->scratch_d_int32_dequant;
    tpu_tiling_stats_t tile_stats = {};

    const int64_t gemm_start_ns = ggml_backend_fpga_now_ns();
#ifdef GGML_FPGA_USE_MOCK
    d_int32.resize((size_t) m * (size_t) n);
    tpu_gemm_int8_int32_tiled(
            a_int8,
            cached.int8_data.data(),
            d_int32.data(),
            (uint32_t) m,
            (uint32_t) n,
            (uint32_t) k,
            &tile_stats);
    tpu_int32_view_t d_int32_view = {};
#else
    tpu_int32_view_t d_int32_view = {};
    int rc = -1;
    if (ggml_backend_fpga_matrix_task_direct_view_enabled()) {
        rc = tpu_gemm_int8_int32_matrix_task_prepacked_b_view_with_cache(
                a_int8,
                cached.int8_data_kn.data(),
                &d_int32_view,
                (uint32_t) m,
                (uint32_t) n,
                (uint32_t) k,
                &tile_stats,
                &cached.packed_b_cache.cache);
    } else {
        d_int32.resize((size_t) m * (size_t) n);
        rc = tpu_gemm_int8_int32_matrix_task_prepacked_b_with_cache(
                a_int8,
                cached.int8_data_kn.data(),
                d_int32.data(),
                (uint32_t) m,
                (uint32_t) n,
                (uint32_t) k,
                &tile_stats,
                &cached.packed_b_cache.cache);
    }
    if (rc != 0) {
        return false;
    }
#endif
    const int64_t gemm_delta_ns = ggml_backend_fpga_now_ns() - gemm_start_ns;
    gemm_ns += gemm_delta_ns;
    ctx->total_gemm_ns += gemm_delta_ns;

    const int64_t tiling_stats_start_ns = ggml_backend_fpga_now_ns();
    ggml_backend_fpga_accumulate_tiling_stats(ctx, tile_stats);
    const int64_t tiling_delta_ns = ggml_backend_fpga_now_ns() - tiling_stats_start_ns;
    tiling_stats_ns += tiling_delta_ns;
    ctx->tiling_stats_accum_ns += tiling_delta_ns;

    const int64_t dequant_start_ns = ggml_backend_fpga_now_ns();
#ifndef GGML_FPGA_USE_MOCK
    const bool copy_view_to_scratch_for_dequant =
        d_int32_view.data != NULL &&
        ggml_backend_fpga_dequant_scratch_copy_enabled();
    if (copy_view_to_scratch_for_dequant) {
        d_int32_dequant_scratch.resize((size_t) n);
    }
#endif
    for (int64_t row = 0; row < m; ++row) {
        char * d_row = (char *) dst->data + row*dst->nb[1];
        const int32_t * d_int32_row =
#ifndef GGML_FPGA_USE_MOCK
            d_int32_view.data != NULL ?
            d_int32_view.data + (size_t) row * d_int32_view.stride :
            d_int32.data() + (size_t) row * n;
#else
            d_int32.data() + (size_t) row * n;
#endif
#ifndef GGML_FPGA_USE_MOCK
        if (copy_view_to_scratch_for_dequant) {
            std::memcpy(d_int32_dequant_scratch.data(), d_int32_row, (size_t) n * sizeof(int32_t));
            d_int32_row = d_int32_dequant_scratch.data();
        }
#endif
        ctx->dequant_path.rows_total++;
        ctx->dequant_path.cols_total += n;
        if (dst->nb[0] == (int64_t) sizeof(float)) {
            ctx->dequant_path.rows_neon++;
        } else {
            ctx->dequant_path.rows_scalar++;
        }
#ifndef GGML_FPGA_USE_MOCK
        if (copy_view_to_scratch_for_dequant) {
            ctx->dequant_path.rows_scratch_copy++;
        }
#endif
        ggml_backend_fpga_dequantize_int32_row(
                d_int32_row,
                n,
                a_scales[(size_t) row],
                cached.qp.data(),
                d_row,
                dst->nb[0]);
    }
    const int64_t dequant_delta_ns = ggml_backend_fpga_now_ns() - dequant_start_ns;
    dequant_ns += dequant_delta_ns;
    ctx->total_dequant_ns += dequant_delta_ns;

    return true;
}

static bool ggml_backend_fpga_try_fuse_ffn_up_gate_dual(
        ggml_backend_fpga_context * ctx,
        ggml_tensor * up_node,
        ggml_tensor * gate_node) {
    if (!ggml_backend_fpga_ffn_sibling_dual_enabled()) {
        return false;
    }
    ctx->ffn_dual_attempts++;

    auto reject = [&]() {
        ctx->ffn_dual_misses++;
        return false;
    };

    if (ggml_backend_fpga_get_phase() != GGML_BACKEND_FPGA_PHASE_DECODE) {
        return reject();
    }
    if (up_node == nullptr || gate_node == nullptr ||
            up_node->op != GGML_OP_MUL_MAT ||
            gate_node->op != GGML_OP_MUL_MAT) {
        return reject();
    }
    if (!ggml_backend_fpga_node_name_is(up_node, "ffn_up") ||
            !ggml_backend_fpga_node_name_is(gate_node, "ffn_gate")) {
        return reject();
    }

    const ggml_tensor * up_w = up_node->src[0];
    const ggml_tensor * gate_w = gate_node->src[0];
    const ggml_tensor * a = up_node->src[1];
    if (a == nullptr || a != gate_node->src[1]) {
        return reject();
    }
    if (up_node->type != GGML_TYPE_F32 || gate_node->type != GGML_TYPE_F32 ||
            a->type != GGML_TYPE_F32) {
        return reject();
    }
    if (!ggml_backend_fpga_supports_src0_type(up_w->type) ||
            !ggml_backend_fpga_supports_src0_type(gate_w->type)) {
        return reject();
    }
    if (!ggml_backend_fpga_can_cache_src0(up_w) ||
            !ggml_backend_fpga_can_cache_src0(gate_w)) {
        return reject();
    }
    if (!ggml_backend_fpga_is_contiguous_2d(up_w) ||
            !ggml_backend_fpga_is_contiguous_2d(gate_w) ||
            !ggml_backend_fpga_is_row_dense_2d_plane(a)) {
        return reject();
    }
    if (up_w->ne[2] != 1 || up_w->ne[3] != 1 ||
            gate_w->ne[2] != 1 || gate_w->ne[3] != 1 ||
            a->ne[2] != 1 || a->ne[3] != 1 ||
            up_node->ne[2] != 1 || up_node->ne[3] != 1 ||
            gate_node->ne[2] != 1 || gate_node->ne[3] != 1) {
        return reject();
    }

    const int64_t m = a->ne[1];
    const int64_t k = up_w->ne[0];
    const int64_t n_up = up_w->ne[1];
    const int64_t n_gate = gate_w->ne[1];
    const int64_t a_row_stride = a->nb[1] / (int64_t) sizeof(float);
    if (m != 1 || k <= 0 || n_up <= 0 || n_gate <= 0 ||
            a->ne[0] != k ||
            a_row_stride < k ||
            gate_w->ne[0] != k ||
            up_node->ne[0] != n_up ||
            gate_node->ne[0] != n_gate ||
            up_node->ne[1] != m ||
            gate_node->ne[1] != m) {
        return reject();
    }
    if (!ggml_backend_fpga_r5_matrix_task_decode_enabled(m, n_up, k, 1) ||
            !ggml_backend_fpga_r5_matrix_task_decode_enabled(m, n_gate, k, 1)) {
        return reject();
    }

    const int64_t wall_start_ns = ggml_backend_fpga_now_ns();
    const int64_t setup_start_ns = ggml_backend_fpga_now_ns();
    ctx->n_mul_mat_calls += 2;
    ctx->min_k = std::min(ctx->min_k, k);
    ctx->min_n = std::min(ctx->min_n, std::min(n_up, n_gate));
    ctx->min_m = std::min(ctx->min_m, m);
    ctx->max_k = std::max(ctx->max_k, k);
    ctx->max_n = std::max(ctx->max_n, std::max(n_up, n_gate));
    ctx->max_m = std::max(ctx->max_m, m);
    ggml_backend_fpga_record_shape(ctx, m, n_up, k, 1);
    ggml_backend_fpga_record_shape(ctx, m, n_gate, k, 1);
    const int64_t setup_ns = ggml_backend_fpga_now_ns() - setup_start_ns;
    ctx->mul_mat_setup_ns += setup_ns;

    int64_t cache_lookup_ns = 0;
    int64_t cache_build_ns = 0;
    weight_cache_entry * up_entry = ggml_backend_fpga_get_or_build_weight_entry(
            ctx,
            up_w,
            (const char *) up_w->data,
            n_up,
            k,
            cache_lookup_ns,
            cache_build_ns);
    weight_cache_entry * gate_entry = ggml_backend_fpga_get_or_build_weight_entry(
            ctx,
            gate_w,
            (const char *) gate_w->data,
            n_gate,
            k,
            cache_lookup_ns,
            cache_build_ns);

    const int64_t prepack_ns =
        ggml_backend_fpga_prepack_weight_entry_for_matrix_task(*up_entry, n_up, k) +
        ggml_backend_fpga_prepack_weight_entry_for_matrix_task(*gate_entry, n_gate, k);
    ctx->matrix_task_prepack_ns += prepack_ns;

    std::vector<float> & a_inv_scales = ctx->scratch_a_inv_scales;
    std::vector<int8_t> & a_int8 = ctx->scratch_a_int8;
    std::vector<fpga_quant_params> & a_qp = ctx->scratch_a_qp;
    const float * a_float = (const float *) a->data;

    const int64_t activation_start_ns = ggml_backend_fpga_now_ns();
    if (ggml_backend_fpga_act_quant_fused_rows_enabled()) {
        ggml_backend_fpga_quantize_rows_and_scales(a_float, m, k, a_row_stride, a_qp, a_inv_scales, a_int8);
    } else {
        ggml_backend_fpga_quantize_row_scales(a_float, m, k, a_row_stride, a_qp, a_inv_scales);
        ggml_backend_fpga_quantize_rows_with_scales(a_float, m, k, a_row_stride, a_inv_scales, a_int8);
    }
    const int64_t activation_ns = ggml_backend_fpga_now_ns() - activation_start_ns;
    ctx->activation_quantize_ns += activation_ns;
    ctx->total_quantize_ns += activation_ns;
    const float a_scale = a_qp.empty() ? 1.0f : a_qp[0].scale;

    int64_t gemm_ns = 0;
    int64_t tiling_stats_ns = 0;
    int64_t dequant_ns = 0;

    if (!ggml_backend_fpga_run_m1_prequantized_matmul_to_node(
                ctx, gate_node, *gate_entry, a_int8.data(), &a_scale, m, n_gate, k,
                gemm_ns, tiling_stats_ns, dequant_ns)) {
        return reject();
    }
    if (!ggml_backend_fpga_run_m1_prequantized_matmul_to_node(
                ctx, up_node, *up_entry, a_int8.data(), &a_scale, m, n_up, k,
                gemm_ns, tiling_stats_ns, dequant_ns)) {
        return reject();
    }

    const int64_t known_ns =
        setup_ns +
        cache_lookup_ns +
        cache_build_ns +
        prepack_ns +
        activation_ns +
        gemm_ns +
        tiling_stats_ns +
        dequant_ns;
    const int64_t wall_ns = ggml_backend_fpga_now_ns() - wall_start_ns;
    if (wall_ns > known_ns) {
        ctx->mul_mat_instance_admin_ns += wall_ns - known_ns;
    }
    ggml_backend_fpga_activation_reuse_clear(ctx);
    ctx->ffn_dual_hits++;
    return true;
}

static void ggml_backend_fpga_mul_mat(
        ggml_backend_fpga_context * ctx,
        ggml_tensor * dst,
        bool activation_reuse_may_store) {
    const int64_t setup_start_ns = ggml_backend_fpga_now_ns();
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    const bool allow_strided = ggml_backend_fpga_strided_mul_mat_enabled();
    const bool allow_vit_attn_strided =
        ggml_backend_fpga_vit_attn_strided_mul_mat_enabled() &&
        ggml_backend_fpga_is_vit_attn_strided_target(dst);
    const bool allow_vit_attn_pl_op =
        ggml_backend_fpga_vit_attn_pl_op_allowed(dst, src1->ne[1], src0->ne[1], src0->ne[0]);
    const bool allow_src1_strided =
        allow_strided ||
        ggml_backend_fpga_src1_strided_mul_mat_enabled() ||
        allow_vit_attn_strided ||
        allow_vit_attn_pl_op;
    GGML_ASSERT(ggml_backend_fpga_is_contiguous_2d(src0) ||
            (allow_strided && ggml_backend_fpga_is_row_dense_2d_plane(src0)));
    GGML_ASSERT(ggml_backend_fpga_is_contiguous_2d(src1) ||
            (allow_src1_strided && ggml_backend_fpga_is_row_dense_2d_plane(src1)));
    GGML_ASSERT(src1->nb[1] % (int64_t) sizeof(float) == 0);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    const int64_t m = src1->ne[1];
    const int64_t k = src0->ne[0];
    const int64_t n = src0->ne[1];
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;
    const int64_t gemm_instances = ne12*ne13;
    const bool cacheable_src0 = ggml_backend_fpga_can_cache_src0(src0);
    const bool output_head_qscale_node =
        src0 != NULL &&
        strstr(src0->name, "output.weight") != NULL &&
        ggml_backend_fpga_output_head_top1_qscale_debug_enabled();
    if (!output_head_qscale_node) {
        ggml_backend_fpga_output_head_qscale_global_finalize(ctx);
    }

    if (ggml_is_empty(dst) || m == 0 || n == 0 || k == 0) {
        ggml_backend_fpga_record_empty_shape(ctx, m, n, k, gemm_instances);
        return;
    }

    // Reuse hot-path buffers across thousands of decode M=1 matmul calls.
    std::vector<float> & w_float = ctx->scratch_w_float;
    std::vector<float> & a_inv_scales = ctx->scratch_a_inv_scales;
    std::vector<int8_t> & a_int8 = ctx->scratch_a_int8;
    std::vector<fpga_quant_params> & a_qp = ctx->scratch_a_qp;
    std::vector<int32_t> & d_int32 = ctx->scratch_d_int32;
    std::vector<int32_t> & d_int32_dequant_scratch = ctx->scratch_d_int32_dequant;
    std::vector<float> a_materialized;

    ctx->n_mul_mat_calls++;
    ctx->min_k = std::min(ctx->min_k, k);
    ctx->min_n = std::min(ctx->min_n, n);
    ctx->min_m = std::min(ctx->min_m, m);
    ctx->max_k = std::max(ctx->max_k, k);
    ctx->max_n = std::max(ctx->max_n, n);
    ctx->max_m = std::max(ctx->max_m, m);
    ggml_backend_fpga_record_shape(ctx, m, n, k, gemm_instances);
    ctx->mul_mat_setup_ns += ggml_backend_fpga_now_ns() - setup_start_ns;

    for (int64_t i13 = 0; i13 < ne13; ++i13) {
        for (int64_t i12 = 0; i12 < ne12; ++i12) {
            const int64_t instance_start_ns = ggml_backend_fpga_now_ns();
            int64_t instance_cache_lookup_ns = 0;
            int64_t instance_src0_build_ns = 0;
            int64_t instance_prepack_ns = 0;
            int64_t instance_gate_ns = 0;
            int64_t instance_activation_lookup_ns = 0;
            int64_t instance_activation_quant_ns = 0;
            int64_t instance_gemm_ns = 0;
            int64_t instance_tiling_stats_ns = 0;
            int64_t instance_dequant_ns = 0;
            int64_t instance_ffn_consume_ns = 0;

            const int64_t i03 = i13/r3;
            const int64_t i02 = i12/r2;

            const char * w_base = (const char *) src0->data + i02*nb02 + i03*nb03;
            weight_cache_entry * cached_ptr = nullptr;
            weight_cache_entry dynamic_src0_entry;
            bool used_dynamic_src0_prepared = false;
            bool cached_ptr_persistent = false;

            if (cacheable_src0) {
                const int64_t cache_lookup_start_ns = ggml_backend_fpga_now_ns();
                auto cache_it = ctx->weight_cache.find(w_base);
                instance_cache_lookup_ns = ggml_backend_fpga_now_ns() - cache_lookup_start_ns;
                ctx->weight_cache_lookup_ns += instance_cache_lookup_ns;
                if (cache_it != ctx->weight_cache.end()) {
                    ctx->weight_cache_hits++;
                } else {
                    ctx->weight_cache_misses++;
                    const int64_t cache_build_start = ggml_backend_fpga_now_ns();

                    weight_cache_entry entry;
                    ggml_backend_fpga_build_src0_entry(src0, w_base, n, k, w_float, entry);

                    instance_src0_build_ns = ggml_backend_fpga_now_ns() - cache_build_start;
                    ctx->weight_cache_build_ns += instance_src0_build_ns;
                    ctx->total_quantize_ns += instance_src0_build_ns;

                    cache_it = ctx->weight_cache.emplace(w_base, std::move(entry)).first;
                }

                cached_ptr = &cache_it->second;
                cached_ptr_persistent = true;
            } else {
                ctx->weight_cache_bypasses++;
                ggml_backend_fpga_record_dynamic_src0(ctx, src0, m, n, k);
                const dynamic_src0_cache_key dynamic_key = { src0, w_base, n, k };
                const bool use_dynamic_cache = ggml_backend_fpga_dynamic_src0_cache_enabled();
                auto dynamic_cache_it = ctx->dynamic_src0_cache.end();
                if (use_dynamic_cache) {
                    dynamic_cache_it = ctx->dynamic_src0_cache.find(dynamic_key);
                }
                if (use_dynamic_cache && dynamic_cache_it != ctx->dynamic_src0_cache.end()) {
                    ctx->dynamic_src0_cache_hits++;
                    cached_ptr = &dynamic_cache_it->second;
                    cached_ptr_persistent = true;
                } else {
                    ctx->dynamic_src0_cache_misses++;
                    const int64_t dynamic_build_start = ggml_backend_fpga_now_ns();
                    ggml_fpga_prepared_activation prepared_src0 = {};
                    const int64_t prepared_src0_row0 = (i02 + i03*ne02) * n;
                    if (ggml_backend_fpga_dynamic_src0_prepared_tensor_allowed(src0) &&
                            ggml_backend_fpga_ffn_streaming_get_rows(src0, prepared_src0_row0, n, k, &prepared_src0)) {
                        ctx->dynamic_src0_prepared_hits++;
                        if (ggml_backend_fpga_dynamic_src0_prepared_check_only_enabled()) {
                            weight_cache_entry prepared_check_entry;
                            ggml_backend_fpga_build_src0_entry(src0, w_base, n, k, w_float, dynamic_src0_entry);
                            ggml_backend_fpga_build_src0_entry_from_prepared(src0, n, k, prepared_src0, prepared_check_entry, true);
                            ggml_backend_fpga_check_prepared_src0_entry(ctx, dynamic_src0_entry, prepared_check_entry);
                        } else {
                            used_dynamic_src0_prepared = true;
                            ggml_backend_fpga_build_src0_entry_from_prepared(src0, n, k, prepared_src0, dynamic_src0_entry, false);
                        }
                    } else {
                        if (ggml_backend_fpga_dynamic_src0_prepared_tensor_allowed(src0)) {
                            ctx->dynamic_src0_prepared_misses++;
                        }
                        ggml_backend_fpga_build_src0_entry(src0, w_base, n, k, w_float, dynamic_src0_entry);
                    }

                    instance_src0_build_ns = ggml_backend_fpga_now_ns() - dynamic_build_start;
                    ctx->dynamic_src0_build_ns += instance_src0_build_ns;
                    if (used_dynamic_src0_prepared) {
                        ctx->dynamic_src0_prepared_import_ns += instance_src0_build_ns;
                    }
                    ctx->total_quantize_ns += instance_src0_build_ns;
                    if (use_dynamic_cache) {
                        const int64_t entry_bytes =
                            (int64_t) dynamic_src0_entry.int8_data.size() +
                            (int64_t) dynamic_src0_entry.int8_data_kn.size() +
                            (int64_t) dynamic_src0_entry.qp.size() * (int64_t) sizeof(fpga_quant_params);
                        auto inserted = ctx->dynamic_src0_cache.emplace(dynamic_key, std::move(dynamic_src0_entry));
                        ctx->dynamic_src0_cache_stores++;
                        ctx->dynamic_src0_cache_store_bytes += entry_bytes;
                        cached_ptr = &inserted.first->second;
                        cached_ptr_persistent = true;
                    } else {
                        cached_ptr = &dynamic_src0_entry;
                    }
                }
            }

            weight_cache_entry & cached = *cached_ptr;
            GGML_ASSERT(cached.N == n && cached.K == k);
            tpu_packed_b_cache_t * packed_b_cache_owner =
                cached_ptr_persistent ? &cached.packed_b_cache.cache : NULL;
#ifdef GGML_FPGA_USE_MOCK
            const int8_t * w_int8_ptr = cached.int8_data.data();
#else
            const int8_t * w_int8_kn_ptr = cached.int8_data_kn.data();
#endif
            const fpga_quant_params * w_qp_ptr = cached.qp.data();

            const int64_t gate_start_ns = ggml_backend_fpga_now_ns();
            const bool use_fused_raw_a = ggml_backend_fpga_fused_raw_a_enabled();
#ifndef GGML_FPGA_USE_MOCK
            if (cacheable_src0 &&
                    !cached.packed_b_cache.matrix_task_prepack_attempted &&
                    ggml_backend_fpga_r5_matrix_task_prepack_enabled() &&
                    cached.matrix_task_prepack_allowed &&
                    cached.matrix_task_decode_nk_allowed &&
                    ggml_backend_fpga_r5_matrix_task_decode_enabled(1, n, k, 1)) {
                cached.packed_b_cache.matrix_task_prepack_attempted = true;
                const int prepack_rc = tpu_matrix_task_prepack_b_with_cache(
                        w_int8_kn_ptr,
                        (uint32_t) n,
                    (uint32_t) k,
                    &cached.packed_b_cache.cache);
                instance_prepack_ns += ggml_backend_fpga_now_ns() - gate_start_ns - instance_gate_ns;
                cached.packed_b_cache.matrix_task_prepack_ok = prepack_rc == 0;
            }
#endif
            const bool try_vit_attn_pl_op = allow_vit_attn_pl_op;
            const bool dynamic_multi_instance = !cacheable_src0 && gemm_instances != 1;
            const bool allow_dynamic_multi_instance_task =
                !dynamic_multi_instance ||
                ggml_backend_fpga_r5_matrix_task_dynamic_multi_instance_enabled();
            const bool try_r5_matrix_task = try_vit_attn_pl_op ||
                (allow_dynamic_multi_instance_task &&
                 ggml_backend_fpga_r5_matrix_task_decode_enabled(
                         m,
                         n,
                         k,
                         gemm_instances));
            const bool use_act_quant_fused_rows = ggml_backend_fpga_act_quant_fused_rows_enabled();
            instance_gate_ns += ggml_backend_fpga_now_ns() - gate_start_ns - instance_prepack_ns;
            ctx->mul_mat_gate_ns += instance_gate_ns;
            ctx->matrix_task_prepack_ns += instance_prepack_ns;
            ggml_fpga_prepared_activation prepared_activation = {};
            bool have_prepared_activation = false;
            bool have_ffn_prepared_activation = false;
            bool have_reused_activation = false;
            bool can_store_reused_activation = false;
            const float * a_float = (const float *) ((const char *) src1->data + i12*nb12 + i13*nb13);
            const void * a_reuse_base = a_float;
            const int64_t a_row_stride = nb11 / (int64_t) sizeof(float);
            const int64_t activation_source_lookup_start_ns = ggml_backend_fpga_now_ns();
            const bool can_use_prepared_activation =
                !use_fused_raw_a &&
                gemm_instances == 1 &&
                m > 1;
            if (can_use_prepared_activation &&
                    ggml_backend_fpga_ffn_streaming_get(src1, m, k, &prepared_activation)) {
                have_prepared_activation = true;
                have_ffn_prepared_activation = true;
            }
            const bool can_use_reused_activation =
                ggml_backend_fpga_activation_reuse_allowed(
                        src1,
                        m,
                        k,
                        a_row_stride,
                        gemm_instances,
                        use_fused_raw_a);
            if (!have_prepared_activation && can_use_reused_activation) {
                have_reused_activation =
                    ggml_backend_fpga_activation_reuse_get(
                            ctx,
                            src1,
                            a_reuse_base,
                            m,
                            k,
                            a_row_stride,
                            &prepared_activation);
                have_prepared_activation = have_reused_activation;
                can_store_reused_activation =
                    !have_reused_activation && activation_reuse_may_store;
            }
            instance_activation_lookup_ns = ggml_backend_fpga_now_ns() - activation_source_lookup_start_ns;
            ctx->activation_source_lookup_ns += instance_activation_lookup_ns;
            if (have_ffn_prepared_activation) {
                ctx->ffn_stream_cache_hits++;
            } else if (can_use_prepared_activation) {
                ctx->ffn_stream_cache_misses++;
            }
            const int64_t activation_quantize_start = ggml_backend_fpga_now_ns();
            int64_t effective_a_row_stride = a_row_stride;
            const bool materialize_strided_a =
                !have_prepared_activation &&
                ggml_backend_fpga_src1_strided_materialize_enabled() &&
                a_row_stride != k;
            if (materialize_strided_a) {
                a_materialized.resize((size_t) m * (size_t) k);
                for (int64_t row = 0; row < m; ++row) {
                    memcpy(a_materialized.data() + (size_t) row * (size_t) k,
                            a_float + row * a_row_stride,
                            (size_t) k * sizeof(float));
                }
                a_float = a_materialized.data();
                effective_a_row_stride = k;
            }
            bool need_a_int8 = !have_prepared_activation && (!use_fused_raw_a || try_r5_matrix_task);
#ifdef FPGA_TILING_DEBUG
            need_a_int8 = true;
#endif
            if (have_prepared_activation) {
                // The prepared activation already carries int8 rows and scales.
            } else if (need_a_int8 && use_act_quant_fused_rows) {
                ggml_backend_fpga_quantize_rows_and_scales(a_float, m, k, effective_a_row_stride, a_qp, a_inv_scales, a_int8);
            } else {
                ggml_backend_fpga_quantize_row_scales(a_float, m, k, effective_a_row_stride, a_qp, a_inv_scales);
            }
            if (need_a_int8 && !use_act_quant_fused_rows) {
                ggml_backend_fpga_quantize_rows_with_scales(a_float, m, k, effective_a_row_stride, a_inv_scales, a_int8);
            }
            instance_activation_quant_ns = ggml_backend_fpga_now_ns() - activation_quantize_start;
            ctx->activation_quantize_ns += instance_activation_quant_ns;
            ctx->total_quantize_ns += instance_activation_quant_ns;

            const int64_t gemm_start = ggml_backend_fpga_now_ns();
            tpu_tiling_stats_t tile_stats = {};

            if (ggml_backend_fpga_mul_mat_submit_trace_enabled()) {
                static uint64_t submit_trace_count = 0;
                submit_trace_count++;
                fprintf(stderr,
                        "[FPGA/MULMAT_SUBMIT] call=%" PRIu64
                        " phase=%s M=%" PRId64 " N=%" PRId64 " K=%" PRId64
                        " inst=%" PRId64 " i12=%" PRId64 " i13=%" PRId64
                        " types=%s/%s/%s src0_dense=%d src1_dense=%d src1_row_dense=%d"
                        " src1_stride=%" PRId64 " effective_a_stride=%" PRId64
                        " cacheable_src0=%d use_fused_raw_a=%d try_r5=%d have_prepared=%d"
                        " tile_family=%s max_nk=%u\n",
                        submit_trace_count,
                        ggml_backend_fpga_phase_name(ggml_backend_fpga_get_phase()),
                        m,
                        n,
                        k,
                        gemm_instances,
                        i12,
                        i13,
                        ggml_type_name(src0->type),
                        ggml_type_name(src1->type),
                        ggml_type_name(dst->type),
                        ggml_backend_fpga_is_dense_2d_plane(src0) ? 1 : 0,
                        ggml_backend_fpga_is_dense_2d_plane(src1) ? 1 : 0,
                        ggml_backend_fpga_is_row_dense_2d_plane(src1) ? 1 : 0,
                        a_row_stride,
                        effective_a_row_stride,
                        cacheable_src0 ? 1 : 0,
                        use_fused_raw_a ? 1 : 0,
                        try_r5_matrix_task ? 1 : 0,
                        have_prepared_activation ? 1 : 0,
                        tpu_tile_family_name(),
                        tpu_get_effective_max_nk());
                fflush(stderr);
            }

#ifdef FPGA_TILING_DEBUG
            std::vector<int32_t> d_int32_ref((size_t) (m * n));
            tpu_gemm_int8_int32_tiled(
                a_int8.data(),
                cached.int8_data.data(),
                d_int32_ref.data(),
                (uint32_t) m,
                (uint32_t) n,
                (uint32_t) k,
                NULL);
#endif

            // Future: GEMV fast path
            // if (m <= GEMV_THRESHOLD) {
            //     mock_tpu_gemv_int8_int32_tiled(a_int8.data(), w_int8.data(),
            //         d_int32.data(), (uint32_t) n, (uint32_t) k, &gemv_stats);
            // } else {
#ifdef GGML_FPGA_USE_MOCK
            d_int32.resize((size_t) (m * n));
            tpu_gemm_int8_int32_tiled(
                have_prepared_activation ? prepared_activation.data : a_int8.data(),
                w_int8_ptr,
                d_int32.data(),
                (uint32_t) m,
                (uint32_t) n,
                (uint32_t) k,
                &tile_stats);
#else
            bool used_r5_matrix_task = false;
            tpu_int32_view_t d_int32_view = {};
            if (try_r5_matrix_task) {
                const int8_t * a_for_r5 = have_prepared_activation ? prepared_activation.data : a_int8.data();
                const bool matrix_task_direct_view = ggml_backend_fpga_matrix_task_direct_view_enabled();
                const bool collect_matrix_task_shape_stats = ggml_backend_fpga_matrix_task_shape_stats_enabled();
                tpu_runtime_stats_t matrix_task_stats_before = {};
                if (collect_matrix_task_shape_stats) {
                    tpu_runtime_get_stats(&matrix_task_stats_before);
                }
                int rc = -1;
                if (matrix_task_direct_view) {
                    rc = tpu_gemm_int8_int32_matrix_task_prepacked_b_view_with_cache(
                        a_for_r5,
                        w_int8_kn_ptr,
                        &d_int32_view,
                        (uint32_t) m,
                        (uint32_t) n,
                        (uint32_t) k,
                        &tile_stats,
                        packed_b_cache_owner);
                } else {
                    d_int32.resize((size_t) (m * n));
                    rc = tpu_gemm_int8_int32_matrix_task_prepacked_b_with_cache(
                        a_for_r5,
                        w_int8_kn_ptr,
                        d_int32.data(),
                        (uint32_t) m,
                        (uint32_t) n,
                        (uint32_t) k,
                        &tile_stats,
                        packed_b_cache_owner);
                }
                if (rc == 0) {
                    used_r5_matrix_task = true;
                } else if (ggml_backend_fpga_r5_matrix_task_strict()) {
                    GGML_ABORT("%s: TPU_V2_R5_MATRIX_TASK_STRICT=1 and matrix-task submit failed\n", __func__);
                }
                if (collect_matrix_task_shape_stats) {
                    tpu_runtime_stats_t matrix_task_stats_after = {};
                    tpu_runtime_get_stats(&matrix_task_stats_after);
                    const tpu_runtime_stats_t matrix_task_delta =
                            ggml_backend_fpga_runtime_stats_delta(matrix_task_stats_after, matrix_task_stats_before);
                    ggml_backend_fpga_record_matrix_task_shape(
                            ctx,
                            src0,
                            m,
                            n,
                            k,
                            gemm_instances,
                            cacheable_src0,
                            dynamic_multi_instance,
                            packed_b_cache_owner != NULL,
                            have_prepared_activation,
                            matrix_task_direct_view,
                            try_vit_attn_pl_op,
                            used_r5_matrix_task,
                            matrix_task_delta);
                }
            }
            if (!used_r5_matrix_task && use_fused_raw_a) {
                d_int32.resize((size_t) (m * n));
                tpu_gemm_f32_int8_int32_tiled_prepacked_b_with_cache(
                    a_float,
                    a_inv_scales.data(),
                    w_int8_kn_ptr,
                    d_int32.data(),
                    (uint32_t) m,
                    (uint32_t) n,
                    (uint32_t) k,
                    &tile_stats,
                    packed_b_cache_owner);
            } else if (!used_r5_matrix_task) {
                d_int32.resize((size_t) (m * n));
                tpu_gemm_int8_int32_tiled_prepacked_b_with_cache(
                    have_prepared_activation ? prepared_activation.data : a_int8.data(),
                    w_int8_kn_ptr,
                    d_int32.data(),
                    (uint32_t) m,
                    (uint32_t) n,
                    (uint32_t) k,
                    &tile_stats,
                    packed_b_cache_owner);
            }
#endif

#ifdef FPGA_TILING_DEBUG
            for (size_t idx = 0; idx < d_int32.size(); ++idx) {
                GGML_ASSERT(d_int32[idx] == d_int32_ref[idx]);
            }
#endif

            const int64_t gemm_call_end_ns = ggml_backend_fpga_now_ns();
            instance_gemm_ns = gemm_call_end_ns - gemm_start;
            ctx->total_gemm_ns += instance_gemm_ns;

            const int64_t tiling_stats_start_ns = ggml_backend_fpga_now_ns();
            ggml_backend_fpga_accumulate_tiling_stats(ctx, tile_stats);
            instance_tiling_stats_ns = ggml_backend_fpga_now_ns() - tiling_stats_start_ns;
            ctx->tiling_stats_accum_ns += instance_tiling_stats_ns;

            const int64_t dequant_start = ggml_backend_fpga_now_ns();
            char * d_base = (char *) dst->data + i12*nb2 + i13*nb3;
#ifndef GGML_FPGA_USE_MOCK
            const bool copy_view_to_scratch_for_dequant =
                d_int32_view.data != NULL &&
                ggml_backend_fpga_dequant_scratch_copy_enabled();
            if (copy_view_to_scratch_for_dequant) {
                d_int32_dequant_scratch.resize((size_t) n);
            }
#endif
            for (int64_t row = 0; row < m; ++row) {
                char * d_row = d_base + row*nb1;
                const float a_scale = have_prepared_activation ?
                    prepared_activation.scales[(size_t) row] :
                    a_qp[(size_t) row].scale;
                const int32_t * d_int32_row =
#ifndef GGML_FPGA_USE_MOCK
                    d_int32_view.data != NULL ?
                    d_int32_view.data + (size_t) row * d_int32_view.stride :
                    d_int32.data() + (size_t) row * n;
#else
                    d_int32.data() + (size_t) row * n;
#endif
#ifndef GGML_FPGA_USE_MOCK
                if (copy_view_to_scratch_for_dequant) {
                    std::memcpy(d_int32_dequant_scratch.data(), d_int32_row, (size_t) n * sizeof(int32_t));
                    d_int32_row = d_int32_dequant_scratch.data();
                }
#endif
                ctx->dequant_path.rows_total++;
                ctx->dequant_path.cols_total += n;
                if (nb0 == (int64_t) sizeof(float)) {
                    ctx->dequant_path.rows_neon++;
                } else {
                    ctx->dequant_path.rows_scalar++;
                }
#ifndef GGML_FPGA_USE_MOCK
                if (copy_view_to_scratch_for_dequant) {
                    ctx->dequant_path.rows_scratch_copy++;
                }
#endif
                ggml_backend_fpga_dequantize_int32_row(
                        d_int32_row,
                        n,
                        a_scale,
                        w_qp_ptr,
                        d_row,
                        nb0);
            }
            instance_dequant_ns = ggml_backend_fpga_now_ns() - dequant_start;
            ctx->total_dequant_ns += instance_dequant_ns;
            if (have_ffn_prepared_activation) {
                const int64_t consume_start_ns = ggml_backend_fpga_now_ns();
                ggml_backend_fpga_ffn_streaming_consume(src1);
                instance_ffn_consume_ns = ggml_backend_fpga_now_ns() - consume_start_ns;
                ctx->ffn_consume_ns += instance_ffn_consume_ns;
            }
            if (need_a_int8 && can_store_reused_activation) {
                ggml_backend_fpga_activation_reuse_store(
                        ctx,
                        src1,
                        a_reuse_base,
                        m,
                        k,
                        a_row_stride,
                        a_int8,
                        a_qp,
                        a_inv_scales);
            }

            const int64_t instance_wall_ns = ggml_backend_fpga_now_ns() - instance_start_ns;
            const int64_t instance_known_ns =
                instance_cache_lookup_ns +
                instance_src0_build_ns +
                instance_prepack_ns +
                instance_gate_ns +
                instance_activation_lookup_ns +
                instance_activation_quant_ns +
                instance_gemm_ns +
                instance_tiling_stats_ns +
                instance_dequant_ns +
                instance_ffn_consume_ns;
            if (instance_wall_ns > instance_known_ns) {
                ctx->mul_mat_instance_admin_ns += instance_wall_ns - instance_known_ns;
            }
        }
    }
}

static const char * ggml_backend_fpga_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return "FPGA";
}

static void ggml_backend_fpga_free(ggml_backend_t backend) {
    auto * ctx = (ggml_backend_fpga_context *) backend->context;
    auto & rejection_stats = ggml_backend_fpga_get_rejection_stats();

    const int64_t total_ns = ctx->total_quantize_ns + ctx->total_gemm_ns + ctx->total_dequant_ns;
    const double quant_ms = ctx->total_quantize_ns / 1e6;
    const double gemm_ms = ctx->total_gemm_ns / 1e6;
    const double dequant_ms = ctx->total_dequant_ns / 1e6;
    const double quant_pct = total_ns > 0 ? 100.0 * ctx->total_quantize_ns / (double) total_ns : 0.0;
    const double gemm_pct = total_ns > 0 ? 100.0 * ctx->total_gemm_ns / (double) total_ns : 0.0;
    const double dequant_pct = total_ns > 0 ? 100.0 * ctx->total_dequant_ns / (double) total_ns : 0.0;
    const double total_hw_calls = (double) ctx->tiling_total_hw_calls;
    const double pct_32x128x128 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_32x128x128 / total_hw_calls : 0.0;
    const double pct_16x128x128 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_16x128x128 / total_hw_calls : 0.0;
    const double pct_1x128x128 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_1x128x128 / total_hw_calls : 0.0;
    const double pct_32x64x64 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_32x64x64 / total_hw_calls : 0.0;
    const double pct_16x64x64 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_16x64x64 / total_hw_calls : 0.0;
    const double pct_1x64x64 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_1x64x64 / total_hw_calls : 0.0;
    const double pct_1x32x32 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_1x32x32 / total_hw_calls : 0.0;
    const double pct_1x16x32 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_1x16x32 / total_hw_calls : 0.0;
    const double pct_32x32x32 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_32x32x32 / total_hw_calls : 0.0;
    const double pct_32x32x16 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_32x32x16 / total_hw_calls : 0.0;
    const double pct_16x32x32 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_16x32x32 / total_hw_calls : 0.0;
    const double pct_32x16x16 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_32x16x16 / total_hw_calls : 0.0;
    const double pct_16x16x16 = total_hw_calls > 0 ? 100.0 * ctx->tiling_calls_16x16x16 / total_hw_calls : 0.0;
    const int64_t total_ab_elements = ctx->tiling_total_useful_elements + ctx->tiling_total_padded_elements;
    const double useful_pct = total_ab_elements > 0 ? 100.0 * ctx->tiling_total_useful_elements / (double) total_ab_elements : 0.0;
    const double padded_pct = total_ab_elements > 0 ? 100.0 * ctx->tiling_total_padded_elements / (double) total_ab_elements : 0.0;
    const int64_t tiling_total_ns =
            ctx->tiling_zero_ns +
            ctx->tiling_copy_a_ns +
            ctx->tiling_copy_b_ns +
            ctx->tiling_copy_c_ns +
            ctx->tiling_tile_call_ns +
            ctx->tiling_copy_out_partial_ns +
            ctx->tiling_copy_out_final_ns;
    size_t weight_cache_int8_bytes = 0;
    size_t weight_cache_scale_bytes = 0;
    size_t weight_cache_packed_b_bytes = 0;
    size_t weight_cache_packed_b_tiles = 0;
    for (const auto & entry : ctx->weight_cache) {
        weight_cache_int8_bytes += entry.second.int8_data.size() * sizeof(int8_t);
        weight_cache_int8_bytes += entry.second.int8_data_kn.size() * sizeof(int8_t);
        weight_cache_scale_bytes += entry.second.qp.size() * sizeof(float);
        for (const auto & packed_entry : entry.second.packed_b_cache.cache) {
            weight_cache_packed_b_tiles++;
            weight_cache_packed_b_bytes += packed_entry.second.size() * sizeof(uint64_t);
        }
    }
    const double weight_cache_int8_mb = weight_cache_int8_bytes / (1024.0 * 1024.0);
    const double weight_cache_scale_mb = weight_cache_scale_bytes / (1024.0 * 1024.0);
    const double weight_cache_packed_b_mb = weight_cache_packed_b_bytes / (1024.0 * 1024.0);
    const tpu_runtime_stats_t runtime_stats = ctx->runtime_stats_accumulated;
    const double readback_count = (double) runtime_stats.prevd_host_d_readback_count;
    const double runtime_read_bytes_mb = runtime_stats.total_read_bytes / (1024.0 * 1024.0);
    const double avg_runtime_read_us = readback_count > 0.0 ? runtime_stats.total_read_us / readback_count : 0.0;
    const double avg_runtime_read_sync_us = readback_count > 0.0 ? runtime_stats.total_read_sync_us / readback_count : 0.0;
    const double avg_runtime_read_copy_us = readback_count > 0.0 ? runtime_stats.total_read_copy_us / readback_count : 0.0;
    const double avg_runtime_read_unpack_us = readback_count > 0.0 ? runtime_stats.total_read_unpack_us / readback_count : 0.0;
    const double avg_runtime_read_mmio_us = readback_count > 0.0 ? runtime_stats.total_read_mmio_us / readback_count : 0.0;
    const double avg_runtime_read_dump_us = readback_count > 0.0 ? runtime_stats.total_read_debug_dump_us / readback_count : 0.0;
    const double avg_runtime_read_bytes = readback_count > 0.0 ? runtime_stats.total_read_bytes / readback_count : 0.0;
    const double runtime_read_mb_per_s = runtime_stats.total_read_us > 0 ? (runtime_read_bytes_mb * 1000000.0) / runtime_stats.total_read_us : 0.0;
    const double read_op_batch_bytes_mb = runtime_stats.read_op_batch_bytes / (1024.0 * 1024.0);
    const double read_async_bytes_mb = runtime_stats.read_async_bytes / (1024.0 * 1024.0);
    const double read_final_desc_bytes_mb = runtime_stats.read_final_desc_bytes / (1024.0 * 1024.0);
    const uint64_t read_source_total_us =
        runtime_stats.read_op_batch_total_us +
        runtime_stats.read_async_total_us +
        runtime_stats.read_final_desc_total_us;
    const uint64_t read_source_sync_us =
        runtime_stats.read_op_batch_sync_us +
        runtime_stats.read_async_sync_us +
        runtime_stats.read_final_desc_sync_us;
    const uint64_t read_source_copy_us =
        runtime_stats.read_op_batch_copy_us +
        runtime_stats.read_async_copy_us +
        runtime_stats.read_final_desc_copy_us;
    const uint64_t read_source_bytes =
        runtime_stats.read_op_batch_bytes +
        runtime_stats.read_async_bytes +
        runtime_stats.read_final_desc_bytes;
    const uint64_t read_source_total_gap_us =
        runtime_stats.total_read_us >= read_source_total_us ? runtime_stats.total_read_us - read_source_total_us : 0;
    const uint64_t read_source_sync_gap_us =
        runtime_stats.total_read_sync_us >= read_source_sync_us ? runtime_stats.total_read_sync_us - read_source_sync_us : 0;
    const uint64_t read_source_copy_gap_us =
        runtime_stats.total_read_copy_us >= read_source_copy_us ? runtime_stats.total_read_copy_us - read_source_copy_us : 0;
    const double read_copy_useful_mb = runtime_stats.read_copy_useful_bytes / (1024.0 * 1024.0);
    const double read_copy_transfer_mb = runtime_stats.read_copy_transfer_bytes / (1024.0 * 1024.0);
    const double read_copy_bulk_candidate_mb = runtime_stats.read_copy_bulk_candidate_bytes / (1024.0 * 1024.0);
    const double read_copy_bulk_memcpy_mb = runtime_stats.read_copy_bulk_memcpy_bytes / (1024.0 * 1024.0);
    const double read_copy_useful_ratio = runtime_stats.read_copy_transfer_bytes > 0 ?
        100.0 * runtime_stats.read_copy_useful_bytes / (double) runtime_stats.read_copy_transfer_bytes : 0.0;
    const double read_copy_bulk_candidate_ratio = runtime_stats.read_copy_job_count > 0 ?
        100.0 * runtime_stats.read_copy_bulk_candidate_jobs / (double) runtime_stats.read_copy_job_count : 0.0;
    const double read_copy_bulk_memcpy_ratio = runtime_stats.read_copy_job_count > 0 ?
        100.0 * runtime_stats.read_copy_bulk_memcpy_jobs / (double) runtime_stats.read_copy_job_count : 0.0;
    const double read_copy_strided_ratio = runtime_stats.read_copy_job_count > 0 ?
        100.0 * runtime_stats.read_copy_strided_jobs / (double) runtime_stats.read_copy_job_count : 0.0;
    const double read_copy_padded_ratio = runtime_stats.read_copy_job_count > 0 ?
        100.0 * runtime_stats.read_copy_padded_jobs / (double) runtime_stats.read_copy_job_count : 0.0;
    const double read_copy_avg_rows = runtime_stats.read_copy_job_count > 0 ?
        runtime_stats.read_copy_row_count / (double) runtime_stats.read_copy_job_count : 0.0;
    const double read_copy_avg_cols = runtime_stats.read_copy_row_count > 0 ?
        runtime_stats.read_copy_elem_count / (double) runtime_stats.read_copy_row_count : 0.0;
    const double read_copy_effective_mb_per_s = runtime_stats.total_read_copy_us > 0 ?
        (read_copy_useful_mb * 1000000.0) / runtime_stats.total_read_copy_us : 0.0;
    const double avg_tiling_copy_out_partial_us = readback_count > 0.0 ? (ctx->tiling_copy_out_partial_ns / 1000.0) / readback_count : 0.0;
    const double avg_tiling_copy_out_final_us = readback_count > 0.0 ? (ctx->tiling_copy_out_final_ns / 1000.0) / readback_count : 0.0;
    const double vanilla_equiv_readback_us = runtime_stats.total_read_us + ctx->tiling_copy_out_partial_ns / 1000.0;
    const double avg_vanilla_equiv_readback_us = readback_count > 0.0 ? vanilla_equiv_readback_us / readback_count : 0.0;
    const double direct_final_ratio = readback_count > 0.0 ? (100.0 * runtime_stats.prevd_direct_final_readback_count / readback_count) : 0.0;
    const double runtime_tile_count = runtime_stats.tile_call_count > 0 ? (double) runtime_stats.tile_call_count : 0.0;
    const double avg_runtime_prepare_us = runtime_tile_count > 0.0 ? runtime_stats.total_prepare_us / runtime_tile_count : 0.0;
    const double avg_runtime_pack_us = runtime_tile_count > 0.0 ? runtime_stats.total_pack_us / runtime_tile_count : 0.0;
    const double avg_runtime_pack_a_us = runtime_tile_count > 0.0 ? runtime_stats.total_pack_a_us / runtime_tile_count : 0.0;
    const double avg_runtime_pack_b_us = runtime_tile_count > 0.0 ? runtime_stats.total_pack_b_us / runtime_tile_count : 0.0;
    const double avg_runtime_copy_us = runtime_tile_count > 0.0 ? runtime_stats.total_copy_us / runtime_tile_count : 0.0;
    const double avg_runtime_copy_a_us = runtime_tile_count > 0.0 ? runtime_stats.total_copy_a_us / runtime_tile_count : 0.0;
    const double avg_runtime_copy_b_us = runtime_tile_count > 0.0 ? runtime_stats.total_copy_b_us / runtime_tile_count : 0.0;
    const double avg_runtime_control_us = runtime_tile_count > 0.0 ? runtime_stats.total_control_us / runtime_tile_count : 0.0;
    const double avg_runtime_preload_us = runtime_tile_count > 0.0 ? runtime_stats.total_preload_us / runtime_tile_count : 0.0;
    const double avg_runtime_start_us = runtime_tile_count > 0.0 ? runtime_stats.total_start_us / runtime_tile_count : 0.0;
    const double avg_runtime_wait_prev_us = runtime_tile_count > 0.0 ? runtime_stats.total_wait_prev_us / runtime_tile_count : 0.0;
    const double avg_runtime_wait_final_us = runtime_tile_count > 0.0 ? runtime_stats.total_wait_final_us / runtime_tile_count : 0.0;
    const double avg_runtime_cleanup_us = runtime_tile_count > 0.0 ? runtime_stats.total_cleanup_us / runtime_tile_count : 0.0;
    const double avg_runtime_other_us = runtime_tile_count > 0.0 ? runtime_stats.total_other_us / runtime_tile_count : 0.0;

#ifdef FPGA_TILING_DEBUG
    if (ggml_backend_fpga_log_summary_enabled()) {
        GGML_LOG_INFO("%s Tiling Debug is on\n", ggml_backend_fpga_log_prefix());
    }
#endif

    if (!ggml_backend_fpga_log_summary_enabled()) {
        tpu_runtime_cleanup();
        delete ctx;
        delete backend;
        return;
    }

    GGML_LOG_INFO("%s Stats:\n", ggml_backend_fpga_log_prefix());
    GGML_LOG_INFO("  mul_mat calls: %" PRId64 "\n", ctx->n_mul_mat_calls);
    GGML_LOG_INFO("  gemm instances: %" PRId64 "\n", ctx->n_gemm_instances);
    GGML_LOG_INFO("  empty mul_mat skipped: %" PRId64 "\n", ctx->n_empty_mul_mat_skipped);
    GGML_LOG_INFO("  empty gemm instances skipped: %" PRId64 "\n", ctx->n_empty_gemm_instances_skipped);
    GGML_LOG_INFO("  quantize: %.3fms (%.0f%%)  gemm: %.3fms (%.0f%%)  dequant: %.3fms (%.0f%%)\n",
            quant_ms, quant_pct, gemm_ms, gemm_pct, dequant_ms, dequant_pct);
    if (ctx->n_mul_mat_calls > 0) {
        GGML_LOG_INFO("  matrix sizes: K=[%" PRId64 "..%" PRId64 "], N=[%" PRId64 "..%" PRId64 "], M=[%" PRId64 "..%" PRId64 "]\n",
                ctx->min_k, ctx->max_k, ctx->min_n, ctx->max_n, ctx->min_m, ctx->max_m);
        ggml_backend_fpga_log_value_counts("K", ctx->k_value_counts);
        ggml_backend_fpga_log_value_counts("N", ctx->n_value_counts);
        ggml_backend_fpga_log_value_counts("M", ctx->m_value_counts);
        ggml_backend_fpga_log_shape_counts("executed gemm shapes", ctx->gemm_shape_counts);
    } else {
        GGML_LOG_INFO("  matrix sizes: none\n");
    }
    if (ctx->n_empty_mul_mat_skipped > 0) {
        ggml_backend_fpga_log_shape_counts("empty gemm shapes skipped", ctx->empty_gemm_shape_counts);
    }
    GGML_LOG_INFO("%s Weight cache stats:\n", ggml_backend_fpga_log_prefix());
    GGML_LOG_INFO("  entries: %zu\n", ctx->weight_cache.size());
    GGML_LOG_INFO("  memory: %.1f MB (INT8 weights) + %.1f MB (scales)\n",
            weight_cache_int8_mb, weight_cache_scale_mb);
    GGML_LOG_INFO("  packed B tiles: %zu (%.1f MB device-packed cache)\n",
            weight_cache_packed_b_tiles, weight_cache_packed_b_mb);
    GGML_LOG_INFO("  build time: %.3f ms\n", ctx->weight_cache_build_ns / 1e6);
    GGML_LOG_INFO("  hits: %" PRId64 "   misses: %" PRId64 "   dynamic bypasses: %" PRId64 "\n",
            ctx->weight_cache_hits, ctx->weight_cache_misses, ctx->weight_cache_bypasses);
    GGML_LOG_INFO("  dynamic src0 cache: enabled=%d hits=%" PRId64 " misses=%" PRId64
            " stores=%" PRId64 " clears=%" PRId64 " resident=%zu store_bytes=%.3f MiB\n",
            ggml_backend_fpga_dynamic_src0_cache_enabled() ? 1 : 0,
            ctx->dynamic_src0_cache_hits,
            ctx->dynamic_src0_cache_misses,
            ctx->dynamic_src0_cache_stores,
            ctx->dynamic_src0_cache_clears,
            ctx->dynamic_src0_cache.size(),
            ctx->dynamic_src0_cache_store_bytes / (1024.0 * 1024.0));
    GGML_LOG_INFO("  dynamic src0 prepared: enabled=%d hits=%" PRId64 " misses=%" PRId64
            " import=%.3f ms\n",
            ggml_backend_fpga_dynamic_src0_prepared_enabled() ? 1 : 0,
            ctx->dynamic_src0_prepared_hits,
            ctx->dynamic_src0_prepared_misses,
            ctx->dynamic_src0_prepared_import_ns / 1e6);
    GGML_LOG_INFO("  dynamic src0 prepared check: check_only=%d entries=%" PRId64
            " int8_bad_entries=%" PRId64 " int8_bad_values=%" PRId64
            " max_i8_abs_diff=%" PRId64 " scale_bad_entries=%" PRId64
            " scale_bad_rows=%" PRId64 " max_scale_abs_diff=%.9g\n",
            ggml_backend_fpga_dynamic_src0_prepared_check_only_enabled() ? 1 : 0,
            ctx->dynamic_src0_prepared_check_entries,
            ctx->dynamic_src0_prepared_check_int8_bad_entries,
            ctx->dynamic_src0_prepared_check_int8_bad_values,
            ctx->dynamic_src0_prepared_check_max_i8_abs_diff,
            ctx->dynamic_src0_prepared_check_scale_bad_entries,
            ctx->dynamic_src0_prepared_check_scale_bad_rows,
            ctx->dynamic_src0_prepared_check_max_scale_abs_diff);
    GGML_LOG_INFO("  dynamic src0 reasons: total=%" PRId64 " op_non_none=%" PRId64
            " view_src=%" PRId64 " input_flag=%" PRId64 "\n",
            ctx->dynamic_src0_stats.total,
            ctx->dynamic_src0_stats.op_non_none,
            ctx->dynamic_src0_stats.view_src,
            ctx->dynamic_src0_stats.input_flag);
    ggml_backend_fpga_log_shape_counts("dynamic src0 shapes", ctx->dynamic_src0_stats.shape_counts);
    ggml_backend_fpga_log_string_counts("dynamic src0 ops", ctx->dynamic_src0_stats.op_counts);
    ggml_backend_fpga_log_string_counts("dynamic src0 names", ctx->dynamic_src0_stats.name_counts);
    if (ggml_backend_fpga_matrix_task_shape_stats_enabled()) {
        ggml_backend_fpga_log_matrix_task_shape_stats(ctx->matrix_task_shape_stats);
    }
    GGML_LOG_INFO("%s Timing breakdown:\n", ggml_backend_fpga_log_prefix());
    GGML_LOG_INFO("  weight cache build: %.3f ms (one-time)\n", ctx->weight_cache_build_ns / 1e6);
    GGML_LOG_INFO("  dynamic src0 quantize: %.3f ms (not cached)\n", ctx->dynamic_src0_build_ns / 1e6);
    GGML_LOG_INFO("  activation quantize: %.3f ms (per-call total)\n", ctx->activation_quantize_ns / 1e6);
    GGML_LOG_INFO("  tiled gemm: %.3f ms\n", ctx->total_gemm_ns / 1e6);
    GGML_LOG_INFO("  dequant output: %.3f ms\n", ctx->total_dequant_ns / 1e6);
    GGML_LOG_INFO("  dequant paths: rows=%" PRId64 " neon=%" PRId64 " scalar=%" PRId64
            " scratch_copy=%" PRId64 " avg_cols=%.2f\n",
            ctx->dequant_path.rows_total,
            ctx->dequant_path.rows_neon,
            ctx->dequant_path.rows_scalar,
            ctx->dequant_path.rows_scratch_copy,
            ctx->dequant_path.rows_total > 0 ?
                (double) ctx->dequant_path.cols_total / (double) ctx->dequant_path.rows_total : 0.0);
    GGML_LOG_INFO("  P23b FFN streaming cache: hits=%" PRId64 " misses=%" PRId64 "\n",
            ctx->ffn_stream_cache_hits,
            ctx->ffn_stream_cache_misses);
    GGML_LOG_INFO("  activation sibling reuse: enabled=%d attempts=%" PRId64
            " hits=%" PRId64 " misses=%" PRId64 " stores=%" PRId64
            " clears=%" PRId64 " store=%.3f ms store_bytes=%.3f MiB\n",
            ggml_backend_fpga_activation_reuse_sibling_enabled() ? 1 : 0,
            ctx->activation_reuse_attempts,
            ctx->activation_reuse_hits,
            ctx->activation_reuse_misses,
            ctx->activation_reuse_stores,
            ctx->activation_reuse_clears,
            ctx->activation_reuse_store_ns / 1e6,
            ctx->activation_reuse_store_bytes / (1024.0 * 1024.0));
    GGML_LOG_INFO("  FFN gate/up dual macro-op: enabled=%d attempts=%" PRId64
            " hits=%" PRId64 " misses=%" PRId64 "\n",
            ggml_backend_fpga_ffn_sibling_dual_enabled() ? 1 : 0,
            ctx->ffn_dual_attempts,
            ctx->ffn_dual_hits,
            ctx->ffn_dual_misses);
    GGML_LOG_INFO("  tiling phases: zero=%.3f ms copy_a=%.3f ms copy_b=%.3f ms copy_c=%.3f ms tile_call=%.3f ms copy_out_partial=%.3f ms copy_out_final=%.3f ms total=%.3f ms\n",
            ctx->tiling_zero_ns / 1e6,
            ctx->tiling_copy_a_ns / 1e6,
            ctx->tiling_copy_b_ns / 1e6,
            ctx->tiling_copy_c_ns / 1e6,
            ctx->tiling_tile_call_ns / 1e6,
            ctx->tiling_copy_out_partial_ns / 1e6,
            ctx->tiling_copy_out_final_ns / 1e6,
            tiling_total_ns / 1e6);
    GGML_LOG_INFO("  runtime phases: calls=%" PRIu64 " prepare=%.3f ms pack=%.3f ms pack_a=%.3f ms pack_b=%.3f ms copy=%.3f ms copy_a=%.3f ms copy_b=%.3f ms control=%.3f ms preload=%.3f ms start=%.3f ms wait_prev=%.3f ms wait_final=%.3f ms sequence=%.3f ms read=%.3f ms cleanup=%.3f ms log=%.3f ms unlock=%.3f ms other=%.3f ms\n",
            runtime_stats.tile_call_count,
            runtime_stats.total_prepare_us / 1000.0,
            runtime_stats.total_pack_us / 1000.0,
            runtime_stats.total_pack_a_us / 1000.0,
            runtime_stats.total_pack_b_us / 1000.0,
            runtime_stats.total_copy_us / 1000.0,
            runtime_stats.total_copy_a_us / 1000.0,
            runtime_stats.total_copy_b_us / 1000.0,
            runtime_stats.total_control_us / 1000.0,
            runtime_stats.total_preload_us / 1000.0,
            runtime_stats.total_start_us / 1000.0,
            runtime_stats.total_wait_prev_us / 1000.0,
            runtime_stats.total_wait_final_us / 1000.0,
            runtime_stats.total_sequence_us / 1000.0,
            runtime_stats.total_read_us / 1000.0,
            runtime_stats.total_cleanup_us / 1000.0,
            runtime_stats.total_log_us / 1000.0,
            runtime_stats.total_unlock_us / 1000.0,
            runtime_stats.total_other_us / 1000.0);
    GGML_LOG_INFO("  runtime phase averages: prepare=%.3f us pack=%.3f us pack_a=%.3f us pack_b=%.3f us copy=%.3f us copy_a=%.3f us copy_b=%.3f us control=%.3f us preload=%.3f us start=%.3f us wait_prev=%.3f us wait_final=%.3f us cleanup=%.3f us other=%.3f us\n",
            avg_runtime_prepare_us,
            avg_runtime_pack_us,
            avg_runtime_pack_a_us,
            avg_runtime_pack_b_us,
            avg_runtime_copy_us,
            avg_runtime_copy_a_us,
            avg_runtime_copy_b_us,
            avg_runtime_control_us,
            avg_runtime_preload_us,
            avg_runtime_start_us,
            avg_runtime_wait_prev_us,
            avg_runtime_wait_final_us,
            avg_runtime_cleanup_us,
            avg_runtime_other_us);
    GGML_LOG_INFO("  runtime start/preload split: preload_sync=%.3f ms preload_wait=%.3f ms preload_setup=%.3f ms preload_xfer=%.3f ms start_sync=%.3f ms start_ready=%.3f ms start_setup=%.3f ms start_mmio=%.3f ms start_xfer=%.3f ms wait_done=%.3f ms\n",
            runtime_stats.total_preload_sync_us / 1000.0,
            runtime_stats.total_preload_wait_us / 1000.0,
            runtime_stats.total_preload_setup_us / 1000.0,
            runtime_stats.total_preload_xfer_us / 1000.0,
            runtime_stats.total_start_sync_us / 1000.0,
            runtime_stats.total_start_ready_us / 1000.0,
            runtime_stats.total_start_setup_us / 1000.0,
            runtime_stats.total_start_mmio_us / 1000.0,
            runtime_stats.total_start_xfer_us / 1000.0,
            runtime_stats.total_wait_done_us / 1000.0);
    GGML_LOG_INFO("  runtime sequences: sequence_program=%" PRIu64 " sequence_fallback=%" PRIu64 " sequence_sync=%.3f ms sequence_ready=%.3f ms sequence_program_time=%.3f ms sequence_wait_done=%.3f ms auto_next_program=%" PRIu64 " auto_next_fallback=%" PRIu64 "\n",
            runtime_stats.prevd_sequence_program_count,
            runtime_stats.prevd_sequence_fallback_count,
            runtime_stats.total_sequence_sync_us / 1000.0,
            runtime_stats.total_sequence_ready_us / 1000.0,
            runtime_stats.total_sequence_program_us / 1000.0,
            runtime_stats.total_sequence_wait_done_us / 1000.0,
            runtime_stats.auto_next_program_count,
            runtime_stats.auto_next_fallback_count);
    GGML_LOG_INFO("  runtime matmul_task: count=%" PRIu64 " fallback=%" PRIu64 " tiles=%" PRIu64 " r5_cycles=%" PRIu64 " fpga_wait_cycles=%" PRIu64 " output_sync=%.3f ms output_copy=%.3f ms b_packed_arena_hits=%" PRIu64 " b_packed_arena_misses=%" PRIu64 "\n",
            runtime_stats.matmul_task_count,
            runtime_stats.matmul_task_fallback_count,
            runtime_stats.matmul_task_tile_count,
            runtime_stats.matmul_task_r5_cycles,
            runtime_stats.matmul_task_fpga_wait_cycles,
            runtime_stats.matmul_task_output_sync_us / 1000.0,
            runtime_stats.matmul_task_output_copy_us / 1000.0,
            runtime_stats.b_packed_arena_hits,
            runtime_stats.b_packed_arena_misses);
    GGML_LOG_INFO("  runtime matmul_task wall: total=%.3f ms entry_setup=%.3f ms lock=%.3f ms reserve=%.3f ms prepare_b_wall=%.3f ms copy_a=%.3f ms sync_a=%.3f ms addr_check=%.3f ms submit=%.3f ms output_sync=%.3f ms output_view=%.3f ms output_copy=%.3f ms post=%.3f ms post_stats=%.3f ms free=%.3f ms residual=%.3f ms\n",
            runtime_stats.matmul_task_total_us / 1000.0,
            runtime_stats.matmul_task_entry_setup_us / 1000.0,
            runtime_stats.matmul_task_lock_us / 1000.0,
            runtime_stats.matmul_task_reserve_us / 1000.0,
            runtime_stats.matmul_task_prepare_b_wall_us / 1000.0,
            runtime_stats.matmul_task_copy_a_wall_us / 1000.0,
            runtime_stats.matmul_task_sync_a_us / 1000.0,
            runtime_stats.matmul_task_addr_check_us / 1000.0,
            runtime_stats.matmul_task_submit_us / 1000.0,
            runtime_stats.matmul_task_output_sync_us / 1000.0,
            runtime_stats.matmul_task_output_view_us / 1000.0,
            runtime_stats.matmul_task_output_copy_us / 1000.0,
            runtime_stats.matmul_task_post_us / 1000.0,
            runtime_stats.matmul_task_post_stats_us / 1000.0,
            runtime_stats.matmul_task_free_us / 1000.0,
            runtime_stats.matmul_task_residual_us / 1000.0);
    GGML_LOG_INFO("  runtime matmul_task output_copy detail: jobs=%" PRIu64 " bytes=%.3f MiB direct=%" PRIu64 "/%.3f ms/%.3f MiB row=%" PRIu64 "/%.3f ms/%.3f MiB rows=%" PRIu64 " padded=%" PRIu64 "/%.3f ms M1=%" PRIu64 "/%.3f ms M32=%" PRIu64 "/%.3f ms N<=64=%" PRIu64 "/%.3f ms N128=%" PRIu64 "/%.3f ms Nother=%" PRIu64 "/%.3f ms\n",
            runtime_stats.matmul_task_output_copy_jobs,
            runtime_stats.matmul_task_output_copy_bytes / (1024.0 * 1024.0),
            runtime_stats.matmul_task_output_copy_direct_jobs,
            runtime_stats.matmul_task_output_copy_direct_us / 1000.0,
            runtime_stats.matmul_task_output_copy_direct_bytes / (1024.0 * 1024.0),
            runtime_stats.matmul_task_output_copy_row_jobs,
            runtime_stats.matmul_task_output_copy_row_us / 1000.0,
            runtime_stats.matmul_task_output_copy_row_bytes / (1024.0 * 1024.0),
            runtime_stats.matmul_task_output_copy_row_rows,
            runtime_stats.matmul_task_output_copy_padded_jobs,
            runtime_stats.matmul_task_output_copy_padded_us / 1000.0,
            runtime_stats.matmul_task_output_copy_m1_jobs,
            runtime_stats.matmul_task_output_copy_m1_us / 1000.0,
            runtime_stats.matmul_task_output_copy_m32_jobs,
            runtime_stats.matmul_task_output_copy_m32_us / 1000.0,
            runtime_stats.matmul_task_output_copy_n64_jobs,
            runtime_stats.matmul_task_output_copy_n64_us / 1000.0,
            runtime_stats.matmul_task_output_copy_n128_jobs,
            runtime_stats.matmul_task_output_copy_n128_us / 1000.0,
            runtime_stats.matmul_task_output_copy_n_other_jobs,
            runtime_stats.matmul_task_output_copy_n_other_us / 1000.0);
    GGML_LOG_INFO("  runtime pl_op_desc: count=%" PRIu64 " tiles=%" PRIu64
            " program=%.3f ms wait=%.3f ms polls=%" PRIu64 " avg_wait=%.3f us\n",
            runtime_stats.pl_op_desc_count,
            runtime_stats.pl_op_desc_tile_count,
            runtime_stats.pl_op_desc_program_us / 1000.0,
            runtime_stats.pl_op_desc_wait_us / 1000.0,
            runtime_stats.pl_op_desc_status_polls,
            runtime_stats.pl_op_desc_count > 0 ?
                (double) runtime_stats.pl_op_desc_wait_us / (double) runtime_stats.pl_op_desc_count : 0.0);
    GGML_LOG_INFO("  runtime readback: count=%" PRIu64 " total=%.3f ms sync=%.3f ms copy=%.3f ms unpack=%.3f ms mmio=%.3f ms debug_dump=%.3f ms bytes=%.3f MiB direct_final=%" PRIu64 " (%.1f%%) paths(outbuf/memcpy/unpack/mmio)=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
            runtime_stats.prevd_host_d_readback_count,
            runtime_stats.total_read_us / 1000.0,
            runtime_stats.total_read_sync_us / 1000.0,
            runtime_stats.total_read_copy_us / 1000.0,
            runtime_stats.total_read_unpack_us / 1000.0,
            runtime_stats.total_read_mmio_us / 1000.0,
            runtime_stats.total_read_debug_dump_us / 1000.0,
            runtime_read_bytes_mb,
            runtime_stats.prevd_direct_final_readback_count,
            direct_final_ratio,
            runtime_stats.total_read_output_buffer_count,
            runtime_stats.total_read_memcpy_count,
            runtime_stats.total_read_unpack_count,
            runtime_stats.total_read_mmio_count);
    GGML_LOG_INFO("  runtime readback sources: op_batch count=%" PRIu64 " total=%.3f ms sync=%.3f ms copy=%.3f ms bytes=%.3f MiB | async count=%" PRIu64 " total=%.3f ms sync=%.3f ms copy=%.3f ms bytes=%.3f MiB | final_desc count=%" PRIu64 " total=%.3f ms sync=%.3f ms copy=%.3f ms unpack=%.3f ms mmio=%.3f ms bytes=%.3f MiB\n",
            runtime_stats.read_op_batch_count,
            runtime_stats.read_op_batch_total_us / 1000.0,
            runtime_stats.read_op_batch_sync_us / 1000.0,
            runtime_stats.read_op_batch_copy_us / 1000.0,
            read_op_batch_bytes_mb,
            runtime_stats.read_async_count,
            runtime_stats.read_async_total_us / 1000.0,
            runtime_stats.read_async_sync_us / 1000.0,
            runtime_stats.read_async_copy_us / 1000.0,
            read_async_bytes_mb,
            runtime_stats.read_final_desc_count,
            runtime_stats.read_final_desc_total_us / 1000.0,
            runtime_stats.read_final_desc_sync_us / 1000.0,
            runtime_stats.read_final_desc_copy_us / 1000.0,
            runtime_stats.read_final_desc_unpack_us / 1000.0,
            runtime_stats.read_final_desc_mmio_us / 1000.0,
            read_final_desc_bytes_mb);
    GGML_LOG_INFO("  runtime readback async exposed_wait: begin=%.3f ms (%" PRIu64 ") stage=%.3f ms (%" PRIu64 ") flush=%.3f ms (%" PRIu64 ") total=%.3f ms\n",
            runtime_stats.read_async_batch_begin_wait_us / 1000.0,
            runtime_stats.read_async_batch_begin_wait_count,
            runtime_stats.read_async_stage_wait_us / 1000.0,
            runtime_stats.read_async_stage_wait_count,
            runtime_stats.read_async_flush_wait_us / 1000.0,
            runtime_stats.read_async_flush_wait_count,
            (runtime_stats.read_async_batch_begin_wait_us +
             runtime_stats.read_async_stage_wait_us +
             runtime_stats.read_async_flush_wait_us) / 1000.0);
    GGML_LOG_INFO("  runtime readback source_check: source_total=%.3f ms total_gap=%.3f ms source_sync=%.3f ms sync_gap=%.3f ms source_copy=%.3f ms copy_gap=%.3f ms source_bytes=%.3f MiB byte_gap=%" PRIu64 " B\n",
            read_source_total_us / 1000.0,
            read_source_total_gap_us / 1000.0,
            read_source_sync_us / 1000.0,
            read_source_sync_gap_us / 1000.0,
            read_source_copy_us / 1000.0,
            read_source_copy_gap_us / 1000.0,
            read_source_bytes / (1024.0 * 1024.0),
            runtime_stats.total_read_bytes >= read_source_bytes ? runtime_stats.total_read_bytes - read_source_bytes : 0);
    GGML_LOG_INFO("  runtime readback copy shapes: jobs=%" PRIu64 " rows=%" PRIu64
            " avg_rows=%.2f avg_cols=%.2f useful=%.3f MiB transfer=%.3f MiB useful_ratio=%.1f%%"
            " padded_jobs=%" PRIu64 " (%.1f%%) strided_jobs=%" PRIu64 " (%.1f%%)"
            " bulk_candidate=%" PRIu64 " (%.1f%%, %.3f MiB) bulk_memcpy=%" PRIu64 " (%.1f%%, %.3f MiB)"
            " row_memcpy_rows=%" PRIu64 " effective_copy_bw=%.3f MiB/s\n",
            runtime_stats.read_copy_job_count,
            runtime_stats.read_copy_row_count,
            read_copy_avg_rows,
            read_copy_avg_cols,
            read_copy_useful_mb,
            read_copy_transfer_mb,
            read_copy_useful_ratio,
            runtime_stats.read_copy_padded_jobs,
            read_copy_padded_ratio,
            runtime_stats.read_copy_strided_jobs,
            read_copy_strided_ratio,
            runtime_stats.read_copy_bulk_candidate_jobs,
            read_copy_bulk_candidate_ratio,
            read_copy_bulk_candidate_mb,
            runtime_stats.read_copy_bulk_memcpy_jobs,
            read_copy_bulk_memcpy_ratio,
            read_copy_bulk_memcpy_mb,
            runtime_stats.read_copy_row_memcpy_rows,
            read_copy_effective_mb_per_s);
    GGML_LOG_INFO("  runtime readback copy bins: rows[1/2-15/16/17-31/32/>32]=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
            " cols[<=16/<=32/<=64/65-127/128/>128]=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
            runtime_stats.read_copy_rows_1,
            runtime_stats.read_copy_rows_2_15,
            runtime_stats.read_copy_rows_16,
            runtime_stats.read_copy_rows_17_31,
            runtime_stats.read_copy_rows_32,
            runtime_stats.read_copy_rows_gt32,
            runtime_stats.read_copy_cols_le16,
            runtime_stats.read_copy_cols_le32,
            runtime_stats.read_copy_cols_le64,
            runtime_stats.read_copy_cols_65_127,
            runtime_stats.read_copy_cols_128,
            runtime_stats.read_copy_cols_gt128);
    GGML_LOG_INFO("  readback averages: runtime_total=%.3f us sync=%.3f us copy=%.3f us unpack=%.3f us mmio=%.3f us debug_dump=%.3f us bytes=%.0f B throughput=%.3f MiB/s vanilla_equiv=%.3f us llama_host_partial=%.3f us llama_host_final=%.3f us direct_final=%.1f%%\n",
            avg_runtime_read_us,
            avg_runtime_read_sync_us,
            avg_runtime_read_copy_us,
            avg_runtime_read_unpack_us,
            avg_runtime_read_mmio_us,
            avg_runtime_read_dump_us,
            avg_runtime_read_bytes,
            runtime_read_mb_per_s,
            avg_vanilla_equiv_readback_us,
            avg_tiling_copy_out_partial_us,
            avg_tiling_copy_out_final_us,
            direct_final_ratio);
    GGML_LOG_INFO("%s FPGA graph wall split:\n", ggml_backend_fpga_log_prefix());
    for (int phase_id = 0; phase_id < GGML_BACKEND_FPGA_PHASE_COUNT; ++phase_id) {
        const ggml_backend_fpga_graph_wall_stats & phase_stats = ctx->graph_wall_stats[phase_id];
        if (phase_stats.graphs == 0) {
            continue;
        }
        const tpu_runtime_stats_t & phase_rt = phase_stats.runtime;
        const uint64_t phase_exposed_read_us =
            phase_rt.read_async_batch_begin_wait_us +
            phase_rt.read_async_stage_wait_us +
            phase_rt.read_async_flush_wait_us;
        GGML_LOG_INFO("  [%s] graphs=%" PRIu64 " nodes=%" PRIu64
                " wall=%.3f ms mul_mat_wall=%.3f ms add_wall=%.3f ms layout_wall=%.3f ms"
                " mul_mat_nodes=%" PRIu64 " add_nodes=%" PRIu64 " layout_nodes=%" PRIu64 "\n",
                ggml_backend_fpga_phase_name((enum ggml_backend_fpga_phase) phase_id),
                phase_stats.graphs,
                phase_stats.nodes,
                phase_stats.wall_ns / 1e6,
                phase_stats.mul_mat_wall_ns / 1e6,
                phase_stats.add_wall_ns / 1e6,
                phase_stats.layout_wall_ns / 1e6,
                phase_stats.mul_mat_nodes,
                phase_stats.add_nodes,
                phase_stats.layout_nodes);
        GGML_LOG_INFO("    mul_mat nested wall: weight_cache=%.3f ms dynamic_src0=%.3f ms"
                " activation=%.3f ms gemm_call=%.3f ms dequant=%.3f ms"
                " setup=%.3f ms cache_lookup=%.3f ms act_src_lookup=%.3f ms"
                " prepack=%.3f ms gates=%.3f ms tiling_stats=%.3f ms"
                " ffn_consume=%.3f ms instance_admin=%.3f ms"
                " residual=%.3f ms nested_overrun=%.3f ms\n",
                phase_stats.weight_cache_ns / 1e6,
                phase_stats.dynamic_src0_ns / 1e6,
                phase_stats.activation_quant_ns / 1e6,
                phase_stats.gemm_call_ns / 1e6,
                phase_stats.dequant_ns / 1e6,
                phase_stats.mul_mat_setup_ns / 1e6,
                phase_stats.weight_cache_lookup_ns / 1e6,
                phase_stats.activation_source_lookup_ns / 1e6,
                phase_stats.matrix_task_prepack_ns / 1e6,
                phase_stats.mul_mat_gate_ns / 1e6,
                phase_stats.tiling_stats_accum_ns / 1e6,
                phase_stats.ffn_consume_ns / 1e6,
                phase_stats.mul_mat_instance_admin_ns / 1e6,
                phase_stats.mul_mat_residual_ns / 1e6,
                phase_stats.mul_mat_nested_overrun_ns / 1e6);
        GGML_LOG_INFO("    runtime nested: calls=%" PRIu64
                " prepare=%.3f ms pack_b=%.3f ms copy_a=%.3f ms copy_b=%.3f ms"
                " sequence=%.3f ms sequence_ready=%.3f ms sequence_program=%.3f ms sequence_wait_done=%.3f ms"
                " pl_wait=%.3f ms output_sync=%.3f ms output_copy=%.3f ms"
                " read_total=%.3f ms read_copy=%.3f ms read_exposed=%.3f ms runtime_other=%.3f ms\n",
                phase_rt.tile_call_count,
                phase_rt.total_prepare_us / 1000.0,
                phase_rt.total_pack_b_us / 1000.0,
                phase_rt.total_copy_a_us / 1000.0,
                phase_rt.total_copy_b_us / 1000.0,
                phase_rt.total_sequence_us / 1000.0,
                phase_rt.total_sequence_ready_us / 1000.0,
                phase_rt.total_sequence_program_us / 1000.0,
                phase_rt.total_sequence_wait_done_us / 1000.0,
                phase_rt.pl_op_desc_wait_us / 1000.0,
                phase_rt.matmul_task_output_sync_us / 1000.0,
                phase_rt.matmul_task_output_copy_us / 1000.0,
                phase_rt.total_read_us / 1000.0,
                phase_rt.total_read_copy_us / 1000.0,
                phase_exposed_read_us / 1000.0,
                phase_rt.total_other_us / 1000.0);
        GGML_LOG_INFO("    matmul_task wall nested: total=%.3f ms entry_setup=%.3f ms"
                " lock=%.3f ms reserve=%.3f ms prepare_b_wall=%.3f ms"
                " copy_a=%.3f ms sync_a=%.3f ms addr_check=%.3f ms"
                " submit=%.3f ms output_sync=%.3f ms output_view=%.3f ms"
                " output_copy=%.3f ms post=%.3f ms post_stats=%.3f ms"
                " free=%.3f ms residual=%.3f ms\n",
                phase_rt.matmul_task_total_us / 1000.0,
                phase_rt.matmul_task_entry_setup_us / 1000.0,
                phase_rt.matmul_task_lock_us / 1000.0,
                phase_rt.matmul_task_reserve_us / 1000.0,
                phase_rt.matmul_task_prepare_b_wall_us / 1000.0,
                phase_rt.matmul_task_copy_a_wall_us / 1000.0,
                phase_rt.matmul_task_sync_a_us / 1000.0,
                phase_rt.matmul_task_addr_check_us / 1000.0,
                phase_rt.matmul_task_submit_us / 1000.0,
                phase_rt.matmul_task_output_sync_us / 1000.0,
                phase_rt.matmul_task_output_view_us / 1000.0,
                phase_rt.matmul_task_output_copy_us / 1000.0,
                phase_rt.matmul_task_post_us / 1000.0,
                phase_rt.matmul_task_post_stats_us / 1000.0,
                phase_rt.matmul_task_free_us / 1000.0,
                phase_rt.matmul_task_residual_us / 1000.0);
        GGML_LOG_INFO("    matmul_task output_copy detail: jobs=%" PRIu64 " bytes=%.3f MiB"
                " direct=%" PRIu64 "/%.3f ms/%.3f MiB row=%" PRIu64 "/%.3f ms/%.3f MiB"
                " rows=%" PRIu64 " padded=%" PRIu64 "/%.3f ms"
                " M1=%" PRIu64 "/%.3f ms M32=%" PRIu64 "/%.3f ms"
                " N<=64=%" PRIu64 "/%.3f ms N128=%" PRIu64 "/%.3f ms"
                " Nother=%" PRIu64 "/%.3f ms\n",
                phase_rt.matmul_task_output_copy_jobs,
                phase_rt.matmul_task_output_copy_bytes / (1024.0 * 1024.0),
                phase_rt.matmul_task_output_copy_direct_jobs,
                phase_rt.matmul_task_output_copy_direct_us / 1000.0,
                phase_rt.matmul_task_output_copy_direct_bytes / (1024.0 * 1024.0),
                phase_rt.matmul_task_output_copy_row_jobs,
                phase_rt.matmul_task_output_copy_row_us / 1000.0,
                phase_rt.matmul_task_output_copy_row_bytes / (1024.0 * 1024.0),
                phase_rt.matmul_task_output_copy_row_rows,
                phase_rt.matmul_task_output_copy_padded_jobs,
                phase_rt.matmul_task_output_copy_padded_us / 1000.0,
                phase_rt.matmul_task_output_copy_m1_jobs,
                phase_rt.matmul_task_output_copy_m1_us / 1000.0,
                phase_rt.matmul_task_output_copy_m32_jobs,
                phase_rt.matmul_task_output_copy_m32_us / 1000.0,
                phase_rt.matmul_task_output_copy_n64_jobs,
                phase_rt.matmul_task_output_copy_n64_us / 1000.0,
                phase_rt.matmul_task_output_copy_n128_jobs,
                phase_rt.matmul_task_output_copy_n128_us / 1000.0,
                phase_rt.matmul_task_output_copy_n_other_jobs,
                phase_rt.matmul_task_output_copy_n_other_us / 1000.0);
    }
    GGML_LOG_INFO("%s Tiling stats:\n", ggml_backend_fpga_log_prefix());
    GGML_LOG_INFO("  tile family: %s (requested max_nk=%u, effective max_nk=%u)\n",
            tpu_tile_family_name(),
            tpu_get_requested_max_nk(),
            tpu_get_effective_max_nk());
    GGML_LOG_INFO("  total hw calls: %" PRId64 "\n", ctx->tiling_total_hw_calls);
    GGML_LOG_INFO("  by config: 32x128x128=%" PRId64 " (%.1f%%)  16x128x128=%" PRId64 " (%.1f%%)  1x128x128=%" PRId64 " (%.1f%%)  32x64x64=%" PRId64 " (%.1f%%)  16x64x64=%" PRId64 " (%.1f%%)  1x64x64=%" PRId64 " (%.1f%%)  1x32x32=%" PRId64 " (%.1f%%)  1x16x32=%" PRId64 " (%.1f%%)  32x32x32=%" PRId64 " (%.1f%%)  32x32x16=%" PRId64 " (%.1f%%)  16x32x32=%" PRId64 " (%.1f%%)  32x16x16=%" PRId64 " (%.1f%%)  16x16x16=%" PRId64 " (%.1f%%)\n",
            ctx->tiling_calls_32x128x128, pct_32x128x128,
            ctx->tiling_calls_16x128x128, pct_16x128x128,
            ctx->tiling_calls_1x128x128, pct_1x128x128,
            ctx->tiling_calls_32x64x64, pct_32x64x64,
            ctx->tiling_calls_16x64x64, pct_16x64x64,
            ctx->tiling_calls_1x64x64, pct_1x64x64,
            ctx->tiling_calls_1x32x32, pct_1x32x32,
            ctx->tiling_calls_1x16x32, pct_1x16x32,
            ctx->tiling_calls_32x32x32, pct_32x32x32,
            ctx->tiling_calls_32x32x16, pct_32x32x16,
            ctx->tiling_calls_16x32x32, pct_16x32x32,
            ctx->tiling_calls_32x16x16, pct_32x16x16,
            ctx->tiling_calls_16x16x16, pct_16x16x16);
    GGML_LOG_INFO("  padding efficiency: %.1f%% useful (%.1f%% wasted on zero-padding)\n",
            useful_pct, padded_pct);

    {
        std::lock_guard<std::mutex> lock(rejection_stats.mutex);
        GGML_LOG_INFO("  rejected mul_mat calls: %" PRId64 " (%zu unique signatures)\n",
                rejection_stats.total_rejected_mul_mat,
                rejection_stats.rejected_mul_mat.size());
        for (const auto & entry : rejection_stats.rejected_mul_mat) {
            GGML_LOG_INFO("    [count=%" PRId64 "] %s\n", entry.second, entry.first.c_str());
        }
        rejection_stats.total_rejected_mul_mat = 0;
        rejection_stats.rejected_mul_mat.clear();
    }

    ggml_backend_fpga_ffn_streaming_log_summary();
    tpu_runtime_cleanup();

    delete ctx;
    delete backend;
}

static void ggml_backend_fpga_add(ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_can_repeat(src1, src0));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const float * src0_data = (const float *) src0->data;
    const float * src1_data = (const float *) src1->data;
    float * dst_data = (float *) dst->data;
    const int64_t total = ggml_nelements(dst);

    const bool same_shape =
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == src0->ne[1] &&
        src1->ne[2] == src0->ne[2] &&
        src1->ne[3] == src0->ne[3];
    if (same_shape) {
        for (int64_t i = 0; i < total; ++i) {
            dst_data[i] = src0_data[i] + src1_data[i];
        }
        return;
    }

    if (ggml_nelements(src1) == 1) {
        const float rhs = src1_data[0];
        for (int64_t i = 0; i < total; ++i) {
            dst_data[i] = src0_data[i] + rhs;
        }
        return;
    }

    const bool bias_1d =
        src1->ne[0] == src0->ne[0] &&
        src1->ne[1] == 1 &&
        src1->ne[2] == 1 &&
        src1->ne[3] == 1;
    if (bias_1d) {
        const int64_t cols = src1->ne[0];
        GGML_ASSERT(cols > 0 && total % cols == 0);
        const int64_t rows = total / cols;
        for (int64_t row = 0; row < rows; ++row) {
            const float * src0_row = src0_data + row*cols;
            float * dst_row = dst_data + row*cols;
            for (int64_t col = 0; col < cols; ++col) {
                dst_row[col] = src0_row[col] + src1_data[col];
            }
        }
        return;
    }

    const char * src0_base = (const char *) src0->data;
    const char * src1_base = (const char *) src1->data;
    char * dst_base = (char *) dst->data;

    for (int64_t i3 = 0; i3 < dst->ne[3]; ++i3) {
        const int64_t j3 = i3 % src1->ne[3];
        for (int64_t i2 = 0; i2 < dst->ne[2]; ++i2) {
            const int64_t j2 = i2 % src1->ne[2];
            for (int64_t i1 = 0; i1 < dst->ne[1]; ++i1) {
                const int64_t j1 = i1 % src1->ne[1];
                const float * src0_row = (const float *) (src0_base + i3*src0->nb[3] + i2*src0->nb[2] + i1*src0->nb[1]);
                const char * src1_row_base = src1_base + j3*src1->nb[3] + j2*src1->nb[2] + j1*src1->nb[1];
                float * dst_row = (float *) (dst_base + i3*dst->nb[3] + i2*dst->nb[2] + i1*dst->nb[1]);

                if (src1->ne[0] == 1) {
                    const float rhs = *(const float *) src1_row_base;
                    for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
                        *(float *) ((char *) dst_row + i0*dst->nb[0]) =
                            *(const float *) ((const char *) src0_row + i0*src0->nb[0]) + rhs;
                    }
                } else {
                    for (int64_t i0 = 0; i0 < dst->ne[0]; ++i0) {
                        const int64_t j0 = i0 % src1->ne[0];
                        *(float *) ((char *) dst_row + i0*dst->nb[0]) =
                            *(const float *) ((const char *) src0_row + i0*src0->nb[0]) +
                            *(const float *) (src1_row_base + j0*src1->nb[0]);
                    }
                }
            }
        }
    }
}

static enum ggml_status ggml_backend_fpga_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    auto * ctx = (ggml_backend_fpga_context *) backend->context;
    enum ggml_backend_fpga_phase phase = ggml_backend_fpga_get_phase();
    if (phase < 0 || phase >= GGML_BACKEND_FPGA_PHASE_COUNT) {
        phase = GGML_BACKEND_FPGA_PHASE_UNKNOWN;
    }
    ggml_backend_fpga_graph_wall_stats & wall_stats = ctx->graph_wall_stats[phase];
    tpu_runtime_stats_t runtime_stats_before = {};
    tpu_runtime_get_stats(&runtime_stats_before);
    const int64_t graph_start_ns = ggml_backend_fpga_now_ns();
    if (ggml_backend_fpga_dynamic_src0_cache_enabled() && !ctx->dynamic_src0_cache.empty()) {
        ctx->dynamic_src0_cache.clear();
        ctx->dynamic_src0_cache_clears++;
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        ggml_tensor * node = cgraph->nodes[i];
        wall_stats.nodes++;
        const int64_t node_start_ns = ggml_backend_fpga_now_ns();

        switch (node->op) {
            case GGML_OP_MUL_MAT: {
                const int64_t before_weight_cache_ns = ctx->weight_cache_build_ns;
                const int64_t before_dynamic_src0_ns = ctx->dynamic_src0_build_ns;
                const int64_t before_activation_ns = ctx->activation_quantize_ns;
                const int64_t before_gemm_ns = ctx->total_gemm_ns;
                const int64_t before_dequant_ns = ctx->total_dequant_ns;
                const int64_t before_setup_ns = ctx->mul_mat_setup_ns;
                const int64_t before_cache_lookup_ns = ctx->weight_cache_lookup_ns;
                const int64_t before_matrix_task_prepack_ns = ctx->matrix_task_prepack_ns;
                const int64_t before_gate_ns = ctx->mul_mat_gate_ns;
                const int64_t before_activation_source_lookup_ns = ctx->activation_source_lookup_ns;
                const int64_t before_tiling_stats_ns = ctx->tiling_stats_accum_ns;
                const int64_t before_ffn_consume_ns = ctx->ffn_consume_ns;
                const int64_t before_instance_admin_ns = ctx->mul_mat_instance_admin_ns;
                bool fused_ffn_sibling = false;
                if (i + 1 < cgraph->n_nodes) {
                    ggml_tensor * next_node = cgraph->nodes[i + 1];
                    if (ggml_backend_fpga_node_name_is(node, "ffn_up")) {
                        fused_ffn_sibling = ggml_backend_fpga_try_fuse_ffn_up_gate_dual(
                                ctx,
                                node,
                                next_node);
                    } else if (ggml_backend_fpga_node_name_is(node, "ffn_gate")) {
                        fused_ffn_sibling = ggml_backend_fpga_try_fuse_ffn_up_gate_dual(
                                ctx,
                                next_node,
                                node);
                    }
                }
                if (!fused_ffn_sibling) {
                    const bool activation_reuse_may_store =
                        ggml_backend_fpga_activation_reuse_future_sibling(cgraph, i, node);
                    ggml_backend_fpga_mul_mat(ctx, node, activation_reuse_may_store);
                }
                const int64_t node_wall_ns = ggml_backend_fpga_now_ns() - node_start_ns;
                const int64_t weight_cache_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->weight_cache_build_ns, before_weight_cache_ns);
                const int64_t dynamic_src0_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->dynamic_src0_build_ns, before_dynamic_src0_ns);
                const int64_t activation_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->activation_quantize_ns, before_activation_ns);
                const int64_t gemm_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->total_gemm_ns, before_gemm_ns);
                const int64_t dequant_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->total_dequant_ns, before_dequant_ns);
                const int64_t setup_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->mul_mat_setup_ns, before_setup_ns);
                const int64_t cache_lookup_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->weight_cache_lookup_ns, before_cache_lookup_ns);
                const int64_t matrix_task_prepack_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->matrix_task_prepack_ns, before_matrix_task_prepack_ns);
                const int64_t gate_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->mul_mat_gate_ns, before_gate_ns);
                const int64_t activation_source_lookup_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->activation_source_lookup_ns, before_activation_source_lookup_ns);
                const int64_t tiling_stats_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->tiling_stats_accum_ns, before_tiling_stats_ns);
                const int64_t ffn_consume_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->ffn_consume_ns, before_ffn_consume_ns);
                const int64_t instance_admin_ns = ggml_backend_fpga_ns_delta_nonnegative(ctx->mul_mat_instance_admin_ns, before_instance_admin_ns);
                const int64_t nested_ns =
                    weight_cache_ns +
                    dynamic_src0_ns +
                    activation_ns +
                    gemm_ns +
                    dequant_ns +
                    setup_ns +
                    cache_lookup_ns +
                    matrix_task_prepack_ns +
                    gate_ns +
                    activation_source_lookup_ns +
                    tiling_stats_ns +
                    ffn_consume_ns +
                    instance_admin_ns;
                wall_stats.mul_mat_nodes += fused_ffn_sibling ? 2 : 1;
                wall_stats.mul_mat_wall_ns += node_wall_ns;
                wall_stats.weight_cache_ns += weight_cache_ns;
                wall_stats.dynamic_src0_ns += dynamic_src0_ns;
                wall_stats.activation_quant_ns += activation_ns;
                wall_stats.gemm_call_ns += gemm_ns;
                wall_stats.dequant_ns += dequant_ns;
                wall_stats.mul_mat_setup_ns += setup_ns;
                wall_stats.weight_cache_lookup_ns += cache_lookup_ns;
                wall_stats.matrix_task_prepack_ns += matrix_task_prepack_ns;
                wall_stats.mul_mat_gate_ns += gate_ns;
                wall_stats.activation_source_lookup_ns += activation_source_lookup_ns;
                wall_stats.tiling_stats_accum_ns += tiling_stats_ns;
                wall_stats.ffn_consume_ns += ffn_consume_ns;
                wall_stats.mul_mat_instance_admin_ns += instance_admin_ns;
                if (node_wall_ns >= nested_ns) {
                    wall_stats.mul_mat_residual_ns += node_wall_ns - nested_ns;
                } else {
                    wall_stats.mul_mat_nested_overrun_ns += nested_ns - node_wall_ns;
                }
                if (fused_ffn_sibling) {
                    wall_stats.nodes++;
                    i++;
                }
                break;
            }
            case GGML_OP_ADD:
                ggml_backend_fpga_add(node);
                wall_stats.add_nodes++;
                wall_stats.add_wall_ns += ggml_backend_fpga_now_ns() - node_start_ns;
                break;
            case GGML_OP_NONE:
            case GGML_OP_RESHAPE:
            case GGML_OP_VIEW:
            case GGML_OP_PERMUTE:
            case GGML_OP_TRANSPOSE:
                wall_stats.layout_nodes++;
                wall_stats.layout_wall_ns += ggml_backend_fpga_now_ns() - node_start_ns;
                break;
            default:
                GGML_ABORT("%s: unsupported op %s\n", __func__, ggml_op_desc(node));
        }
    }
    wall_stats.graphs++;
    wall_stats.wall_ns += ggml_backend_fpga_now_ns() - graph_start_ns;

    tpu_runtime_stats_t runtime_stats_after = {};
    tpu_runtime_get_stats(&runtime_stats_after);
    const tpu_runtime_stats_t runtime_delta =
            ggml_backend_fpga_runtime_stats_delta(runtime_stats_after, runtime_stats_before);
    ggml_backend_fpga_runtime_stats_add(ctx->runtime_stats_accumulated, runtime_delta);
    ggml_backend_fpga_runtime_stats_add(wall_stats.runtime, runtime_delta);

    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_fpga_i = {
    /* .get_name                = */ ggml_backend_fpga_get_name,
    /* .free                    = */ ggml_backend_fpga_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_fpga_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

static ggml_guid_t ggml_backend_fpga_guid(void) {
    static ggml_guid guid = { 0x63, 0x92, 0x30, 0x15, 0x1d, 0x57, 0x47, 0x72, 0xb2, 0xe3, 0x50, 0xea, 0x63, 0x67, 0x44, 0x01 };
    return &guid;
}

static ggml_backend_t ggml_backend_fpga_init(void) {
    ggml_backend_fpga_setenv_default("XILINX_XRT", "/usr");
    ggml_backend_fpga_setenv_default("MTMD_BACKEND_DEVICE", "FPGA");
    ggml_backend_fpga_setenv_default("TPU_SOC_PLATFORM", "kv260");
    ggml_backend_fpga_setenv_default("GGML_FPGA_MAX_NK", "128");
    ggml_backend_fpga_setenv_default("GGML_FPGA_WEIGHT_CACHE_PREWARM", "1");
    ggml_backend_fpga_setenv_default("GGML_FPGA_ACT_QUANT_FUSED_ROWS", "1");
    ggml_backend_fpga_setenv_default("GGML_FPGA_ACT_QUANT_NEON", "1");
    ggml_backend_fpga_setenv_default("GGML_FPGA_ACT_QUANT_PARALLEL", "1");
    ggml_backend_fpga_setenv_default("GGML_FPGA_ACT_QUANT_THREADS", "4");
    ggml_backend_fpga_setenv_default("GGML_FPGA_ACT_QUANT_PARALLEL_MIN_ELEMS", "262144");
    ggml_backend_fpga_setenv_default("GGML_FPGA_FFN_STREAMING_CPU", "1");
    ggml_backend_fpga_setenv_default("TPU_R5_WAIT_STATUS", "1");
    ggml_backend_fpga_setenv_default("TPU_R5_MATMUL_TASK", "1");
    ggml_backend_fpga_setenv_default("TPU_R5_MATMUL_TIMEOUT_MS", "600000");
    ggml_backend_fpga_setenv_default("TPU_V2_R5_MATRIX_TASK_DECODE", "1");
    ggml_backend_fpga_setenv_default("TPU_V2_R5_MATRIX_TASK_DECODE_MAX_M", "1");
    ggml_backend_fpga_setenv_default("TPU_V2_R5_MATRIX_TASK_DECODE_MIN_N", "64");
    ggml_backend_fpga_setenv_default("TPU_V2_R5_MATRIX_TASK_DECODE_MIN_K", "64");
    ggml_backend_fpga_setenv_default("TPU_V2_R5_MATRIX_TASK_PREPACK", "1");
    ggml_backend_fpga_setenv_default("TPU_V2_R5_MATRIX_TASK_AB_PRELOAD", "1");
    ggml_backend_fpga_setenv_default("TPU_V2_R5_MATRIX_TASK_DESC_PRELOAD", "0");
    ggml_backend_fpga_setenv_default("TPU_V2_R5_MATRIX_TASK_SEQ_INNER_K", "0");
    ggml_backend_fpga_setenv_default("TPU_V2_R5_MATRIX_TASK_SEQ_INNER_K_MAX_M", "1");
    ggml_backend_fpga_setenv_default("TPU_V2_R5_MATRIX_TASK_SEQ_INNER_K_MAX_RUNS", "8");
    ggml_backend_fpga_setenv_default("TPU_V2_R5_MATRIX_TASK_SEQ_INNER_K_MIN_K", "128");
    ggml_backend_fpga_setenv_default("TPU_V2_A53_SEQUENCE", "1");
    ggml_backend_fpga_setenv_default("TPU_V2_PL_OP_DESCRIPTOR", "1");
    ggml_backend_fpga_setenv_default("TPU_V2_PL_OP_DESCRIPTOR_MAX_M", "1");

    if (tpu_runtime_init() != 0) {
        GGML_LOG_ERROR("%s Failed to initialize TPU runtime\n", ggml_backend_fpga_log_prefix());
        return nullptr;
    }

    auto * ctx = new ggml_backend_fpga_context;

    return new ggml_backend {
        /* .guid    = */ ggml_backend_fpga_guid(),
        /* .iface   = */ ggml_backend_fpga_i,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_fpga_reg(), 0),
        /* .context = */ ctx,
    };
}

static const char * ggml_backend_fpga_device_get_name(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return "FPGA";
}

static const char * ggml_backend_fpga_device_get_description(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
#ifdef GGML_FPGA_USE_MOCK
    return "Mock FPGA INT8 GEMM";
#else
    return "Real FPGA INT8 GEMM";
#endif
}

static void ggml_backend_fpga_device_get_memory(ggml_backend_dev_t dev, size_t * free_mem, size_t * total_mem) {
    GGML_UNUSED(dev);
    *free_mem = 0;
    *total_mem = 0;
}

static enum ggml_backend_dev_type ggml_backend_fpga_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

static void ggml_backend_fpga_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_fpga_device_get_name(dev);
    props->description = ggml_backend_fpga_device_get_description(dev);
    props->type = ggml_backend_fpga_device_get_type(dev);
    props->device_id = NULL;
    ggml_backend_fpga_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ true,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_fpga_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(dev);
    GGML_UNUSED(params);
    return ggml_backend_fpga_init();
}

static ggml_backend_buffer_type_t ggml_backend_fpga_device_get_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cpu_buffer_type();
}

static ggml_backend_buffer_t ggml_backend_fpga_device_buffer_from_host_ptr(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_UNUSED(dev);
    GGML_UNUSED(max_tensor_size);
    return ggml_backend_cpu_buffer_from_ptr(ptr, size);
}

static bool ggml_backend_fpga_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_UNUSED(dev);

    switch (op->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
            return true;

        case GGML_OP_MUL_MAT: {
            std::string reason;
            const bool supported = ggml_backend_fpga_supports_mul_mat(op, &reason);
            if (!supported) {
                ggml_backend_fpga_record_rejected_mul_mat(op, reason);
            }
            return supported;
        }
        case GGML_OP_ADD:
            return ggml_backend_fpga_supports_add(op);

        default:
            return false;
    }
}

static bool ggml_backend_fpga_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_host(buft);
}

static const ggml_backend_device_i ggml_backend_fpga_device_i = {
    /* .get_name             = */ ggml_backend_fpga_device_get_name,
    /* .get_description      = */ ggml_backend_fpga_device_get_description,
    /* .get_memory           = */ ggml_backend_fpga_device_get_memory,
    /* .get_type             = */ ggml_backend_fpga_device_get_type,
    /* .get_props            = */ ggml_backend_fpga_device_get_props,
    /* .init_backend         = */ ggml_backend_fpga_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_fpga_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_fpga_device_buffer_from_host_ptr,
    /* .supports_op          = */ ggml_backend_fpga_device_supports_op,
    /* .supports_buft        = */ ggml_backend_fpga_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

static const char * ggml_backend_fpga_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return "FPGA";
}

static size_t ggml_backend_fpga_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return 1;
}

static ggml_backend_dev_t ggml_backend_fpga_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(index == 0);

    static ggml_backend_device dev = {
        /* .iface   = */ ggml_backend_fpga_device_i,
        /* .reg     = */ reg,
        /* .context = */ NULL,
    };

    return &dev;
}

static void * ggml_backend_fpga_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    GGML_UNUSED(name);
    return NULL;
}

static const ggml_backend_reg_i ggml_backend_fpga_reg_i = {
    /* .get_name         = */ ggml_backend_fpga_reg_get_name,
    /* .get_device_count = */ ggml_backend_fpga_reg_get_device_count,
    /* .get_device       = */ ggml_backend_fpga_reg_get_device,
    /* .get_proc_address = */ ggml_backend_fpga_get_proc_address,
};

ggml_backend_reg_t ggml_backend_fpga_reg(void) {
    static ggml_backend_reg reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_fpga_reg_i,
        /* .context     = */ NULL,
    };

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_fpga_reg)
