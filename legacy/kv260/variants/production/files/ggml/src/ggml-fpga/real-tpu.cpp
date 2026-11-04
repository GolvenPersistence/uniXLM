#include "real-tpu.h"

#include "tpu-dispatch.h"
#include "tpu-tiling.h"

#include "tpu_soc_fp32.h"
#include "tpu_soc_v2_int8.h"
#include "tpu_vanilla_regs.h"
#include "tpu_r5_ctrl.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <algorithm>
#include <thread>
#include <unordered_map>
#include <vector>

static const uint32_t TPU_SEQUENCE_MAX_RUNS = 8;
static const uint32_t TPU_ASYNC_D_MAX_BUFS = 3;
static tpu_soc_fp32_t g_tpu_ctx;
static int g_tpu_initialized = 0;
static int g_tpu_buffer_pool_initialized = 0;
static std::mutex g_tpu_mutex;
static tpu_soc_fp32_buffer_t g_tpu_a_buf[TPU_SEQUENCE_MAX_RUNS];
static tpu_soc_fp32_buffer_t g_tpu_b_buf[TPU_SEQUENCE_MAX_RUNS];
static std::vector<uint64_t> g_tpu_a_stage[TPU_SEQUENCE_MAX_RUNS];
static std::vector<uint64_t> g_tpu_b_stage[TPU_SEQUENCE_MAX_RUNS];
static tpu_soc_fp32_buffer_t g_tpu_d_buf;
static tpu_soc_fp32_buffer_t g_tpu_op_d_buf;
static tpu_soc_fp32_buffer_t g_tpu_async_d_buf[TPU_ASYNC_D_MAX_BUFS];
static tpu_soc_fp32_buffer_t g_tpu_matmul_a_buf;
static tpu_soc_fp32_buffer_t g_tpu_matmul_b_buf;
static tpu_soc_fp32_buffer_t g_tpu_matmul_c_buf;
static size_t g_tpu_a_buf_bytes = 0;
static size_t g_tpu_b_buf_bytes = 0;
static size_t g_tpu_d_buf_bytes = 0;
static size_t g_tpu_op_d_buf_bytes = 0;
static size_t g_tpu_async_d_buf_bytes = 0;
static size_t g_tpu_matmul_a_buf_bytes = 0;
static size_t g_tpu_matmul_b_buf_bytes = 0;
static size_t g_tpu_matmul_c_buf_bytes = 0;
static int g_tpu_output_buffer_enabled = 0;
static std::mutex g_tpu_op_mutex;
static std::mutex g_tpu_async_mutex;
static std::condition_variable g_tpu_async_cv;
static std::thread g_tpu_async_worker;
static uint64_t g_tile_call_count = 0;
static uint64_t g_buffer_pool_init_us = 0;
static uint64_t g_total_pack_us = 0;
static uint64_t g_total_pack_a_us = 0;
static uint64_t g_total_pack_b_us = 0;
static uint64_t g_total_prepare_us = 0;
static uint64_t g_total_alloc_us = 0;
static uint64_t g_total_copy_us = 0;
static uint64_t g_total_copy_a_us = 0;
static uint64_t g_total_copy_b_us = 0;
static uint64_t g_b_device_cache_hits = 0;
static uint64_t g_b_device_cache_misses = 0;
static uint64_t g_b_device_cache_fallbacks = 0;
static uint64_t g_b_device_cache_budget_rejects = 0;
static uint64_t g_b_device_cache_aperture_rejects = 0;
static uint64_t g_b_device_cache_saved_bytes = 0;
static uint64_t g_b_device_cache_resident_bytes = 0;
static uint64_t g_b_matrix_cache_budget_rejects = 0;
static uint64_t g_b_matrix_cache_resident_bytes = 0;
static uint64_t g_b_packed_arena_hits = 0;
static uint64_t g_b_packed_arena_misses = 0;
static uint64_t g_matmul_task_count = 0;
static uint64_t g_matmul_task_tile_count = 0;
static uint64_t g_matmul_task_fallback_count = 0;
static uint64_t g_matmul_task_r5_cycles = 0;
static uint64_t g_matmul_task_fpga_wait_cycles = 0;
static uint64_t g_matmul_task_total_us = 0;
static uint64_t g_matmul_task_entry_setup_us = 0;
static uint64_t g_matmul_task_lock_us = 0;
static uint64_t g_matmul_task_reserve_us = 0;
static uint64_t g_matmul_task_prepare_b_wall_us = 0;
static uint64_t g_matmul_task_copy_a_wall_us = 0;
static uint64_t g_matmul_task_sync_a_us = 0;
static uint64_t g_matmul_task_addr_check_us = 0;
static uint64_t g_matmul_task_submit_us = 0;
static uint64_t g_matmul_task_output_sync_us = 0;
static uint64_t g_matmul_task_output_view_us = 0;
static uint64_t g_matmul_task_output_copy_us = 0;
static uint64_t g_matmul_task_output_copy_jobs = 0;
static uint64_t g_matmul_task_output_copy_bytes = 0;
static uint64_t g_matmul_task_output_copy_direct_jobs = 0;
static uint64_t g_matmul_task_output_copy_direct_us = 0;
static uint64_t g_matmul_task_output_copy_direct_bytes = 0;
static uint64_t g_matmul_task_output_copy_row_jobs = 0;
static uint64_t g_matmul_task_output_copy_row_us = 0;
static uint64_t g_matmul_task_output_copy_row_bytes = 0;
static uint64_t g_matmul_task_output_copy_row_rows = 0;
static uint64_t g_matmul_task_output_copy_padded_jobs = 0;
static uint64_t g_matmul_task_output_copy_padded_us = 0;
static uint64_t g_matmul_task_output_copy_m1_jobs = 0;
static uint64_t g_matmul_task_output_copy_m1_us = 0;
static uint64_t g_matmul_task_output_copy_m32_jobs = 0;
static uint64_t g_matmul_task_output_copy_m32_us = 0;
static uint64_t g_matmul_task_output_copy_n64_jobs = 0;
static uint64_t g_matmul_task_output_copy_n64_us = 0;
static uint64_t g_matmul_task_output_copy_n128_jobs = 0;
static uint64_t g_matmul_task_output_copy_n128_us = 0;
static uint64_t g_matmul_task_output_copy_n_other_jobs = 0;
static uint64_t g_matmul_task_output_copy_n_other_us = 0;
static uint64_t g_matmul_task_post_us = 0;
static uint64_t g_matmul_task_post_stats_us = 0;
static uint64_t g_matmul_task_free_us = 0;
static uint64_t g_matmul_task_residual_us = 0;
static uint32_t g_matmul_task_fence_id = 0;
static uint64_t g_pl_op_desc_count = 0;
static uint64_t g_pl_op_desc_tile_count = 0;
static uint64_t g_pl_op_desc_program_us = 0;
static uint64_t g_pl_op_desc_wait_us = 0;
static uint64_t g_pl_op_desc_status_polls = 0;
static uint64_t g_total_control_us = 0;
static uint64_t g_total_clear_us = 0;
static uint64_t g_total_log_us = 0;
static uint64_t g_total_execute_us = 0;
static uint64_t g_total_sync_us = 0;
static uint64_t g_total_soft_reset_us = 0;
static uint64_t g_total_cdma_reset_us = 0;
static uint64_t g_total_ready_wait_us = 0;
static uint64_t g_total_path_setup_us = 0;
static uint64_t g_total_start_us = 0;
static uint64_t g_total_post_start_delay_us = 0;
static uint64_t g_total_transfer_us = 0;
static uint64_t g_total_tail_fixup_us = 0;
static uint64_t g_total_wait_done_us = 0;
static uint64_t g_total_preload_sync_us = 0;
static uint64_t g_total_preload_write_ready_wait_us = 0;
static uint64_t g_total_preload_path_setup_us = 0;
static uint64_t g_total_preload_transfer_us = 0;
static uint64_t g_total_execute_gap_us = 0;
static uint64_t g_total_read_us = 0;
static uint64_t g_total_read_sync_us = 0;
static uint64_t g_total_read_copy_us = 0;
static uint64_t g_total_read_unpack_us = 0;
static uint64_t g_total_read_mmio_us = 0;
static uint64_t g_total_read_debug_dump_us = 0;
static uint64_t g_total_read_bytes = 0;
static uint64_t g_total_read_output_buffer_count = 0;
static uint64_t g_total_read_memcpy_count = 0;
static uint64_t g_total_read_unpack_count = 0;
static uint64_t g_total_read_mmio_count = 0;
static uint64_t g_read_op_batch_count = 0;
static uint64_t g_read_op_batch_total_us = 0;
static uint64_t g_read_op_batch_sync_us = 0;
static uint64_t g_read_op_batch_copy_us = 0;
static uint64_t g_read_op_batch_bytes = 0;
static uint64_t g_read_async_count = 0;
static uint64_t g_read_async_total_us = 0;
static uint64_t g_read_async_sync_us = 0;
static uint64_t g_read_async_copy_us = 0;
static uint64_t g_read_async_bytes = 0;
static uint64_t g_read_async_batch_begin_wait_us = 0;
static uint64_t g_read_async_batch_begin_wait_count = 0;
static uint64_t g_read_async_stage_wait_us = 0;
static uint64_t g_read_async_stage_wait_count = 0;
static uint64_t g_read_async_flush_wait_us = 0;
static uint64_t g_read_async_flush_wait_count = 0;
static uint64_t g_read_final_desc_count = 0;
static uint64_t g_read_final_desc_total_us = 0;
static uint64_t g_read_final_desc_sync_us = 0;
static uint64_t g_read_final_desc_copy_us = 0;
static uint64_t g_read_final_desc_unpack_us = 0;
static uint64_t g_read_final_desc_mmio_us = 0;
static uint64_t g_read_final_desc_bytes = 0;
static uint64_t g_read_copy_job_count = 0;
static uint64_t g_read_copy_row_count = 0;
static uint64_t g_read_copy_elem_count = 0;
static uint64_t g_read_copy_useful_bytes = 0;
static uint64_t g_read_copy_transfer_bytes = 0;
static uint64_t g_read_copy_padded_jobs = 0;
static uint64_t g_read_copy_strided_jobs = 0;
static uint64_t g_read_copy_bulk_candidate_jobs = 0;
static uint64_t g_read_copy_bulk_candidate_bytes = 0;
static uint64_t g_read_copy_bulk_memcpy_jobs = 0;
static uint64_t g_read_copy_bulk_memcpy_bytes = 0;
static uint64_t g_read_copy_row_memcpy_rows = 0;
static uint64_t g_read_copy_rows_1 = 0;
static uint64_t g_read_copy_rows_2_15 = 0;
static uint64_t g_read_copy_rows_16 = 0;
static uint64_t g_read_copy_rows_17_31 = 0;
static uint64_t g_read_copy_rows_32 = 0;
static uint64_t g_read_copy_rows_gt32 = 0;
static uint64_t g_read_copy_cols_le16 = 0;
static uint64_t g_read_copy_cols_le32 = 0;
static uint64_t g_read_copy_cols_le64 = 0;
static uint64_t g_read_copy_cols_65_127 = 0;
static uint64_t g_read_copy_cols_128 = 0;
static uint64_t g_read_copy_cols_gt128 = 0;
static uint64_t g_total_free_us = 0;
static uint64_t g_total_other_us = 0;
static uint64_t g_total_tile_us = 0;
static std::vector<int32_t> g_tpu_output_view_scratch;

typedef struct {
    uint64_t total_us;
    uint64_t pack_total_us;
    uint64_t pack_a_us;
    uint64_t pack_b_us;
    uint64_t copy_total_us;
    uint64_t copy_a_us;
    uint64_t copy_b_us;
    size_t a_bytes;
    size_t b_bytes;
    int used_b_pack;
    int used_b_kn_cache;
} tpu_prepare_ab_report_t;

typedef struct {
    uint64_t total_us;
    uint64_t sync_us;
    uint64_t copy_us;
    uint64_t unpack_us;
    uint64_t mmio_us;
    uint64_t dump_us;
    uint64_t bytes;
    int used_output_buffer;
    int used_memcpy;
    int used_unpack;
    int used_mmio;
} tpu_readback_stage_report_t;

struct tpu_device_b_cache_key_t {
    const void * owner;
    const void * source;
    size_t bytes;
    uint32_t source_n;
    uint32_t n0;
    uint32_t k0;
    uint32_t n_actual;
    uint32_t k_actual;
    uint32_t n;
    uint32_t k;

    bool operator==(const tpu_device_b_cache_key_t & other) const {
        return owner == other.owner &&
               source == other.source &&
               bytes == other.bytes &&
               source_n == other.source_n &&
               n0 == other.n0 &&
               k0 == other.k0 &&
               n_actual == other.n_actual &&
               k_actual == other.k_actual &&
               n == other.n &&
               k == other.k;
    }
};

struct tpu_device_b_cache_key_hash_t {
    size_t operator()(const tpu_device_b_cache_key_t & key) const {
        uint64_t h = (uint64_t)(uintptr_t) key.owner;
        h = h * 1315423911u + (uint64_t)(uintptr_t) key.source;
        h = h * 1315423911u + (uint64_t) key.bytes;
        h = h * 1315423911u + key.source_n;
        h = h * 1315423911u + key.n0;
        h = h * 1315423911u + key.k0;
        h = h * 1315423911u + key.n_actual;
        h = h * 1315423911u + key.k_actual;
        h = h * 1315423911u + key.n;
        h = h * 1315423911u + key.k;
        return (size_t) h;
    }
};

struct tpu_device_b_cache_entry_t {
    tpu_soc_fp32_buffer_t buf;
};

static std::unordered_map<tpu_device_b_cache_key_t, tpu_device_b_cache_entry_t, tpu_device_b_cache_key_hash_t> g_tpu_device_b_cache;

struct tpu_device_b_matrix_cache_key_t {
    const void * owner;
    const void * source;
    uint32_t source_n;
    uint32_t source_k;
    uint32_t tile_n;
    uint32_t tile_k;
    uint32_t pack_layout;

    bool operator==(const tpu_device_b_matrix_cache_key_t & other) const {
        return owner == other.owner &&
               source == other.source &&
               source_n == other.source_n &&
               source_k == other.source_k &&
               tile_n == other.tile_n &&
               tile_k == other.tile_k &&
               pack_layout == other.pack_layout;
    }
};

struct tpu_device_b_matrix_cache_key_hash_t {
    size_t operator()(const tpu_device_b_matrix_cache_key_t & key) const {
        uint64_t h = (uint64_t)(uintptr_t) key.owner;
        h = h * 1315423911u + (uint64_t)(uintptr_t) key.source;
        h = h * 1315423911u + key.source_n;
        h = h * 1315423911u + key.source_k;
        h = h * 1315423911u + key.tile_n;
        h = h * 1315423911u + key.tile_k;
        h = h * 1315423911u + key.pack_layout;
        return (size_t) h;
    }
};

struct tpu_device_b_matrix_cache_entry_t {
    void * owner;
    const int8_t * source_ptr;
    tpu_soc_fp32_buffer_t buf;
    int borrowed_buf;
    uint32_t source_n;
    uint32_t source_k;
    uint32_t tile_n;
    uint32_t tile_k;
    uint32_t pack_layout;
    uint32_t n_tiles;
    uint32_t k_tiles;
    size_t tile_bytes;
    size_t tile_words;
    size_t total_bytes;
};

static std::unordered_map<
    tpu_device_b_matrix_cache_key_t,
    tpu_device_b_matrix_cache_entry_t,
    tpu_device_b_matrix_cache_key_hash_t> g_tpu_device_b_matrix_cache;
static std::unordered_map<void *, tpu_device_b_matrix_cache_entry_t *> g_tpu_device_b_matrix_owner_cache;

typedef struct {
    int active;
    std::thread::id owner;
    tpu_accum_key_t key;
    uint32_t run_index;
    uint32_t pending_c_mode;
    uint32_t pending_d_mode;
    int pending_slot;
    int next_preloaded_valid;
} tpu_prevd_chain_state_t;

static tpu_prevd_chain_state_t g_prevd_chain = {};
static uint64_t g_prevd_chain_runs = 0;
static uint64_t g_prevd_over_signoff_runs = 0;
static uint64_t g_prevd_ab_preload_count = 0;
static uint64_t g_prevd_host_c_upload_count = 0;
static uint64_t g_prevd_host_d_readback_count = 0;
static uint64_t g_prevd_direct_final_readback_count = 0;
static uint64_t g_prevd_auto_next_program_count = 0;
static uint64_t g_prevd_auto_next_fallback_count = 0;
static uint64_t g_prevd_sequence_program_count = 0;
static uint64_t g_prevd_sequence_fallback_count = 0;
static uint64_t g_total_sequence_us = 0;
static uint64_t g_total_sequence_sync_us = 0;
static uint64_t g_total_sequence_ready_us = 0;
static uint64_t g_total_sequence_program_us = 0;
static uint64_t g_total_sequence_wait_done_us = 0;
static uint64_t g_total_wait_prev_us = 0;
static uint64_t g_total_wait_final_us = 0;
static uint64_t g_total_preload_wall_us = 0;
static uint64_t g_total_start_wall_us = 0;
static uint64_t g_total_cleanup_us = 0;
static uint64_t g_total_unlock_us = 0;

struct tpu_op_readback_tile_t {
    size_t offset_bytes;
    size_t packed_bytes;
    int32_t * dst;
    uint32_t packed_rows;
    uint32_t packed_cols;
    uint32_t copy_rows;
    uint32_t copy_cols;
    uint32_t dst_stride;
};

static int g_tpu_op_readback_active = 0;
static size_t g_tpu_op_readback_used_bytes = 0;
static std::vector<tpu_op_readback_tile_t> g_tpu_op_readback_tiles;

struct tpu_async_readback_job_t {
    uint32_t buf_idx;
    size_t bytes;
    int32_t * dst;
    uint32_t packed_rows;
    uint32_t packed_cols;
    uint32_t copy_rows;
    uint32_t copy_cols;
    uint32_t dst_stride;
};

static int g_tpu_async_worker_started = 0;
static int g_tpu_async_stop = 0;
static int g_tpu_async_failed = 0;
static int g_tpu_async_readback_active = 0;
static uint32_t g_tpu_async_ring_count = 0;
static uint32_t g_tpu_async_next_buf = 0;
static int g_tpu_async_buf_busy[TPU_ASYNC_D_MAX_BUFS] = {};
static std::deque<tpu_async_readback_job_t> g_tpu_async_jobs;
static uint64_t g_tpu_async_jobs_enqueued = 0;
static uint64_t g_tpu_async_jobs_completed = 0;

static const uint32_t TPU_R5_MATMUL_TILE_M_LOCAL = 32u;
static const uint32_t TPU_R5_MATMUL_TILE_N_LOCAL = 128u;
static const uint32_t TPU_R5_MATMUL_TILE_K_LOCAL = 128u;

static const char * tpu_env_or_unset(const char * name) {
    const char * value = getenv(name);
    return (value != NULL && *value != '\0') ? value : "<unset>";
}

static int tpu_env_is_truthy_or_default(const char * name, int default_value) {
    const char * value = getenv(name);

    if (value == NULL || *value == '\0') {
        return default_value;
    }

    return strcmp(value, "0") != 0;
}

static unsigned tpu_env_u32_or_default(const char * name, unsigned default_value) {
    const char * value = getenv(name);
    char * endptr = NULL;
    unsigned long parsed = 0;

    if (value == NULL || *value == '\0') {
        return default_value;
    }

    errno = 0;
    parsed = strtoul(value, &endptr, 0);
    if (errno != 0 || endptr == value || (endptr != NULL && *endptr != '\0') || parsed > UINT32_MAX) {
        return default_value;
    }

    return (unsigned) parsed;
}

static uint32_t tpu_pl_op_tile_n(void) {
    static uint32_t cached = 0u;
    if (cached != 0u) {
        return cached;
    }
    const uint32_t value = tpu_env_u32_or_default("TPU_V2_PL_OP_TILE_N", TPU_R5_MATMUL_TILE_N_LOCAL);
    cached = value == 64u ? 64u : TPU_R5_MATMUL_TILE_N_LOCAL;
    return cached;
}

static uint32_t tpu_pl_op_tile_k(void) {
    static uint32_t cached = 0u;
    if (cached != 0u) {
        return cached;
    }
    const uint32_t value = tpu_env_u32_or_default("TPU_V2_PL_OP_TILE_K", TPU_R5_MATMUL_TILE_K_LOCAL);
    cached = value == 64u ? 64u : TPU_R5_MATMUL_TILE_K_LOCAL;
    return cached;
}

static int tpu_skip_sequence_soft_reset_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_SKIP_SEQUENCE_SOFT_RESET", 1);
    }
    return cached;
}

static int tpu_skip_single_tile_soft_reset_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_SKIP_SINGLE_TILE_SOFT_RESET", 0);
    }
    return cached;
}

static int tpu_device_b_cache_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_DEVICE_B_CACHE", 1);
    }
    return cached;
}

static int tpu_r5_matmul_task_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_R5_MATMUL_TASK", 1) && tpu_r5_ctrl_enabled();
    }
    return cached;
}

static int tpu_pl_op_descriptor_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_PL_OP_DESCRIPTOR", 1);
    }
    return cached;
}

enum {
    TPU_B_PACK_LAYOUT_PACK2 = 0u,
    TPU_B_PACK_LAYOUT_B16_TILE_MAJOR = 1u,
    TPU_B_PACK_LAYOUT_B16_BANKFRIENDLY = 2u,
};

static int tpu_pl_op_b16_layout_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_PL_OP_B16_LAYOUT", 0);
    }
    return cached;
}

static int tpu_pl_op_b16_bankfriendly_layout_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_PL_OP_B16_BANKFRIENDLY_LAYOUT", 0);
    }
    return cached;
}

static uint32_t tpu_select_matmul_b_pack_layout(
        int use_pl_op_desc,
        uint32_t tile_n,
        uint32_t tile_k) {
    if (use_pl_op_desc &&
        tile_n == TPU_R5_MATMUL_TILE_N_LOCAL &&
        tile_k == TPU_R5_MATMUL_TILE_K_LOCAL) {
        if (tpu_pl_op_b16_bankfriendly_layout_enabled()) {
            return TPU_B_PACK_LAYOUT_B16_BANKFRIENDLY;
        }
        if (tpu_pl_op_b16_layout_enabled()) {
            return TPU_B_PACK_LAYOUT_B16_TILE_MAJOR;
        }
    }
    return TPU_B_PACK_LAYOUT_PACK2;
}

static int tpu_matrix_task_skip_a_sync_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_MATMUL_TASK_SKIP_A_SYNC", 1);
    }
    return cached;
}

static int tpu_matrix_task_skip_output_sync_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_MATMUL_TASK_SKIP_OUTPUT_SYNC", 0);
    }
    return cached;
}

static int tpu_matrix_task_tiling_stats_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_MATMUL_TASK_TILING_STATS", 0);
    }
    return cached;
}

static int tpu_matrix_task_detail_timing_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_MATMUL_TASK_DETAIL_TIMING", 0);
    }
    return cached;
}

static int tpu_matrix_task_reuse_transient_b_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_MATRIX_TASK_REUSE_TRANSIENT_B", 0);
    }
    return cached;
}

static int tpu_pl_op_descriptor_log_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_PL_OP_DESCRIPTOR_LOG", 0);
    }
    return cached;
}

static int tpu_pl_op_a_tile_contig_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_PL_OP_A_TILE_CONTIG", 0);
    }
    return cached;
}

static size_t tpu_device_b_cache_max_bytes(void) {
    static size_t cached = (size_t) -1;
    if (cached != (size_t) -1) {
        return cached;
    }
    const unsigned max_mb = tpu_env_u32_or_default("TPU_V2_DEVICE_B_CACHE_MAX_MB", 256u);
    if (max_mb == 0u) {
        cached = 0u;
        return cached;
    }
    cached = (size_t) max_mb * 1024u * 1024u;
    return cached;
}

static size_t tpu_matrix_task_b_cache_max_bytes(void) {
    static size_t cached = (size_t) -1;
    if (cached != (size_t) -1) {
        return cached;
    }
    const unsigned max_mb = tpu_env_u32_or_default("TPU_V2_MATRIX_TASK_B_CACHE_MAX_MB", 256u);
    if (max_mb == 0u) {
        cached = 0u;
        return cached;
    }
    cached = (size_t) max_mb * 1024u * 1024u;
    return cached;
}

static int tpu_log_init_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("GGML_FPGA_LOG_INIT", 0);
    }
    return cached;
}

static int tpu_log_tile_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("GGML_FPGA_LOG_TILE", 0);
    }
    return cached;
}

static unsigned tpu_log_tile_every(void) {
    static unsigned cached = 0u;
    if (cached != 0u) {
        return cached;
    }
    const unsigned every = tpu_env_u32_or_default("GGML_FPGA_LOG_TILE_EVERY", 1u);
    cached = every == 0u ? 1u : every;
    return cached;
}

static uint32_t tpu_pl_op_descriptor_max_m(void) {
    static uint32_t cached = 0u;
    if (cached != 0u) {
        return cached;
    }
    cached = tpu_env_u32_or_default("TPU_V2_PL_OP_DESCRIPTOR_MAX_M", 1u);
    if (cached == 0u) {
        cached = 1u;
    }
    return cached;
}

static unsigned tpu_r5_matmul_timeout_ms(void) {
    static unsigned cached = 0u;
    if (cached != 0u) {
        return cached;
    }
    cached = tpu_env_u32_or_default("TPU_R5_MATMUL_TIMEOUT_MS", 600000u);
    if (cached == 0u) {
        cached = 600000u;
    }
    return cached;
}

static uint32_t tpu_pl_op_poll_sleep_us(void) {
    static int initialized = 0;
    static uint32_t cached = 0u;
    if (!initialized) {
        cached = tpu_env_u32_or_default("TPU_V2_PL_OP_POLL_SLEEP_US", 0u);
        initialized = 1;
    }
    return cached;
}

static uint32_t tpu_pl_op_poll_spins(void) {
    static int initialized = 0;
    static uint32_t cached = 0u;
    if (!initialized) {
        cached = tpu_env_u32_or_default("TPU_V2_PL_OP_POLL_SPINS", 0u);
        initialized = 1;
    }
    return cached;
}

static void tpu_pl_op_poll_pause(uint32_t spins) {
    for (uint32_t i = 0; i < spins; ++i) {
#if defined(__aarch64__)
        __asm__ __volatile__("yield" ::: "memory");
#else
        __asm__ __volatile__("" ::: "memory");
#endif
    }
}

static int tpu_matmul_output_cacheable_bo_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        cached = tpu_env_is_truthy_or_default("TPU_V2_MATMUL_OUTPUT_CACHEABLE_BO", 0);
    }
    return cached;
}

static bool tpu_should_log_tile(uint64_t call_idx) {
    if (!tpu_log_tile_enabled()) {
        return false;
    }
    const unsigned every = tpu_log_tile_every();
    return every <= 1u || call_idx == 1u || (call_idx % every) == 0u;
}

static uint64_t tpu_ns_to_us(uint64_t ns) {
    return (ns + 500u) / 1000u;
}

static uint64_t tpu_elapsed_us(const std::chrono::steady_clock::time_point & start,
                               const std::chrono::steady_clock::time_point & end) {
    return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

static std::chrono::steady_clock::time_point tpu_detail_now(int enabled) {
    return enabled ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
}

static uint64_t tpu_detail_elapsed_us(
        int enabled,
        const std::chrono::steady_clock::time_point & start) {
    return enabled ? tpu_elapsed_us(start, std::chrono::steady_clock::now()) : 0u;
}

static uint64_t * tpu_stage_words(std::vector<uint64_t> & stage, size_t byte_count) {
    const size_t word_count = (byte_count + sizeof(uint64_t) - 1u) / sizeof(uint64_t);
    if (stage.size() < word_count) {
        stage.resize(word_count);
    }
    return stage.data();
}

static size_t tpu_align_up_size(size_t value, size_t align) {
    return (value + align - 1u) & ~(align - 1u);
}

static size_t tpu_tile_d_transfer_bytes(uint32_t packed_rows, uint32_t packed_cols) {
    return (size_t) tpu_soc_v2_int32_word_count(packed_rows, packed_cols) * sizeof(uint64_t);
}

static void tpu_copy_int32_row_inline(int32_t * dst, const int32_t * src, uint32_t count);

static void tpu_record_read_copy_shape(uint32_t copy_rows,
                                       uint32_t copy_cols,
                                       uint32_t packed_rows,
                                       uint32_t packed_cols,
                                       uint32_t dst_stride,
                                       size_t transfer_bytes,
                                       bool bulk_memcpy) {
    const uint64_t useful_bytes = (uint64_t) copy_rows * (uint64_t) copy_cols * sizeof(int32_t);
    g_read_copy_job_count++;
    g_read_copy_row_count += copy_rows;
    g_read_copy_elem_count += (uint64_t) copy_rows * (uint64_t) copy_cols;
    g_read_copy_useful_bytes += useful_bytes;
    g_read_copy_transfer_bytes += transfer_bytes;
    if (copy_rows != packed_rows || copy_cols != packed_cols) {
        g_read_copy_padded_jobs++;
    }
    if (dst_stride != copy_cols) {
        g_read_copy_strided_jobs++;
    }
    if (copy_rows > 0u && copy_cols > 0u && dst_stride == copy_cols && packed_cols == copy_cols) {
        g_read_copy_bulk_candidate_jobs++;
        g_read_copy_bulk_candidate_bytes += useful_bytes;
    }
    if (bulk_memcpy) {
        g_read_copy_bulk_memcpy_jobs++;
        g_read_copy_bulk_memcpy_bytes += useful_bytes;
    } else {
        g_read_copy_row_memcpy_rows += copy_rows;
    }

    if (copy_rows == 1u) {
        g_read_copy_rows_1++;
    } else if (copy_rows < 16u) {
        g_read_copy_rows_2_15++;
    } else if (copy_rows == 16u) {
        g_read_copy_rows_16++;
    } else if (copy_rows < 32u) {
        g_read_copy_rows_17_31++;
    } else if (copy_rows == 32u) {
        g_read_copy_rows_32++;
    } else {
        g_read_copy_rows_gt32++;
    }

    if (copy_cols <= 16u) {
        g_read_copy_cols_le16++;
    } else if (copy_cols <= 32u) {
        g_read_copy_cols_le32++;
    } else if (copy_cols <= 64u) {
        g_read_copy_cols_le64++;
    } else if (copy_cols < 128u) {
        g_read_copy_cols_65_127++;
    } else if (copy_cols == 128u) {
        g_read_copy_cols_128++;
    } else {
        g_read_copy_cols_gt128++;
    }
}

static void tpu_copy_int32_rect(int32_t * dst,
                                const int32_t * src,
                                uint32_t copy_rows,
                                uint32_t copy_cols,
                                uint32_t packed_rows,
                                uint32_t packed_cols,
                                uint32_t dst_stride,
                                size_t transfer_bytes) {
    const bool bulk_memcpy = copy_rows > 0u &&
            copy_cols > 0u &&
            dst_stride == copy_cols &&
            packed_cols == copy_cols;
    tpu_record_read_copy_shape(copy_rows,
                               copy_cols,
                               packed_rows,
                               packed_cols,
                               dst_stride,
                               transfer_bytes,
                               bulk_memcpy);
    if (bulk_memcpy) {
        memcpy(dst, src, (size_t) copy_rows * (size_t) copy_cols * sizeof(int32_t));
        return;
    }
    for (uint32_t row = 0; row < copy_rows; ++row) {
        tpu_copy_int32_row_inline(dst + (size_t) row * dst_stride,
                                  src + (size_t) row * packed_cols,
                                  copy_cols);
    }
}

static void tpu_copy_int32_row_inline(int32_t * dst, const int32_t * src, uint32_t count) {
    uint32_t i = 0;
    for (; i + 8u <= count; i += 8u) {
        dst[i + 0u] = src[i + 0u];
        dst[i + 1u] = src[i + 1u];
        dst[i + 2u] = src[i + 2u];
        dst[i + 3u] = src[i + 3u];
        dst[i + 4u] = src[i + 4u];
        dst[i + 5u] = src[i + 5u];
        dst[i + 6u] = src[i + 6u];
        dst[i + 7u] = src[i + 7u];
    }
    for (; i < count; ++i) {
        dst[i] = src[i];
    }
}

static int tpu_output_addr_range_valid(uint64_t addr, size_t bytes) {
    if (bytes == 0u || addr > UINT32_MAX) {
        return 0;
    }
    const uint64_t end = addr + (uint64_t) bytes - 1u;
    if (end < addr || end > UINT32_MAX) {
        return 0;
    }
    return addr >= g_tpu_ctx.output_ddr_base && end <= g_tpu_ctx.output_ddr_high;
}

static size_t tpu_op_readback_capacity_bytes(uint32_t M, uint32_t N) {
    const uint32_t tile_m = 32u;
    const uint32_t tile_n = tpu_get_effective_max_nk();
    const uint32_t mt = (M + tile_m - 1u) / tile_m;
    const uint32_t nt = (N + tile_n - 1u) / tile_n;
    const size_t per_tile = tpu_tile_d_transfer_bytes(tile_m, tile_n);

    return tpu_align_up_size((size_t) mt * nt * per_tile, 4096u);
}

static int tpu_op_readback_batch_begin(uint32_t M, uint32_t N) {
    g_tpu_op_readback_active = 0;
    g_tpu_op_readback_used_bytes = 0;
    g_tpu_op_readback_tiles.clear();

    if (M != 1u || !tpu_env_is_truthy_or_default("TPU_V2_M1_OP_READBACK_BATCH", 1)) {
        return 0;
    }

    const size_t needed = tpu_op_readback_capacity_bytes(M, N);
    if (needed == 0u) {
        return 0;
    }
    if (!g_tpu_initialized && tpu_runtime_init() != 0) {
        return 0;
    }
    if (g_tpu_op_d_buf.handle == NULL || g_tpu_op_d_buf_bytes < needed) {
        tpu_soc_fp32_free_buffer(&g_tpu_op_d_buf);
        g_tpu_op_d_buf_bytes = 0;
        if (tpu_soc_fp32_alloc_buffer(&g_tpu_ctx, &g_tpu_op_d_buf, needed) != 0) {
            return 0;
        }
        g_tpu_op_d_buf_bytes = g_tpu_op_d_buf.size;
    }
    if (!tpu_output_addr_range_valid(g_tpu_op_d_buf.device_addr, needed)) {
        if (tpu_log_init_enabled()) {
            fprintf(stderr,
                    "[FPGA] M=1 op readback batch disabled: staging BO [0x%08" PRIx64 ", 0x%08" PRIx64
                    "] outside output aperture [0x%08x, 0x%08x]\n",
                    g_tpu_op_d_buf.device_addr,
                    g_tpu_op_d_buf.device_addr + (uint64_t) needed - 1u,
                    g_tpu_ctx.output_ddr_base,
                    g_tpu_ctx.output_ddr_high);
        }
        return 0;
    }

    g_tpu_op_readback_active = 1;
    return 1;
}

static void tpu_op_readback_batch_abort(void) {
    g_tpu_op_readback_active = 0;
    g_tpu_op_readback_used_bytes = 0;
    g_tpu_op_readback_tiles.clear();
    g_tpu_output_buffer_enabled = 0;
}

static int tpu_op_readback_batch_stage_desc(const tpu_tile_execute_desc_t * desc,
                                            tpu_readback_stage_report_t * report_out,
                                            tpu_tile_execute_result_t * result) {
    if (report_out != NULL) {
        memset(report_out, 0, sizeof(*report_out));
    }
    if (!g_tpu_op_readback_active || desc == NULL || desc->d_final == NULL ||
        desc->d_actual_m == 0u || desc->d_actual_n == 0u ||
        desc->d_actual_m > desc->m || desc->d_actual_n > desc->n ||
        desc->d_final_stride < desc->d_actual_n) {
        return 0;
    }

    const size_t tile_bytes = tpu_tile_d_transfer_bytes(desc->m, desc->n);
    const size_t offset = tpu_align_up_size(g_tpu_op_readback_used_bytes, 64u);
    const uint64_t target = g_tpu_op_d_buf.device_addr + (uint64_t) offset;

    if (offset + tile_bytes > g_tpu_op_d_buf_bytes ||
        !tpu_output_addr_range_valid(target, tile_bytes)) {
        return -1;
    }
    if (tpu_soc_v2_set_output_target_addr(&g_tpu_ctx, (uint32_t) target) != 0) {
        return -1;
    }

    tpu_op_readback_tile_t tile = {};
    tile.offset_bytes = offset;
    tile.packed_bytes = tile_bytes;
    tile.dst = desc->d_final;
    tile.packed_rows = desc->m;
    tile.packed_cols = desc->n;
    tile.copy_rows = desc->d_actual_m;
    tile.copy_cols = desc->d_actual_n;
    tile.dst_stride = desc->d_final_stride;
    g_tpu_op_readback_tiles.push_back(tile);
    g_tpu_op_readback_used_bytes = offset + tile_bytes;
    g_tpu_output_buffer_enabled = 1;

    if (result != NULL) {
        result->readback_direct_final = 1;
    }
    if (report_out != NULL) {
        report_out->used_output_buffer = 1;
        report_out->used_memcpy = 1;
    }
    return 1;
}

static int tpu_op_readback_batch_flush(void) {
    if (!g_tpu_op_readback_active) {
        return 0;
    }
    const auto total_start = std::chrono::steady_clock::now();
    uint64_t sync_us = 0;
    uint64_t copy_us = 0;
    const size_t bytes = g_tpu_op_readback_used_bytes;

    if (bytes != 0u && !g_tpu_op_readback_tiles.empty()) {
        const auto sync_start = std::chrono::steady_clock::now();
        if (tpu_soc_fp32_buffer_sync_from_device(&g_tpu_op_d_buf, bytes) != 0) {
            tpu_op_readback_batch_abort();
            return -1;
        }
        const auto sync_end = std::chrono::steady_clock::now();
        sync_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(sync_end - sync_start).count();

        const auto copy_start = std::chrono::steady_clock::now();
        const uint8_t * base = (const uint8_t *) g_tpu_op_d_buf.mapping;
        for (const tpu_op_readback_tile_t & tile : g_tpu_op_readback_tiles) {
            const int32_t * src = (const int32_t *) (const void *) (base + tile.offset_bytes);
            tpu_copy_int32_rect(tile.dst,
                                src,
                                tile.copy_rows,
                                tile.copy_cols,
                                tile.packed_rows,
                                tile.packed_cols,
                                tile.dst_stride,
                                tile.packed_bytes);
        }
        const auto copy_end = std::chrono::steady_clock::now();
        copy_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count();
    }

    const auto total_end = std::chrono::steady_clock::now();
    const uint64_t total_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
    g_total_read_us += total_us;
    g_total_read_sync_us += sync_us;
    g_total_read_copy_us += copy_us;
    g_total_read_bytes += bytes;
    g_total_read_output_buffer_count++;
    g_total_read_memcpy_count++;
    g_prevd_host_d_readback_count++;
    g_prevd_direct_final_readback_count++;
    g_read_op_batch_count++;
    g_read_op_batch_total_us += total_us;
    g_read_op_batch_sync_us += sync_us;
    g_read_op_batch_copy_us += copy_us;
    g_read_op_batch_bytes += bytes;

    g_tpu_op_readback_active = 0;
    g_tpu_op_readback_used_bytes = 0;
    g_tpu_op_readback_tiles.clear();
    g_tpu_output_buffer_enabled = 0;
    return 0;
}

static bool tpu_async_readback_any_busy_locked(void) {
    for (uint32_t i = 0; i < TPU_ASYNC_D_MAX_BUFS; ++i) {
        if (g_tpu_async_buf_busy[i]) {
            return true;
        }
    }
    return false;
}

static uint32_t tpu_async_readback_ring_count(void) {
    unsigned requested = tpu_env_u32_or_default("TPU_V2_ASYNC_READBACK_BUFS", 2u);
    if (requested < 2u) {
        requested = 2u;
    }
    if (requested > TPU_ASYNC_D_MAX_BUFS) {
        requested = TPU_ASYNC_D_MAX_BUFS;
    }
    return (uint32_t) requested;
}

static int tpu_async_readback_process_job(const tpu_async_readback_job_t & job,
                                          uint64_t * total_us_out,
                                          uint64_t * sync_us_out,
                                          uint64_t * copy_us_out) {
    const auto total_start = std::chrono::steady_clock::now();
    uint64_t sync_us = 0;
    uint64_t copy_us = 0;

    const auto sync_start = std::chrono::steady_clock::now();
    if (tpu_soc_fp32_buffer_sync_from_device(&g_tpu_async_d_buf[job.buf_idx], job.bytes) != 0) {
        return -1;
    }
    const auto sync_end = std::chrono::steady_clock::now();
    sync_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(sync_end - sync_start).count();

    const auto copy_start = std::chrono::steady_clock::now();
    const uint8_t * base = (const uint8_t *) g_tpu_async_d_buf[job.buf_idx].mapping;
    const int32_t * src = (const int32_t *) (const void *) base;
    tpu_copy_int32_rect(job.dst,
                        src,
                        job.copy_rows,
                        job.copy_cols,
                        job.packed_rows,
                        job.packed_cols,
                        job.dst_stride,
                        job.bytes);
    const auto copy_end = std::chrono::steady_clock::now();
    copy_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count();

    const auto total_end = std::chrono::steady_clock::now();
    if (total_us_out != NULL) {
        *total_us_out = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
    }
    if (sync_us_out != NULL) {
        *sync_us_out = sync_us;
    }
    if (copy_us_out != NULL) {
        *copy_us_out = copy_us;
    }
    return 0;
}

static void tpu_async_readback_worker_main(void) {
    for (;;) {
        tpu_async_readback_job_t job = {};
        {
            std::unique_lock<std::mutex> lock(g_tpu_async_mutex);
            g_tpu_async_cv.wait(lock, [] {
                return g_tpu_async_stop || !g_tpu_async_jobs.empty();
            });
            if (g_tpu_async_stop && g_tpu_async_jobs.empty()) {
                return;
            }
            job = g_tpu_async_jobs.front();
            g_tpu_async_jobs.pop_front();
        }

        uint64_t total_us = 0;
        uint64_t sync_us = 0;
        uint64_t copy_us = 0;
        const int rc = tpu_async_readback_process_job(job, &total_us, &sync_us, &copy_us);

        {
            std::lock_guard<std::mutex> lock(g_tpu_async_mutex);
            if (rc != 0) {
                g_tpu_async_failed = 1;
            } else {
                g_total_read_us += total_us;
                g_total_read_sync_us += sync_us;
                g_total_read_copy_us += copy_us;
                g_total_read_bytes += job.bytes;
                g_total_read_output_buffer_count++;
                g_total_read_memcpy_count++;
                g_prevd_host_d_readback_count++;
                g_prevd_direct_final_readback_count++;
                g_read_async_count++;
                g_read_async_total_us += total_us;
                g_read_async_sync_us += sync_us;
                g_read_async_copy_us += copy_us;
                g_read_async_bytes += job.bytes;
                g_tpu_async_jobs_completed++;
            }
            if (job.buf_idx < TPU_ASYNC_D_MAX_BUFS) {
                g_tpu_async_buf_busy[job.buf_idx] = 0;
            }
            g_tpu_async_cv.notify_all();
        }
    }
}

static int tpu_async_readback_ensure_worker(void) {
    std::lock_guard<std::mutex> lock(g_tpu_async_mutex);
    if (g_tpu_async_worker_started) {
        return 0;
    }
    g_tpu_async_stop = 0;
    try {
        g_tpu_async_worker = std::thread(tpu_async_readback_worker_main);
    } catch (...) {
        errno = EAGAIN;
        return -1;
    }
    g_tpu_async_worker_started = 1;
    return 0;
}

static void tpu_async_readback_shutdown(void) {
    {
        std::lock_guard<std::mutex> lock(g_tpu_async_mutex);
        g_tpu_async_readback_active = 0;
        g_tpu_async_stop = 1;
        g_tpu_async_cv.notify_all();
    }
    if (g_tpu_async_worker_started && g_tpu_async_worker.joinable()) {
        g_tpu_async_worker.join();
    }
    {
        std::lock_guard<std::mutex> lock(g_tpu_async_mutex);
        g_tpu_async_jobs.clear();
        memset(g_tpu_async_buf_busy, 0, sizeof(g_tpu_async_buf_busy));
        g_tpu_async_worker_started = 0;
        g_tpu_async_stop = 0;
        g_tpu_async_failed = 0;
        g_tpu_async_readback_active = 0;
        g_tpu_async_ring_count = 0;
        g_tpu_async_next_buf = 0;
    }
}

static int tpu_async_readback_batch_begin(uint32_t M, uint32_t N) {
    (void) N;

    {
        std::unique_lock<std::mutex> lock(g_tpu_async_mutex);
        g_tpu_async_readback_active = 0;
        g_tpu_async_ring_count = 0;
        g_tpu_async_next_buf = 0;
        const bool begin_wait_needed =
            !g_tpu_async_jobs.empty() || tpu_async_readback_any_busy_locked();
        const auto begin_wait_start = std::chrono::steady_clock::now();
        g_tpu_async_cv.wait(lock, [] {
            return g_tpu_async_jobs.empty() && !tpu_async_readback_any_busy_locked();
        });
        if (begin_wait_needed) {
            const uint64_t wait_us = tpu_elapsed_us(begin_wait_start, std::chrono::steady_clock::now());
            g_read_async_batch_begin_wait_us += wait_us;
            g_read_async_batch_begin_wait_count++;
        }
        g_tpu_async_failed = 0;
    }

    if (!tpu_env_is_truthy_or_default("TPU_V2_ASYNC_READBACK", 1) || M == 1u) {
        return 0;
    }
    if (!g_tpu_initialized && tpu_runtime_init() != 0) {
        return 0;
    }

    const uint32_t ring_count = tpu_async_readback_ring_count();
    const size_t needed = tpu_tile_d_transfer_bytes(32u, tpu_get_effective_max_nk());
    if (needed == 0u) {
        return 0;
    }

    bool need_alloc = g_tpu_async_d_buf_bytes < needed;
    for (uint32_t i = 0; i < ring_count; ++i) {
        if (g_tpu_async_d_buf[i].handle == NULL) {
            need_alloc = true;
        }
    }

    if (need_alloc) {
        for (uint32_t i = 0; i < TPU_ASYNC_D_MAX_BUFS; ++i) {
            tpu_soc_fp32_free_buffer(&g_tpu_async_d_buf[i]);
            memset(&g_tpu_async_d_buf[i], 0, sizeof(g_tpu_async_d_buf[i]));
        }
        g_tpu_async_d_buf_bytes = 0;
        for (uint32_t i = 0; i < ring_count; ++i) {
            if (tpu_soc_fp32_alloc_buffer(&g_tpu_ctx, &g_tpu_async_d_buf[i], needed) != 0) {
                for (uint32_t j = 0; j < TPU_ASYNC_D_MAX_BUFS; ++j) {
                    tpu_soc_fp32_free_buffer(&g_tpu_async_d_buf[j]);
                    memset(&g_tpu_async_d_buf[j], 0, sizeof(g_tpu_async_d_buf[j]));
                }
                return 0;
            }
        }
        g_tpu_async_d_buf_bytes = g_tpu_async_d_buf[0].size;
    }

    for (uint32_t i = 0; i < ring_count; ++i) {
        if (!tpu_output_addr_range_valid(g_tpu_async_d_buf[i].device_addr, needed)) {
            if (tpu_log_init_enabled()) {
                fprintf(stderr,
                        "[FPGA] async readback disabled: D BO%u [0x%08" PRIx64 ", 0x%08" PRIx64
                        "] outside output aperture [0x%08x, 0x%08x]\n",
                        i,
                        g_tpu_async_d_buf[i].device_addr,
                        g_tpu_async_d_buf[i].device_addr + (uint64_t) needed - 1u,
                        g_tpu_ctx.output_ddr_base,
                        g_tpu_ctx.output_ddr_high);
            }
            return 0;
        }
    }

    if (tpu_async_readback_ensure_worker() != 0) {
        return 0;
    }

    {
        std::lock_guard<std::mutex> lock(g_tpu_async_mutex);
        g_tpu_async_readback_active = 1;
        g_tpu_async_ring_count = ring_count;
        g_tpu_async_next_buf = 0;
        memset(g_tpu_async_buf_busy, 0, sizeof(g_tpu_async_buf_busy));
    }
    return 1;
}

static int tpu_async_readback_stage_desc(const tpu_tile_execute_desc_t * desc,
                                         tpu_readback_stage_report_t * report_out,
                                         tpu_tile_execute_result_t * result,
                                         tpu_async_readback_job_t * job_out) {
    if (report_out != NULL) {
        memset(report_out, 0, sizeof(*report_out));
    }
    if (job_out != NULL) {
        memset(job_out, 0, sizeof(*job_out));
    }
    if (!g_tpu_async_readback_active) {
        return 0;
    }
    if (desc == NULL || desc->d_final == NULL || job_out == NULL ||
        desc->d_actual_m == 0u || desc->d_actual_n == 0u ||
        desc->d_actual_m > desc->m || desc->d_actual_n > desc->n ||
        desc->d_final_stride < desc->d_actual_n) {
        return 0;
    }

    const size_t tile_bytes = tpu_tile_d_transfer_bytes(desc->m, desc->n);
    uint32_t buf_idx = 0;
    uint64_t stage_wait_us = 0;
    uint64_t stage_wait_count = 0;
    {
        std::unique_lock<std::mutex> lock(g_tpu_async_mutex);
        if (g_tpu_async_failed || g_tpu_async_ring_count == 0u) {
            return -1;
        }
        for (;;) {
            buf_idx = g_tpu_async_next_buf;
            if (!g_tpu_async_buf_busy[buf_idx]) {
                g_tpu_async_buf_busy[buf_idx] = 1;
                g_tpu_async_next_buf = (buf_idx + 1u) % g_tpu_async_ring_count;
                break;
            }
            const auto wait_start = std::chrono::steady_clock::now();
            g_tpu_async_cv.wait(lock);
            stage_wait_us += tpu_elapsed_us(wait_start, std::chrono::steady_clock::now());
            stage_wait_count++;
            if (g_tpu_async_failed) {
                return -1;
            }
        }
        if (stage_wait_count != 0u) {
            g_read_async_stage_wait_us += stage_wait_us;
            g_read_async_stage_wait_count += stage_wait_count;
        }
    }

    const uint64_t target = g_tpu_async_d_buf[buf_idx].device_addr;
    if (tile_bytes > g_tpu_async_d_buf[buf_idx].size ||
        !tpu_output_addr_range_valid(target, tile_bytes) ||
        tpu_soc_v2_set_output_target_addr(&g_tpu_ctx, (uint32_t) target) != 0) {
        std::lock_guard<std::mutex> lock(g_tpu_async_mutex);
        g_tpu_async_buf_busy[buf_idx] = 0;
        g_tpu_async_cv.notify_all();
        return -1;
    }

    job_out->buf_idx = buf_idx;
    job_out->bytes = tile_bytes;
    job_out->dst = desc->d_final;
    job_out->packed_rows = desc->m;
    job_out->packed_cols = desc->n;
    job_out->copy_rows = desc->d_actual_m;
    job_out->copy_cols = desc->d_actual_n;
    job_out->dst_stride = desc->d_final_stride;
    g_tpu_output_buffer_enabled = 1;

    if (result != NULL) {
        result->readback_direct_final = 1;
    }
    if (report_out != NULL) {
        report_out->used_output_buffer = 1;
        report_out->used_memcpy = 1;
    }
    return 1;
}

static int tpu_async_readback_enqueue(const tpu_async_readback_job_t * job) {
    if (job == NULL || job->bytes == 0u || job->buf_idx >= TPU_ASYNC_D_MAX_BUFS) {
        errno = EINVAL;
        return -1;
    }
    std::lock_guard<std::mutex> lock(g_tpu_async_mutex);
    if (!g_tpu_async_readback_active || g_tpu_async_failed) {
        g_tpu_async_buf_busy[job->buf_idx] = 0;
        g_tpu_async_cv.notify_all();
        return -1;
    }
    g_tpu_async_jobs.push_back(*job);
    g_tpu_async_jobs_enqueued++;
    g_tpu_async_cv.notify_all();
    return 0;
}

static int tpu_async_readback_batch_flush(void) {
    std::unique_lock<std::mutex> lock(g_tpu_async_mutex);
    if (!g_tpu_async_readback_active) {
        return 0;
    }
    const bool flush_wait_needed =
        !g_tpu_async_jobs.empty() || tpu_async_readback_any_busy_locked();
    const auto flush_wait_start = std::chrono::steady_clock::now();
    g_tpu_async_cv.wait(lock, [] {
        return g_tpu_async_jobs.empty() && !tpu_async_readback_any_busy_locked();
    });
    if (flush_wait_needed) {
        const uint64_t wait_us = tpu_elapsed_us(flush_wait_start, std::chrono::steady_clock::now());
        g_read_async_flush_wait_us += wait_us;
        g_read_async_flush_wait_count++;
    }
    const int failed = g_tpu_async_failed;
    g_tpu_async_readback_active = 0;
    g_tpu_async_ring_count = 0;
    g_tpu_async_next_buf = 0;
    if (tpu_env_is_truthy_or_default("TPU_V2_ASYNC_READBACK_LOG", 0)) {
        fprintf(stderr,
                "[FPGA] async readback batch flush: enqueued=%" PRIu64 " completed=%" PRIu64 " failed=%d\n",
                g_tpu_async_jobs_enqueued,
                g_tpu_async_jobs_completed,
                failed);
    }
    return failed ? -1 : 0;
}

static void tpu_clear_stage_buffers(void) {
    for (uint32_t i = 0; i < TPU_SEQUENCE_MAX_RUNS; ++i) {
        std::vector<uint64_t>().swap(g_tpu_a_stage[i]);
        std::vector<uint64_t>().swap(g_tpu_b_stage[i]);
    }
}

static void tpu_log_device_b_cache(void) {
    if (!tpu_log_init_enabled() && !tpu_env_is_truthy_or_default("TPU_V2_LOG_B_CACHE", 0)) {
        return;
    }
    if (g_b_device_cache_hits == 0 && g_b_device_cache_misses == 0 && g_b_device_cache_fallbacks == 0) {
        return;
    }

    fprintf(stderr,
            "[FPGA] Device B cache: entries=%zu hits=%" PRIu64 " misses=%" PRIu64
            " fallbacks=%" PRIu64 " budget_rejects=%" PRIu64
            " aperture_rejects=%" PRIu64 " resident=%.3f MiB saved_copy=%.3f MiB max=%.3f MiB\n",
            g_tpu_device_b_cache.size(),
            g_b_device_cache_hits,
            g_b_device_cache_misses,
            g_b_device_cache_fallbacks,
            g_b_device_cache_budget_rejects,
            g_b_device_cache_aperture_rejects,
            g_b_device_cache_resident_bytes / (1024.0 * 1024.0),
            g_b_device_cache_saved_bytes / (1024.0 * 1024.0),
            tpu_device_b_cache_max_bytes() / (1024.0 * 1024.0));
    if (g_b_packed_arena_hits != 0 || g_b_packed_arena_misses != 0) {
        fprintf(stderr,
                "[FPGA] Matrix-task B arena: entries=%zu hits=%" PRIu64
                " misses=%" PRIu64 " budget_rejects=%" PRIu64
                " resident=%.3f MiB max=%.3f MiB\n",
                g_tpu_device_b_matrix_cache.size(),
                g_b_packed_arena_hits,
                g_b_packed_arena_misses,
                g_b_matrix_cache_budget_rejects,
                g_b_matrix_cache_resident_bytes / (1024.0 * 1024.0),
                tpu_matrix_task_b_cache_max_bytes() / (1024.0 * 1024.0));
    }
    if (g_pl_op_desc_count != 0) {
        fprintf(stderr,
                "[FPGA] PL op descriptor: calls=%" PRIu64 " tiles=%" PRIu64
                " program=%.3f ms wait=%.3f ms polls=%" PRIu64
                " poll_sleep_us=%u"
                " avg_wait=%.3f us\n",
                g_pl_op_desc_count,
                g_pl_op_desc_tile_count,
                g_pl_op_desc_program_us / 1000.0,
                g_pl_op_desc_wait_us / 1000.0,
                g_pl_op_desc_status_polls,
                tpu_pl_op_poll_sleep_us(),
                (double) g_pl_op_desc_wait_us / (double) g_pl_op_desc_count);
    }
}

static void tpu_clear_device_b_cache(void) {
    for (auto & it : g_tpu_device_b_cache) {
        tpu_soc_fp32_free_buffer(&it.second.buf);
    }
    g_tpu_device_b_cache.clear();
    for (auto & it : g_tpu_device_b_matrix_cache) {
        tpu_soc_fp32_free_buffer(&it.second.buf);
    }
    g_tpu_device_b_matrix_cache.clear();
    g_tpu_device_b_matrix_owner_cache.clear();
    g_b_device_cache_resident_bytes = 0;
    g_b_matrix_cache_resident_bytes = 0;
}

static void tpu_dump_ctx_failure(tpu_soc_fp32_t * ctx, const char * label) {
    if (ctx == NULL) {
        return;
    }

    fprintf(stderr, "%s\n", label);
    tpu_soc_fp32_dump_cdma_trace(stderr, &ctx->last_cdma_trace);

    tpu_soc_fp32_regs_t regs;
    if (tpu_soc_fp32_read_regs(ctx, &regs) == 0) {
        fprintf(stderr, "Register snapshot:\n");
        tpu_soc_fp32_dump_regs(stderr, &regs);
    }
}

static int tpu_runtime_init_buffer_pool(void) {
    size_t a_bytes = 0;
    size_t b_bytes = 0;
    size_t c_bytes = 0;
    size_t d_bytes = 0;
    size_t d_buf_bytes = 0;
    int output_buffer_enabled = 0;
    const uint32_t max_m = 32u;
    const uint32_t max_nk = tpu_get_effective_max_nk();
    const uint32_t max_n = max_nk;
    const uint32_t max_k = max_nk;

    if (g_tpu_buffer_pool_initialized) {
        return 0;
    }

    if (tpu_soc_v2_validate_sizes(max_m, max_n, max_k, &a_bytes, &b_bytes, &c_bytes, &d_bytes) != 0) {
        if (max_nk > 32u) {
            fprintf(stderr,
                    "[FPGA] GGML_FPGA_MAX_NK=%u requested tile family %s, "
                    "but current runtime rejected max tile M=%u N=%u K=%u\n",
                    tpu_get_requested_max_nk(),
                    tpu_tile_family_name(),
                    max_m,
                    max_n,
                    max_k);
        }
        perror("tpu_runtime_init_buffer_pool: tpu_soc_v2_validate_sizes");
        return -1;
    }
    d_buf_bytes = c_bytes;
    (void) d_bytes;

    if (tpu_log_init_enabled()) {
        fprintf(stderr,
                "[FPGA] buffer pool init family=%s requested_max_nk=%u effective_max_nk=%u "
                "max_tile=%ux%ux%u bytes(A/B/Dpacked)=%zu/%zu/%zu no_host_c=1\n",
                tpu_tile_family_name(),
                tpu_get_requested_max_nk(),
                tpu_get_effective_max_nk(),
                max_m,
                max_n,
                max_k,
                a_bytes,
                b_bytes,
                d_buf_bytes);
    }

    memset(&g_tpu_a_buf, 0, sizeof(g_tpu_a_buf));
    memset(&g_tpu_b_buf, 0, sizeof(g_tpu_b_buf));
    memset(&g_tpu_d_buf, 0, sizeof(g_tpu_d_buf));
    tpu_clear_stage_buffers();

    const auto alloc_start = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < TPU_SEQUENCE_MAX_RUNS; ++i) {
        if (tpu_soc_fp32_alloc_buffer(&g_tpu_ctx, &g_tpu_a_buf[i], a_bytes) != 0 ||
            tpu_soc_fp32_alloc_buffer(&g_tpu_ctx, &g_tpu_b_buf[i], b_bytes) != 0) {
            perror("tpu_runtime_init_buffer_pool: tpu_soc_fp32_alloc_buffer");
            tpu_dump_ctx_failure(&g_tpu_ctx, "Buffer-pool allocation failure; runtime state:");
            for (uint32_t j = 0; j <= i; ++j) {
                tpu_soc_fp32_free_buffer(&g_tpu_b_buf[j]);
                tpu_soc_fp32_free_buffer(&g_tpu_a_buf[j]);
            }
            memset(&g_tpu_a_buf, 0, sizeof(g_tpu_a_buf));
            memset(&g_tpu_b_buf, 0, sizeof(g_tpu_b_buf));
            memset(&g_tpu_d_buf, 0, sizeof(g_tpu_d_buf));
            tpu_clear_stage_buffers();
            return -1;
        }
    }
    if (tpu_soc_fp32_alloc_buffer(&g_tpu_ctx, &g_tpu_d_buf, d_buf_bytes) != 0) {
        perror("tpu_runtime_init_buffer_pool: tpu_soc_fp32_alloc_buffer");
        tpu_dump_ctx_failure(&g_tpu_ctx, "Buffer-pool allocation failure; runtime state:");
        for (uint32_t i = 0; i < TPU_SEQUENCE_MAX_RUNS; ++i) {
            tpu_soc_fp32_free_buffer(&g_tpu_b_buf[i]);
            tpu_soc_fp32_free_buffer(&g_tpu_a_buf[i]);
        }
        memset(&g_tpu_a_buf, 0, sizeof(g_tpu_a_buf));
        memset(&g_tpu_b_buf, 0, sizeof(g_tpu_b_buf));
        memset(&g_tpu_d_buf, 0, sizeof(g_tpu_d_buf));
        tpu_clear_stage_buffers();
        return -1;
    }
    const auto alloc_end = std::chrono::steady_clock::now();

    if (tpu_soc_v2_try_enable_output_buffer(&g_tpu_ctx, &g_tpu_d_buf, &output_buffer_enabled) != 0) {
        perror("tpu_runtime_init_buffer_pool: tpu_soc_v2_try_enable_output_buffer");
        tpu_dump_ctx_failure(&g_tpu_ctx, "Output-buffer enable failure during buffer-pool init; runtime state:");
        tpu_soc_fp32_free_buffer(&g_tpu_d_buf);
        for (uint32_t i = 0; i < TPU_SEQUENCE_MAX_RUNS; ++i) {
            tpu_soc_fp32_free_buffer(&g_tpu_b_buf[i]);
            tpu_soc_fp32_free_buffer(&g_tpu_a_buf[i]);
        }
        memset(&g_tpu_a_buf, 0, sizeof(g_tpu_a_buf));
        memset(&g_tpu_b_buf, 0, sizeof(g_tpu_b_buf));
        memset(&g_tpu_d_buf, 0, sizeof(g_tpu_d_buf));
        tpu_clear_stage_buffers();
        return -1;
    }

    g_buffer_pool_init_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(alloc_end - alloc_start).count();
    g_tpu_a_buf_bytes = a_bytes;
    g_tpu_b_buf_bytes = b_bytes;
    g_tpu_d_buf_bytes = d_buf_bytes;
    g_tpu_output_buffer_enabled = output_buffer_enabled;
    g_tpu_buffer_pool_initialized = 1;

    if (tpu_log_init_enabled()) {
        fprintf(stderr,
                "[FPGA] buffer pool ready family=%s max_nk=%u A/B slots=%u A_bytes=%zu B_bytes=%zu D=%zu bytes (init=%.3f ms, output_buffer=%s)\n",
                tpu_tile_family_name(),
                max_nk,
                TPU_SEQUENCE_MAX_RUNS,
                g_tpu_a_buf_bytes,
                g_tpu_b_buf_bytes,
                g_tpu_d_buf_bytes,
                g_buffer_pool_init_us / 1000.0,
                g_tpu_output_buffer_enabled ? "on" : "off");
    }

    return 0;
}

int tpu_runtime_init(void) {
    std::lock_guard<std::mutex> lock(g_tpu_mutex);

    if (g_tpu_initialized) {
        return 0;
    }

    memset(&g_tpu_ctx, 0, sizeof(g_tpu_ctx));

    if (geteuid() != 0) {
        errno = EACCES;
        perror("tpu_runtime_init: please run as root so /dev/mem and XRT are accessible");
        return -1;
    }

    if (getenv("XILINX_XRT") == NULL) {
        (void) setenv("XILINX_XRT", "/usr", 0);
    }
    if (getenv("MTMD_BACKEND_DEVICE") == NULL) {
        (void) setenv("MTMD_BACKEND_DEVICE", "FPGA", 0);
    }
    if (getenv("TPU_SOC_PLATFORM") == NULL) {
        (void) setenv("TPU_SOC_PLATFORM", "kv260", 0);
    }
    if (getenv("TPU_HW_INGRESS") == NULL) {
        (void) setenv("TPU_HW_INGRESS", "1", 0);
    }
    if (getenv("TPU_V2_WAIT_INGRESS_SAFE") == NULL) {
        (void) setenv("TPU_V2_WAIT_INGRESS_SAFE", "1", 0);
    }
    if (getenv("TPU_V2_OUTPUT_READ_MODE") == NULL) {
        (void) setenv("TPU_V2_OUTPUT_READ_MODE", "memcpy", 0);
    }
    if (getenv("TPU_V2_SKIP_CLEAR_IF_OUTBUF") == NULL) {
        (void) setenv("TPU_V2_SKIP_CLEAR_IF_OUTBUF", "1", 0);
    }
    if (getenv("TPU_V2_SKIP_SEQUENCE_SOFT_RESET") == NULL) {
        (void) setenv("TPU_V2_SKIP_SEQUENCE_SOFT_RESET", "1", 0);
    }
    if (getenv("TPU_V2_DEVICE_B_CACHE") == NULL) {
        (void) setenv("TPU_V2_DEVICE_B_CACHE", "1", 0);
    }
    if (getenv("TPU_V2_DEVICE_B_CACHE_MAX_MB") == NULL) {
        (void) setenv("TPU_V2_DEVICE_B_CACHE_MAX_MB", "256", 0);
    }
    if (getenv("TPU_V2_MATRIX_TASK_B_CACHE_MAX_MB") == NULL) {
        (void) setenv("TPU_V2_MATRIX_TASK_B_CACHE_MAX_MB", "256", 0);
    }
    if (getenv("TPU_V2_ASYNC_READBACK") == NULL) {
        (void) setenv("TPU_V2_ASYNC_READBACK", "1", 0);
    }
    if (getenv("TPU_V2_ASYNC_READBACK_BUFS") == NULL) {
        (void) setenv("TPU_V2_ASYNC_READBACK_BUFS", "2", 0);
    }
    if (getenv("TPU_V2_PL_OP_DESCRIPTOR") == NULL) {
        (void) setenv("TPU_V2_PL_OP_DESCRIPTOR", "1", 0);
    }
    if (getenv("TPU_V2_PL_OP_DESCRIPTOR_MAX_M") == NULL) {
        (void) setenv("TPU_V2_PL_OP_DESCRIPTOR_MAX_M", "1", 0);
    }
    if (getenv("TPU_V2_R5_MATRIX_TASK_AB_PRELOAD") == NULL) {
        (void) setenv("TPU_V2_R5_MATRIX_TASK_AB_PRELOAD", "1", 0);
    }
    if (getenv("TPU_V2_R5_MATRIX_TASK_DESC_PRELOAD") == NULL) {
        (void) setenv("TPU_V2_R5_MATRIX_TASK_DESC_PRELOAD", "0", 0);
    }
    if (getenv("TPU_V2_R5_MATRIX_TASK_SEQ_INNER_K") == NULL) {
        (void) setenv("TPU_V2_R5_MATRIX_TASK_SEQ_INNER_K", "0", 0);
    }
    if (getenv("TPU_V2_R5_MATRIX_TASK_SEQ_INNER_K_MAX_M") == NULL) {
        (void) setenv("TPU_V2_R5_MATRIX_TASK_SEQ_INNER_K_MAX_M", "1", 0);
    }
    if (getenv("TPU_V2_R5_MATRIX_TASK_SEQ_INNER_K_MAX_RUNS") == NULL) {
        (void) setenv("TPU_V2_R5_MATRIX_TASK_SEQ_INNER_K_MAX_RUNS", "8", 0);
    }
    if (getenv("TPU_V2_R5_MATRIX_TASK_SEQ_INNER_K_MIN_K") == NULL) {
        (void) setenv("TPU_V2_R5_MATRIX_TASK_SEQ_INNER_K_MIN_K", "128", 0);
    }
    if (getenv("TPU_V2_A53_SEQUENCE") == NULL) {
        (void) setenv("TPU_V2_A53_SEQUENCE", "1", 0);
    }

    if (tpu_log_init_enabled()) {
        fprintf(stderr,
                "[FPGA] runtime init config XILINX_XRT=%s MTMD_BACKEND_DEVICE=%s GGML_FPGA_MAX_NK=%s family=%s "
                "TPU_SOC_PLATFORM=%s TPU_HW_INGRESS=%s TPU_GEMV_M1=%s TPU_V2_WAIT_INGRESS_SAFE=%s "
                "TPU_V2_OUTPUT_READ_MODE=%s TPU_V2_SKIP_CLEAR_IF_OUTBUF=%s TPU_V2_SKIP_SEQUENCE_SOFT_RESET=%s "
                "TPU_V2_DEVICE_B_CACHE=%s TPU_V2_DEVICE_B_CACHE_MAX_MB=%s "
                "TPU_V2_MATRIX_TASK_B_CACHE_MAX_MB=%s\n",
                tpu_env_or_unset("XILINX_XRT"),
                tpu_env_or_unset("MTMD_BACKEND_DEVICE"),
                tpu_env_or_unset("GGML_FPGA_MAX_NK"),
                tpu_tile_family_name(),
                tpu_env_or_unset("TPU_SOC_PLATFORM"),
                tpu_env_or_unset("TPU_HW_INGRESS"),
                tpu_env_or_unset("TPU_GEMV_M1"),
                tpu_env_or_unset("TPU_V2_WAIT_INGRESS_SAFE"),
                tpu_env_or_unset("TPU_V2_OUTPUT_READ_MODE"),
                tpu_env_or_unset("TPU_V2_SKIP_CLEAR_IF_OUTBUF"),
                tpu_env_or_unset("TPU_V2_SKIP_SEQUENCE_SOFT_RESET"),
                tpu_env_or_unset("TPU_V2_DEVICE_B_CACHE"),
                tpu_env_or_unset("TPU_V2_DEVICE_B_CACHE_MAX_MB"),
                tpu_env_or_unset("TPU_V2_MATRIX_TASK_B_CACHE_MAX_MB"));
    }

    if (tpu_soc_fp32_open(
                &g_tpu_ctx,
                TPU_SOC_FP32_CSR_BASE_DEFAULT,
                TPU_SOC_FP32_CDMA_BASE_DEFAULT,
                TPU_SOC_FP32_BRAM_BASE_DEFAULT,
                0u,
                0u,
                "/dev/mem") != 0) {
        perror("tpu_runtime_init: tpu_soc_fp32_open");
        return -1;
    }
    if (tpu_log_init_enabled()) {
        fprintf(stderr, "[FPGA] runtime init stage=open ok\n");
    }

    if (tpu_soc_fp32_preflight(&g_tpu_ctx, NULL) != 0) {
        perror("tpu_runtime_init: tpu_soc_fp32_preflight");
        tpu_dump_ctx_failure(&g_tpu_ctx, "Preflight failed; runtime state:");
        tpu_soc_fp32_close(&g_tpu_ctx);
        memset(&g_tpu_ctx, 0, sizeof(g_tpu_ctx));
        return -1;
    }
    if (tpu_log_init_enabled()) {
        fprintf(stderr, "[FPGA] runtime init stage=preflight ok\n");
    }

    if (tpu_runtime_init_buffer_pool() != 0) {
        if (tpu_log_init_enabled()) {
            fprintf(stderr, "[FPGA] runtime init stage=buffer_pool failed\n");
        }
        tpu_soc_fp32_close(&g_tpu_ctx);
        memset(&g_tpu_ctx, 0, sizeof(g_tpu_ctx));
        return -1;
    }

    if (tpu_log_init_enabled()) {
        fprintf(stderr,
                "[FPGA] real TPU PREV_D dual-AB runtime active (XILINX_XRT=%s)\n",
                getenv("XILINX_XRT"));
    }

    g_tpu_initialized = 1;
    return 0;
}

void tpu_runtime_cleanup(void) {
    std::lock_guard<std::mutex> lock(g_tpu_mutex);

    if (!g_tpu_initialized) {
        return;
    }

    tpu_log_device_b_cache();
    tpu_clear_device_b_cache();
    tpu_op_readback_batch_abort();
    tpu_async_readback_shutdown();

    tpu_soc_fp32_free_buffer(&g_tpu_matmul_c_buf);
    tpu_soc_fp32_free_buffer(&g_tpu_matmul_b_buf);
    tpu_soc_fp32_free_buffer(&g_tpu_matmul_a_buf);
    tpu_soc_fp32_free_buffer(&g_tpu_d_buf);
    tpu_soc_fp32_free_buffer(&g_tpu_op_d_buf);
    for (uint32_t i = 0; i < TPU_ASYNC_D_MAX_BUFS; ++i) {
        tpu_soc_fp32_free_buffer(&g_tpu_async_d_buf[i]);
    }
    for (uint32_t i = 0; i < TPU_SEQUENCE_MAX_RUNS; ++i) {
        tpu_soc_fp32_free_buffer(&g_tpu_b_buf[i]);
        tpu_soc_fp32_free_buffer(&g_tpu_a_buf[i]);
    }
    memset(&g_tpu_a_buf, 0, sizeof(g_tpu_a_buf));
    memset(&g_tpu_b_buf, 0, sizeof(g_tpu_b_buf));
    memset(&g_tpu_d_buf, 0, sizeof(g_tpu_d_buf));
    memset(&g_tpu_op_d_buf, 0, sizeof(g_tpu_op_d_buf));
    memset(&g_tpu_async_d_buf, 0, sizeof(g_tpu_async_d_buf));
    memset(&g_tpu_matmul_a_buf, 0, sizeof(g_tpu_matmul_a_buf));
    memset(&g_tpu_matmul_b_buf, 0, sizeof(g_tpu_matmul_b_buf));
    memset(&g_tpu_matmul_c_buf, 0, sizeof(g_tpu_matmul_c_buf));
    tpu_clear_stage_buffers();
    g_tpu_a_buf_bytes = 0;
    g_tpu_b_buf_bytes = 0;
    g_tpu_d_buf_bytes = 0;
    g_tpu_op_d_buf_bytes = 0;
    g_tpu_async_d_buf_bytes = 0;
    g_tpu_matmul_a_buf_bytes = 0;
    g_tpu_matmul_b_buf_bytes = 0;
    g_tpu_matmul_c_buf_bytes = 0;
    g_tpu_output_buffer_enabled = 0;
    g_tpu_buffer_pool_initialized = 0;

    tpu_soc_fp32_close(&g_tpu_ctx);
    memset(&g_tpu_ctx, 0, sizeof(g_tpu_ctx));
    g_tpu_initialized = 0;
}

void tpu_runtime_get_stats(tpu_runtime_stats_t * stats) {
    if (stats == NULL) {
        return;
    }

    memset(stats, 0, sizeof(*stats));
    stats->tile_call_count = g_tile_call_count;
    stats->prevd_chain_runs = g_prevd_chain_runs;
    stats->prevd_over_signoff_runs = g_prevd_over_signoff_runs;
    stats->prevd_ab_preload_count = g_prevd_ab_preload_count;
    stats->prevd_host_c_upload_count = g_prevd_host_c_upload_count;
    stats->prevd_host_d_readback_count = g_prevd_host_d_readback_count;
    stats->prevd_direct_final_readback_count = g_prevd_direct_final_readback_count;
    stats->total_wait_done_us = g_total_wait_done_us;
    stats->total_read_us = g_total_read_us;
    stats->total_read_sync_us = g_total_read_sync_us;
    stats->total_read_copy_us = g_total_read_copy_us;
    stats->total_read_unpack_us = g_total_read_unpack_us;
    stats->total_read_mmio_us = g_total_read_mmio_us;
    stats->total_read_debug_dump_us = g_total_read_debug_dump_us;
    stats->total_read_bytes = g_total_read_bytes;
    stats->total_read_output_buffer_count = g_total_read_output_buffer_count;
    stats->total_read_memcpy_count = g_total_read_memcpy_count;
    stats->total_read_unpack_count = g_total_read_unpack_count;
    stats->total_read_mmio_count = g_total_read_mmio_count;
    stats->read_op_batch_count = g_read_op_batch_count;
    stats->read_op_batch_total_us = g_read_op_batch_total_us;
    stats->read_op_batch_sync_us = g_read_op_batch_sync_us;
    stats->read_op_batch_copy_us = g_read_op_batch_copy_us;
    stats->read_op_batch_bytes = g_read_op_batch_bytes;
    stats->read_async_count = g_read_async_count;
    stats->read_async_total_us = g_read_async_total_us;
    stats->read_async_sync_us = g_read_async_sync_us;
    stats->read_async_copy_us = g_read_async_copy_us;
    stats->read_async_bytes = g_read_async_bytes;
    stats->read_async_batch_begin_wait_us = g_read_async_batch_begin_wait_us;
    stats->read_async_batch_begin_wait_count = g_read_async_batch_begin_wait_count;
    stats->read_async_stage_wait_us = g_read_async_stage_wait_us;
    stats->read_async_stage_wait_count = g_read_async_stage_wait_count;
    stats->read_async_flush_wait_us = g_read_async_flush_wait_us;
    stats->read_async_flush_wait_count = g_read_async_flush_wait_count;
    stats->read_final_desc_count = g_read_final_desc_count;
    stats->read_final_desc_total_us = g_read_final_desc_total_us;
    stats->read_final_desc_sync_us = g_read_final_desc_sync_us;
    stats->read_final_desc_copy_us = g_read_final_desc_copy_us;
    stats->read_final_desc_unpack_us = g_read_final_desc_unpack_us;
    stats->read_final_desc_mmio_us = g_read_final_desc_mmio_us;
    stats->read_final_desc_bytes = g_read_final_desc_bytes;
    stats->read_copy_job_count = g_read_copy_job_count;
    stats->read_copy_row_count = g_read_copy_row_count;
    stats->read_copy_elem_count = g_read_copy_elem_count;
    stats->read_copy_useful_bytes = g_read_copy_useful_bytes;
    stats->read_copy_transfer_bytes = g_read_copy_transfer_bytes;
    stats->read_copy_padded_jobs = g_read_copy_padded_jobs;
    stats->read_copy_strided_jobs = g_read_copy_strided_jobs;
    stats->read_copy_bulk_candidate_jobs = g_read_copy_bulk_candidate_jobs;
    stats->read_copy_bulk_candidate_bytes = g_read_copy_bulk_candidate_bytes;
    stats->read_copy_bulk_memcpy_jobs = g_read_copy_bulk_memcpy_jobs;
    stats->read_copy_bulk_memcpy_bytes = g_read_copy_bulk_memcpy_bytes;
    stats->read_copy_row_memcpy_rows = g_read_copy_row_memcpy_rows;
    stats->read_copy_rows_1 = g_read_copy_rows_1;
    stats->read_copy_rows_2_15 = g_read_copy_rows_2_15;
    stats->read_copy_rows_16 = g_read_copy_rows_16;
    stats->read_copy_rows_17_31 = g_read_copy_rows_17_31;
    stats->read_copy_rows_32 = g_read_copy_rows_32;
    stats->read_copy_rows_gt32 = g_read_copy_rows_gt32;
    stats->read_copy_cols_le16 = g_read_copy_cols_le16;
    stats->read_copy_cols_le32 = g_read_copy_cols_le32;
    stats->read_copy_cols_le64 = g_read_copy_cols_le64;
    stats->read_copy_cols_65_127 = g_read_copy_cols_65_127;
    stats->read_copy_cols_128 = g_read_copy_cols_128;
    stats->read_copy_cols_gt128 = g_read_copy_cols_gt128;
    stats->total_prepare_us = g_total_prepare_us;
    stats->total_pack_us = g_total_pack_us;
    stats->total_pack_a_us = g_total_pack_a_us;
    stats->total_pack_b_us = g_total_pack_b_us;
    stats->total_copy_us = g_total_copy_us;
    stats->total_copy_a_us = g_total_copy_a_us;
    stats->total_copy_b_us = g_total_copy_b_us;
    stats->total_control_us = g_total_control_us;
    stats->total_preload_us = g_total_preload_wall_us;
    stats->total_preload_sync_us = g_total_preload_sync_us;
    stats->total_preload_wait_us = g_total_preload_write_ready_wait_us;
    stats->total_preload_setup_us = g_total_preload_path_setup_us;
    stats->total_preload_xfer_us = g_total_preload_transfer_us;
    stats->total_start_us = g_total_start_wall_us;
    stats->total_start_sync_us = g_total_sync_us;
    stats->total_start_ready_us = g_total_ready_wait_us;
    stats->total_start_setup_us = g_total_path_setup_us;
    stats->total_start_mmio_us = g_total_start_us;
    stats->total_start_xfer_us = g_total_transfer_us;
    stats->total_wait_prev_us = g_total_wait_prev_us;
    stats->total_wait_final_us = g_total_wait_final_us;
    stats->total_cleanup_us = g_total_cleanup_us;
    stats->total_log_us = g_total_log_us;
    stats->total_unlock_us = g_total_unlock_us;
    stats->total_other_us = g_total_other_us;
    stats->prevd_sequence_program_count = g_prevd_sequence_program_count;
    stats->prevd_sequence_fallback_count = g_prevd_sequence_fallback_count;
    stats->total_sequence_us = g_total_sequence_us;
    stats->total_sequence_sync_us = g_total_sequence_sync_us;
    stats->total_sequence_ready_us = g_total_sequence_ready_us;
    stats->total_sequence_program_us = g_total_sequence_program_us;
    stats->total_sequence_wait_done_us = g_total_sequence_wait_done_us;
    stats->auto_next_program_count = g_prevd_auto_next_program_count;
    stats->auto_next_fallback_count = g_prevd_auto_next_fallback_count;
    stats->matmul_task_count = g_matmul_task_count;
    stats->matmul_task_tile_count = g_matmul_task_tile_count;
    stats->matmul_task_fallback_count = g_matmul_task_fallback_count;
    stats->matmul_task_r5_cycles = g_matmul_task_r5_cycles;
    stats->matmul_task_fpga_wait_cycles = g_matmul_task_fpga_wait_cycles;
    stats->matmul_task_total_us = g_matmul_task_total_us;
    stats->matmul_task_entry_setup_us = g_matmul_task_entry_setup_us;
    stats->matmul_task_lock_us = g_matmul_task_lock_us;
    stats->matmul_task_reserve_us = g_matmul_task_reserve_us;
    stats->matmul_task_prepare_b_wall_us = g_matmul_task_prepare_b_wall_us;
    stats->matmul_task_copy_a_wall_us = g_matmul_task_copy_a_wall_us;
    stats->matmul_task_sync_a_us = g_matmul_task_sync_a_us;
    stats->matmul_task_addr_check_us = g_matmul_task_addr_check_us;
    stats->matmul_task_submit_us = g_matmul_task_submit_us;
    stats->matmul_task_output_sync_us = g_matmul_task_output_sync_us;
    stats->matmul_task_output_view_us = g_matmul_task_output_view_us;
    stats->matmul_task_output_copy_us = g_matmul_task_output_copy_us;
    stats->matmul_task_output_copy_jobs = g_matmul_task_output_copy_jobs;
    stats->matmul_task_output_copy_bytes = g_matmul_task_output_copy_bytes;
    stats->matmul_task_output_copy_direct_jobs = g_matmul_task_output_copy_direct_jobs;
    stats->matmul_task_output_copy_direct_us = g_matmul_task_output_copy_direct_us;
    stats->matmul_task_output_copy_direct_bytes = g_matmul_task_output_copy_direct_bytes;
    stats->matmul_task_output_copy_row_jobs = g_matmul_task_output_copy_row_jobs;
    stats->matmul_task_output_copy_row_us = g_matmul_task_output_copy_row_us;
    stats->matmul_task_output_copy_row_bytes = g_matmul_task_output_copy_row_bytes;
    stats->matmul_task_output_copy_row_rows = g_matmul_task_output_copy_row_rows;
    stats->matmul_task_output_copy_padded_jobs = g_matmul_task_output_copy_padded_jobs;
    stats->matmul_task_output_copy_padded_us = g_matmul_task_output_copy_padded_us;
    stats->matmul_task_output_copy_m1_jobs = g_matmul_task_output_copy_m1_jobs;
    stats->matmul_task_output_copy_m1_us = g_matmul_task_output_copy_m1_us;
    stats->matmul_task_output_copy_m32_jobs = g_matmul_task_output_copy_m32_jobs;
    stats->matmul_task_output_copy_m32_us = g_matmul_task_output_copy_m32_us;
    stats->matmul_task_output_copy_n64_jobs = g_matmul_task_output_copy_n64_jobs;
    stats->matmul_task_output_copy_n64_us = g_matmul_task_output_copy_n64_us;
    stats->matmul_task_output_copy_n128_jobs = g_matmul_task_output_copy_n128_jobs;
    stats->matmul_task_output_copy_n128_us = g_matmul_task_output_copy_n128_us;
    stats->matmul_task_output_copy_n_other_jobs = g_matmul_task_output_copy_n_other_jobs;
    stats->matmul_task_output_copy_n_other_us = g_matmul_task_output_copy_n_other_us;
    stats->matmul_task_post_us = g_matmul_task_post_us;
    stats->matmul_task_post_stats_us = g_matmul_task_post_stats_us;
    stats->matmul_task_free_us = g_matmul_task_free_us;
    stats->matmul_task_residual_us = g_matmul_task_residual_us;
    stats->pl_op_desc_count = g_pl_op_desc_count;
    stats->pl_op_desc_tile_count = g_pl_op_desc_tile_count;
    stats->pl_op_desc_program_us = g_pl_op_desc_program_us;
    stats->pl_op_desc_wait_us = g_pl_op_desc_wait_us;
    stats->pl_op_desc_status_polls = g_pl_op_desc_status_polls;
    stats->b_packed_arena_hits = g_b_packed_arena_hits;
    stats->b_packed_arena_misses = g_b_packed_arena_misses;
}

static const char * tpu_a_layout_name(tpu_a_layout_t layout) {
    switch (layout) {
        case TPU_A_LAYOUT_RAW_MK: return "raw-mk";
        case TPU_A_LAYOUT_MK:
        default: return "mk-pack";
    }
}

static const char * tpu_b_layout_name(tpu_b_layout_t layout) {
    switch (layout) {
        case TPU_B_LAYOUT_KN: return "kn-cache";
        case TPU_B_LAYOUT_DEVICE_PACKED: return "device-packed";
        case TPU_B_LAYOUT_NK:
        default: return "nk-pack";
    }
}

static const char * tpu_c_source_name(tpu_c_source_t source) {
    switch (source) {
        case TPU_C_SOURCE_ZERO: return "ZERO";
        case TPU_C_SOURCE_DEVICE_ACCUM: return "PREV_D";
        case TPU_C_SOURCE_HOST: return "HOST";
        default: return "unknown";
    }
}

static const char * tpu_d_dest_name(tpu_d_dest_t dest) {
    switch (dest) {
        case TPU_D_DEST_HOST_READBACK: return "READBACK";
        case TPU_D_DEST_DEVICE_ACCUM: return "RETAIN";
        default: return "unknown";
    }
}

static int tpu_accum_key_equal(const tpu_accum_key_t * a, const tpu_accum_key_t * b) {
    return a->op_id == b->op_id &&
           a->m0 == b->m0 &&
           a->n0 == b->n0 &&
           a->m == b->m &&
           a->n == b->n;
}

static uint32_t tpu_desc_c_mode(const tpu_tile_execute_desc_t * desc) {
    return desc->c_source == TPU_C_SOURCE_ZERO ? TPU_SOC_V2_C_MODE_ZERO : TPU_SOC_V2_C_MODE_PREV_D;
}

static uint32_t tpu_desc_d_mode(const tpu_tile_execute_desc_t * desc) {
    return desc->d_dest == TPU_D_DEST_DEVICE_ACCUM ? TPU_SOC_V2_D_MODE_RETAIN : TPU_SOC_V2_D_MODE_READBACK;
}

static void tpu_accumulate_execute_report(const tpu_soc_v2_execute_report_t * report, uint64_t wall_us) {
    const uint64_t sync_us = tpu_ns_to_us(report->sync_ns);
    const uint64_t soft_reset_us = tpu_ns_to_us(report->soft_reset_ns);
    const uint64_t cdma_reset_us = tpu_ns_to_us(report->cdma_reset_ns);
    const uint64_t ready_wait_us = tpu_ns_to_us(report->ready_wait_ns);
    const uint64_t path_setup_us = tpu_ns_to_us(report->path_setup_ns);
    const uint64_t start_us = tpu_ns_to_us(report->start_ns);
    const uint64_t post_start_delay_us = tpu_ns_to_us(report->post_start_delay_ns);
    const uint64_t transfer_us = tpu_ns_to_us(report->transfer_ns);
    const uint64_t tail_fixup_us = tpu_ns_to_us(report->tail_fixup_ns);
    const uint64_t wait_done_us = tpu_ns_to_us(report->wait_done_ns);
    const uint64_t tracked_us = sync_us +
                                soft_reset_us +
                                cdma_reset_us +
                                ready_wait_us +
                                path_setup_us +
                                start_us +
                                post_start_delay_us +
                                transfer_us +
                                tail_fixup_us +
                                wait_done_us;

    g_total_execute_us += wall_us;
    g_total_sync_us += sync_us;
    g_total_soft_reset_us += soft_reset_us;
    g_total_cdma_reset_us += cdma_reset_us;
    g_total_ready_wait_us += ready_wait_us;
    g_total_path_setup_us += path_setup_us;
    g_total_start_us += start_us;
    g_total_post_start_delay_us += post_start_delay_us;
    g_total_transfer_us += transfer_us;
    g_total_tail_fixup_us += tail_fixup_us;
    g_total_wait_done_us += wait_done_us;
    g_total_execute_gap_us += wall_us >= tracked_us ? (wall_us - tracked_us) : 0;
}

static void tpu_accumulate_read_report(const tpu_soc_v2_read_result_report_t * report) {
    if (report == NULL) {
        return;
    }

    const uint64_t total_us = tpu_ns_to_us(report->total_ns);
    const uint64_t sync_us = tpu_ns_to_us(report->sync_ns);
    const uint64_t copy_us = tpu_ns_to_us(report->copy_ns);
    const uint64_t unpack_us = tpu_ns_to_us(report->unpack_ns);
    const uint64_t mmio_us = tpu_ns_to_us(report->mmio_read_ns);

    g_total_read_us += total_us;
    g_total_read_sync_us += sync_us;
    g_total_read_copy_us += copy_us;
    g_total_read_unpack_us += unpack_us;
    g_total_read_mmio_us += mmio_us;
    g_total_read_bytes += report->bytes;
    g_total_read_output_buffer_count += report->used_output_buffer ? 1u : 0u;
    g_total_read_memcpy_count += report->used_memcpy ? 1u : 0u;
    g_total_read_unpack_count += report->used_unpack ? 1u : 0u;
    g_total_read_mmio_count += report->used_mmio ? 1u : 0u;
    g_read_final_desc_count++;
    g_read_final_desc_total_us += total_us;
    g_read_final_desc_sync_us += sync_us;
    g_read_final_desc_copy_us += copy_us;
    g_read_final_desc_unpack_us += unpack_us;
    g_read_final_desc_mmio_us += mmio_us;
    g_read_final_desc_bytes += report->bytes;
}

static void tpu_add_prepare_ab_report(tpu_prepare_ab_report_t * dst, const tpu_prepare_ab_report_t * src) {
    if (dst == NULL || src == NULL) {
        return;
    }

    dst->total_us += src->total_us;
    dst->pack_total_us += src->pack_total_us;
    dst->pack_a_us += src->pack_a_us;
    dst->pack_b_us += src->pack_b_us;
    dst->copy_total_us += src->copy_total_us;
    dst->copy_a_us += src->copy_a_us;
    dst->copy_b_us += src->copy_b_us;
    dst->a_bytes += src->a_bytes;
    dst->b_bytes += src->b_bytes;
    dst->used_b_pack |= src->used_b_pack;
    dst->used_b_kn_cache |= src->used_b_kn_cache;
}

static int tpu_try_get_device_cached_b(
        const tpu_tile_execute_desc_t * desc,
        size_t b_bytes,
        tpu_soc_fp32_buffer_t * prepared_b_buf_out,
        uint64_t * copy_b_us_out) {
    if (prepared_b_buf_out == NULL ||
        desc == NULL ||
        desc->b_layout != TPU_B_LAYOUT_DEVICE_PACKED ||
        desc->b_cache_owner == NULL ||
        !tpu_device_b_cache_enabled()) {
        return 0;
    }

    const tpu_device_b_cache_key_t key = {
        desc->b_cache_owner,
        desc->b_cache_source,
        b_bytes,
        desc->b_cache_source_n,
        desc->b_cache_n0,
        desc->b_cache_k0,
        desc->b_cache_n_actual,
        desc->b_cache_k_actual,
        desc->n,
        desc->k,
    };
    auto it = g_tpu_device_b_cache.find(key);
    if (it != g_tpu_device_b_cache.end()) {
        *prepared_b_buf_out = it->second.buf;
        g_b_device_cache_hits++;
        g_b_device_cache_saved_bytes += b_bytes;
        return 1;
    }

    const size_t max_cache_bytes = tpu_device_b_cache_max_bytes();
    if (max_cache_bytes == 0u || b_bytes > max_cache_bytes ||
        g_b_device_cache_resident_bytes + b_bytes > max_cache_bytes) {
        g_b_device_cache_budget_rejects++;
        return 0;
    }

    auto inserted = g_tpu_device_b_cache.emplace(key, tpu_device_b_cache_entry_t{});
    it = inserted.first;
    if (tpu_soc_fp32_alloc_buffer(&g_tpu_ctx, &it->second.buf, b_bytes) != 0) {
        g_tpu_device_b_cache.erase(it);
        g_b_device_cache_fallbacks++;
        return 0;
    }
    if (tpu_soc_fp32_validate_input_ddr_addr(it->second.buf.device_addr, b_bytes) != 0) {
        tpu_soc_fp32_free_buffer(&it->second.buf);
        g_tpu_device_b_cache.erase(it);
        g_b_device_cache_aperture_rejects++;
        return 0;
    }

    const auto copy_start = std::chrono::steady_clock::now();
    memcpy(it->second.buf.mapping, desc->b, b_bytes);
    const auto copy_end = std::chrono::steady_clock::now();
    const uint64_t copy_b_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(copy_end - copy_start).count();
    if (copy_b_us_out != NULL) {
        *copy_b_us_out += copy_b_us;
    }

    *prepared_b_buf_out = it->second.buf;
    g_b_device_cache_misses++;
    g_b_device_cache_resident_bytes += b_bytes;
    return 1;
}

static int tpu_prepare_ab_slot(const tpu_tile_execute_desc_t * desc,
                               int slot,
                               uint64_t * pack_us_out,
                               uint64_t * copy_us_out,
                               tpu_prepare_ab_report_t * report,
                               tpu_soc_fp32_buffer_t * prepared_b_buf_out) {
    size_t a_bytes = 0;
    size_t b_bytes = 0;
    size_t c_bytes = 0;
    size_t d_bytes = 0;

    if (report != NULL) {
        memset(report, 0, sizeof(*report));
    }
    if (desc == NULL || slot < 0 || slot >= (int) TPU_SEQUENCE_MAX_RUNS) {
        errno = EINVAL;
        return -1;
    }

    if (tpu_soc_v2_validate_sizes(desc->m, desc->n, desc->k, &a_bytes, &b_bytes, &c_bytes, &d_bytes) != 0) {
        return -1;
    }
    if (!g_tpu_buffer_pool_initialized ||
        a_bytes > g_tpu_a_buf[slot].size ||
        b_bytes > g_tpu_b_buf[slot].size ||
        c_bytes > g_tpu_d_buf.size) {
        return -1;
    }

    uint64_t * a_stage = desc->a_layout == TPU_A_LAYOUT_RAW_MK ? NULL : tpu_stage_words(g_tpu_a_stage[slot], a_bytes);
    uint64_t * b_stage = desc->b_layout == TPU_B_LAYOUT_DEVICE_PACKED ? NULL : tpu_stage_words(g_tpu_b_stage[slot], b_bytes);

    if (desc->a_layout == TPU_A_LAYOUT_RAW_MK && desc->a_bytes > a_bytes) {
        errno = EINVAL;
        return -1;
    }

    const auto prepare_start = std::chrono::steady_clock::now();

    uint64_t pack_a_us = 0;
    if (desc->a_layout != TPU_A_LAYOUT_RAW_MK) {
        const auto pack_a_start = std::chrono::steady_clock::now();
        tpu_soc_v2_pack_a_int8_row_major(desc->a, desc->m, desc->k, a_stage);
        const auto pack_a_end = std::chrono::steady_clock::now();
        pack_a_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(pack_a_end - pack_a_start).count();
    }

    uint64_t pack_b_us = 0;
    if (desc->b_layout != TPU_B_LAYOUT_DEVICE_PACKED) {
        const auto pack_b_start = std::chrono::steady_clock::now();
        if (desc->b_layout == TPU_B_LAYOUT_KN) {
            tpu_soc_v2_pack_b_int8_kn_raw(desc->b, desc->n, desc->k, b_stage);
        } else {
            tpu_soc_v2_pack_b_int8_nk_to_kn(desc->b, desc->n, desc->k, b_stage);
        }
        const auto pack_b_end = std::chrono::steady_clock::now();
        pack_b_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(pack_b_end - pack_b_start).count();
    }
    const uint64_t pack_us = pack_a_us + pack_b_us;
    g_total_pack_us += pack_us;
    g_total_pack_a_us += pack_a_us;
    g_total_pack_b_us += pack_b_us;
    if (pack_us_out != NULL) {
        *pack_us_out += pack_us;
    }

    const auto copy_a_start = std::chrono::steady_clock::now();
    if (desc->a_layout == TPU_A_LAYOUT_RAW_MK) {
        if (desc->a_bytes < a_bytes) {
            memset(g_tpu_a_buf[slot].mapping, 0, a_bytes);
        }
        memcpy(g_tpu_a_buf[slot].mapping, desc->a, desc->a_bytes);
    } else {
        memcpy(g_tpu_a_buf[slot].mapping, a_stage, a_bytes);
    }
    const auto copy_a_end = std::chrono::steady_clock::now();
    const uint64_t copy_a_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(copy_a_end - copy_a_start).count();

    tpu_soc_fp32_buffer_t prepared_b_buf = g_tpu_b_buf[slot];
    uint64_t copy_b_us = 0;
    /*
     * Device-B cache returns an alternate BO. Only the sequence-controller
     * path consumes that BO via prepared_b_buf_out. Single/direct and
     * auto-next paths still pass g_tpu_b_buf[slot] to the RTL, so a cache hit
     * there would skip the slot memcpy and leave hardware reading stale B.
     */
    const int can_use_external_b_buf = prepared_b_buf_out != NULL;
    const int used_device_b_cache = can_use_external_b_buf ?
        tpu_try_get_device_cached_b(desc, b_bytes, &prepared_b_buf, &copy_b_us) : 0;
    if (!used_device_b_cache) {
        const auto copy_b_start = std::chrono::steady_clock::now();
        memcpy(g_tpu_b_buf[slot].mapping,
               desc->b_layout == TPU_B_LAYOUT_DEVICE_PACKED ? desc->b : (const int8_t *) b_stage,
               b_bytes);
        const auto copy_b_end = std::chrono::steady_clock::now();
        copy_b_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(copy_b_end - copy_b_start).count();
    }
    const uint64_t copy_us = copy_a_us + copy_b_us;
    g_total_copy_us += copy_us;
    g_total_copy_a_us += copy_a_us;
    g_total_copy_b_us += copy_b_us;
    if (copy_us_out != NULL) {
        *copy_us_out += copy_us;
    }
    if (prepared_b_buf_out != NULL) {
        *prepared_b_buf_out = prepared_b_buf;
    }

    const auto prepare_end = std::chrono::steady_clock::now();
    const uint64_t prepare_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(prepare_end - prepare_start).count();
    g_total_prepare_us += prepare_us;

    if (report != NULL) {
        report->total_us = prepare_us;
        report->pack_total_us = pack_us;
        report->pack_a_us = pack_a_us;
        report->pack_b_us = pack_b_us;
        report->copy_total_us = copy_us;
        report->copy_a_us = copy_a_us;
        report->copy_b_us = copy_b_us;
        report->a_bytes = a_bytes;
        report->b_bytes = b_bytes;
        report->used_b_pack = desc->b_layout == TPU_B_LAYOUT_NK;
        report->used_b_kn_cache = desc->b_layout == TPU_B_LAYOUT_KN || desc->b_layout == TPU_B_LAYOUT_DEVICE_PACKED;
    }

    (void) d_bytes;
    return 0;
}


static int tpu_start_desc_no_wait(
        const tpu_tile_execute_desc_t * desc,
        int slot,
        int use_preloaded_ab,
        tpu_soc_v2_execute_report_t * report,
        uint64_t * wall_us_out) {
    g_tpu_ctx.v2_c_mode = tpu_desc_c_mode(desc);
    g_tpu_ctx.v2_d_mode = tpu_desc_d_mode(desc);

    const auto start = std::chrono::steady_clock::now();
    const int rc = tpu_soc_v2_start_int8_int32_hw_ingress_no_wait(
            &g_tpu_ctx,
            &g_tpu_a_buf[slot],
            &g_tpu_b_buf[slot],
            desc->m,
            desc->n,
            desc->k,
            0,
            use_preloaded_ab,
            10000u,
            report);
    const auto end = std::chrono::steady_clock::now();
    const uint64_t wall_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    if (wall_us_out != NULL) {
        *wall_us_out += wall_us;
    }
    g_total_start_wall_us += wall_us;
    return rc;
}

static int tpu_wait_active(uint32_t c_mode, uint32_t d_mode, tpu_soc_v2_execute_report_t * report, uint64_t * wall_us_out) {
    const auto start = std::chrono::steady_clock::now();
    const int rc = tpu_soc_v2_wait_int8_int32_active(&g_tpu_ctx, c_mode, d_mode, 10000u, report);
    const auto end = std::chrono::steady_clock::now();
    if (wall_us_out != NULL) {
        *wall_us_out += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }
    if (rc != 0 && (tpu_log_tile_enabled() || tpu_log_init_enabled())) {
        const int saved_errno = errno;
        tpu_dump_ctx_failure(&g_tpu_ctx, "[FPGA] active wait failure; runtime state:");
        errno = saved_errno;
    }
    return rc;
}

static int tpu_preload_ab(const tpu_tile_execute_desc_t * desc,
                          int slot,
                          tpu_soc_v2_ab_preload_report_t * report_out,
                          uint64_t * wall_us_out) {
    tpu_soc_v2_ab_preload_report_t report = {};
    const auto start = std::chrono::steady_clock::now();
    const int rc = tpu_soc_v2_preload_ab_int8_int32_with_report(
            &g_tpu_ctx,
            &g_tpu_a_buf[slot],
            &g_tpu_b_buf[slot],
            desc->m,
            desc->n,
            desc->k,
            10000u,
            &report);
    const auto end = std::chrono::steady_clock::now();
    const uint64_t wall_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    if (wall_us_out != NULL) {
        *wall_us_out += wall_us;
    }
    g_total_preload_wall_us += wall_us;
    if (report_out != NULL) {
        *report_out = report;
    }

    if (rc == 0) {
        g_prevd_ab_preload_count++;
        g_total_preload_sync_us += tpu_ns_to_us(report.sync_ns);
        g_total_preload_write_ready_wait_us += tpu_ns_to_us(report.write_ready_wait_ns);
        g_total_preload_path_setup_us += tpu_ns_to_us(report.path_setup_ns);
        g_total_preload_transfer_us += tpu_ns_to_us(report.transfer_ns);
    } else if (tpu_log_tile_enabled() || tpu_log_init_enabled()) {
        const int saved_errno = errno;
        tpu_dump_ctx_failure(&g_tpu_ctx, "[FPGA] A/B preload failure; runtime state:");
        errno = saved_errno;
    }
    return rc;
}

static int tpu_program_next_ab(const tpu_tile_execute_desc_t * next_desc,
                               int slot,
                               tpu_soc_v2_ab_preload_report_t * report_out,
                               uint64_t * wall_us_out) {
    tpu_soc_v2_ab_preload_report_t report = {};
    const auto start = std::chrono::steady_clock::now();
    const int rc = tpu_soc_v2_program_next_ab_preload_int8_int32_with_report(
            &g_tpu_ctx,
            &g_tpu_a_buf[slot],
            &g_tpu_b_buf[slot],
            next_desc->m,
            next_desc->n,
            next_desc->k,
            &report);
    const auto end = std::chrono::steady_clock::now();
    const uint64_t wall_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    if (wall_us_out != NULL) {
        *wall_us_out += wall_us;
    }
    g_total_preload_wall_us += wall_us;
    if (report_out != NULL) {
        *report_out = report;
    }

    if (rc == 0) {
        g_prevd_auto_next_program_count++;
        g_total_preload_sync_us += tpu_ns_to_us(report.sync_ns);
        g_total_preload_path_setup_us += tpu_ns_to_us(report.path_setup_ns);
    } else if (tpu_log_tile_enabled() || tpu_log_init_enabled()) {
        const int saved_errno = errno;
        tpu_dump_ctx_failure(&g_tpu_ctx, "[FPGA] next A/B descriptor arm failure; runtime state:");
        errno = saved_errno;
    }
    return rc;
}

static int tpu_read_final_desc(const tpu_tile_execute_desc_t * desc,
                               tpu_readback_stage_report_t * report_out,
                               tpu_tile_execute_result_t * result) {
    const unsigned dump_raw_output_bytes = tpu_env_u32_or_default("TPU_V2_DUMP_RAW_OUTPUT_BYTES", 0u);
    tpu_soc_v2_read_result_report_t report = {};
    tpu_readback_stage_report_t stage = {};

    if (report_out != NULL) {
        memset(report_out, 0, sizeof(*report_out));
    }
    if (result != NULL) {
        result->readback_direct_final = 0;
    }

    if (desc->d == NULL && desc->d_final == NULL) {
        errno = EINVAL;
        return -1;
    }

    const auto total_start = std::chrono::steady_clock::now();
    if (g_tpu_output_buffer_enabled) {
        if (desc->d_final != NULL && desc->d_actual_m != 0u && desc->d_actual_n != 0u && desc->d_final_stride >= desc->d_actual_n) {
            if (tpu_soc_v2_read_result_int32_from_buffer_with_rect_report(
                    &g_tpu_d_buf,
                    desc->d_final,
                    desc->m,
                    desc->n,
                    desc->d_actual_m,
                    desc->d_actual_n,
                    desc->d_final_stride,
                    &report) != 0) {
                return -1;
            }
            if (result != NULL) {
                result->readback_direct_final = 1;
            }
            g_prevd_direct_final_readback_count++;
        } else {
            if (tpu_soc_v2_read_result_int32_from_buffer_with_report(&g_tpu_d_buf, desc->d, desc->m * desc->n, &report) != 0) {
                return -1;
            }
        }
        tpu_accumulate_read_report(&report);
        if (dump_raw_output_bytes != 0u) {
            const auto dump_start = std::chrono::steady_clock::now();
            if (tpu_soc_v2_dump_output_buffer_bytes(stderr, &g_tpu_d_buf, desc->m * desc->n, dump_raw_output_bytes) != 0) {
                return -1;
            }
            const auto dump_end = std::chrono::steady_clock::now();
            stage.dump_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(dump_end - dump_start).count();
            g_total_read_debug_dump_us += stage.dump_us;
        }
    } else {
        if (tpu_soc_v2_read_result_int32_with_report(&g_tpu_ctx, desc->d, desc->m * desc->n, &report) != 0) {
            return -1;
        }
        tpu_accumulate_read_report(&report);
    }
    const auto total_end = std::chrono::steady_clock::now();
    g_prevd_host_d_readback_count++;

    stage.total_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(total_end - total_start).count();
    stage.sync_us = tpu_ns_to_us(report.sync_ns);
    stage.copy_us = tpu_ns_to_us(report.copy_ns);
    stage.unpack_us = tpu_ns_to_us(report.unpack_ns);
    stage.mmio_us = tpu_ns_to_us(report.mmio_read_ns);
    stage.bytes = report.bytes;
    stage.used_output_buffer = report.used_output_buffer;
    stage.used_memcpy = report.used_memcpy;
    stage.used_unpack = report.used_unpack;
    stage.used_mmio = report.used_mmio;

    if (report_out != NULL) {
        *report_out = stage;
    }
    return 0;
}
static int tpu_real_sequence_execute_cached(
    const tpu_tile_execute_desc_t * descs,
    uint32_t desc_count,
    tpu_tile_execute_result_t * result) {
    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }
    if (descs == NULL || desc_count < 2u || desc_count > TPU_SEQUENCE_MAX_RUNS) {
        errno = EINVAL;
        return -1;
    }
    if (!g_tpu_initialized && tpu_runtime_init() != 0) {
        return -1;
    }

    const tpu_tile_execute_desc_t * first = &descs[0];
    const tpu_tile_execute_desc_t * last = &descs[desc_count - 1u];
    const bool start_prev_d = first->c_source == TPU_C_SOURCE_DEVICE_ACCUM;
    const bool final_retain = last->d_dest == TPU_D_DEST_DEVICE_ACCUM;
    if ((first->c_source != TPU_C_SOURCE_ZERO && !start_prev_d) ||
        (last->d_dest != TPU_D_DEST_HOST_READBACK && !final_retain) ||
        (!final_retain && last->d_final == NULL) ||
        first->a == NULL || first->b == NULL) {
        errno = EINVAL;
        return -1;
    }
    for (uint32_t i = 0; i < desc_count; ++i) {
        const tpu_tile_execute_desc_t * desc = &descs[i];
        if (desc->a == NULL || desc->b == NULL ||
            desc->m != first->m || desc->n != first->n || desc->k != first->k ||
            desc->a_layout != first->a_layout ||
            desc->b_layout != first->b_layout ||
            !tpu_accum_key_equal(&desc->accum_key, &first->accum_key) ||
            desc->c_source != (i == 0u ? first->c_source : TPU_C_SOURCE_DEVICE_ACCUM) ||
            desc->d_dest != (i + 1u == desc_count ? last->d_dest : TPU_D_DEST_DEVICE_ACCUM)) {
            errno = EINVAL;
            return -1;
        }
    }

    const std::thread::id self = std::this_thread::get_id();
    const bool already_owns_chain = g_prevd_chain.active && g_prevd_chain.owner == self;
    bool locked_here = false;
    if (!already_owns_chain) {
        g_tpu_mutex.lock();
        locked_here = true;
    }

    auto fail = [&]() -> int {
        g_prevd_chain.active = 0;
        g_prevd_chain.owner = std::thread::id();
        if (already_owns_chain || locked_here) {
            g_tpu_mutex.unlock();
        }
        return -1;
    };

    if (!start_prev_d && g_prevd_chain.active && !already_owns_chain) {
        errno = EBUSY;
        return fail();
    }
    if (start_prev_d &&
        (!g_prevd_chain.active || !already_owns_chain ||
         !tpu_accum_key_equal(&g_prevd_chain.key, &first->accum_key))) {
        errno = ENOENT;
        return fail();
    }

    const uint64_t tile_call_idx = ++g_tile_call_count;
    const bool log_tile = tpu_should_log_tile(tile_call_idx);
    uint64_t pack_us = 0;
    uint64_t copy_us = 0;
    uint64_t read_wall_us = 0;
    tpu_prepare_ab_report_t prepare_report = {};
    tpu_readback_stage_report_t read_report = {};
    tpu_soc_v2_sequence_report_t sequence_report = {};
    tpu_soc_fp32_buffer_t sequence_b_bufs[TPU_SEQUENCE_MAX_RUNS] = {};
    tpu_async_readback_job_t async_readback_job = {};
    bool staged_readback = false;
    bool async_staged_readback = false;

    const auto sequence_call_start = std::chrono::steady_clock::now();
    const auto control_start = std::chrono::steady_clock::now();
    const int skip_sequence_soft_reset = tpu_skip_sequence_soft_reset_enabled();
    if (!start_prev_d) {
        if ((!skip_sequence_soft_reset && tpu_soc_v2_soft_reset(&g_tpu_ctx, 10000u) != 0) ||
            tpu_soc_v2_clear_prev_d_slot(&g_tpu_ctx) != 0) {
            return fail();
        }
        g_tpu_output_buffer_enabled = 0;
    }
    if (!final_retain && !g_tpu_op_readback_active && !g_tpu_async_readback_active && g_tpu_d_buf.handle != NULL &&
        (tpu_env_is_truthy_or_default("TPU_V2_FORCE_PER_TILE_OUTBUF_REENABLE", 0) || !g_tpu_output_buffer_enabled) &&
        tpu_soc_v2_try_enable_output_buffer(&g_tpu_ctx, &g_tpu_d_buf, &g_tpu_output_buffer_enabled) != 0) {
        return fail();
    }
    const auto control_end = std::chrono::steady_clock::now();
    g_total_control_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(control_end - control_start).count();

    for (uint32_t i = 0; i < desc_count; ++i) {
        tpu_prepare_ab_report_t slot_prepare_report = {};
        if (tpu_prepare_ab_slot(&descs[i], (int) i, &pack_us, &copy_us, &slot_prepare_report, &sequence_b_bufs[i]) != 0) {
            return fail();
        }
        tpu_add_prepare_ab_report(&prepare_report, &slot_prepare_report);
    }

    g_tpu_ctx.v2_c_mode = start_prev_d ? TPU_SOC_V2_C_MODE_PREV_D : TPU_SOC_V2_C_MODE_ZERO;
    g_tpu_ctx.v2_d_mode = final_retain ? TPU_SOC_V2_D_MODE_RETAIN : TPU_SOC_V2_D_MODE_READBACK;
    if (!final_retain) {
        const int stage_rc = tpu_op_readback_batch_stage_desc(last, &read_report, result);
        if (stage_rc < 0 || (g_tpu_op_readback_active && stage_rc == 0)) {
            return fail();
        }
        staged_readback = stage_rc > 0;
        if (!staged_readback) {
            const int async_stage_rc = tpu_async_readback_stage_desc(last, &read_report, result, &async_readback_job);
            if (async_stage_rc < 0 || (g_tpu_async_readback_active && async_stage_rc == 0)) {
                return fail();
            }
            async_staged_readback = async_stage_rc > 0;
            staged_readback = async_staged_readback;
        }
    }
    const auto sequence_start = std::chrono::steady_clock::now();
    const int sequence_rc = tpu_soc_v2_run_prev_d_sequence_prepacked_int8_int32_ex_with_report(
            &g_tpu_ctx,
            g_tpu_a_buf,
            sequence_b_bufs,
            desc_count,
            first->m,
            first->n,
            first->k,
            start_prev_d ? 1 : 0,
            final_retain ? 1 : 0,
            10000u,
            &sequence_report);
    const auto sequence_end = std::chrono::steady_clock::now();
    if (sequence_rc != 0) {
        g_prevd_sequence_fallback_count++;
        if (tpu_log_tile_enabled() || tpu_log_init_enabled()) {
            const int saved_errno = errno;
            tpu_dump_ctx_failure(&g_tpu_ctx, "[FPGA] sequence controller failure; falling back to single-tile path:");
            errno = saved_errno;
        }
        return fail();
    }

    g_prevd_sequence_program_count++;
    g_prevd_chain_runs += desc_count;
    if (desc_count > 5u) {
        g_prevd_over_signoff_runs += desc_count - 5u;
    }
    const uint64_t sequence_wall_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(sequence_end - sequence_start).count();
    g_total_sequence_us += sequence_wall_us;
    g_total_sequence_sync_us += tpu_ns_to_us(sequence_report.sync_ns);
    g_total_sequence_ready_us += tpu_ns_to_us(sequence_report.ready_wait_ns);
    g_total_sequence_program_us += tpu_ns_to_us(sequence_report.program_ns);
    g_total_sequence_wait_done_us += tpu_ns_to_us(sequence_report.wait_done_ns);
    g_total_ready_wait_us += tpu_ns_to_us(sequence_report.ready_wait_ns);
    g_total_path_setup_us += tpu_ns_to_us(sequence_report.program_ns);
    g_total_wait_done_us += tpu_ns_to_us(sequence_report.wait_done_ns);
    g_total_execute_us += sequence_wall_us;

    uint64_t cleanup_us = 0;
    if (!final_retain) {
        if (!staged_readback && tpu_read_final_desc(last, &read_report, result) != 0) {
            return fail();
        }
        if (async_staged_readback && tpu_async_readback_enqueue(&async_readback_job) != 0) {
            return fail();
        }
        read_wall_us = read_report.total_us;
        const auto cleanup_start = std::chrono::steady_clock::now();
        if (tpu_soc_v2_clear_prev_d_slot(&g_tpu_ctx) != 0) {
            return fail();
        }
        const auto cleanup_end = std::chrono::steady_clock::now();
        cleanup_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(cleanup_end - cleanup_start).count();
        g_total_cleanup_us += cleanup_us;
        g_prevd_chain.active = 0;
        g_prevd_chain.owner = std::thread::id();
    } else {
        const uint32_t final_run_index = start_prev_d ?
                (g_prevd_chain.run_index + desc_count) :
                (desc_count - 1u);
        g_prevd_chain.active = 1;
        g_prevd_chain.owner = self;
        g_prevd_chain.key = first->accum_key;
        g_prevd_chain.run_index = final_run_index;
        g_prevd_chain.pending_c_mode = tpu_desc_c_mode(last);
        g_prevd_chain.pending_d_mode = tpu_desc_d_mode(last);
        g_prevd_chain.pending_slot = 0;
        g_prevd_chain.next_preloaded_valid = 0;
    }
    if (result != NULL) {
        result->accum_input_hit = start_prev_d || desc_count > 1u;
        result->accum_output_valid = final_retain ? 1 : 0;
        result->readback_done = final_retain ? 0 : 1;
        result->bytes_device_to_host_d = final_retain ? 0u :
                (uint64_t) last->accum_key.m * last->accum_key.n * sizeof(int32_t);
    }

    const auto sequence_call_end = std::chrono::steady_clock::now();
    const uint64_t sequence_call_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(sequence_call_end - sequence_call_start).count();
    const uint64_t control_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(control_end - control_start).count();
    const uint64_t phase_us = prepare_report.total_us + control_us + sequence_wall_us + read_wall_us + cleanup_us;
    g_total_tile_us += sequence_call_us;
    g_total_other_us += sequence_call_us >= phase_us ? (sequence_call_us - phase_us) : 0;

    if (log_tile) {
        fprintf(stderr,
                "[FPGA] PREV_D sequence #%" PRIu64 " runs=%u M=%u N=%u K=%u a_layout=%s b_layout=%s skip_sequence_soft_reset=%d output_buffer=%s seq_status=0x%08x preload_hw_count=%u read_total=%" PRIu64 " us direct_final=%s\n",
                tile_call_idx,
                desc_count,
                first->m,
                first->n,
                first->k,
                tpu_a_layout_name(first->a_layout),
                tpu_b_layout_name(first->b_layout),
                skip_sequence_soft_reset,
                g_tpu_output_buffer_enabled ? "on" : "off",
                sequence_report.seq_status,
                sequence_report.preload_hw_count,
                read_wall_us,
                (result != NULL && result->readback_direct_final) ? "on" : "off");
    }
    if (!final_retain && (already_owns_chain || locked_here)) {
        g_tpu_mutex.unlock();
        locked_here = false;
    } else if (final_retain) {
        locked_here = false;
    }
    return 0;
}

static int tpu_real_tile_execute_cached(
        const tpu_tile_execute_desc_t * desc,
        tpu_tile_execute_result_t * result) {
    if (result != NULL) {
        memset(result, 0, sizeof(*result));
    }
    if (desc == NULL || desc->a == NULL || desc->b == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (desc->c_source == TPU_C_SOURCE_HOST) {
        return -1;
    }
    if (desc->d_dest == TPU_D_DEST_HOST_READBACK && desc->d == NULL && desc->d_final == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (!g_tpu_initialized && tpu_runtime_init() != 0) {
        return -1;
    }

    const std::thread::id self = std::this_thread::get_id();
    const bool already_owns_chain = g_prevd_chain.active && g_prevd_chain.owner == self;
    bool locked_here = false;
    if (!already_owns_chain) {
        g_tpu_mutex.lock();
        locked_here = true;
    }

    const uint64_t tile_call_idx = ++g_tile_call_count;
    const bool log_tile = tpu_should_log_tile(tile_call_idx);
    const bool first_run = desc->c_source == TPU_C_SOURCE_ZERO;
    const bool final_run = desc->d_dest == TPU_D_DEST_HOST_READBACK;
    const char * preload_mode = first_run ? "none" : "explicit";
    uint64_t pack_us = 0;
    uint64_t copy_us = 0;
    uint64_t execute_wall_us = 0;
    uint64_t start_log_us = 0;
    uint64_t control_us = 0;
    uint64_t wait_prev_wall_us = 0;
    uint64_t wait_final_wall_us = 0;
    uint64_t preload_wall_us = 0;
    uint64_t read_wall_us = 0;
    uint64_t cleanup_us = 0;
    uint64_t unlock_us = 0;
    int rc = -1;
    tpu_prepare_ab_report_t prepare_report = {};
    tpu_readback_stage_report_t read_report = {};
    tpu_soc_v2_ab_preload_report_t preload_report = {};
    tpu_soc_v2_execute_report_t start_report = {};
    tpu_soc_v2_execute_report_t wait_report = {};
    tpu_soc_v2_execute_report_t final_wait_report = {};
    tpu_async_readback_job_t async_readback_job = {};
    bool staged_readback = false;
    bool async_staged_readback = false;

    const auto tile_start = std::chrono::steady_clock::now();

    auto fail = [&]() -> int {
        g_prevd_chain.active = 0;
        g_prevd_chain.owner = std::thread::id();
        if (already_owns_chain || locked_here) {
            g_tpu_mutex.unlock();
        }
        return -1;
    };

    if (first_run && g_prevd_chain.active && !already_owns_chain) {
        errno = EBUSY;
        return fail();
    }
    if (!first_run) {
        if (!g_prevd_chain.active || !already_owns_chain ||
            !tpu_accum_key_equal(&g_prevd_chain.key, &desc->accum_key)) {
            errno = ENOENT;
            return fail();
        }
    }

    if (log_tile) {
        const auto log_start = std::chrono::steady_clock::now();
        fprintf(stderr,
                "[FPGA] PREV_D tile #%" PRIu64 " run=%u M=%u N=%u K=%u C=%s D=%s a_layout=%s b_layout=%s output_buffer=%s\n",
                tile_call_idx,
                first_run ? 0u : g_prevd_chain.run_index + 1u,
                desc->m,
                desc->n,
                desc->k,
                tpu_c_source_name(desc->c_source),
                tpu_d_dest_name(desc->d_dest),
                tpu_a_layout_name(desc->a_layout),
                tpu_b_layout_name(desc->b_layout),
                g_tpu_output_buffer_enabled ? "on" : "off");
        const auto log_end = std::chrono::steady_clock::now();
        start_log_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(log_end - log_start).count();
        g_total_log_us += start_log_us;
    }

    const auto control_start = std::chrono::steady_clock::now();
    if (first_run) {
        if ((!tpu_skip_single_tile_soft_reset_enabled() &&
             tpu_soc_v2_soft_reset(&g_tpu_ctx, 10000u) != 0) ||
            tpu_soc_v2_clear_prev_d_slot(&g_tpu_ctx) != 0) {
            return fail();
        }
        g_tpu_output_buffer_enabled = 0;
    }
    if (final_run && !g_tpu_op_readback_active && !g_tpu_async_readback_active && g_tpu_d_buf.handle != NULL &&
        (tpu_env_is_truthy_or_default("TPU_V2_FORCE_PER_TILE_OUTBUF_REENABLE", 0) || !g_tpu_output_buffer_enabled) &&
        tpu_soc_v2_try_enable_output_buffer(&g_tpu_ctx, &g_tpu_d_buf, &g_tpu_output_buffer_enabled) != 0) {
        return fail();
    }
    const auto control_end = std::chrono::steady_clock::now();
    control_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(control_end - control_start).count();
    g_total_control_us += control_us;

    if (first_run) {
        const int slot = 0;
        const int next_slot = 1;
        const bool has_next = desc->next_desc != NULL;

        tpu_prepare_ab_report_t slot_prepare_report = {};
        if (tpu_prepare_ab_slot(desc, slot, &pack_us, &copy_us, &slot_prepare_report, NULL) != 0) {
            return fail();
        }
        tpu_add_prepare_ab_report(&prepare_report, &slot_prepare_report);
        if (has_next) {
            tpu_prepare_ab_report_t next_prepare_report = {};
            if (tpu_prepare_ab_slot(desc->next_desc, next_slot, &pack_us, &copy_us, &next_prepare_report, NULL) != 0) {
                return fail();
            }
            tpu_add_prepare_ab_report(&prepare_report, &next_prepare_report);
            if (tpu_program_next_ab(desc->next_desc, next_slot, &preload_report, &preload_wall_us) == 0) {
                preload_mode = "auto-next";
            } else {
                g_prevd_auto_next_fallback_count++;
                preload_mode = "explicit";
                if (tpu_preload_ab(desc->next_desc, next_slot, &preload_report, &preload_wall_us) != 0) {
                    return fail();
                }
            }
        }
        if (final_run) {
            const int stage_rc = tpu_op_readback_batch_stage_desc(desc, &read_report, result);
            if (stage_rc < 0 || (g_tpu_op_readback_active && stage_rc == 0)) {
                return fail();
            }
            staged_readback = stage_rc > 0;
            if (!staged_readback) {
                const int async_stage_rc = tpu_async_readback_stage_desc(desc, &read_report, result, &async_readback_job);
                if (async_stage_rc < 0 || (g_tpu_async_readback_active && async_stage_rc == 0)) {
                    return fail();
                }
                async_staged_readback = async_stage_rc > 0;
                staged_readback = async_staged_readback;
            }
        }
        if (tpu_start_desc_no_wait(desc, slot, 0, &start_report, &execute_wall_us) != 0) {
            return fail();
        }
        tpu_accumulate_execute_report(&start_report, execute_wall_us);
        g_prevd_chain_runs++;
        if (g_prevd_chain.run_index >= 5u) {
            g_prevd_over_signoff_runs++;
        }

        if (final_run) {
            tpu_soc_v2_execute_report_t wait_report = {};
            if (tpu_wait_active(tpu_desc_c_mode(desc), tpu_desc_d_mode(desc), &wait_report, &wait_final_wall_us) != 0 ||
                (!staged_readback && tpu_read_final_desc(desc, &read_report, result) != 0)) {
                return fail();
            }
            if (async_staged_readback && tpu_async_readback_enqueue(&async_readback_job) != 0) {
                return fail();
            }
            const auto cleanup_start = std::chrono::steady_clock::now();
            if (tpu_soc_v2_clear_prev_d_slot(&g_tpu_ctx) != 0) {
                return fail();
            }
            const auto cleanup_end = std::chrono::steady_clock::now();
            cleanup_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(cleanup_end - cleanup_start).count();
            g_total_cleanup_us += cleanup_us;
            tpu_accumulate_execute_report(&wait_report, wait_final_wall_us);
            read_wall_us = read_report.total_us;
            if (result != NULL) {
                result->readback_done = 1;
                result->bytes_device_to_host_d = (uint64_t) desc->accum_key.m * desc->accum_key.n * sizeof(int32_t);
            }
            rc = 0;
            const auto unlock_start = std::chrono::steady_clock::now();
            g_tpu_mutex.unlock();
            const auto unlock_end = std::chrono::steady_clock::now();
            unlock_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(unlock_end - unlock_start).count();
            g_total_unlock_us += unlock_us;
            locked_here = false;
        } else {
            g_prevd_chain.active = 1;
            g_prevd_chain.owner = self;
            g_prevd_chain.key = desc->accum_key;
            g_prevd_chain.run_index = 0;
            g_prevd_chain.pending_c_mode = tpu_desc_c_mode(desc);
            g_prevd_chain.pending_d_mode = tpu_desc_d_mode(desc);
            g_prevd_chain.pending_slot = slot;
            g_prevd_chain.next_preloaded_valid = has_next ? 1 : 0;
            if (result != NULL) {
                result->accum_output_valid = 1;
            }
            rc = 0;
            locked_here = false;
        }
    } else {
        const int active_slot = g_prevd_chain.pending_slot;
        const int slot = active_slot == 0 ? 1 : 0;
        const int use_preloaded_ab = g_prevd_chain.next_preloaded_valid;
        const bool has_next = desc->next_desc != NULL;
        const int next_slot = active_slot;
        int next_preloaded_valid = 0;

        if (!use_preloaded_ab) {
            tpu_prepare_ab_report_t slot_prepare_report = {};
            if (tpu_prepare_ab_slot(desc, slot, &pack_us, &copy_us, &slot_prepare_report, NULL) != 0) {
                return fail();
            }
            tpu_add_prepare_ab_report(&prepare_report, &slot_prepare_report);
        }

        if (tpu_wait_active(g_prevd_chain.pending_c_mode, g_prevd_chain.pending_d_mode, &wait_report, &wait_prev_wall_us) != 0) {
            return fail();
        }
        tpu_accumulate_execute_report(&wait_report, wait_prev_wall_us);

        if (has_next) {
            tpu_prepare_ab_report_t next_prepare_report = {};
            if (tpu_prepare_ab_slot(desc->next_desc, next_slot, &pack_us, &copy_us, &next_prepare_report, NULL) != 0) {
                return fail();
            }
            tpu_add_prepare_ab_report(&prepare_report, &next_prepare_report);
            if (tpu_program_next_ab(desc->next_desc, next_slot, &preload_report, &preload_wall_us) == 0) {
                preload_mode = "auto-next";
                next_preloaded_valid = 1;
            } else {
                g_prevd_auto_next_fallback_count++;
                preload_mode = "explicit";
                if (tpu_preload_ab(desc->next_desc, next_slot, &preload_report, &preload_wall_us) != 0) {
                    return fail();
                }
                next_preloaded_valid = 1;
            }
        }

        const uint32_t run_index = g_prevd_chain.run_index + 1u;
        if (run_index >= 5u) {
            g_prevd_over_signoff_runs++;
        }
        if (final_run) {
            const int stage_rc = tpu_op_readback_batch_stage_desc(desc, &read_report, result);
            if (stage_rc < 0 || (g_tpu_op_readback_active && stage_rc == 0)) {
                return fail();
            }
            staged_readback = stage_rc > 0;
            if (!staged_readback) {
                const int async_stage_rc = tpu_async_readback_stage_desc(desc, &read_report, result, &async_readback_job);
                if (async_stage_rc < 0 || (g_tpu_async_readback_active && async_stage_rc == 0)) {
                    return fail();
                }
                async_staged_readback = async_stage_rc > 0;
                staged_readback = async_staged_readback;
            }
        }
        if (tpu_start_desc_no_wait(desc, slot, use_preloaded_ab, &start_report, &execute_wall_us) != 0) {
            return fail();
        }
        tpu_accumulate_execute_report(&start_report, execute_wall_us);
        g_prevd_chain_runs++;

        if (final_run) {
            if (tpu_wait_active(tpu_desc_c_mode(desc), tpu_desc_d_mode(desc), &final_wait_report, &wait_final_wall_us) != 0 ||
                (!staged_readback && tpu_read_final_desc(desc, &read_report, result) != 0)) {
                return fail();
            }
            if (async_staged_readback && tpu_async_readback_enqueue(&async_readback_job) != 0) {
                return fail();
            }
            const auto cleanup_start = std::chrono::steady_clock::now();
            if (tpu_soc_v2_clear_prev_d_slot(&g_tpu_ctx) != 0) {
                return fail();
            }
            const auto cleanup_end = std::chrono::steady_clock::now();
            cleanup_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(cleanup_end - cleanup_start).count();
            g_total_cleanup_us += cleanup_us;
            tpu_accumulate_execute_report(&final_wait_report, wait_final_wall_us);
            read_wall_us = read_report.total_us;
            g_prevd_chain.active = 0;
            g_prevd_chain.owner = std::thread::id();
            g_prevd_chain.next_preloaded_valid = 0;
            if (result != NULL) {
                result->accum_input_hit = 1;
                result->readback_done = 1;
                result->bytes_device_to_host_d = (uint64_t) desc->accum_key.m * desc->accum_key.n * sizeof(int32_t);
            }
            rc = 0;
            const auto unlock_start = std::chrono::steady_clock::now();
            g_tpu_mutex.unlock();
            const auto unlock_end = std::chrono::steady_clock::now();
            unlock_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(unlock_end - unlock_start).count();
            g_total_unlock_us += unlock_us;
            locked_here = false;
        } else {
            g_prevd_chain.run_index = run_index;
            g_prevd_chain.pending_c_mode = tpu_desc_c_mode(desc);
            g_prevd_chain.pending_d_mode = tpu_desc_d_mode(desc);
            g_prevd_chain.pending_slot = slot;
            g_prevd_chain.next_preloaded_valid = next_preloaded_valid;
            if (result != NULL) {
                result->accum_input_hit = 1;
                result->accum_output_valid = 1;
            }
            rc = 0;
        }
    }

    g_total_wait_prev_us += wait_prev_wall_us;
    g_total_wait_final_us += wait_final_wall_us;
    const auto tile_end = std::chrono::steady_clock::now();
    const uint64_t tile_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(tile_end - tile_start).count();
    const uint64_t phase_us =
            prepare_report.total_us +
            control_us +
            preload_wall_us +
            wait_prev_wall_us +
            execute_wall_us +
            wait_final_wall_us +
            read_wall_us +
            cleanup_us +
            start_log_us +
            unlock_us;
    const uint64_t other_us = tile_us >= phase_us ? (tile_us - phase_us) : 0;
    g_total_other_us += other_us;
    g_total_tile_us += tile_us;

    if (log_tile) {
        fprintf(stderr,
                "[FPGA] PREV_D tile #%" PRIu64 " done prepare=%" PRIu64 " us pack=%" PRIu64 " us pack_a=%" PRIu64 " us pack_b=%" PRIu64 " us copy=%" PRIu64 " us copy_a=%" PRIu64 " us copy_b=%" PRIu64 " us control=%" PRIu64 " us preload_total=%" PRIu64 " us preload_sync=%" PRIu64 " us preload_wrwait=%" PRIu64 " us preload_setup=%" PRIu64 " us preload_xfer=%" PRIu64 " us wait_prev=%" PRIu64 " us start_total=%" PRIu64 " us start_sync=%" PRIu64 " us start_ready=%" PRIu64 " us start_setup=%" PRIu64 " us start_mmio=%" PRIu64 " us start_xfer=%" PRIu64 " us wait_final=%" PRIu64 " us read_total=%" PRIu64 " us read_sync=%" PRIu64 " us read_copy=%" PRIu64 " us read_unpack=%" PRIu64 " us read_mmio=%" PRIu64 " us read_dump=%" PRIu64 " us cleanup=%" PRIu64 " us log=%" PRIu64 " us unlock=%" PRIu64 " us other=%" PRIu64 " us total=%" PRIu64 " us a_path=%s b_path=%s outbuf=%s direct_final=%s preload_mode=%s\n",
                tile_call_idx,
                prepare_report.total_us,
                prepare_report.pack_total_us,
                prepare_report.pack_a_us,
                prepare_report.pack_b_us,
                prepare_report.copy_total_us,
                prepare_report.copy_a_us,
                prepare_report.copy_b_us,
                control_us,
                preload_wall_us,
                tpu_ns_to_us(preload_report.sync_ns),
                tpu_ns_to_us(preload_report.write_ready_wait_ns),
                tpu_ns_to_us(preload_report.path_setup_ns),
                tpu_ns_to_us(preload_report.transfer_ns),
                wait_prev_wall_us,
                execute_wall_us,
                tpu_ns_to_us(start_report.sync_ns),
                tpu_ns_to_us(start_report.ready_wait_ns),
                tpu_ns_to_us(start_report.path_setup_ns),
                tpu_ns_to_us(start_report.start_ns),
                tpu_ns_to_us(start_report.transfer_ns),
                wait_final_wall_us,
                read_report.total_us,
                read_report.sync_us,
                read_report.copy_us,
                read_report.unpack_us,
                read_report.mmio_us,
                read_report.dump_us,
                cleanup_us,
                start_log_us,
                unlock_us,
                other_us,
                tile_us,
                tpu_a_layout_name(desc->a_layout),
                prepare_report.used_b_kn_cache ? "kn-cache" : (prepare_report.used_b_pack ? "nk-pack" : "unknown"),
                read_report.used_output_buffer ? "on" : "off",
                (final_run && result != NULL && result->readback_direct_final) ? "on" : "off",
                preload_mode);
        fprintf(stderr,
                "[FPGA] PREV_D cumulative runs=%" PRIu64 " over_signoff_runs=%" PRIu64 " "
                "host_c_upload_count=%" PRIu64 " host_d_readback_count=%" PRIu64 " direct_final_count=%" PRIu64 " direct_final_ratio=%.1f%% auto_next_program_count=%" PRIu64 " auto_next_fallback_count=%" PRIu64 " ab_preload_count=%" PRIu64 " "
                "preload_sync=%.3f ms preload_wrwait=%.3f ms preload_setup=%.3f ms preload_transfer=%.3f ms "
                "wait_done=%.3f ms read_total=%.3f ms read_sync=%.3f ms read_copy=%.3f ms read_unpack=%.3f ms read_mmio=%.3f ms read_dump=%.3f ms read_bytes=%.3f MiB paths(outbuf/memcpy/unpack/mmio)=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
                g_prevd_chain_runs,
                g_prevd_over_signoff_runs,
                g_prevd_host_c_upload_count,
                g_prevd_host_d_readback_count,
                g_prevd_direct_final_readback_count,
                g_prevd_host_d_readback_count > 0 ? (100.0 * g_prevd_direct_final_readback_count / g_prevd_host_d_readback_count) : 0.0,
                g_prevd_auto_next_program_count,
                g_prevd_auto_next_fallback_count,
                g_prevd_ab_preload_count,
                g_total_preload_sync_us / 1000.0,
                g_total_preload_write_ready_wait_us / 1000.0,
                g_total_preload_path_setup_us / 1000.0,
                g_total_preload_transfer_us / 1000.0,
                g_total_wait_done_us / 1000.0,
                g_total_read_us / 1000.0,
                g_total_read_sync_us / 1000.0,
                g_total_read_copy_us / 1000.0,
                g_total_read_unpack_us / 1000.0,
                g_total_read_mmio_us / 1000.0,
                g_total_read_debug_dump_us / 1000.0,
                g_total_read_bytes / (1024.0 * 1024.0),
                g_total_read_output_buffer_count,
                g_total_read_memcpy_count,
                g_total_read_unpack_count,
                g_total_read_mmio_count);
    }

    if (locked_here) {
        g_tpu_mutex.unlock();
    }
    return rc;
}

static uint32_t tpu_ceil_div_u32(uint32_t value, uint32_t divisor) {
    return divisor == 0u ? 0u : (value + divisor - 1u) / divisor;
}

static int tpu_reserve_matmul_buffer(
        tpu_soc_fp32_buffer_t * buf,
        size_t * capacity,
        size_t required,
        int output_buffer) {
    if (buf == NULL || capacity == NULL || required == 0u) {
        errno = EINVAL;
        return -1;
    }
    if (buf->handle != NULL && *capacity >= required) {
        return 0;
    }

    tpu_soc_fp32_free_buffer(buf);
    memset(buf, 0, sizeof(*buf));
    *capacity = 0u;
    const int cacheable = output_buffer && tpu_matmul_output_cacheable_bo_enabled();
    if (tpu_soc_fp32_alloc_buffer_ex(&g_tpu_ctx, buf, required, cacheable) != 0) {
        return -1;
    }
    if (output_buffer) {
        if (tpu_soc_fp32_validate_output_ddr_addr(buf->device_addr, required) != 0) {
            tpu_soc_fp32_free_buffer(buf);
            memset(buf, 0, sizeof(*buf));
            return -1;
        }
    } else if (tpu_soc_fp32_validate_input_ddr_addr(buf->device_addr, required) != 0) {
        tpu_soc_fp32_free_buffer(buf);
        memset(buf, 0, sizeof(*buf));
        return -1;
    }
    *capacity = required;
    return 0;
}

static int tpu_pack_b_kn_tile_to_words(
        const int8_t * source_kn,
        uint32_t source_n,
        uint32_t source_k,
        uint32_t n0,
        uint32_t k0,
        uint32_t n_actual,
        uint32_t k_actual,
        uint32_t tile_n,
        uint32_t tile_k,
        uint32_t pack_layout,
        uint64_t * dst_words,
        size_t dst_word_count) {
    if (source_kn == NULL || dst_words == NULL ||
        source_n == 0u || source_k == 0u ||
        tile_n == 0u || tile_k == 0u ||
        n_actual > tile_n || k_actual > tile_k ||
        n0 > source_n || k0 > source_k ||
        n_actual > source_n - n0 || k_actual > source_k - k0) {
        errno = EINVAL;
        return -1;
    }

    memset(dst_words, 0, dst_word_count * sizeof(uint64_t));
    if (pack_layout == TPU_B_PACK_LAYOUT_B16_TILE_MAJOR ||
        pack_layout == TPU_B_PACK_LAYOUT_B16_BANKFRIENDLY) {
        if (tile_k % 16u != 0u) {
            errno = EINVAL;
            return -1;
        }
        if (pack_layout == TPU_B_PACK_LAYOUT_B16_BANKFRIENDLY) {
            if (tile_n % 8u != 0u) {
                errno = EINVAL;
                return -1;
            }
            const uint32_t kg_count = tile_k / 16u;
            const uint32_t nb_count = tile_n / 8u;
            const uint32_t entries_per_lane = kg_count * nb_count * 2u;
            const size_t required_words = (size_t) 4u * entries_per_lane * 2u;
            if (required_words > dst_word_count) {
                errno = EINVAL;
                return -1;
            }
            for (uint32_t kg = 0; kg < kg_count; ++kg) {
                const uint32_t k16 = kg * 16u;
                const uint32_t valid_k = (k16 < k_actual) ?
                    (k_actual - k16 < 16u ? k_actual - k16 : 16u) : 0u;
                for (uint32_t nb = 0; nb < nb_count; ++nb) {
                    const uint32_t n_block = nb * 8u;
                    uint64_t words[4][2][2] = {};
                    for (uint32_t kk = 0; kk < valid_k; ++kk) {
                        const int8_t * row = source_kn + (k0 + k16 + kk) * source_n + n0 + n_block;
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
            return 0;
        }
        for (uint32_t k16 = 0; k16 < tile_k; k16 += 16u) {
            for (uint32_t nn = 0; nn < tile_n; ++nn) {
                for (uint32_t kk = 0; kk < 16u; ++kk) {
                    const uint32_t tile_k_idx = k16 + kk;
                    const uint8_t value =
                        (nn < n_actual && tile_k_idx < k_actual) ?
                        (uint8_t) source_kn[(k0 + tile_k_idx) * source_n + n0 + nn] : 0u;
                    const uint32_t linear_idx = ((k16 / 16u) * tile_n + nn) * 16u + kk;
                    const size_t word_idx = linear_idx / 8u;
                    const uint32_t lane = linear_idx % 8u;
                    if (word_idx >= dst_word_count) {
                        errno = EINVAL;
                        return -1;
                    }
                    dst_words[word_idx] |= ((uint64_t) value) << (lane * 8u);
                }
            }
        }
        return 0;
    }
    if (pack_layout != TPU_B_PACK_LAYOUT_PACK2) {
        errno = EINVAL;
        return -1;
    }
    for (uint32_t kk = 0; kk < k_actual; ++kk) {
        for (uint32_t nn = 0; nn < n_actual; ++nn) {
            const uint8_t value = (uint8_t) source_kn[(k0 + kk) * source_n + n0 + nn];
            const uint32_t linear_idx = kk * tile_n + nn;
            const size_t word_idx = linear_idx / 8u;
            const uint32_t lane = linear_idx % 8u;
            if (word_idx >= dst_word_count) {
                errno = EINVAL;
                return -1;
            }
            dst_words[word_idx] |= ((uint64_t) value) << (lane * 8u);
        }
    }
    return 0;
}

static int tpu_build_b_packed_arena(
        const int8_t * b_kn,
        uint32_t N,
        uint32_t K,
        uint32_t tile_n,
        uint32_t tile_k,
        uint32_t pack_layout,
        int reuse_transient_buf,
        tpu_device_b_matrix_cache_entry_t * entry,
        uint64_t * pack_us_out) {
    const size_t tile_bytes = (size_t) tile_n * (size_t) tile_k;
    const size_t tile_words = tile_bytes / sizeof(uint64_t);
    const uint32_t n_tiles = tpu_ceil_div_u32(N, tile_n);
    const uint32_t k_tiles = tpu_ceil_div_u32(K, tile_k);
    const size_t total_words = tile_words * (size_t) n_tiles * (size_t) k_tiles;
    const size_t total_bytes = total_words * sizeof(uint64_t);
    const size_t max_cache_bytes = tpu_device_b_cache_max_bytes();
    const auto build_start = std::chrono::steady_clock::now();
    std::vector<uint64_t> staging;

    if (b_kn == NULL || entry == NULL || N == 0u || K == 0u ||
        n_tiles == 0u || k_tiles == 0u || total_words == 0u ||
        max_cache_bytes == 0u || total_bytes > max_cache_bytes) {
        errno = EINVAL;
        return -1;
    }

    staging.resize(total_words);
    for (uint32_t nt = 0; nt < n_tiles; ++nt) {
        const uint32_t tile_n0 = nt * tile_n;
        const uint32_t n_actual = N - tile_n0 < tile_n ? N - tile_n0 : tile_n;
        for (uint32_t kt = 0; kt < k_tiles; ++kt) {
            const uint32_t tile_k0 = kt * tile_k;
            const uint32_t k_actual = K - tile_k0 < tile_k ? K - tile_k0 : tile_k;
            const size_t tile_index = (size_t) nt * k_tiles + kt;
            if (tpu_pack_b_kn_tile_to_words(
                        b_kn,
                        N,
                        K,
                        tile_n0,
                        tile_k0,
                        n_actual,
                        k_actual,
                        tile_n,
                        tile_k,
                        pack_layout,
                        staging.data() + tile_index * tile_words,
                        tile_words) != 0) {
                return -1;
            }
        }
    }

    memset(entry, 0, sizeof(*entry));
    entry->source_ptr = b_kn;
    entry->source_n = N;
    entry->source_k = K;
    entry->tile_n = tile_n;
    entry->tile_k = tile_k;
    entry->pack_layout = pack_layout;
    entry->n_tiles = n_tiles;
    entry->k_tiles = k_tiles;
    entry->tile_bytes = tile_bytes;
    entry->tile_words = tile_words;
    entry->total_bytes = total_bytes;

    if (reuse_transient_buf) {
        if (tpu_reserve_matmul_buffer(&g_tpu_matmul_b_buf, &g_tpu_matmul_b_buf_bytes, total_bytes, 0) != 0) {
            memset(entry, 0, sizeof(*entry));
            return -1;
        }
        entry->buf = g_tpu_matmul_b_buf;
        entry->borrowed_buf = 1;
    } else {
        if (tpu_soc_fp32_alloc_buffer(&g_tpu_ctx, &entry->buf, total_bytes) != 0) {
            return -1;
        }
        if (tpu_soc_fp32_validate_input_ddr_addr(entry->buf.device_addr, total_bytes) != 0) {
            tpu_soc_fp32_free_buffer(&entry->buf);
            memset(entry, 0, sizeof(*entry));
            return -1;
        }
    }
    memcpy(entry->buf.mapping, staging.data(), total_bytes);
    if (tpu_soc_fp32_buffer_sync_to_device(&entry->buf, total_bytes) != 0) {
        if (!entry->borrowed_buf) {
            tpu_soc_fp32_free_buffer(&entry->buf);
        }
        memset(entry, 0, sizeof(*entry));
        return -1;
    }

    if (pack_us_out != NULL) {
        const auto build_end = std::chrono::steady_clock::now();
        *pack_us_out += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(build_end - build_start).count();
    }
    return 0;
}

static int tpu_prepare_matmul_b_arena(
        const int8_t * b_kn,
        uint32_t N,
        uint32_t K,
        void * owner,
        tpu_device_b_matrix_cache_entry_t * transient_entry,
        tpu_device_b_matrix_cache_entry_t ** entry_out,
        uint64_t * pack_us_out) {
    if (entry_out == NULL || transient_entry == NULL) {
        errno = EINVAL;
        return -1;
    }
    *entry_out = NULL;

    const int use_pl_op_desc_for_pack = tpu_pl_op_descriptor_enabled();
    const uint32_t tile_n = use_pl_op_desc_for_pack ? tpu_pl_op_tile_n() : TPU_R5_MATMUL_TILE_N_LOCAL;
    const uint32_t tile_k = use_pl_op_desc_for_pack ? tpu_pl_op_tile_k() : TPU_R5_MATMUL_TILE_K_LOCAL;
    const uint32_t pack_layout = tpu_select_matmul_b_pack_layout(use_pl_op_desc_for_pack, tile_n, tile_k);
    const tpu_device_b_matrix_cache_key_t key = {
        owner,
        b_kn,
        N,
        K,
        tile_n,
        tile_k,
        pack_layout,
    };

    if (owner != NULL) {
        auto fast = g_tpu_device_b_matrix_owner_cache.find(owner);
        if (fast != g_tpu_device_b_matrix_owner_cache.end()) {
            tpu_device_b_matrix_cache_entry_t * entry = fast->second;
            if (entry != NULL &&
                entry->owner == owner &&
                entry->source_ptr == b_kn &&
                entry->source_n == N &&
                entry->source_k == K &&
                entry->tile_n == tile_n &&
                entry->tile_k == tile_k &&
                entry->pack_layout == pack_layout) {
                *entry_out = entry;
                g_b_packed_arena_hits++;
                return 0;
            }
        }
        auto hit = g_tpu_device_b_matrix_cache.find(key);
        if (hit != g_tpu_device_b_matrix_cache.end()) {
            *entry_out = &hit->second;
            g_tpu_device_b_matrix_owner_cache[owner] = &hit->second;
            g_b_packed_arena_hits++;
            return 0;
        }
    }

    const int reuse_transient_buf =
        owner == NULL && tpu_matrix_task_reuse_transient_b_enabled();
    if (tpu_build_b_packed_arena(
                b_kn,
                N,
                K,
                tile_n,
                tile_k,
                pack_layout,
                reuse_transient_buf,
                transient_entry,
                pack_us_out) != 0) {
        return -1;
    }
    g_b_packed_arena_misses++;
    transient_entry->owner = owner;
    transient_entry->source_ptr = b_kn;

    if (owner != NULL) {
        if (g_b_matrix_cache_resident_bytes + transient_entry->total_bytes > tpu_matrix_task_b_cache_max_bytes()) {
            g_b_matrix_cache_budget_rejects++;
            *entry_out = transient_entry;
            return 0;
        }
        auto inserted = g_tpu_device_b_matrix_cache.emplace(key, *transient_entry);
        if (inserted.second) {
            g_b_matrix_cache_resident_bytes += transient_entry->total_bytes;
            *entry_out = &inserted.first->second;
            g_tpu_device_b_matrix_owner_cache[owner] = &inserted.first->second;
            memset(transient_entry, 0, sizeof(*transient_entry));
            return 0;
        }
        tpu_soc_fp32_free_buffer(&transient_entry->buf);
        memset(transient_entry, 0, sizeof(*transient_entry));
        *entry_out = &inserted.first->second;
        g_tpu_device_b_matrix_owner_cache[owner] = &inserted.first->second;
        g_b_packed_arena_hits++;
        return 0;
    }

    *entry_out = transient_entry;
    return 0;
}

int tpu_matrix_task_prepack_b_with_cache(
    const int8_t  * b_kn,
    uint32_t N,
    uint32_t K,
    void * b_arena_owner) {
    tpu_device_b_matrix_cache_entry_t transient_b = {};
    tpu_device_b_matrix_cache_entry_t * b_entry = NULL;
    uint64_t b_pack_us = 0;
    int rc = -1;

    if (!tpu_r5_matmul_task_enabled()) {
        errno = ENOSYS;
        return -1;
    }
    if (b_kn == NULL || b_arena_owner == NULL || N == 0u || K == 0u) {
        errno = EINVAL;
        return -1;
    }
    if (!g_tpu_initialized && tpu_runtime_init() != 0) {
        return -1;
    }

    std::lock_guard<std::mutex> op_lock(g_tpu_op_mutex);
    std::lock_guard<std::mutex> lock(g_tpu_mutex);
    if (g_prevd_chain.active) {
        errno = EBUSY;
        return -1;
    }
    if (tpu_prepare_matmul_b_arena(
                b_kn,
                N,
                K,
                b_arena_owner,
                &transient_b,
                &b_entry,
                &b_pack_us) != 0 ||
            b_entry == NULL) {
        goto done;
    }
    g_total_pack_b_us += b_pack_us;
    g_total_pack_us += b_pack_us;

    // A transient entry means the arena budget rejected persistent caching.
    // Treat that as a prepack failure so the caller knows decode would still
    // pay the cold B-pack cost later.
    if (transient_b.buf.handle != NULL) {
        errno = ENOMEM;
        goto done;
    }

    rc = 0;

done:
    if (transient_b.buf.handle != NULL) {
        if (!transient_b.borrowed_buf) {
            tpu_soc_fp32_free_buffer(&transient_b.buf);
        }
        memset(&transient_b, 0, sizeof(transient_b));
    }
    return rc;
}

static void tpu_accumulate_matrix_task_tiling_stats(
        tpu_tiling_stats_t * stats,
        uint32_t M,
        uint32_t N,
        uint32_t K) {
    if (stats == NULL) {
        return;
    }
    const uint32_t tile_m = TPU_R5_MATMUL_TILE_M_LOCAL;
    const uint32_t tile_n = tpu_pl_op_descriptor_enabled() ? tpu_pl_op_tile_n() : TPU_R5_MATMUL_TILE_N_LOCAL;
    const uint32_t tile_k = tpu_pl_op_descriptor_enabled() ? tpu_pl_op_tile_k() : TPU_R5_MATMUL_TILE_K_LOCAL;
    const uint32_t m_tiles = tpu_ceil_div_u32(M, tile_m);
    const uint32_t n_tiles = tpu_ceil_div_u32(N, tile_n);
    const uint32_t k_tiles = tpu_ceil_div_u32(K, tile_k);
    const int64_t total_tiles = (int64_t) m_tiles * (int64_t) n_tiles * (int64_t) k_tiles;
    const int64_t useful = (int64_t) M * (int64_t) N * (int64_t) K;
    const int64_t padded = total_tiles * (int64_t) tile_m * (int64_t) tile_n * (int64_t) tile_k;

    if (total_tiles <= 0 || useful <= 0 || padded < useful) {
        return;
    }
    stats->total_hw_calls += total_tiles;
    if (tile_m == 32u && tile_n == 128u && tile_k == 128u) {
        stats->calls_32x128x128 += total_tiles;
    } else if (tile_m == 16u && tile_n == 128u && tile_k == 128u) {
        stats->calls_16x128x128 += total_tiles;
    } else if (tile_m == 1u && tile_n == 128u && tile_k == 128u) {
        stats->calls_1x128x128 += total_tiles;
    } else if (tile_m == 32u && tile_n == 64u && tile_k == 64u) {
        stats->calls_32x64x64 += total_tiles;
    } else if (tile_m == 16u && tile_n == 64u && tile_k == 64u) {
        stats->calls_16x64x64 += total_tiles;
    } else if (tile_m == 1u && tile_n == 64u && tile_k == 64u) {
        stats->calls_1x64x64 += total_tiles;
    } else if (tile_m == 1u && tile_n == 32u && tile_k == 32u) {
        stats->calls_1x32x32 += total_tiles;
    } else if (tile_m == 1u && tile_n == 16u && tile_k == 32u) {
        stats->calls_1x16x32 += total_tiles;
    } else if (tile_m == 32u && tile_n == 32u && tile_k == 32u) {
        stats->calls_32x32x32 += total_tiles;
    } else if (tile_m == 32u && tile_n == 32u && tile_k == 16u) {
        stats->calls_32x32x16 += total_tiles;
    } else if (tile_m == 16u && tile_n == 32u && tile_k == 32u) {
        stats->calls_16x32x32 += total_tiles;
    } else if (tile_m == 32u && tile_n == 16u && tile_k == 16u) {
        stats->calls_32x16x16 += total_tiles;
    } else if (tile_m == 16u && tile_n == 16u && tile_k == 16u) {
        stats->calls_16x16x16 += total_tiles;
    }
    stats->total_useful_elements += useful;
    stats->total_padded_elements += padded - useful;
}

static int tpu_run_pl_op_descriptor(
        uint32_t a_base,
        uint32_t b_base,
        uint32_t c_base,
        uint32_t M,
        uint32_t padded_N,
        uint32_t padded_K,
        uint32_t a_tile_stride,
        uint32_t c_tile_stride,
        uint32_t b_k_stride,
        uint32_t b_n_stride,
        uint32_t op_ctrl_flags,
        uint32_t expected_tiles,
        unsigned timeout_ms,
        uint32_t * status_out) {
    if (timeout_ms == 0u) {
        timeout_ms = 600000u;
    }

    const auto program_start = std::chrono::steady_clock::now();
    if (axil_mmio_write32(&g_tpu_ctx.csr_mmio, TPU_VANILLA_REG_OP_A_BASE, a_base) != 0 ||
        axil_mmio_write32(&g_tpu_ctx.csr_mmio, TPU_VANILLA_REG_OP_B_BASE, b_base) != 0 ||
        axil_mmio_write32(&g_tpu_ctx.csr_mmio, TPU_VANILLA_REG_OP_C_BASE, c_base) != 0 ||
        axil_mmio_write32(&g_tpu_ctx.csr_mmio, TPU_VANILLA_REG_OP_M, M) != 0 ||
        axil_mmio_write32(&g_tpu_ctx.csr_mmio, TPU_VANILLA_REG_OP_N, padded_N) != 0 ||
        axil_mmio_write32(&g_tpu_ctx.csr_mmio, TPU_VANILLA_REG_OP_K, padded_K) != 0 ||
        axil_mmio_write32(&g_tpu_ctx.csr_mmio, TPU_VANILLA_REG_OP_A_TILE_STRIDE, a_tile_stride) != 0 ||
        axil_mmio_write32(&g_tpu_ctx.csr_mmio, TPU_VANILLA_REG_OP_C_TILE_STRIDE, c_tile_stride) != 0 ||
        axil_mmio_write32(&g_tpu_ctx.csr_mmio, TPU_VANILLA_REG_OP_B_K_STRIDE, b_k_stride) != 0 ||
        axil_mmio_write32(&g_tpu_ctx.csr_mmio, TPU_VANILLA_REG_OP_B_N_STRIDE, b_n_stride) != 0 ||
        axil_mmio_write32(&g_tpu_ctx.csr_mmio, TPU_VANILLA_REG_OP_CTRL,
                          TPU_VANILLA_OP_CTRL_START | op_ctrl_flags) != 0) {
        return -1;
    }
    const auto program_end = std::chrono::steady_clock::now();
    g_pl_op_desc_program_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(program_end - program_start).count();

    uint32_t status = 0u;
    const uint64_t deadline_ms = (uint64_t) timeout_ms +
        (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    const auto wait_start = std::chrono::steady_clock::now();
    uint64_t polls = 0u;
    const uint32_t poll_sleep_us = tpu_pl_op_poll_sleep_us();
    const uint32_t poll_spins = tpu_pl_op_poll_spins();
    for (;;) {
        if (axil_mmio_read32(&g_tpu_ctx.csr_mmio, TPU_VANILLA_REG_OP_STATUS, &status) != 0) {
            return -1;
        }
        ++polls;
        if ((status & TPU_VANILLA_OP_STATUS_ERROR) != 0u ||
            (status & TPU_VANILLA_OP_STATUS_DONE) != 0u) {
            break;
        }
        const uint64_t now_ms = (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        if (now_ms > deadline_ms) {
            if (status_out != NULL) {
                *status_out = status;
            }
            errno = ETIMEDOUT;
            return -1;
        }
        if (poll_sleep_us != 0u) {
            std::this_thread::sleep_for(std::chrono::microseconds(poll_sleep_us));
        } else if (poll_spins != 0u) {
            tpu_pl_op_poll_pause(poll_spins);
        }
    }
    const auto wait_end = std::chrono::steady_clock::now();
    g_pl_op_desc_wait_us += (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(wait_end - wait_start).count();
    g_pl_op_desc_status_polls += polls;

    if (status_out != NULL) {
        *status_out = status;
    }
    if ((status & TPU_VANILLA_OP_STATUS_ERROR) != 0u) {
        errno = EIO;
        return -1;
    }
    const uint32_t completed =
        (status & TPU_VANILLA_OP_STATUS_COMPLETED_MASK) >> TPU_VANILLA_OP_STATUS_COMPLETED_SHIFT;
    if (expected_tiles != 0u && completed != expected_tiles) {
        errno = EIO;
        return -1;
    }
    return 0;
}

void tpu_gemm_int8_int32(
    const int8_t  * a,
    const int8_t  * b,
    const int32_t * c,
    int32_t       * d,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
    if (c != NULL) {
        for (uint32_t i = 0; i < m*n; ++i) {
            if (c[i] != 0) {
                fprintf(stderr, "tpu_gemm_int8_int32: HOST_C is not supported by the PREV_D runtime\n");
                abort();
            }
        }
    }
    tpu_tile_execute_desc_t desc = {};
    tpu_tile_execute_result_t result = {};
    desc.a = a;
    desc.b = b;
    desc.d = d;
    desc.m = m;
    desc.n = n;
    desc.k = k;
    desc.b_layout = TPU_B_LAYOUT_NK;
    desc.c_source = TPU_C_SOURCE_ZERO;
    desc.d_dest = TPU_D_DEST_HOST_READBACK;
    desc.accum_key = { ++g_tile_call_count, 0u, 0u, m, n };
    if (tpu_real_tile_execute_cached(&desc, &result) != 0) {
        perror("tpu_gemm_int8_int32");
        abort();
    }
}

void tpu_gemm_int8_int32_view(
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
    g_tpu_output_view_scratch.resize((size_t) m * (size_t) n);
    tpu_gemm_int8_int32(a, b, c, g_tpu_output_view_scratch.data(), m, n, k);
    out_view->data = g_tpu_output_view_scratch.data();
    out_view->m = m;
    out_view->n = n;
    out_view->stride = n;
    out_view->elem_count = m * n;
}

void tpu_gemm_int8_int32_prepacked_b(
    const int8_t  * a,
    const int8_t  * b_kn,
    const int32_t * c,
    int32_t       * d,
    uint32_t m,
    uint32_t n,
    uint32_t k) {
    if (c != NULL) {
        for (uint32_t i = 0; i < m*n; ++i) {
            if (c[i] != 0) {
                fprintf(stderr, "tpu_gemm_int8_int32_prepacked_b: HOST_C is not supported by the PREV_D runtime\n");
                abort();
            }
        }
    }
    tpu_tile_execute_desc_t desc = {};
    tpu_tile_execute_result_t result = {};
    desc.a = a;
    desc.b = b_kn;
    desc.d = d;
    desc.m = m;
    desc.n = n;
    desc.k = k;
    desc.b_layout = TPU_B_LAYOUT_KN;
    desc.c_source = TPU_C_SOURCE_ZERO;
    desc.d_dest = TPU_D_DEST_HOST_READBACK;
    desc.accum_key = { ++g_tile_call_count, 0u, 0u, m, n };
    if (tpu_real_tile_execute_cached(&desc, &result) != 0) {
        perror("tpu_gemm_int8_int32_prepacked_b");
        abort();
    }
}

void tpu_gemm_int8_int32_prepacked_b_view(
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
    g_tpu_output_view_scratch.resize((size_t) m * (size_t) n);
    tpu_gemm_int8_int32_prepacked_b(a, b_kn, c, g_tpu_output_view_scratch.data(), m, n, k);
    out_view->data = g_tpu_output_view_scratch.data();
    out_view->m = m;
    out_view->n = n;
    out_view->stride = n;
    out_view->elem_count = m * n;
}

static int tpu_gemm_int8_int32_matrix_task_prepacked_b_impl(
    const int8_t  * a_mk,
    const int8_t  * b_kn,
    int32_t       * d_mn,
    tpu_int32_view_t * out_view,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    void * b_arena_owner) {
    const auto impl_start = std::chrono::steady_clock::now();
    const int detail_timing = tpu_matrix_task_detail_timing_enabled();
    const uint32_t tile_m = TPU_R5_MATMUL_TILE_M_LOCAL;
    const uint32_t pl_op_desc_max_m = tpu_pl_op_descriptor_max_m();
    const int use_pl_op_desc = tpu_pl_op_descriptor_enabled() && M <= pl_op_desc_max_m;
    const uint32_t tile_n = use_pl_op_desc ? tpu_pl_op_tile_n() : TPU_R5_MATMUL_TILE_N_LOCAL;
    const uint32_t tile_k = use_pl_op_desc ? tpu_pl_op_tile_k() : TPU_R5_MATMUL_TILE_K_LOCAL;
    const size_t a_bytes = (size_t) M * (size_t) K * sizeof(int8_t);
    const size_t c_bytes = (size_t) M * (size_t) N * sizeof(int32_t);
    const uint32_t n_tiles = tpu_ceil_div_u32(N, tile_n);
    const uint32_t k_tiles = tpu_ceil_div_u32(K, tile_k);
    const uint32_t m_tiles = tpu_ceil_div_u32(M, tile_m);
    const uint32_t padded_n = n_tiles * tile_n;
    const uint32_t padded_k = k_tiles * tile_k;
    const int use_a_tile_contig =
        use_pl_op_desc &&
        tpu_pl_op_a_tile_contig_enabled() &&
        M > 1u &&
        k_tiles > 1u;
    const size_t a_device_rows = use_a_tile_contig ? (size_t) m_tiles * (size_t) tile_m : (size_t) M;
    const size_t a_device_bytes = a_device_rows * (size_t) padded_k * sizeof(int8_t);
    const size_t c_device_bytes = (size_t) M * (size_t) padded_n * sizeof(int32_t);
    const uint32_t total_tiles = m_tiles * n_tiles * k_tiles;
    tpu_device_b_matrix_cache_entry_t transient_b = {};
    tpu_device_b_matrix_cache_entry_t * b_entry = NULL;
    tpu_r5_matmul_task_t task = {};
    tpu_r5_matmul_result_t result = {};
    const unsigned timeout_ms = tpu_r5_matmul_timeout_ms();
    uint32_t pl_op_status = 0u;
    uint32_t completed_tiles = total_tiles;
    uint64_t b_pack_us = 0;
    uint64_t entry_setup_us = 0;
    uint64_t lock_us = 0;
    uint64_t reserve_us = 0;
    uint64_t prepare_b_wall_us = 0;
    uint64_t copy_a_us = 0;
    uint64_t sync_a_us = 0;
    uint64_t addr_check_us = 0;
    uint64_t submit_us = 0;
    uint64_t output_sync_us = 0;
    uint64_t output_view_us = 0;
    uint64_t output_copy_us = 0;
    uint64_t post_us = 0;
    uint64_t post_stats_us = 0;
    uint64_t free_us = 0;
    int rc = -1;
    std::chrono::steady_clock::time_point addr_check_start;

    if (!use_pl_op_desc && !tpu_r5_matmul_task_enabled()) {
        g_matmul_task_fallback_count++;
        errno = ENOSYS;
        return -1;
    }
    if (out_view != NULL) {
        memset(out_view, 0, sizeof(*out_view));
    }
    if (a_mk == NULL || b_kn == NULL || (d_mn == NULL && out_view == NULL) ||
        M == 0u || N == 0u || K == 0u ||
        a_bytes == 0u || c_bytes == 0u ||
        a_device_bytes == 0u || c_device_bytes == 0u ||
        n_tiles == 0u || k_tiles == 0u || m_tiles == 0u ||
        padded_n == 0u || padded_k == 0u) {
        g_matmul_task_fallback_count++;
        errno = EINVAL;
        return -1;
    }
    if (!g_tpu_initialized && tpu_runtime_init() != 0) {
        g_matmul_task_fallback_count++;
        return -1;
    }
    if (detail_timing) {
        entry_setup_us = tpu_elapsed_us(impl_start, std::chrono::steady_clock::now());
    }

    const auto lock_start = tpu_detail_now(detail_timing);
    std::lock_guard<std::mutex> op_lock(g_tpu_op_mutex);
    std::lock_guard<std::mutex> lock(g_tpu_mutex);
    lock_us = tpu_detail_elapsed_us(detail_timing, lock_start);
    const auto reserve_start = tpu_detail_now(detail_timing);
    if (g_prevd_chain.active) {
        g_matmul_task_fallback_count++;
        errno = EBUSY;
        return -1;
    }
    if (tpu_reserve_matmul_buffer(&g_tpu_matmul_a_buf, &g_tpu_matmul_a_buf_bytes, a_device_bytes, 0) != 0 ||
        tpu_reserve_matmul_buffer(&g_tpu_matmul_c_buf, &g_tpu_matmul_c_buf_bytes, c_device_bytes, 1) != 0) {
        g_matmul_task_fallback_count++;
        return -1;
    }
    reserve_us = tpu_detail_elapsed_us(detail_timing, reserve_start);

    const auto prepare_b_start = tpu_detail_now(detail_timing);
    const int prepare_b_rc = tpu_prepare_matmul_b_arena(
            b_kn, N, K, b_arena_owner, &transient_b, &b_entry, &b_pack_us);
    prepare_b_wall_us = tpu_detail_elapsed_us(detail_timing, prepare_b_start);
    if (prepare_b_rc != 0 || b_entry == NULL) {
        if (transient_b.buf.handle != NULL) {
            if (!transient_b.borrowed_buf) {
                tpu_soc_fp32_free_buffer(&transient_b.buf);
            }
            memset(&transient_b, 0, sizeof(transient_b));
        }
        g_matmul_task_fallback_count++;
        return -1;
    }

    const auto copy_a_start = tpu_detail_now(detail_timing);
    if (use_a_tile_contig) {
        memset(g_tpu_matmul_a_buf.mapping, 0, a_device_bytes);
        uint8_t * dst = (uint8_t *) g_tpu_matmul_a_buf.mapping;
        const uint32_t tile_k_bytes = tile_k * (uint32_t) sizeof(int8_t);
        for (uint32_t mt = 0; mt < m_tiles; ++mt) {
            for (uint32_t kt = 0; kt < k_tiles; ++kt) {
                const uint32_t src_k = kt * tile_k;
                const uint32_t copy_k = (src_k < K) ? std::min(tile_k, K - src_k) : 0u;
                uint8_t * tile_dst = dst + ((size_t) mt * (size_t) k_tiles + (size_t) kt) *
                                            (size_t) tile_m * (size_t) tile_k_bytes;
                if (copy_k == 0u) {
                    continue;
                }
                for (uint32_t row_off = 0; row_off < tile_m; ++row_off) {
                    const uint32_t row = mt * tile_m + row_off;
                    if (row >= M) {
                        break;
                    }
                    memcpy(tile_dst + (size_t) row_off * (size_t) tile_k_bytes,
                           a_mk + (size_t) row * (size_t) K + (size_t) src_k,
                           (size_t) copy_k * sizeof(int8_t));
                }
            }
        }
    } else if (padded_k == K) {
        memcpy(g_tpu_matmul_a_buf.mapping, a_mk, a_bytes);
    } else {
        memset(g_tpu_matmul_a_buf.mapping, 0, a_device_bytes);
        uint8_t * dst = (uint8_t *) g_tpu_matmul_a_buf.mapping;
        for (uint32_t row = 0; row < M; ++row) {
            memcpy(dst + (size_t) row * (size_t) padded_k,
                   a_mk + (size_t) row * (size_t) K,
                   (size_t) K * sizeof(int8_t));
        }
    }
    copy_a_us = tpu_detail_elapsed_us(detail_timing, copy_a_start);
    g_total_copy_a_us += copy_a_us;
    g_total_copy_us += copy_a_us;
    g_total_pack_b_us += b_pack_us;
    g_total_pack_us += b_pack_us;

    const auto sync_a_start = tpu_detail_now(detail_timing);
    if (!tpu_matrix_task_skip_a_sync_enabled()) {
        if (tpu_soc_fp32_buffer_sync_to_device(&g_tpu_matmul_a_buf, a_device_bytes) != 0) {
            goto done;
        }
    }
    sync_a_us = tpu_detail_elapsed_us(detail_timing, sync_a_start);
    addr_check_start = tpu_detail_now(detail_timing);
    if (g_tpu_matmul_a_buf.device_addr > UINT32_MAX ||
        b_entry->buf.device_addr > UINT32_MAX ||
        g_tpu_matmul_c_buf.device_addr > UINT32_MAX ||
        g_tpu_ctx.csr_mmio.base_addr > UINT32_MAX) {
        errno = ERANGE;
        goto done;
    }
    addr_check_us = tpu_detail_elapsed_us(detail_timing, addr_check_start);

    if (use_pl_op_desc) {
        const auto submit_start = std::chrono::steady_clock::now();
        const uint32_t a_tile_stride = padded_k * tile_m * (uint32_t) sizeof(int8_t);
        const uint32_t c_tile_stride = padded_n * tile_m * (uint32_t) sizeof(int32_t);
        const uint32_t b_k_stride = tile_n * tile_k * (uint32_t) sizeof(int8_t);
        const uint32_t b_n_stride = k_tiles * b_k_stride;
        const uint32_t op_ctrl_flags = use_a_tile_contig ? TPU_VANILLA_OP_CTRL_A_TILE_CONTIG : 0u;
        const int submit_rc = tpu_run_pl_op_descriptor(
                    (uint32_t) g_tpu_matmul_a_buf.device_addr,
                    (uint32_t) b_entry->buf.device_addr,
                    (uint32_t) g_tpu_matmul_c_buf.device_addr,
                    M,
                    padded_n,
                    padded_k,
                    a_tile_stride,
                    c_tile_stride,
                    b_k_stride,
                    b_n_stride,
                    op_ctrl_flags,
                    total_tiles,
                    timeout_ms,
                    &pl_op_status);
        submit_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - submit_start).count();
        if (submit_rc != 0) {
            fprintf(stderr,
                    "[FPGA/PL-OP] op descriptor failed errno=%d M=%u N=%u K=%u "
                    "padded_N=%u padded_K=%u A=0x%08x B=0x%08x C=0x%08x "
                    "strides A=%u C=%u BK=%u BN=%u flags=0x%08x tiles=%u:%u:%u total=%u "
                    "status=0x%08x\n",
                    errno,
                    M, N, K,
                    padded_n,
                    padded_k,
                    (uint32_t) g_tpu_matmul_a_buf.device_addr,
                    (uint32_t) b_entry->buf.device_addr,
                    (uint32_t) g_tpu_matmul_c_buf.device_addr,
                    a_tile_stride,
                    c_tile_stride,
                    b_k_stride,
                    b_n_stride,
                    op_ctrl_flags,
                    m_tiles,
                    n_tiles,
                    k_tiles,
                    total_tiles,
                    pl_op_status);
            goto done;
        }
        completed_tiles =
            (pl_op_status & TPU_VANILLA_OP_STATUS_COMPLETED_MASK) >> TPU_VANILLA_OP_STATUS_COMPLETED_SHIFT;
        if (tpu_pl_op_descriptor_log_enabled()) {
            fprintf(stderr,
                    "[FPGA/PL-OP] done M=%u N=%u K=%u padded_N=%u padded_K=%u "
                    "a_tile_contig=%d tiles=%u status=0x%08x\n",
                    M, N, K, padded_n, padded_k, use_a_tile_contig, completed_tiles, pl_op_status);
        }
    } else {
        const auto submit_start = std::chrono::steady_clock::now();
        task.magic = TPU_R5_MATMUL_TASK_MAGIC;
        task.version = TPU_R5_MATMUL_TASK_VERSION;
        task.flags = TPU_R5_MATMUL_FLAG_A_INT8_RAW;
        task.fence_id = ++g_matmul_task_fence_id;
        if (task.fence_id == 0u) {
            task.fence_id = ++g_matmul_task_fence_id;
        }
        task.a_base = (uint32_t) g_tpu_matmul_a_buf.device_addr;
        task.b_packed_base = (uint32_t) b_entry->buf.device_addr;
        task.c_base = (uint32_t) g_tpu_matmul_c_buf.device_addr;
        task.m = M;
        task.n = padded_n;
        task.k = padded_k;
        task.lda_bytes = padded_k * sizeof(int8_t);
        task.ldc_bytes = padded_n * sizeof(int32_t);
        task.tile_m = tile_m;
        task.tile_n = tile_n;
        task.tile_k = tile_k;
        task.b_tile_bytes = (uint32_t) ((size_t) tile_n * (size_t) tile_k);
        task.b_n_tiles = n_tiles;
        task.b_k_tiles = k_tiles;
        task.status = TPU_R5_MATMUL_STATUS_SUBMITTED;
        task.total_tiles = total_tiles;

        const int submit_rc = tpu_r5_ctrl_run_matmul_task((uint32_t) g_tpu_ctx.csr_mmio.base_addr,
                                        &task,
                                        timeout_ms,
                                        &result);
        submit_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - submit_start).count();
        if (submit_rc != 0) {
            fprintf(stderr,
                    "[FPGA/R5] matrix-task submit failed errno=%d M=%u N=%u K=%u "
                    "A=0x%08x B=0x%08x C=0x%08x lda=%u ldc=%u "
                    "tiles=%u:%u:%u total=%u fence=%u "
                    "result status=%u error=%u completed=%u/%u done_fence=%u "
                    "r5_cycles=%u fpga_wait_cycles=%u last_hw_status=0x%08x\n",
                    errno,
                    M, N, K,
                    task.a_base,
                    task.b_packed_base,
                    task.c_base,
                    task.lda_bytes,
                    task.ldc_bytes,
                    m_tiles,
                    n_tiles,
                    k_tiles,
                    total_tiles,
                    task.fence_id,
                    result.status,
                    result.error,
                    result.completed_tiles,
                    result.total_tiles,
                    result.done_fence,
                    result.r5_cycles,
                    result.fpga_wait_cycles,
                    result.last_hw_status);
            goto done;
        }
        if (result.error != TPU_R5_MATMUL_ERR_NONE ||
            result.status != TPU_R5_MATMUL_STATUS_DONE ||
            result.done_fence != task.fence_id ||
            result.completed_tiles != total_tiles) {
            fprintf(stderr,
                    "[FPGA/R5] matrix-task bad result M=%u N=%u K=%u "
                    "tiles=%u:%u:%u total=%u fence=%u "
                    "result status=%u error=%u completed=%u/%u done_fence=%u "
                    "r5_cycles=%u fpga_wait_cycles=%u last_hw_status=0x%08x\n",
                    M, N, K,
                    m_tiles,
                    n_tiles,
                    k_tiles,
                    total_tiles,
                    task.fence_id,
                    result.status,
                    result.error,
                    result.completed_tiles,
                    result.total_tiles,
                    result.done_fence,
                    result.r5_cycles,
                    result.fpga_wait_cycles,
                    result.last_hw_status);
            errno = EIO;
            goto done;
        }
        completed_tiles = result.completed_tiles;
    }

    {
        const auto sync_start = std::chrono::steady_clock::now();
        if (!tpu_matrix_task_skip_output_sync_enabled()) {
            if (tpu_soc_fp32_buffer_sync_from_device(&g_tpu_matmul_c_buf, c_device_bytes) != 0) {
                goto done;
            }
        }
        const auto sync_end = std::chrono::steady_clock::now();
        output_sync_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(sync_end - sync_start).count();
        g_matmul_task_output_sync_us += output_sync_us;

        if (out_view != NULL) {
            const auto view_start = tpu_detail_now(detail_timing);
            out_view->data = (const int32_t *) g_tpu_matmul_c_buf.mapping;
            out_view->m = M;
            out_view->n = N;
            out_view->stride = padded_n;
            out_view->elem_count = M * padded_n;
            output_view_us = tpu_detail_elapsed_us(detail_timing, view_start);
        } else {
            const auto copy_start = tpu_detail_now(detail_timing);
            if (padded_n == N) {
                memcpy(d_mn, g_tpu_matmul_c_buf.mapping, c_bytes);
            } else {
                const uint8_t * src = (const uint8_t *) g_tpu_matmul_c_buf.mapping;
                for (uint32_t row = 0; row < M; ++row) {
                    memcpy(d_mn + (size_t) row * (size_t) N,
                           src + (size_t) row * (size_t) padded_n * sizeof(int32_t),
                           (size_t) N * sizeof(int32_t));
                }
            }
            output_copy_us = tpu_detail_elapsed_us(detail_timing, copy_start);
            g_matmul_task_output_copy_jobs++;
            g_matmul_task_output_copy_bytes += c_bytes;
            if (padded_n == N) {
                g_matmul_task_output_copy_direct_jobs++;
                g_matmul_task_output_copy_direct_us += output_copy_us;
                g_matmul_task_output_copy_direct_bytes += c_bytes;
            } else {
                g_matmul_task_output_copy_row_jobs++;
                g_matmul_task_output_copy_row_us += output_copy_us;
                g_matmul_task_output_copy_row_bytes += c_bytes;
                g_matmul_task_output_copy_row_rows += M;
                g_matmul_task_output_copy_padded_jobs++;
                g_matmul_task_output_copy_padded_us += output_copy_us;
            }
            if (M == 1u) {
                g_matmul_task_output_copy_m1_jobs++;
                g_matmul_task_output_copy_m1_us += output_copy_us;
            } else if (M == 32u) {
                g_matmul_task_output_copy_m32_jobs++;
                g_matmul_task_output_copy_m32_us += output_copy_us;
            }
            if (N <= 64u) {
                g_matmul_task_output_copy_n64_jobs++;
                g_matmul_task_output_copy_n64_us += output_copy_us;
            } else if (N == 128u) {
                g_matmul_task_output_copy_n128_jobs++;
                g_matmul_task_output_copy_n128_us += output_copy_us;
            } else {
                g_matmul_task_output_copy_n_other_jobs++;
                g_matmul_task_output_copy_n_other_us += output_copy_us;
            }
        }
        g_matmul_task_output_copy_us += output_copy_us;
        if (stats != NULL) {
            stats->total_copy_a_ns += (int64_t) copy_a_us * 1000;
            stats->total_copy_b_ns += (int64_t) b_pack_us * 1000;
            stats->total_copy_out_final_ns += (int64_t) output_copy_us * 1000;
        }
    }

    {
        const auto post_start = tpu_detail_now(detail_timing);
        g_matmul_task_count++;
        g_matmul_task_tile_count += completed_tiles;
        if (use_pl_op_desc) {
            g_pl_op_desc_count++;
            g_pl_op_desc_tile_count += completed_tiles;
        } else {
            g_matmul_task_r5_cycles += result.r5_cycles;
            g_matmul_task_fpga_wait_cycles += result.fpga_wait_cycles;
        }
        if (tpu_matrix_task_tiling_stats_enabled()) {
            const auto post_stats_start = std::chrono::steady_clock::now();
            tpu_accumulate_matrix_task_tiling_stats(stats, M, N, K);
            post_stats_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - post_stats_start).count();
        }
        post_us = tpu_detail_elapsed_us(detail_timing, post_start);
    }
    rc = 0;

done:
    if (rc != 0) {
        g_matmul_task_fallback_count++;
    }
    if (transient_b.buf.handle != NULL) {
        const auto free_start = tpu_detail_now(detail_timing);
        if (!transient_b.borrowed_buf) {
            tpu_soc_fp32_free_buffer(&transient_b.buf);
        }
        memset(&transient_b, 0, sizeof(transient_b));
        free_us = tpu_detail_elapsed_us(detail_timing, free_start);
    }
    {
        const uint64_t total_us = (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - impl_start).count();
        const uint64_t known_us =
            entry_setup_us +
            lock_us +
            reserve_us +
            prepare_b_wall_us +
            copy_a_us +
            sync_a_us +
            addr_check_us +
            submit_us +
            output_sync_us +
            output_view_us +
            output_copy_us +
            post_us +
            free_us;
        g_matmul_task_total_us += total_us;
        g_matmul_task_entry_setup_us += entry_setup_us;
        g_matmul_task_lock_us += lock_us;
        g_matmul_task_reserve_us += reserve_us;
        g_matmul_task_prepare_b_wall_us += prepare_b_wall_us;
        g_matmul_task_copy_a_wall_us += copy_a_us;
        g_matmul_task_sync_a_us += sync_a_us;
        g_matmul_task_addr_check_us += addr_check_us;
        g_matmul_task_submit_us += submit_us;
        g_matmul_task_post_us += post_us;
        g_matmul_task_post_stats_us += post_stats_us;
        g_matmul_task_output_view_us += output_view_us;
        g_matmul_task_free_us += free_us;
        if (total_us > known_us) {
            g_matmul_task_residual_us += total_us - known_us;
        }
    }
    return rc;
}

int tpu_gemm_int8_int32_matrix_task_prepacked_b_with_cache(
    const int8_t  * a_mk,
    const int8_t  * b_kn,
    int32_t       * d_mn,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    void * b_arena_owner) {
    return tpu_gemm_int8_int32_matrix_task_prepacked_b_impl(
            a_mk, b_kn, d_mn, NULL, M, N, K, stats, b_arena_owner);
}

int tpu_gemm_int8_int32_matrix_task_prepacked_b_view_with_cache(
    const int8_t    * a_mk,
    const int8_t    * b_kn,
    tpu_int32_view_t * out_view,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    void * b_arena_owner) {
    return tpu_gemm_int8_int32_matrix_task_prepacked_b_impl(
            a_mk, b_kn, NULL, out_view, M, N, K, stats, b_arena_owner);
}

void tpu_gemm_int8_int32_tiled(
    const int8_t  * a,
    const int8_t  * b,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats) {
    std::lock_guard<std::mutex> op_lock(g_tpu_op_mutex);
    const int async_batch = tpu_async_readback_batch_begin(M, N);
    const int batch = async_batch > 0 ? 0 : tpu_op_readback_batch_begin(M, N);
    tpu_gemm_int8_int32_tiled_device_psum_impl(a, b, d, M, N, K, stats, tpu_real_tile_execute_cached, tpu_real_sequence_execute_cached);
    if (batch > 0 && tpu_op_readback_batch_flush() != 0) {
        abort();
    }
    if (async_batch > 0 && tpu_async_readback_batch_flush() != 0) {
        abort();
    }
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
    std::lock_guard<std::mutex> op_lock(g_tpu_op_mutex);
    const int async_batch = tpu_async_readback_batch_begin(M, N);
    const int batch = async_batch > 0 ? 0 : tpu_op_readback_batch_begin(M, N);
    tpu_gemm_int8_int32_tiled_prepacked_b_device_psum_impl(
        a,
        b_kn,
        d,
        M,
        N,
        K,
        stats,
        tpu_real_tile_execute_cached,
        tpu_real_sequence_execute_cached,
        packed_b_cache);
    if (batch > 0 && tpu_op_readback_batch_flush() != 0) {
        abort();
    }
    if (async_batch > 0 && tpu_async_readback_batch_flush() != 0) {
        abort();
    }
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
    std::lock_guard<std::mutex> op_lock(g_tpu_op_mutex);
    const int async_batch = tpu_async_readback_batch_begin(M, N);
    const int batch = async_batch > 0 ? 0 : tpu_op_readback_batch_begin(M, N);
    tpu_gemm_f32_int8_int32_tiled_prepacked_b_device_psum_impl(
        a,
        a_inv_scales,
        b_kn,
        d,
        M,
        N,
        K,
        stats,
        tpu_real_tile_execute_cached,
        tpu_real_sequence_execute_cached,
        packed_b_cache);
    if (batch > 0 && tpu_op_readback_batch_flush() != 0) {
        abort();
    }
    if (async_batch > 0 && tpu_async_readback_batch_flush() != 0) {
        abort();
    }
}

void tpu_gemm_int8_int32_tiled_prepacked_b(
    const int8_t  * a,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats) {
    tpu_gemm_int8_int32_tiled_prepacked_b_with_cache(
        a,
        b_kn,
        d,
        M,
        N,
        K,
        stats,
        NULL);
}
