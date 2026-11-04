#pragma once

#include <stddef.h>
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
    int64_t total_input_padded_elements;
    int64_t total_input_useful_elements;
    int64_t total_zero_ns;
    int64_t total_copy_a_ns;
    int64_t total_copy_b_ns;
    int64_t total_copy_c_ns;
    int64_t total_tile_call_ns;
    int64_t total_copy_out_partial_ns;
    int64_t total_copy_out_final_ns;
    int64_t direct_final_hits;
    int64_t direct_final_misses;
} tpu_tiling_stats_t;

typedef struct {
    const int32_t *data;
    uint32_t m;
    uint32_t n;
    uint32_t stride;
    uint32_t elem_count;
} tpu_int32_view_t;

typedef struct {
    uint64_t tile_call_count;
    uint64_t prevd_chain_runs;
    uint64_t prevd_over_signoff_runs;
    uint64_t prevd_ab_preload_count;
    uint64_t prevd_host_c_upload_count;
    uint64_t prevd_host_d_readback_count;
    uint64_t prevd_direct_final_readback_count;
    uint64_t total_wait_done_us;
    uint64_t total_read_us;
    uint64_t total_read_sync_us;
    uint64_t total_read_copy_us;
    uint64_t total_read_unpack_us;
    uint64_t total_read_mmio_us;
    uint64_t total_read_debug_dump_us;
    uint64_t total_read_bytes;
    uint64_t total_read_output_buffer_count;
    uint64_t total_read_memcpy_count;
    uint64_t total_read_unpack_count;
    uint64_t total_read_mmio_count;
    uint64_t read_op_batch_count;
    uint64_t read_op_batch_total_us;
    uint64_t read_op_batch_sync_us;
    uint64_t read_op_batch_copy_us;
    uint64_t read_op_batch_bytes;
    uint64_t read_async_count;
    uint64_t read_async_total_us;
    uint64_t read_async_sync_us;
    uint64_t read_async_copy_us;
    uint64_t read_async_bytes;
    uint64_t read_async_batch_begin_wait_us;
    uint64_t read_async_batch_begin_wait_count;
    uint64_t read_async_stage_wait_us;
    uint64_t read_async_stage_wait_count;
    uint64_t read_async_flush_wait_us;
    uint64_t read_async_flush_wait_count;
    uint64_t read_final_desc_count;
    uint64_t read_final_desc_total_us;
    uint64_t read_final_desc_sync_us;
    uint64_t read_final_desc_copy_us;
    uint64_t read_final_desc_unpack_us;
    uint64_t read_final_desc_mmio_us;
    uint64_t read_final_desc_bytes;
    uint64_t read_copy_job_count;
    uint64_t read_copy_row_count;
    uint64_t read_copy_elem_count;
    uint64_t read_copy_useful_bytes;
    uint64_t read_copy_transfer_bytes;
    uint64_t read_copy_padded_jobs;
    uint64_t read_copy_strided_jobs;
    uint64_t read_copy_bulk_candidate_jobs;
    uint64_t read_copy_bulk_candidate_bytes;
    uint64_t read_copy_bulk_memcpy_jobs;
    uint64_t read_copy_bulk_memcpy_bytes;
    uint64_t read_copy_row_memcpy_rows;
    uint64_t read_copy_rows_1;
    uint64_t read_copy_rows_2_15;
    uint64_t read_copy_rows_16;
    uint64_t read_copy_rows_17_31;
    uint64_t read_copy_rows_32;
    uint64_t read_copy_rows_gt32;
    uint64_t read_copy_cols_le16;
    uint64_t read_copy_cols_le32;
    uint64_t read_copy_cols_le64;
    uint64_t read_copy_cols_65_127;
    uint64_t read_copy_cols_128;
    uint64_t read_copy_cols_gt128;
    uint64_t total_prepare_us;
    uint64_t total_pack_us;
    uint64_t total_pack_a_us;
    uint64_t total_pack_b_us;
    uint64_t total_copy_us;
    uint64_t total_copy_a_us;
    uint64_t total_copy_b_us;
    uint64_t total_control_us;
    uint64_t total_preload_us;
    uint64_t total_preload_sync_us;
    uint64_t total_preload_wait_us;
    uint64_t total_preload_setup_us;
    uint64_t total_preload_xfer_us;
    uint64_t total_start_us;
    uint64_t total_start_sync_us;
    uint64_t total_start_ready_us;
    uint64_t total_start_setup_us;
    uint64_t total_start_mmio_us;
    uint64_t total_start_xfer_us;
    uint64_t total_wait_prev_us;
    uint64_t total_wait_final_us;
    uint64_t total_cleanup_us;
    uint64_t total_log_us;
    uint64_t total_unlock_us;
    uint64_t total_other_us;
    uint64_t prevd_sequence_program_count;
    uint64_t prevd_sequence_fallback_count;
    uint64_t total_sequence_us;
    uint64_t total_sequence_sync_us;
    uint64_t total_sequence_ready_us;
    uint64_t total_sequence_program_us;
    uint64_t total_sequence_wait_done_us;
    uint64_t auto_next_program_count;
    uint64_t auto_next_fallback_count;
    uint64_t matmul_task_count;
    uint64_t matmul_task_tile_count;
    uint64_t matmul_task_fallback_count;
    uint64_t matmul_task_r5_cycles;
    uint64_t matmul_task_fpga_wait_cycles;
    uint64_t matmul_task_total_us;
    uint64_t matmul_task_entry_setup_us;
    uint64_t matmul_task_lock_us;
    uint64_t matmul_task_reserve_us;
    uint64_t matmul_task_prepare_b_wall_us;
    uint64_t matmul_task_copy_a_wall_us;
    uint64_t matmul_task_sync_a_us;
    uint64_t matmul_task_addr_check_us;
    uint64_t matmul_task_submit_us;
    uint64_t matmul_task_output_sync_us;
    uint64_t matmul_task_output_view_us;
    uint64_t matmul_task_output_copy_us;
    uint64_t matmul_task_output_copy_jobs;
    uint64_t matmul_task_output_copy_bytes;
    uint64_t matmul_task_output_copy_direct_jobs;
    uint64_t matmul_task_output_copy_direct_us;
    uint64_t matmul_task_output_copy_direct_bytes;
    uint64_t matmul_task_output_copy_row_jobs;
    uint64_t matmul_task_output_copy_row_us;
    uint64_t matmul_task_output_copy_row_bytes;
    uint64_t matmul_task_output_copy_row_rows;
    uint64_t matmul_task_output_copy_padded_jobs;
    uint64_t matmul_task_output_copy_padded_us;
    uint64_t matmul_task_output_copy_m1_jobs;
    uint64_t matmul_task_output_copy_m1_us;
    uint64_t matmul_task_output_copy_m32_jobs;
    uint64_t matmul_task_output_copy_m32_us;
    uint64_t matmul_task_output_copy_n64_jobs;
    uint64_t matmul_task_output_copy_n64_us;
    uint64_t matmul_task_output_copy_n128_jobs;
    uint64_t matmul_task_output_copy_n128_us;
    uint64_t matmul_task_output_copy_n_other_jobs;
    uint64_t matmul_task_output_copy_n_other_us;
    uint64_t matmul_task_post_us;
    uint64_t matmul_task_post_stats_us;
    uint64_t matmul_task_free_us;
    uint64_t matmul_task_residual_us;
    uint64_t pl_op_desc_count;
    uint64_t pl_op_desc_tile_count;
    uint64_t pl_op_desc_program_us;
    uint64_t pl_op_desc_wait_us;
    uint64_t pl_op_desc_status_polls;
    uint64_t b_packed_arena_hits;
    uint64_t b_packed_arena_misses;
} tpu_runtime_stats_t;

// Logical layout of A bytes provided by the tiling layer. RAW_MK points to
// host memory containing padded row-major int8 bytes, not a device address.
typedef enum {
    TPU_A_LAYOUT_MK = 0,
    TPU_A_LAYOUT_RAW_MK = 1,
} tpu_a_layout_t;

// Logical/device layout of B bytes provided by the tiling layer. Raw NK/KN
// layouts are packed into the device ABI by the real runtime before launch.
typedef enum {
    TPU_B_LAYOUT_NK = 0,
    TPU_B_LAYOUT_KN = 1,
    TPU_B_LAYOUT_DEVICE_PACKED = 2,
} tpu_b_layout_t;

typedef enum {
    TPU_C_SOURCE_ZERO = 0,
    TPU_C_SOURCE_HOST = 1,
    TPU_C_SOURCE_DEVICE_ACCUM = 2,
} tpu_c_source_t;

typedef enum {
    TPU_D_DEST_HOST_READBACK = 0,
    TPU_D_DEST_DEVICE_ACCUM = 1,
} tpu_d_dest_t;

typedef struct {
    uint64_t op_id;
    uint32_t m0;
    uint32_t n0;
    uint32_t m;
    uint32_t n;
} tpu_accum_key_t;

typedef struct tpu_tile_execute_desc {
    const int8_t * a;
    const int8_t * b;
    const int32_t * c;
    int32_t * d;
    int32_t * d_final;
    uint32_t d_final_stride;
    uint32_t d_actual_m;
    uint32_t d_actual_n;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    tpu_a_layout_t a_layout;
    size_t a_bytes;
    tpu_b_layout_t b_layout;
    const void * b_cache_owner;
    const int8_t * b_cache_source;
    uint32_t b_cache_source_n;
    uint32_t b_cache_n0;
    uint32_t b_cache_k0;
    uint32_t b_cache_n_actual;
    uint32_t b_cache_k_actual;
    tpu_c_source_t c_source;
    tpu_d_dest_t d_dest;
    tpu_accum_key_t accum_key;
    const struct tpu_tile_execute_desc * next_desc;
} tpu_tile_execute_desc_t;

typedef struct {
    int accum_input_hit;
    int accum_output_valid;
    int readback_done;
    int readback_direct_final;
    uint64_t bytes_host_to_device_c;
    uint64_t bytes_device_to_host_d;
} tpu_tile_execute_result_t;

typedef int (*tpu_tile_execute_cached_fn)(
    const tpu_tile_execute_desc_t * desc,
    tpu_tile_execute_result_t * result);

typedef int (*tpu_tile_sequence_execute_cached_fn)(
    const tpu_tile_execute_desc_t * descs,
    uint32_t desc_count,
    tpu_tile_execute_result_t * result);

// Single hardware-granularity GEMM call.
// D[m,n] = A[m,k] * B[n,k]^T + C[m,n]
// Current board-validated configs keep M choices but only support 64/128 N/K families:
//   1x64x64, 16x64x64, 32x64x64, 1x128x128, 16x128x128, 32x128x128
// A PREV_D chain must use one hardware M/N/K config for all runs.
void tpu_gemm_int8_int32(
    const int8_t  * a,
    const int8_t  * b,
    const int32_t * c,
    int32_t       * d,
    uint32_t m,
    uint32_t n,
    uint32_t k);

void tpu_gemm_int8_int32_view(
    const int8_t   * a,
    const int8_t   * b,
    const int32_t  * c,
    tpu_int32_view_t * out_view,
    uint32_t m,
    uint32_t n,
    uint32_t k);

// Same GEMM semantics as above, but B is provided in transposed byte layout
// [K, N] row-major. This is used to skip the per-tile NK->KN repack on
// constant weights that have been pre-transposed by the upper layer.
void tpu_gemm_int8_int32_prepacked_b(
    const int8_t  * a,
    const int8_t  * b_kn,
    const int32_t * c,
    int32_t       * d,
    uint32_t m,
    uint32_t n,
    uint32_t k);

void tpu_gemm_int8_int32_prepacked_b_view(
    const int8_t   * a,
    const int8_t   * b_kn,
    const int32_t  * c,
    tpu_int32_view_t * out_view,
    uint32_t m,
    uint32_t n,
    uint32_t k);

// Tiled GEMM for arbitrary M/N/K. Internally tiles and calls
// tpu_gemm_int8_int32() for each hardware-valid tile.
void tpu_gemm_int8_int32_tiled(
    const int8_t  * a,
    const int8_t  * b,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats);

// Tiled GEMM variant that can use a full-matrix transposed B cache [K, N]
// row-major to avoid per-tile B repacking.
void tpu_gemm_int8_int32_tiled_prepacked_b(
    const int8_t  * a,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats);

void tpu_gemm_int8_int32_tiled_prepacked_b_with_cache(
    const int8_t  * a,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    void * packed_b_cache);

void tpu_gemm_f32_int8_int32_tiled_prepacked_b_with_cache(
    const float   * a,
    const float   * a_inv_scales,
    const int8_t  * b_kn,
    int32_t       * d,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    void * packed_b_cache);

int tpu_gemm_int8_int32_matrix_task_prepacked_b_with_cache(
    const int8_t  * a_mk,
    const int8_t  * b_kn,
    int32_t       * d_mn,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    void * b_arena_owner);

int tpu_gemm_int8_int32_matrix_task_prepacked_b_view_with_cache(
    const int8_t    * a_mk,
    const int8_t    * b_kn,
    tpu_int32_view_t * out_view,
    uint32_t M,
    uint32_t N,
    uint32_t K,
    tpu_tiling_stats_t * stats,
    void * b_arena_owner);

int tpu_matrix_task_prepack_b_with_cache(
    const int8_t  * b_kn,
    uint32_t N,
    uint32_t K,
    void * b_arena_owner);

// Future: vector accelerator path for M=1 or small M.
// void tpu_gemv_int8_int32(
//     const int8_t  * a,
//     const int8_t  * b,
//     const int32_t * c,
//     int32_t       * d,
//     uint32_t n,
//     uint32_t k);

int tpu_runtime_init(void);
void tpu_runtime_cleanup(void);
void tpu_runtime_get_stats(tpu_runtime_stats_t * stats);

#ifdef __cplusplus
}
#endif
