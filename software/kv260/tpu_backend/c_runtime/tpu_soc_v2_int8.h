#pragma once

#include <stddef.h>
#include <stdio.h>
#include <stdint.h>

#include "tpu_soc_fp32.h"
#include "tpu_vanilla_regs.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TPU_SOC_V2_SOFT_RESET_HOLD_CYCLES_FAST   200u
#define TPU_SOC_V2_SOFT_RESET_HOLD_CYCLES_LEGACY 50000u

typedef struct {
    uint32_t hold_cycles;
    uint32_t fallback_hold_cycles;
    const char *label;
} tpu_soc_v2_soft_reset_policy_t;

typedef struct {
    uint32_t c_mode;
    uint32_t d_mode;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t precision_mode;
    const void *a_addr;
    const void *b_addr;
    const void *c_addr;
    void *d_addr;
    int clear_prev_d;
    uint64_t csr_base;
    uint64_t cdma_base;
    uint64_t bram_base;
    unsigned device_index;
    unsigned mem_group;
    unsigned timeout_ms;
    int sparse;
    int run_preflight;
    const char *devmem_path;
} tpu_soc_v2_run_args_t;

typedef struct {
    size_t d_bytes;
    tpu_soc_fp32_cdma_trace_t cdma_trace;
    tpu_soc_fp32_regs_t initial_regs;
    tpu_soc_fp32_regs_t final_regs;
    int have_initial_regs;
    int have_final_regs;
} tpu_soc_v2_run_report_t;

typedef struct {
    const char *selected_path;
    uint64_t execute_total_ns;
    uint64_t sync_ns;
    uint64_t soft_reset_ns;
    uint64_t cdma_reset_ns;
    uint64_t ready_wait_ns;
    uint64_t path_setup_ns;
    uint64_t start_ns;
    uint64_t post_start_delay_ns;
    uint64_t transfer_ns;
    uint64_t tail_fixup_ns;
    uint64_t wait_done_ns;
} tpu_soc_v2_execute_report_t;

typedef struct {
    uint64_t sync_ns;
    uint64_t write_ready_wait_ns;
    uint64_t path_setup_ns;
    uint64_t transfer_ns;
    int saw_compute_busy;
} tpu_soc_v2_ab_preload_report_t;

typedef struct {
    uint64_t sync_ns;
    uint64_t ready_wait_ns;
    uint64_t program_ns;
    uint64_t wait_done_ns;
    uint64_t execute_total_ns;
    uint32_t seq_status;
    uint32_t preload_hw_cycles[4];
    uint32_t preload_hw_count;
    uint32_t preload_hw_status;
} tpu_soc_v2_sequence_report_t;

typedef struct {
    uint64_t sync_ns;
    uint64_t copy_ns;
    uint64_t unpack_ns;
    uint64_t mmio_read_ns;
    uint64_t total_ns;
    uint64_t bytes;
    uint32_t elem_count;
    uint32_t word_count;
    int used_output_buffer;
    int used_memcpy;
    int used_unpack;
    int used_mmio;
} tpu_soc_v2_read_result_report_t;

enum {
    TPU_SOC_V2_C_MODE_HOST = TPU_VANILLA_C_MODE_HOST,
    TPU_SOC_V2_C_MODE_ZERO = TPU_VANILLA_C_MODE_ZERO,
    TPU_SOC_V2_C_MODE_PREV_D = TPU_VANILLA_C_MODE_PREV_D
};

enum {
    TPU_SOC_V2_D_MODE_READBACK = TPU_VANILLA_D_MODE_READBACK,
    TPU_SOC_V2_D_MODE_RETAIN = TPU_VANILLA_D_MODE_RETAIN
};

void tpu_soc_v2_run_args_init(tpu_soc_v2_run_args_t *args);
void tpu_soc_v2_run_report_init(tpu_soc_v2_run_report_t *report);

uint32_t tpu_soc_v2_int8_word_count(uint32_t rows, uint32_t cols);
uint32_t tpu_soc_v2_int32_word_count(uint32_t rows, uint32_t cols);

int tpu_soc_v2_validate_sizes(uint32_t m,
                              uint32_t n,
                              uint32_t k,
                              size_t *a_bytes,
                              size_t *b_bytes,
                              size_t *c_bytes,
                              size_t *d_bytes);

void tpu_soc_v2_pack_a_int8_row_major(const int8_t *src_mk,
                                      uint32_t m,
                                      uint32_t k,
                                      uint64_t *dst_words);
void tpu_soc_v2_pack_b_int8_nk_to_kn(const int8_t *src_nk,
                                     uint32_t n,
                                     uint32_t k,
                                     uint64_t *dst_words);
void tpu_soc_v2_pack_b_int8_kn_raw(const int8_t *src_kn,
                                   uint32_t n,
                                   uint32_t k,
                                   uint64_t *dst_words);
void tpu_soc_v2_pack_c_int32_row_major(const int32_t *src_mn,
                                       uint32_t m,
                                       uint32_t n,
                                       uint64_t *dst_words);

void tpu_soc_v2_reference_gemm_int8_int32(const int8_t *a_mk,
                                          const int8_t *b_nk,
                                          const int32_t *c_mn,
                                          int32_t *d_mn,
                                          uint32_t m,
                                          uint32_t n,
                                          uint32_t k);

int tpu_soc_v2_clear_output_region(tpu_soc_fp32_t *ctx, uint32_t m, uint32_t n);
int tpu_soc_v2_read_result_int32(tpu_soc_fp32_t *ctx, int32_t *dst, uint32_t elem_count);
int tpu_soc_v2_read_result_int32_with_report(tpu_soc_fp32_t *ctx,
                                             int32_t *dst,
                                             uint32_t elem_count,
                                             tpu_soc_v2_read_result_report_t *report);
int tpu_soc_v2_read_result_int32_from_buffer(tpu_soc_fp32_buffer_t *buf,
                                             int32_t *dst,
                                             uint32_t elem_count);
int tpu_soc_v2_read_result_int32_from_buffer_with_report(tpu_soc_fp32_buffer_t *buf,
                                                         int32_t *dst,
                                                         uint32_t elem_count,
                                                         tpu_soc_v2_read_result_report_t *report);
int tpu_soc_v2_read_result_int32_from_buffer_with_rect_report(tpu_soc_fp32_buffer_t *buf,
                                                              int32_t *dst,
                                                              uint32_t packed_rows,
                                                              uint32_t packed_cols,
                                                              uint32_t copy_rows,
                                                              uint32_t copy_cols,
                                                              uint32_t dst_stride,
                                                              tpu_soc_v2_read_result_report_t *report);
int tpu_soc_v2_read_result_int32_from_buffer_with_rect(tpu_soc_fp32_buffer_t *buf,
                                                       int32_t *dst,
                                                       uint32_t packed_rows,
                                                       uint32_t packed_cols,
                                                       uint32_t copy_rows,
                                                       uint32_t copy_cols,
                                                       uint32_t dst_stride);

int tpu_soc_v2_read_result_int32_from_buffer_with_layout_report(tpu_soc_fp32_buffer_t *buf,
                                                                int32_t *dst,
                                                                uint32_t elem_count,
                                                                uint32_t rows,
                                                                uint32_t cols,
                                                                uint32_t dst_stride,
                                                                tpu_soc_v2_read_result_report_t *report);
int tpu_soc_v2_read_result_int32_from_buffer_with_layout(tpu_soc_fp32_buffer_t *buf,
                                                         int32_t *dst,
                                                         uint32_t elem_count,
                                                         uint32_t rows,
                                                         uint32_t cols,
                                                         uint32_t dst_stride);

int tpu_soc_v2_dump_output_buffer_bytes(FILE *stream,
                                        tpu_soc_fp32_buffer_t *buf,
                                        uint32_t elem_count,
                                        uint32_t max_bytes);
int tpu_soc_v2_set_output_target_addr(tpu_soc_fp32_t *ctx, uint32_t addr);
int tpu_soc_v2_get_output_target_addr(tpu_soc_fp32_t *ctx, uint32_t *addr);
int tpu_soc_v2_set_reuse_mode(tpu_soc_fp32_t *ctx,
                              uint32_t c_mode,
                              uint32_t d_mode,
                              int clear_prev_d);
int tpu_soc_v2_get_reuse_status(tpu_soc_fp32_t *ctx,
                                uint32_t *status_out);
int tpu_soc_v2_clear_prev_d_slot(tpu_soc_fp32_t *ctx);
int tpu_soc_v2_try_enable_output_buffer(tpu_soc_fp32_t *ctx,
                                        const tpu_soc_fp32_buffer_t *d_buf,
                                        int *enabled_out);
int tpu_soc_v2_resolve_soft_reset_policy(tpu_soc_v2_soft_reset_policy_t *policy);
int tpu_soc_v2_soft_reset(tpu_soc_fp32_t *ctx, unsigned timeout_ms);

int tpu_soc_v2_execute_int8_int32(tpu_soc_fp32_t *ctx,
                                  const tpu_soc_fp32_buffer_t *a_buf,
                                  const tpu_soc_fp32_buffer_t *b_buf,
                                  const tpu_soc_fp32_buffer_t *c_buf,
                                  uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  int sparse,
                                  unsigned timeout_ms);
int tpu_soc_v2_execute_int8_int32_with_report(tpu_soc_fp32_t *ctx,
                                              const tpu_soc_fp32_buffer_t *a_buf,
                                              const tpu_soc_fp32_buffer_t *b_buf,
                                              const tpu_soc_fp32_buffer_t *c_buf,
                                              uint32_t m,
                                              uint32_t n,
                                              uint32_t k,
                                              int sparse,
                                              unsigned timeout_ms,
                                              tpu_soc_v2_execute_report_t *report);

int tpu_soc_v2_start_int8_int32_hw_ingress_no_wait(tpu_soc_fp32_t *ctx,
                                                   const tpu_soc_fp32_buffer_t *a_buf,
                                                   const tpu_soc_fp32_buffer_t *b_buf,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   int sparse,
                                                   int use_preloaded_ab,
                                                   unsigned timeout_ms,
                                                   tpu_soc_v2_execute_report_t *report);

int tpu_soc_v2_preload_ab_int8_int32_with_report(tpu_soc_fp32_t *ctx,
                                                 const tpu_soc_fp32_buffer_t *a_buf,
                                                 const tpu_soc_fp32_buffer_t *b_buf,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 unsigned timeout_ms,
                                                 tpu_soc_v2_ab_preload_report_t *report);
int tpu_soc_v2_preload_ab_int8_int32_presynced_with_report(tpu_soc_fp32_t *ctx,
                                                           const tpu_soc_fp32_buffer_t *a_buf,
                                                           const tpu_soc_fp32_buffer_t *b_buf,
                                                           uint32_t m,
                                                           uint32_t n,
                                                           uint32_t k,
                                                           unsigned timeout_ms,
                                                           tpu_soc_v2_ab_preload_report_t *report);
int tpu_soc_v2_preload_ab_int8_int32_start_with_report(tpu_soc_fp32_t *ctx,
                                                       const tpu_soc_fp32_buffer_t *a_buf,
                                                       const tpu_soc_fp32_buffer_t *b_buf,
                                                       uint32_t m,
                                                       uint32_t n,
                                                       uint32_t k,
                                                       unsigned timeout_ms,
                                                       tpu_soc_v2_ab_preload_report_t *report);
int tpu_soc_v2_preload_ab_int8_int32_presynced_start_with_report(tpu_soc_fp32_t *ctx,
                                                                 const tpu_soc_fp32_buffer_t *a_buf,
                                                                 const tpu_soc_fp32_buffer_t *b_buf,
                                                                 uint32_t m,
                                                                 uint32_t n,
                                                                 uint32_t k,
                                                                 unsigned timeout_ms,
                                                                 tpu_soc_v2_ab_preload_report_t *report);
int tpu_soc_v2_program_next_ab_preload_int8_int32_with_report(tpu_soc_fp32_t *ctx,
                                                              const tpu_soc_fp32_buffer_t *a_buf,
                                                              const tpu_soc_fp32_buffer_t *b_buf,
                                                              uint32_t m,
                                                              uint32_t n,
                                                              uint32_t k,
                                                              tpu_soc_v2_ab_preload_report_t *report);
int tpu_soc_v2_program_next_ab_preload_int8_int32_presynced_with_report(tpu_soc_fp32_t *ctx,
                                                                        const tpu_soc_fp32_buffer_t *a_buf,
                                                                        const tpu_soc_fp32_buffer_t *b_buf,
                                                                        uint32_t m,
                                                                        uint32_t n,
                                                                        uint32_t k,
                                                                        tpu_soc_v2_ab_preload_report_t *report);

int tpu_soc_v2_run_prev_d_sequence_prepacked_int8_int32_with_report(tpu_soc_fp32_t *ctx,
                                                                    const tpu_soc_fp32_buffer_t *a_bufs,
                                                                    const tpu_soc_fp32_buffer_t *b_bufs,
                                                                    uint32_t run_count,
                                                                    uint32_t m,
                                                                    uint32_t n,
                                                                    uint32_t k,
                                                                    unsigned timeout_ms,
                                                                    tpu_soc_v2_sequence_report_t *report);
int tpu_soc_v2_run_prev_d_sequence_prepacked_int8_int32_ex_with_report(tpu_soc_fp32_t *ctx,
                                                                       const tpu_soc_fp32_buffer_t *a_bufs,
                                                                       const tpu_soc_fp32_buffer_t *b_bufs,
                                                                       uint32_t run_count,
                                                                       uint32_t m,
                                                                       uint32_t n,
                                                                       uint32_t k,
                                                                       int start_prev_d,
                                                                       int final_retain,
                                                                       unsigned timeout_ms,
                                                                       tpu_soc_v2_sequence_report_t *report);

int tpu_soc_v2_wait_int8_int32_active(tpu_soc_fp32_t *ctx,
                                      uint32_t c_mode,
                                      uint32_t d_mode,
                                      unsigned timeout_ms,
                                      tpu_soc_v2_execute_report_t *report);

/*
 * High-level one-shot entrypoint for application code.
 *
 * Inputs:
 * - a_addr: host virtual address of A, int8 row-major MxK
 * - b_addr: host virtual address of B, int8 row-major NxK
 * - c_addr: host virtual address of C, int32 row-major MxN
 * - d_addr: caller-provided output buffer, int32 row-major MxN
 * - precision_mode: currently only TPU_VANILLA_PM_INT8_INT32 is supported
 *
 * The function opens the runtime, optionally runs preflight, packs A/B/C,
 * allocates XRT buffers, executes the TPU flow, reads D back, and stores the
 * result into d_addr.
 */
int tpu_soc_v2_run(const tpu_soc_v2_run_args_t *args, tpu_soc_v2_run_report_t *report);

#ifdef __cplusplus
}
#endif
