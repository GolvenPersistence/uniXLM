#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TPU_R5_SEQ_MAX_RUNS 8u
#define TPU_R5_SEQ_BATCH_MAX_COMMANDS 64u

#define TPU_R5_MATMUL_TASK_MAGIC   0x4d544d31u
#define TPU_R5_MATMUL_TASK_VERSION 1u
#define TPU_R5_MATMUL_TILE_M       32u
#define TPU_R5_MATMUL_TILE_N       128u
#define TPU_R5_MATMUL_TILE_K       128u
#define TPU_R5_MATMUL_B_TILE_BYTES (TPU_R5_MATMUL_TILE_N * TPU_R5_MATMUL_TILE_K)

#define TPU_R5_MATMUL_FLAG_A_INT8_RAW  (1u << 0)
#define TPU_R5_MATMUL_FLAG_ZERO_ACC    (1u << 8)
#define TPU_R5_MATMUL_FLAG_ACCUM       (1u << 9)
#define TPU_R5_MATMUL_FLAG_FINAL_STORE (1u << 10)

#define TPU_R5_MATMUL_STATUS_EMPTY     0u
#define TPU_R5_MATMUL_STATUS_SUBMITTED 1u
#define TPU_R5_MATMUL_STATUS_RUNNING   2u
#define TPU_R5_MATMUL_STATUS_DONE      3u
#define TPU_R5_MATMUL_STATUS_ERROR     4u

#define TPU_R5_MATMUL_ERR_NONE         0u
#define TPU_R5_MATMUL_ERR_BAD_ARG      3u
#define TPU_R5_MATMUL_ERR_TIMEOUT      2u
#define TPU_R5_MATMUL_ERR_HW_ERROR     4u

typedef struct {
    uint32_t config_word;
    uint32_t seq_ctrl;
    uint32_t run_count;
    uint32_t a_addrs[TPU_R5_SEQ_MAX_RUNS];
    uint32_t b_addrs[TPU_R5_SEQ_MAX_RUNS];
} tpu_r5_sequence_cmd_t;

typedef struct {
    uint32_t seq_status;
    uint32_t cache_status;
    uint32_t preload01;
    uint32_t preload23;
    uint32_t preload_status;
    uint32_t error;
} tpu_r5_sequence_result_t;

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t flags;
    uint32_t fence_id;
    uint32_t a_base;
    uint32_t b_packed_base;
    uint32_t c_base;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda_bytes;
    uint32_t ldc_bytes;
    uint32_t tile_m;
    uint32_t tile_n;
    uint32_t tile_k;
    uint32_t b_tile_bytes;
    uint32_t b_n_tiles;
    uint32_t b_k_tiles;
    uint32_t status;
    uint32_t error;
    uint32_t completed_tiles;
    uint32_t total_tiles;
    uint32_t done_fence;
    uint32_t r5_cycles;
    uint32_t fpga_wait_cycles;
    uint32_t last_hw_status;
    uint32_t scratch_m0;
    uint32_t scratch_n0;
    uint32_t scratch_k0;
    uint32_t scratch_nt;
    uint32_t scratch_kt;
    uint32_t reserved0;
} tpu_r5_matmul_task_t;

typedef struct {
    uint32_t status;
    uint32_t error;
    uint32_t completed_tiles;
    uint32_t total_tiles;
    uint32_t done_fence;
    uint32_t r5_cycles;
    uint32_t fpga_wait_cycles;
    uint32_t last_hw_status;
} tpu_r5_matmul_result_t;

int tpu_r5_ctrl_enabled(void);
int tpu_r5_ctrl_unavailable_errno(int err);
uint32_t tpu_r5_ctrl_status_reg_offset(uint32_t default_offset);
int tpu_r5_ctrl_wait_mask(uint32_t csr_base,
                          uint32_t reg_offset,
                          uint32_t set_mask,
                          uint32_t clear_mask,
                          unsigned timeout_ms,
                          uint32_t *last_value);
int tpu_r5_ctrl_wait_pair_mask(uint32_t csr_base,
                               uint32_t reg_offset,
                               uint32_t set_mask_0,
                               uint32_t clear_mask_0,
                               uint32_t set_mask_1,
                               uint32_t clear_mask_1,
                               unsigned timeout_ms,
                               uint32_t *last_value);
int tpu_r5_ctrl_run_sequence(uint32_t csr_base,
                              uint32_t config_word,
                              uint32_t seq_ctrl,
                              const uint32_t *a_addrs,
                              const uint32_t *b_addrs,
                              uint32_t run_count,
                              unsigned timeout_ms,
                              uint32_t *seq_status,
                              uint32_t *cache_status,
                              uint32_t *preload01,
                              uint32_t *preload23,
                              uint32_t *preload_status,
                              uint32_t *r5_cycles);
int tpu_r5_ctrl_run_sequence_batch(uint32_t csr_base,
                                   const tpu_r5_sequence_cmd_t *cmds,
                                   uint32_t command_count,
                                   unsigned timeout_ms,
                                   tpu_r5_sequence_result_t *results,
                                   uint32_t *completed_count,
                                   uint32_t *r5_cycles);
int tpu_r5_ctrl_run_matmul_task(uint32_t csr_base,
                                const tpu_r5_matmul_task_t *task,
                                unsigned timeout_ms,
                                tpu_r5_matmul_result_t *result);
void tpu_r5_ctrl_close(void);

#ifdef __cplusplus
}
#endif
