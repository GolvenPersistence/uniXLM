#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include "axil_mmio.h"
#include "xrt_cpp_bridge.h"

#ifdef __cplusplus
extern "C" {
#endif

#define TPU_SOC_FP32_CSR_BASE_DEFAULT  0x00B0000000ull
#define TPU_SOC_FP32_CDMA_BASE_DEFAULT 0x00B0010000ull
#define TPU_SOC_FP32_BRAM_BASE_DEFAULT 0x00A0000000ull

#define TPU_SOC_FP32_CSR_SPAN  0x1000u
#define TPU_SOC_FP32_CDMA_SPAN 0x10000u
#define TPU_SOC_FP32_BRAM_SPAN 0x2000u

#define TPU_SOC_FP32_CDMA_TPU_BASE 0x10000000u

#define TPU_SOC_FP32_ZU_DDR_BASE_DEFAULT    0x40000000u
#define TPU_SOC_FP32_ZU_DDR_RANGE_DEFAULT   0x40000000u
#define TPU_SOC_FP32_KV260_DDR_BASE_DEFAULT 0x20000000u
#define TPU_SOC_FP32_KV260_DDR_RANGE_DEFAULT 0x20000000u

#define TPU_SOC_FP32_CDMA_MATRIX_A (TPU_SOC_FP32_CDMA_TPU_BASE + 0x0000u)
#define TPU_SOC_FP32_CDMA_MATRIX_B (TPU_SOC_FP32_CDMA_TPU_BASE + 0x4000u)
#define TPU_SOC_FP32_CDMA_MATRIX_C (TPU_SOC_FP32_CDMA_TPU_BASE + 0x8000u)

#define TPU_SOC_FP32_CDMA_CR     0x00u
#define TPU_SOC_FP32_CDMA_SR     0x04u
#define TPU_SOC_FP32_CDMA_CDESC  0x08u
#define TPU_SOC_FP32_CDMA_CDESC_MSB 0x0cu
#define TPU_SOC_FP32_CDMA_TDESC  0x10u
#define TPU_SOC_FP32_CDMA_TDESC_MSB 0x14u
#define TPU_SOC_FP32_CDMA_SA     0x18u
#define TPU_SOC_FP32_CDMA_SA_MSB 0x1cu
#define TPU_SOC_FP32_CDMA_DA     0x20u
#define TPU_SOC_FP32_CDMA_DA_MSB 0x24u
#define TPU_SOC_FP32_CDMA_BTT    0x28u

#define TPU_SOC_FP32_CDMA_CR_RESET (1u << 2)
#define TPU_SOC_FP32_CDMA_CR_SGMODE (1u << 3)
#define TPU_SOC_FP32_CDMA_SR_IDLE  (1u << 1)
#define TPU_SOC_FP32_CDMA_SR_SGINCLD (1u << 3)
#define TPU_SOC_FP32_CDMA_SR_IOC   (1u << 1)
#define TPU_SOC_FP32_CDMA_SR_ERR_MASK 0x70u
#define TPU_SOC_FP32_CDMA_SR_SG_ERR_MASK 0x770u

#define TPU_SOC_FP32_CDMA_BD_ALIGN 0x40u
#define TPU_SOC_FP32_CDMA_BD_NDESC_OFFSET 0x00u
#define TPU_SOC_FP32_CDMA_BD_NDESC_MSB_OFFSET 0x04u
#define TPU_SOC_FP32_CDMA_BD_BUFSRC_OFFSET 0x08u
#define TPU_SOC_FP32_CDMA_BD_BUFSRC_MSB_OFFSET 0x0cu
#define TPU_SOC_FP32_CDMA_BD_BUFDST_OFFSET 0x10u
#define TPU_SOC_FP32_CDMA_BD_BUFDST_MSB_OFFSET 0x14u
#define TPU_SOC_FP32_CDMA_BD_CTRL_LEN_OFFSET 0x18u
#define TPU_SOC_FP32_CDMA_BD_STS_OFFSET 0x1cu
#define TPU_SOC_FP32_CDMA_BD_PHYS_ADDR_OFFSET 0x20u
#define TPU_SOC_FP32_CDMA_BD_PHYS_ADDR_MSB_OFFSET 0x24u
#define TPU_SOC_FP32_CDMA_BD_ISLITE_OFFSET 0x28u
#define TPU_SOC_FP32_CDMA_BD_HASDRE_OFFSET 0x2cu
#define TPU_SOC_FP32_CDMA_BD_WORDLEN_OFFSET 0x30u
#define TPU_SOC_FP32_CDMA_BD_MAX_LEN_OFFSET 0x34u
#define TPU_SOC_FP32_CDMA_BD_ADDRLEN_OFFSET 0x38u
#define TPU_SOC_FP32_CDMA_BD_WORDS 16u
#define TPU_SOC_FP32_CDMA_BD_BYTES 64u
#define TPU_SOC_FP32_CDMA_BD_CTRL_LEN_MASK 0x007fffffu
#define TPU_SOC_FP32_CDMA_BD_STS_COMPLETE_MASK 0x80000000u
#define TPU_SOC_FP32_CDMA_BD_STS_ERR_MASK 0x70000000u

typedef struct {
    xrt_cpp_bo_handle_t handle;
    void *mapping;
    uint64_t device_addr;
    size_t size;
} tpu_soc_fp32_buffer_t;

typedef struct {
    const char *label;
    uint64_t src_addr;
    uint64_t dst_addr;
    size_t size_bytes;
} tpu_soc_fp32_cdma_segment_t;

typedef struct {
    uint32_t ctrl;
    uint32_t status;
    uint32_t config;
    uint32_t dbg_done;
    uint32_t dbg_last_addr;
    uint32_t dbg_flow;
    uint32_t dbg_counts;
    uint32_t dbg_counts2;
    uint32_t dbg_counts3;
    uint32_t dbg_counts4;
    uint32_t dbg_counts5;
    uint32_t dbg_counts6;
    uint32_t dbg_acc_first;
    uint32_t dbg_acc_last;
    uint32_t dbg_cfg1;
    uint32_t dbg_cfg2;
    uint32_t dbg_cfg3;
    uint32_t dbg_acc_ctrl0;
    uint32_t dbg_acc_ctrl1;
    uint32_t dbg_acc_ctrl2;
    uint32_t dbg_axi_aw;
    uint32_t dbg_axi_w;
    uint32_t dbg_axi_wlast;
    uint32_t dbg_axi_ram_wr;
    uint32_t dbg_axi_last;
    uint32_t dbg_axi_aw_beats;
    uint32_t dbg_axi_short_burst;
    uint32_t dbg_axi_short_miss;
    uint32_t dbg_axi_totals;
    uint32_t dbg_axi_none;
    uint32_t dbg_acc_pipe0;
    uint32_t dbg_acc_pipe1;
    uint32_t dbg_acc_pipe2;
    uint32_t store_target_addr;
    uint32_t dbg_ingress;
} tpu_soc_fp32_regs_t;

#define TPU_SOC_FP32_CDMA_TRACE_MAX_TRANSFERS 8u
#define TPU_SOC_FP32_CDMA_LABEL_MAX           16u

typedef struct {
    int attempted;
    int rc;
    int err;
    uint32_t cr_before;
    uint32_t sr_before;
    uint32_t cr_after;
    uint32_t sr_after;
} tpu_soc_fp32_cdma_reset_log_t;

void tpu_soc_fp32_get_input_ddr_aperture(uint32_t *base, uint32_t *high);
void tpu_soc_fp32_get_output_ddr_aperture(uint32_t *base, uint32_t *high);
int tpu_soc_fp32_validate_input_ddr_addr(uint64_t addr, size_t size_bytes);
int tpu_soc_fp32_validate_output_ddr_addr(uint64_t addr, size_t size_bytes);

typedef struct {
    char label[TPU_SOC_FP32_CDMA_LABEL_MAX];
    uint64_t src_addr;
    uint64_t dst_addr;
    uint64_t start_ns;
    uint64_t done_ns;
    uint64_t duration_ns;
    size_t size_bytes;
    unsigned timeout_ms;
    uint32_t cr_before;
    uint32_t sr_before;
    uint32_t cr_after;
    uint32_t sr_after;
    uint32_t poll_count;
    int rc;
    int err;
    int timed_out;
    int has_failure_regs;
    tpu_soc_fp32_regs_t failure_regs;
} tpu_soc_fp32_cdma_transfer_log_t;

typedef struct {
    tpu_soc_fp32_cdma_reset_log_t reset;
    tpu_soc_fp32_cdma_transfer_log_t transfers[TPU_SOC_FP32_CDMA_TRACE_MAX_TRANSFERS];
    unsigned transfer_count;
    int first_failure_index;
} tpu_soc_fp32_cdma_trace_t;

typedef struct {
    axil_mmio_t csr_mmio;
    axil_mmio_t cdma_mmio;
    axil_mmio_t bram_mmio;
    xrt_cpp_device_handle_t xrt_device;
    unsigned device_index;
    unsigned mem_group;
    tpu_soc_fp32_cdma_trace_t last_cdma_trace;
    uint32_t input_ddr_base;
    uint32_t input_ddr_high;
    int input_ddr_high_window_valid;
    uint32_t input_ddr_high_window_base;
    uint32_t input_ddr_high_window_high;
    uint32_t output_ddr_base;
    uint32_t output_ddr_high;
    int output_ddr_high_window_valid;
    uint32_t output_ddr_high_window_base;
    uint32_t output_ddr_high_window_high;
    int v2_exec_cfg_valid;
    int v2_trace_cdma_path;
    int v2_tail_fixup;
    int v2_use_sg;
    int v2_hw_ingress;
    int v2_gemv_m1;
    int v2_transfer_only;
    uint32_t v2_hw_load_mask;
    unsigned v2_post_start_delay_us;
    unsigned v2_between_cdma_delay_us_ab;
    unsigned v2_between_cdma_delay_us_bc;
    unsigned v2_post_cdma_delay_us;
    int v2_shape_cache_valid;
    uint32_t v2_cached_m;
    uint32_t v2_cached_n;
    uint32_t v2_cached_k;
    size_t v2_cached_a_bytes;
    size_t v2_cached_b_bytes;
    size_t v2_cached_c_bytes;
    size_t v2_cached_d_bytes;
    int v2_a_addr_cache_valid;
    int v2_b_addr_cache_valid;
    int v2_c_addr_cache_valid;
    uint64_t v2_cached_a_addr;
    uint64_t v2_cached_b_addr;
    uint64_t v2_cached_c_addr;
    size_t v2_cached_a_size;
    size_t v2_cached_b_size;
    size_t v2_cached_c_size;
    uint32_t v2_c_mode;
    uint32_t v2_d_mode;
    int v2_prev_d_shape_valid;
    uint32_t v2_prev_d_m;
    uint32_t v2_prev_d_n;
} tpu_soc_fp32_t;

int tpu_soc_fp32_open(tpu_soc_fp32_t *ctx,
                      uint64_t csr_base,
                      uint64_t cdma_base,
                      uint64_t bram_base,
                      unsigned device_index,
                      unsigned mem_group,
                      const char *devmem_path);
void tpu_soc_fp32_close(tpu_soc_fp32_t *ctx);

int tpu_soc_fp32_alloc_buffer(tpu_soc_fp32_t *ctx, tpu_soc_fp32_buffer_t *buf, size_t size_bytes);
int tpu_soc_fp32_alloc_buffer_ex(tpu_soc_fp32_t *ctx, tpu_soc_fp32_buffer_t *buf, size_t size_bytes, int cacheable);
void tpu_soc_fp32_free_buffer(tpu_soc_fp32_buffer_t *buf);
int tpu_soc_fp32_buffer_sync_to_device(tpu_soc_fp32_buffer_t *buf, size_t size_bytes);
int tpu_soc_fp32_buffer_sync_from_device(tpu_soc_fp32_buffer_t *buf, size_t size_bytes);

int tpu_soc_fp32_read_regs(tpu_soc_fp32_t *ctx, tpu_soc_fp32_regs_t *regs);
void tpu_soc_fp32_dump_regs(FILE *stream, const tpu_soc_fp32_regs_t *regs);
void tpu_soc_fp32_cdma_trace_clear(tpu_soc_fp32_cdma_trace_t *trace);
const char *tpu_soc_fp32_cdma_first_failure_label(const tpu_soc_fp32_cdma_trace_t *trace);
void tpu_soc_fp32_dump_cdma_trace(FILE *stream, const tpu_soc_fp32_cdma_trace_t *trace);

int tpu_soc_fp32_preflight(tpu_soc_fp32_t *ctx, tpu_soc_fp32_regs_t *regs_out);
int tpu_soc_fp32_soft_reset(tpu_soc_fp32_t *ctx, unsigned timeout_ms);
int tpu_soc_fp32_soft_reset_with_hold_cycles(tpu_soc_fp32_t *ctx,
                                             unsigned timeout_ms,
                                             uint32_t hold_cycles);
int tpu_soc_fp32_soft_reset_with_hold_cycles_probe(tpu_soc_fp32_t *ctx,
                                                   unsigned timeout_ms,
                                                   uint32_t hold_cycles,
                                                   int *saw_active_out,
                                                   uint32_t *last_status_out);
int tpu_soc_fp32_cdma_reset(tpu_soc_fp32_t *ctx);
int tpu_soc_fp32_cdma_reset_with_trace(tpu_soc_fp32_t *ctx, tpu_soc_fp32_cdma_trace_t *trace);
int tpu_soc_fp32_cdma_transfer_with_trace(tpu_soc_fp32_t *ctx,
                                          tpu_soc_fp32_cdma_trace_t *trace,
                                          const char *label,
                                          uint64_t src_addr,
                                          uint64_t dst_addr,
                                          size_t size_bytes,
                                          unsigned timeout_ms);
int tpu_soc_fp32_cdma_transfer_sg_chain_with_trace(tpu_soc_fp32_t *ctx,
                                                   tpu_soc_fp32_cdma_trace_t *trace,
                                                   const char *label,
                                                   const tpu_soc_fp32_cdma_segment_t *segments,
                                                   unsigned segment_count,
                                                   unsigned timeout_ms);
int tpu_soc_fp32_cdma_tail_fixup_with_trace(tpu_soc_fp32_t *ctx,
                                            tpu_soc_fp32_cdma_trace_t *trace,
                                            const char *label,
                                            uint64_t src_addr,
                                            uint64_t dst_addr,
                                            size_t size_bytes,
                                            unsigned timeout_ms);
int tpu_soc_fp32_execute(tpu_soc_fp32_t *ctx,
                         const tpu_soc_fp32_buffer_t *a_buf,
                         const tpu_soc_fp32_buffer_t *b_buf,
                         const tpu_soc_fp32_buffer_t *c_buf,
                         uint32_t m,
                         uint32_t n,
                         uint32_t k,
                         int sparse,
                         unsigned timeout_ms);
int tpu_soc_fp16_execute(tpu_soc_fp32_t *ctx,
                         const tpu_soc_fp32_buffer_t *a_buf,
                         const tpu_soc_fp32_buffer_t *b_buf,
                         const tpu_soc_fp32_buffer_t *c_buf,
                         uint32_t m,
                         uint32_t n,
                         uint32_t k,
                         int sparse,
                         unsigned timeout_ms);
int tpu_soc_fp16_mix_execute(tpu_soc_fp32_t *ctx,
                             const tpu_soc_fp32_buffer_t *a_buf,
                             const tpu_soc_fp32_buffer_t *b_buf,
                             const tpu_soc_fp32_buffer_t *c_buf,
                             uint32_t m,
                             uint32_t n,
                             uint32_t k,
                             int sparse,
                             unsigned timeout_ms);
int tpu_soc_fp32_read_result(tpu_soc_fp32_t *ctx, float *dst, size_t element_count);
int tpu_soc_fp32_read_result_words64(tpu_soc_fp32_t *ctx, uint64_t *dst, size_t word_count);

#ifdef __cplusplus
}
#endif
