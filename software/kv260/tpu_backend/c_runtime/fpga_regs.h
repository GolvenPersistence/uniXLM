#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GEMM_REG_CTRL        0x00u
#define GEMM_REG_STATUS      0x04u
#define GEMM_REG_DESC_ADDR_L 0x08u
#define GEMM_REG_DESC_ADDR_H 0x0cu
#define GEMM_REG_CYCLES      0x10u
#define GEMM_REG_DONE_TASK   0x14u

#define GEMM_CTRL_START      (1u << 0)
#define GEMM_CTRL_SOFT_RESET (1u << 1)

#define GEMM_STATUS_BUSY       (1u << 0)
#define GEMM_STATUS_DONE       (1u << 1)
#define GEMM_STATUS_FETCH_BUSY (1u << 2)

typedef struct {
    uint8_t task_id;
    uint8_t b_int4_mode;
    uint16_t m;
    uint16_t n;
    uint16_t k;
    uint32_t a_addr;
    uint32_t b_addr;
    uint32_t c_addr;
    uint16_t a_stride;
    uint16_t b_stride;
    uint16_t c_stride;
} gemm_desc_v1_t;

void gemm_pack_desc_words(const gemm_desc_v1_t *desc, uint32_t out_words[7]);

#ifdef __cplusplus
}
#endif
