#pragma once

#include <stddef.h>
#include <stdint.h>

#include "axil_mmio.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    axil_mmio_t mmio;
} axidma_simple_t;

enum {
    AXIDMA_MM2S_DMACR   = 0x00u,
    AXIDMA_MM2S_DMASR   = 0x04u,
    AXIDMA_MM2S_SA_LSB  = 0x18u,
    AXIDMA_MM2S_SA_MSB  = 0x1Cu,
    AXIDMA_MM2S_LENGTH  = 0x28u,

    AXIDMA_S2MM_DMACR   = 0x30u,
    AXIDMA_S2MM_DMASR   = 0x34u,
    AXIDMA_S2MM_DA_LSB  = 0x48u,
    AXIDMA_S2MM_DA_MSB  = 0x4Cu,
    AXIDMA_S2MM_LENGTH  = 0x58u,
};

enum {
    AXIDMA_DMACR_RS       = (1u << 0),
    AXIDMA_DMACR_RESET    = (1u << 2),

    AXIDMA_DMASR_HALTED   = (1u << 0),
    AXIDMA_DMASR_IDLE     = (1u << 1),
    AXIDMA_DMASR_IOC_IRQ  = (1u << 12),
    AXIDMA_DMASR_DLY_IRQ  = (1u << 13),
    AXIDMA_DMASR_ERR_IRQ  = (1u << 14),

    AXIDMA_DMASR_ERR_MASK =
        (1u << 4) | (1u << 5) | (1u << 6) |
        (1u << 8) | (1u << 9) | (1u << 10) |
        AXIDMA_DMASR_ERR_IRQ,

    AXIDMA_DMASR_IRQ_MASK =
        AXIDMA_DMASR_IOC_IRQ | AXIDMA_DMASR_DLY_IRQ | AXIDMA_DMASR_ERR_IRQ,
};

int axidma_simple_open(axidma_simple_t *dma, uint64_t base_addr, size_t span, const char *devmem_path);
void axidma_simple_close(axidma_simple_t *dma);

int axidma_simple_reset(axidma_simple_t *dma, unsigned timeout_ms);
int axidma_simple_start_s2mm(axidma_simple_t *dma, uint64_t dst_phys, uint32_t length_bytes);
int axidma_simple_start_mm2s(axidma_simple_t *dma, uint64_t src_phys, uint32_t length_bytes);
int axidma_simple_wait_mm2s(axidma_simple_t *dma, unsigned timeout_ms, uint32_t *final_status);
int axidma_simple_wait_s2mm(axidma_simple_t *dma, unsigned timeout_ms, uint32_t *final_status);
int axidma_simple_transfer(axidma_simple_t *dma,
                           uint64_t src_phys,
                           uint64_t dst_phys,
                           uint32_t length_bytes,
                           unsigned timeout_ms);

#ifdef __cplusplus
}
#endif
