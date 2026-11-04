#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int fd;
    uint64_t base_addr;
    size_t span;
    size_t map_span;
    size_t page_offset;
    void *mapping;
    volatile uint32_t *regs;
} axil_mmio_t;

int axil_mmio_open(axil_mmio_t *ctx, uint64_t base_addr, size_t span, const char *devmem_path);
void axil_mmio_close(axil_mmio_t *ctx);
int axil_mmio_write32(axil_mmio_t *ctx, uint32_t offset, uint32_t value);
int axil_mmio_read32(axil_mmio_t *ctx, uint32_t offset, uint32_t *value);
int axil_mmio_write64(axil_mmio_t *ctx, uint32_t offset, uint64_t value);
int axil_mmio_read64(axil_mmio_t *ctx, uint32_t offset, uint64_t *value);

#ifdef __cplusplus
}
#endif
