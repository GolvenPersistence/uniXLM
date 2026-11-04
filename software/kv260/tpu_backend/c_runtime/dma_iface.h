#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t phys_addr;
    void *virt_addr;
    size_t size;
} dma_buffer_t;

typedef struct dma_iface dma_iface_t;

struct dma_iface {
    void *user;
    int (*alloc_buffer)(dma_iface_t *iface, dma_buffer_t *buf, size_t size, size_t align);
    void (*free_buffer)(dma_iface_t *iface, dma_buffer_t *buf);
    int (*to_device)(dma_iface_t *iface, uint64_t src_phys, size_t size_bytes);
    int (*from_device)(dma_iface_t *iface, uint64_t dst_phys, size_t size_bytes);
    int (*wait_tx)(dma_iface_t *iface, unsigned timeout_ms);
    int (*wait_rx)(dma_iface_t *iface, unsigned timeout_ms);
};

int dma_iface_not_implemented(void);

#ifdef __cplusplus
}
#endif
