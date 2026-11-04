#include "axil_mmio.h"

#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

int axil_mmio_open(axil_mmio_t *ctx, uint64_t base_addr, size_t span, const char *devmem_path) {
    long page_size;
    uint64_t page_mask;
    uint64_t page_base;
    const char *path = devmem_path ? devmem_path : "/dev/mem";

    if (ctx == NULL || span == 0) {
        errno = EINVAL;
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->fd = -1;
    ctx->base_addr = base_addr;
    ctx->span = span;

    page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        errno = EINVAL;
        return -1;
    }

    page_mask = (uint64_t)page_size - 1u;
    page_base = base_addr & ~page_mask;
    ctx->page_offset = (size_t)(base_addr - page_base);
    ctx->map_span = ctx->page_offset + span;

    ctx->fd = open(path, O_RDWR | O_SYNC);
    if (ctx->fd < 0) {
        return -1;
    }

    ctx->mapping = mmap(NULL, ctx->map_span, PROT_READ | PROT_WRITE, MAP_SHARED, ctx->fd, (off_t)page_base);
    if (ctx->mapping == MAP_FAILED) {
        ctx->mapping = NULL;
        close(ctx->fd);
        ctx->fd = -1;
        return -1;
    }

    ctx->regs = (volatile uint32_t *)((uint8_t *)ctx->mapping + ctx->page_offset);
    return 0;
}

void axil_mmio_close(axil_mmio_t *ctx) {
    if (ctx == NULL) {
        return;
    }
    if (ctx->mapping != NULL) {
        munmap(ctx->mapping, ctx->map_span);
    }
    if (ctx->fd >= 0) {
        close(ctx->fd);
    }
    memset(ctx, 0, sizeof(*ctx));
    ctx->fd = -1;
}

int axil_mmio_write32(axil_mmio_t *ctx, uint32_t offset, uint32_t value) {
    if (ctx == NULL || ctx->regs == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (((size_t)offset + sizeof(uint32_t)) > ctx->span) {
        errno = ERANGE;
        return -1;
    }
    ctx->regs[offset / 4u] = value;
    return 0;
}

int axil_mmio_read32(axil_mmio_t *ctx, uint32_t offset, uint32_t *value) {
    if (ctx == NULL || ctx->regs == NULL || value == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (((size_t)offset + sizeof(uint32_t)) > ctx->span) {
        errno = ERANGE;
        return -1;
    }
    *value = ctx->regs[offset / 4u];
    return 0;
}

int axil_mmio_write64(axil_mmio_t *ctx, uint32_t offset, uint64_t value) {
    volatile uint8_t *base;

    if (ctx == NULL || ctx->regs == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (((size_t)offset + sizeof(uint64_t)) > ctx->span || (offset & 0x7u) != 0u) {
        errno = ERANGE;
        return -1;
    }

    base = (volatile uint8_t *)ctx->regs;
#if defined(__aarch64__) || defined(__x86_64__)
    *(volatile uint64_t *)(void *)(base + offset) = value;
#else
    ((volatile uint32_t *)(void *)(base + offset))[0] = (uint32_t)(value & 0xffffffffu);
    ((volatile uint32_t *)(void *)(base + offset))[1] = (uint32_t)(value >> 32);
#endif
    return 0;
}

int axil_mmio_read64(axil_mmio_t *ctx, uint32_t offset, uint64_t *value) {
    volatile uint8_t *base;

    if (ctx == NULL || ctx->regs == NULL || value == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (((size_t)offset + sizeof(uint64_t)) > ctx->span || (offset & 0x7u) != 0u) {
        errno = ERANGE;
        return -1;
    }

    base = (volatile uint8_t *)ctx->regs;
#if defined(__aarch64__) || defined(__x86_64__)
    *value = *(volatile uint64_t *)(void *)(base + offset);
#else
    {
        volatile uint32_t *p32 = (volatile uint32_t *)(void *)(base + offset);
        *value = ((uint64_t)p32[1] << 32) | (uint64_t)p32[0];
    }
#endif
    return 0;
}
