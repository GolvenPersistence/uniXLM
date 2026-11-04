#define _POSIX_C_SOURCE 200809L

#include "axidma_simple.h"

#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

static void sleep_1ms(void) {
    struct timespec req;
    req.tv_sec = 0;
    req.tv_nsec = 1000 * 1000;
    (void)nanosleep(&req, NULL);
}

static int write_reg(axidma_simple_t *dma, uint32_t off, uint32_t value) {
    return axil_mmio_write32(&dma->mmio, off, value);
}

static int read_reg(axidma_simple_t *dma, uint32_t off, uint32_t *value) {
    return axil_mmio_read32(&dma->mmio, off, value);
}

static int timed_out(const struct timespec *start, unsigned timeout_ms) {
    struct timespec now;
    uint64_t elapsed_ms;

    if (timeout_ms == 0u) {
        return 0;
    }

    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0) {
        return 0;
    }

    elapsed_ms = (uint64_t)(now.tv_sec - start->tv_sec) * 1000u;
    elapsed_ms += (uint64_t)(now.tv_nsec - start->tv_nsec) / 1000000u;
    return elapsed_ms > timeout_ms;
}

static int wait_reset_clear(axidma_simple_t *dma, uint32_t dmacr_off, unsigned timeout_ms) {
    struct timespec start;
    uint32_t value = 0;

    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        memset(&start, 0, sizeof(start));
    }

    for (;;) {
        if (read_reg(dma, dmacr_off, &value) != 0) {
            return -1;
        }
        if ((value & AXIDMA_DMACR_RESET) == 0u) {
            return 0;
        }
        if (timed_out(&start, timeout_ms)) {
            errno = ETIMEDOUT;
            return -1;
        }
        sleep_1ms();
    }
}

static int wait_channel_done(axidma_simple_t *dma,
                             uint32_t dmasr_off,
                             unsigned timeout_ms,
                             uint32_t *final_status) {
    struct timespec start;
    uint32_t status = 0;

    if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        memset(&start, 0, sizeof(start));
    }

    for (;;) {
        if (read_reg(dma, dmasr_off, &status) != 0) {
            return -1;
        }

        if ((status & AXIDMA_DMASR_ERR_MASK) != 0u) {
            if ((status & AXIDMA_DMASR_IRQ_MASK) != 0u) {
                (void)write_reg(dma, dmasr_off, status & AXIDMA_DMASR_IRQ_MASK);
            }
            if (final_status != NULL) {
                *final_status = status;
            }
            errno = EIO;
            return -1;
        }

        if (((status & AXIDMA_DMASR_IOC_IRQ) != 0u) || ((status & AXIDMA_DMASR_IDLE) != 0u)) {
            if ((status & AXIDMA_DMASR_IRQ_MASK) != 0u) {
                if (write_reg(dma, dmasr_off, status & AXIDMA_DMASR_IRQ_MASK) != 0) {
                    return -1;
                }
            }
            if (final_status != NULL) {
                *final_status = status;
            }
            return 0;
        }

        if (timed_out(&start, timeout_ms)) {
            errno = ETIMEDOUT;
            return -1;
        }
        sleep_1ms();
    }
}

int axidma_simple_open(axidma_simple_t *dma, uint64_t base_addr, size_t span, const char *devmem_path) {
    if (dma == NULL) {
        errno = EINVAL;
        return -1;
    }
    memset(dma, 0, sizeof(*dma));
    return axil_mmio_open(&dma->mmio, base_addr, span, devmem_path);
}

void axidma_simple_close(axidma_simple_t *dma) {
    if (dma == NULL) {
        return;
    }
    axil_mmio_close(&dma->mmio);
}

int axidma_simple_reset(axidma_simple_t *dma, unsigned timeout_ms) {
    if (dma == NULL) {
        errno = EINVAL;
        return -1;
    }

    if (write_reg(dma, AXIDMA_MM2S_DMACR, AXIDMA_DMACR_RESET) != 0 ||
        write_reg(dma, AXIDMA_S2MM_DMACR, AXIDMA_DMACR_RESET) != 0) {
        return -1;
    }

    if (wait_reset_clear(dma, AXIDMA_MM2S_DMACR, timeout_ms) != 0 ||
        wait_reset_clear(dma, AXIDMA_S2MM_DMACR, timeout_ms) != 0) {
        return -1;
    }

    if (write_reg(dma, AXIDMA_MM2S_DMASR, AXIDMA_DMASR_IRQ_MASK) != 0 ||
        write_reg(dma, AXIDMA_S2MM_DMASR, AXIDMA_DMASR_IRQ_MASK) != 0) {
        return -1;
    }

    if (write_reg(dma, AXIDMA_MM2S_DMACR, AXIDMA_DMACR_RS) != 0 ||
        write_reg(dma, AXIDMA_S2MM_DMACR, AXIDMA_DMACR_RS) != 0) {
        return -1;
    }

    return 0;
}

int axidma_simple_start_s2mm(axidma_simple_t *dma, uint64_t dst_phys, uint32_t length_bytes) {
    if (dma == NULL || length_bytes == 0u) {
        errno = EINVAL;
        return -1;
    }
    if (write_reg(dma, AXIDMA_S2MM_DMACR, AXIDMA_DMACR_RS) != 0 ||
        write_reg(dma, AXIDMA_S2MM_DA_LSB, (uint32_t)(dst_phys & 0xffffffffu)) != 0 ||
        write_reg(dma, AXIDMA_S2MM_DA_MSB, (uint32_t)(dst_phys >> 32)) != 0 ||
        write_reg(dma, AXIDMA_S2MM_LENGTH, length_bytes) != 0) {
        return -1;
    }
    return 0;
}

int axidma_simple_start_mm2s(axidma_simple_t *dma, uint64_t src_phys, uint32_t length_bytes) {
    if (dma == NULL || length_bytes == 0u) {
        errno = EINVAL;
        return -1;
    }
    if (write_reg(dma, AXIDMA_MM2S_DMACR, AXIDMA_DMACR_RS) != 0 ||
        write_reg(dma, AXIDMA_MM2S_SA_LSB, (uint32_t)(src_phys & 0xffffffffu)) != 0 ||
        write_reg(dma, AXIDMA_MM2S_SA_MSB, (uint32_t)(src_phys >> 32)) != 0 ||
        write_reg(dma, AXIDMA_MM2S_LENGTH, length_bytes) != 0) {
        return -1;
    }
    return 0;
}

int axidma_simple_wait_mm2s(axidma_simple_t *dma, unsigned timeout_ms, uint32_t *final_status) {
    if (dma == NULL) {
        errno = EINVAL;
        return -1;
    }
    return wait_channel_done(dma, AXIDMA_MM2S_DMASR, timeout_ms, final_status);
}

int axidma_simple_wait_s2mm(axidma_simple_t *dma, unsigned timeout_ms, uint32_t *final_status) {
    if (dma == NULL) {
        errno = EINVAL;
        return -1;
    }
    return wait_channel_done(dma, AXIDMA_S2MM_DMASR, timeout_ms, final_status);
}

int axidma_simple_transfer(axidma_simple_t *dma,
                           uint64_t src_phys,
                           uint64_t dst_phys,
                           uint32_t length_bytes,
                           unsigned timeout_ms) {
    uint32_t status = 0;

    if (dma == NULL || length_bytes == 0u) {
        errno = EINVAL;
        return -1;
    }

    if (axidma_simple_start_s2mm(dma, dst_phys, length_bytes) != 0) {
        return -1;
    }
    if (axidma_simple_start_mm2s(dma, src_phys, length_bytes) != 0) {
        return -1;
    }
    if (axidma_simple_wait_mm2s(dma, timeout_ms, &status) != 0) {
        return -1;
    }
    if (axidma_simple_wait_s2mm(dma, timeout_ms, &status) != 0) {
        return -1;
    }
    return 0;
}
