#define _POSIX_C_SOURCE 200809L

#include "tpu_soc_fp32.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <strings.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "tpu_vanilla_regs.h"

static int failf(int err, const char *fmt, ...) {
    va_list ap;

    errno = err;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    return -1;
}

static uint64_t monotonic_ms(void) {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }

    return (uint64_t)ts.tv_sec * 1000ull + (uint64_t)ts.tv_nsec / 1000000ull;
}

static uint64_t monotonic_ns(void) {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }

    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void sleep_1ms(void) {
    struct timespec ts;

    ts.tv_sec = 0;
    ts.tv_nsec = 1000000L;
    (void)nanosleep(&ts, NULL);
}

static void sleep_us(unsigned us) {
    struct timespec ts;
    uint64_t start_ns;
    uint64_t deadline_ns;

    if (us == 0u) {
        return;
    }
    if (us <= 50u) {
        start_ns = monotonic_ns();
        deadline_ns = start_ns + (uint64_t)us * 1000ull;
        while (monotonic_ns() < deadline_ns) {
        }
        return;
    }

    ts.tv_sec = us / 1000000u;
    ts.tv_nsec = (long)(us % 1000000u) * 1000L;
    (void)nanosleep(&ts, NULL);
}

static void sleep_ms(unsigned ms) {
    while (ms-- != 0u) {
        sleep_1ms();
    }
}

static uint32_t env_u32_aperture_or_default(const char *name, uint32_t fallback) {
    const char *value = getenv(name);
    char *end = NULL;
    unsigned long long parsed = 0ull;

    if (value == NULL || *value == '\0') {
        return fallback;
    }

    errno = 0;
    parsed = strtoull(value, &end, 0);
    if (errno != 0 || end == value || (end != NULL && *end != '\0') || parsed > UINT32_MAX) {
        return fallback;
    }
    return (uint32_t)parsed;
}

static void platform_default_ddr_window(uint32_t *base_out, uint32_t *range_out) {
    const char *platform = getenv("TPU_SOC_PLATFORM");
    uint32_t base = TPU_SOC_FP32_ZU_DDR_BASE_DEFAULT;
    uint32_t range = TPU_SOC_FP32_ZU_DDR_RANGE_DEFAULT;

    if (platform != NULL) {
        if (strcasecmp(platform, "kv260") == 0 || strcasecmp(platform, "kria") == 0) {
            base = TPU_SOC_FP32_KV260_DDR_BASE_DEFAULT;
            range = TPU_SOC_FP32_KV260_DDR_RANGE_DEFAULT;
        } else if (strcasecmp(platform, "zu") == 0 || strcasecmp(platform, "pynq") == 0) {
            base = TPU_SOC_FP32_ZU_DDR_BASE_DEFAULT;
            range = TPU_SOC_FP32_ZU_DDR_RANGE_DEFAULT;
        }
    }

    if (base_out != NULL) {
        *base_out = base;
    }
    if (range_out != NULL) {
        *range_out = range;
    }
}

static void resolve_ddr_aperture(const char *base_env,
                                 const char *range_env,
                                 uint32_t *base_out,
                                 uint32_t *high_out) {
    uint32_t default_base = 0u;
    uint32_t default_range = 0u;
    uint32_t base = 0u;
    uint32_t range = 0u;
    uint64_t high = 0ull;

    platform_default_ddr_window(&default_base, &default_range);
    base = env_u32_aperture_or_default(base_env, default_base);
    range = env_u32_aperture_or_default(range_env, default_range);
    if (range == 0u) {
        range = default_range;
    }

    high = (uint64_t)base + (uint64_t)range - 1ull;
    if (high > UINT32_MAX) {
        high = UINT32_MAX;
    }

    if (base_out != NULL) {
        *base_out = base;
    }
    if (high_out != NULL) {
        *high_out = (uint32_t)high;
    }
}

void tpu_soc_fp32_get_input_ddr_aperture(uint32_t *base, uint32_t *high) {
    resolve_ddr_aperture("TPU_SOC_INPUT_DDR_BASE",
                         "TPU_SOC_INPUT_DDR_RANGE",
                         base,
                         high);
}

void tpu_soc_fp32_get_output_ddr_aperture(uint32_t *base, uint32_t *high) {
    resolve_ddr_aperture("TPU_SOC_OUTPUT_DDR_BASE",
                         "TPU_SOC_OUTPUT_DDR_RANGE",
                         base,
                         high);
}

static int validate_ddr_addr(uint64_t addr,
                             size_t size_bytes,
                             uint32_t base,
                             uint32_t high,
                             const char *label) {
    uint64_t end_addr;

    if (size_bytes == 0u) {
        return failf(EINVAL, "%s transfer size must be non-zero", label);
    }
    if (addr > UINT32_MAX) {
        return failf(ERANGE,
                     "%s device address 0x%016" PRIx64 " exceeds 32-bit AXI aperture",
                     label,
                     addr);
    }

    end_addr = addr + size_bytes - 1u;
    if (addr < base || end_addr > high) {
        return failf(ERANGE,
                     "%s device range [0x%08" PRIx64 ", 0x%08" PRIx64 "] is outside DDR aperture [0x%08x, 0x%08x]",
                     label,
                     addr,
                     end_addr,
                     base,
                     high);
    }

    return 0;
}

int tpu_soc_fp32_validate_input_ddr_addr(uint64_t addr, size_t size_bytes) {
    uint32_t base = 0u;
    uint32_t high = 0u;

    tpu_soc_fp32_get_input_ddr_aperture(&base, &high);
    return validate_ddr_addr(addr, size_bytes, base, high, "input buffer");
}

int tpu_soc_fp32_validate_output_ddr_addr(uint64_t addr, size_t size_bytes) {
    uint32_t base = 0u;
    uint32_t high = 0u;

    tpu_soc_fp32_get_output_ddr_aperture(&base, &high);
    return validate_ddr_addr(addr, size_bytes, base, high, "output buffer");
}

static unsigned env_u32_or_default(const char *name, unsigned default_value) {
    const char *value;
    char *endptr = NULL;
    unsigned long parsed;

    if (name == NULL) {
        return default_value;
    }

    value = getenv(name);
    if (value == NULL || *value == '\0') {
        return default_value;
    }

    errno = 0;
    parsed = strtoul(value, &endptr, 0);
    if (errno != 0 || endptr == value || (endptr != NULL && *endptr != '\0') || parsed > UINT_MAX) {
        return default_value;
    }

    return (unsigned)parsed;
}

static int env_is_truthy(const char *value) {
    if (value == NULL || *value == '\0') {
        return 0;
    }
    return strcmp(value, "0") != 0;
}

static unsigned poll_spin_budget_default(void) {
    return env_u32_or_default("TPU_POLL_SPIN_BUDGET", 4096u);
}

static unsigned poll_sleep_us_default(void) {
    return env_u32_or_default("TPU_POLL_SLEEP_US", 1u);
}

static unsigned config_to_ctrl_delay_us_default(void) {
    return env_u32_or_default("TPU_CONFIG_TO_CTRL_DELAY_US", 50u);
}

static void poll_wait_step(unsigned *spin_budget, unsigned sleep_us_value) {
    if (spin_budget != NULL && *spin_budget != 0u) {
        --(*spin_budget);
        return;
    }
    if (sleep_us_value != 0u) {
        sleep_us(sleep_us_value);
    }
}

static int validate_soft_reset_hold_cycles(uint32_t hold_cycles) {
    if (hold_cycles == 0u || hold_cycles > TPU_VANILLA_SOFT_RESET_HOLD_MASK) {
        return failf(EINVAL,
                     "invalid soft reset hold cycles=%" PRIu32 " (expected 1..%u)",
                     hold_cycles,
                     TPU_VANILLA_SOFT_RESET_HOLD_MASK);
    }
    return 0;
}

static int set_soft_reset_hold_cycles(tpu_soc_fp32_t *ctx, uint32_t hold_cycles) {
    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (validate_soft_reset_hold_cycles(hold_cycles) != 0) {
        return -1;
    }

    if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_SOFT_RESET_CFG, hold_cycles) != 0) {
        return -1;
    }
    return 0;
}

static int configure_soft_reset_cycles_from_env(tpu_soc_fp32_t *ctx) {
    const char *value;
    char *endptr = NULL;
    unsigned long parsed;
    uint32_t hold_cycles;

    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }

    value = getenv("TPU_SOFT_RESET_HOLD_CYCLES");
    if (value == NULL || *value == '\0') {
        return 0;
    }

    errno = 0;
    parsed = strtoul(value, &endptr, 0);
    if (errno != 0 || endptr == value || (endptr != NULL && *endptr != '\0') ||
        parsed > TPU_VANILLA_SOFT_RESET_HOLD_MASK) {
        return failf(EINVAL,
                     "invalid TPU_SOFT_RESET_HOLD_CYCLES=%s (expected 1..%u)",
                     value,
                     TPU_VANILLA_SOFT_RESET_HOLD_MASK);
    }

    hold_cycles = (uint32_t)parsed & TPU_VANILLA_SOFT_RESET_HOLD_MASK;
    if (hold_cycles == 0u) {
        hold_cycles = 1u;
    }
    return set_soft_reset_hold_cycles(ctx, hold_cycles);
}

static int program_config_then_start(tpu_soc_fp32_t *ctx,
                                     uint32_t config_word,
                                     uint32_t ctrl_word,
                                     unsigned timeout_ms) {
    uint64_t deadline;
    uint32_t observed = 0u;
    unsigned spin_budget;
    unsigned sleep_us_value;
    unsigned config_to_ctrl_delay_us;

    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (timeout_ms == 0u) {
        timeout_ms = 10000u;
    }
    spin_budget = poll_spin_budget_default();
    sleep_us_value = poll_sleep_us_default();
    config_to_ctrl_delay_us = config_to_ctrl_delay_us_default();

    if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_CONFIG, config_word) != 0) {
        return -1;
    }

    deadline = monotonic_ms() + (uint64_t)timeout_ms;
    for (;;) {
        if (axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_CONFIG, &observed) != 0) {
            return -1;
        }
        if (observed == config_word) {
            break;
        }
        if (monotonic_ms() > deadline) {
            return failf(ETIMEDOUT,
                         "timeout waiting for CONFIG readback 0x%08x (last=0x%08x)",
                         config_word,
                         observed);
        }
        poll_wait_step(&spin_budget, sleep_us_value);
    }
    if (config_to_ctrl_delay_us != 0u) {
        sleep_us(config_to_ctrl_delay_us);
    }

    if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_CTRL, ctrl_word) != 0) {
        return -1;
    }

    return 0;
}

static int wait_status_mask(tpu_soc_fp32_t *ctx,
                            uint32_t set_mask,
                            uint32_t clear_mask,
                            unsigned timeout_ms,
                            uint32_t *last_status) {
    uint64_t deadline = monotonic_ms() + (uint64_t)timeout_ms;
    uint32_t status = 0;
    unsigned spin_budget = poll_spin_budget_default();
    unsigned sleep_us_value = poll_sleep_us_default();

    for (;;) {
        if (axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_STATUS, &status) != 0) {
            return -1;
        }

        if ((status & set_mask) == set_mask && (status & clear_mask) == 0u) {
            if (last_status != NULL) {
                *last_status = status;
            }
            return 0;
        }

        if (monotonic_ms() > deadline) {
            return failf(ETIMEDOUT,
                         "timeout waiting for STATUS set=0x%08x clear=0x%08x (last=0x%08x)",
                         set_mask, clear_mask, status);
        }

        poll_wait_step(&spin_budget, sleep_us_value);
    }
}

static int validate_matrix_sizes_with_bytes(uint32_t m,
                                            uint32_t n,
                                            uint32_t k,
                                            size_t a_elem_bytes,
                                            size_t b_elem_bytes,
                                            size_t c_elem_bytes,
                                            size_t d_elem_bytes,
                                            size_t *a_bytes,
                                            size_t *b_bytes,
                                            size_t *c_bytes,
                                            size_t *d_bytes) {
    size_t a_sz;
    size_t b_sz;
    size_t c_sz;
    size_t d_sz;

    if (m == 0u || n == 0u || k == 0u) {
        return failf(EINVAL, "matrix dimensions must be non-zero");
    }
    if (m > 255u || n > 255u || k > 255u) {
        return failf(EINVAL, "matrix dimensions exceed 8-bit CSR packing: M=%u N=%u K=%u", m, n, k);
    }
    if (a_elem_bytes == 0u || b_elem_bytes == 0u || c_elem_bytes == 0u || d_elem_bytes == 0u) {
        return failf(EINVAL, "matrix element width must be non-zero");
    }

    a_sz = (size_t)m * (size_t)k * a_elem_bytes;
    b_sz = (size_t)k * (size_t)n * b_elem_bytes;
    c_sz = (size_t)m * (size_t)n * c_elem_bytes;
    d_sz = (size_t)m * (size_t)n * d_elem_bytes;

    if (a_sz > TPU_VANILLA_MATRIX_WINDOW_STRIDE) {
        return failf(ERANGE, "A matrix is too large for TPU input window: %zu bytes", a_sz);
    }
    if (b_sz > TPU_VANILLA_MATRIX_WINDOW_STRIDE) {
        return failf(ERANGE, "B matrix is too large for TPU input window: %zu bytes", b_sz);
    }
    if (c_sz > TPU_VANILLA_MATRIX_WINDOW_STRIDE) {
        return failf(ERANGE, "C matrix is too large for TPU input window: %zu bytes", c_sz);
    }
    if (d_sz > TPU_SOC_FP32_BRAM_SPAN) {
        return failf(ERANGE, "output matrix exceeds result BRAM capacity: %zu bytes", d_sz);
    }

    if (a_bytes != NULL) {
        *a_bytes = a_sz;
    }
    if (b_bytes != NULL) {
        *b_bytes = b_sz;
    }
    if (c_bytes != NULL) {
        *c_bytes = c_sz;
    }
    if (d_bytes != NULL) {
        *d_bytes = d_sz;
    }
    return 0;
}

static int validate_dma_source_addr(uint64_t addr, size_t size_bytes) {
    return tpu_soc_fp32_validate_input_ddr_addr(addr, size_bytes);
}

static int cdma_read_cr_sr(tpu_soc_fp32_t *ctx, uint32_t *cr_out, uint32_t *sr_out) {
    uint32_t cr = 0u;
    uint32_t sr = 0u;

    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (axil_mmio_read32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_CR, &cr) != 0 ||
        axil_mmio_read32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_SR, &sr) != 0) {
        return -1;
    }
    if (cr_out != NULL) {
        *cr_out = cr;
    }
    if (sr_out != NULL) {
        *sr_out = sr;
    }
    return 0;
}

static tpu_soc_fp32_cdma_transfer_log_t *cdma_trace_append(tpu_soc_fp32_cdma_trace_t *trace,
                                                           const char *label) {
    tpu_soc_fp32_cdma_transfer_log_t *entry;

    if (trace == NULL || trace->transfer_count >= TPU_SOC_FP32_CDMA_TRACE_MAX_TRANSFERS) {
        return NULL;
    }

    entry = &trace->transfers[trace->transfer_count++];
    memset(entry, 0, sizeof(*entry));
    if (label != NULL && label[0] != '\0') {
        (void)snprintf(entry->label, sizeof(entry->label), "%s", label);
    }
    return entry;
}

static void cdma_trace_store_after_regs(tpu_soc_fp32_t *ctx,
                                        tpu_soc_fp32_cdma_transfer_log_t *entry,
                                        uint32_t fallback_sr) {
    if (entry == NULL) {
        return;
    }
    if (cdma_read_cr_sr(ctx, &entry->cr_after, &entry->sr_after) != 0) {
        entry->sr_after = fallback_sr;
    }
}

static void cdma_trace_store_failure_regs(tpu_soc_fp32_t *ctx,
                                          tpu_soc_fp32_cdma_trace_t *trace,
                                          tpu_soc_fp32_cdma_transfer_log_t *entry,
                                          unsigned index) {
    if (trace != NULL && trace->first_failure_index < 0) {
        trace->first_failure_index = (int)index;
    }
    if (ctx != NULL && entry != NULL && tpu_soc_fp32_read_regs(ctx, &entry->failure_regs) == 0) {
        entry->has_failure_regs = 1;
    }
}

static void cdma_bd_write32(void *bd_base, uint32_t offset, uint32_t value) {
    uint8_t *base = (uint8_t *)bd_base;

    *(uint32_t *)(void *)(base + offset) = value;
}

static void cdma_bd_write64(void *bd_base, uint32_t offset, uint64_t value) {
    cdma_bd_write32(bd_base, offset, (uint32_t)(value & 0xffffffffu));
    cdma_bd_write32(bd_base, offset + 4u, (uint32_t)(value >> 32));
}

static void cdma_bd_prepare(void *bd_base,
                            uint64_t bd_phys_addr,
                            uint64_t next_phys_addr,
                            const tpu_soc_fp32_cdma_segment_t *segment) {
    memset(bd_base, 0, TPU_SOC_FP32_CDMA_BD_BYTES);
    cdma_bd_write64(bd_base, TPU_SOC_FP32_CDMA_BD_NDESC_OFFSET, next_phys_addr);
    cdma_bd_write64(bd_base, TPU_SOC_FP32_CDMA_BD_BUFSRC_OFFSET, segment->src_addr);
    cdma_bd_write64(bd_base, TPU_SOC_FP32_CDMA_BD_BUFDST_OFFSET, segment->dst_addr);
    cdma_bd_write32(bd_base,
                    TPU_SOC_FP32_CDMA_BD_CTRL_LEN_OFFSET,
                    (uint32_t)segment->size_bytes & TPU_SOC_FP32_CDMA_BD_CTRL_LEN_MASK);
    cdma_bd_write32(bd_base, TPU_SOC_FP32_CDMA_BD_STS_OFFSET, 0u);
    cdma_bd_write64(bd_base, TPU_SOC_FP32_CDMA_BD_PHYS_ADDR_OFFSET, bd_phys_addr);
    cdma_bd_write32(bd_base, TPU_SOC_FP32_CDMA_BD_ISLITE_OFFSET, 0u);
    cdma_bd_write32(bd_base, TPU_SOC_FP32_CDMA_BD_HASDRE_OFFSET, 0u);
    cdma_bd_write32(bd_base, TPU_SOC_FP32_CDMA_BD_WORDLEN_OFFSET, 8u);
    cdma_bd_write32(bd_base, TPU_SOC_FP32_CDMA_BD_MAX_LEN_OFFSET, TPU_SOC_FP32_CDMA_BD_CTRL_LEN_MASK);
    cdma_bd_write32(bd_base, TPU_SOC_FP32_CDMA_BD_ADDRLEN_OFFSET, 32u);
}

void tpu_soc_fp32_cdma_trace_clear(tpu_soc_fp32_cdma_trace_t *trace) {
    if (trace == NULL) {
        return;
    }
    memset(trace, 0, sizeof(*trace));
    trace->first_failure_index = -1;
}

const char *tpu_soc_fp32_cdma_first_failure_label(const tpu_soc_fp32_cdma_trace_t *trace) {
    unsigned index;

    if (trace == NULL || trace->first_failure_index < 0) {
        return NULL;
    }
    index = (unsigned)trace->first_failure_index;
    if (index >= trace->transfer_count || trace->transfers[index].label[0] == '\0') {
        return NULL;
    }
    return trace->transfers[index].label;
}

void tpu_soc_fp32_dump_cdma_trace(FILE *stream, const tpu_soc_fp32_cdma_trace_t *trace) {
    FILE *out = (stream != NULL) ? stream : stdout;
    const char *first_failure;
    unsigned i;

    if (trace == NULL) {
        fprintf(out, "CDMA trace: unavailable\n");
        return;
    }

    fprintf(out,
            "CDMA reset: attempted=%d rc=%d err=%d pre(CR=0x%08" PRIx32 ",SR=0x%08" PRIx32 ") post(CR=0x%08" PRIx32 ",SR=0x%08" PRIx32 ")\n",
            trace->reset.attempted,
            trace->reset.rc,
            trace->reset.err,
            trace->reset.cr_before,
            trace->reset.sr_before,
            trace->reset.cr_after,
            trace->reset.sr_after);

    if (trace->transfer_count == 0u) {
        fprintf(out, "CDMA transfers: none recorded\n");
    }

    for (i = 0; i < trace->transfer_count; ++i) {
        const tpu_soc_fp32_cdma_transfer_log_t *entry = &trace->transfers[i];
        const char *label = (entry->label[0] != '\0') ? entry->label : "unnamed";

        fprintf(out,
                "CDMA[%u] %s src=0x%016" PRIx64 " dst=0x%016" PRIx64 " bytes=%zu timeout_ms=%u polls=%" PRIu32 " dur_us=%.3f rc=%d err=%d timed_out=%d\n",
                i,
                label,
                entry->src_addr,
                entry->dst_addr,
                entry->size_bytes,
                entry->timeout_ms,
                entry->poll_count,
                (double)entry->duration_ns / 1000.0,
                entry->rc,
                entry->err,
                entry->timed_out);
        fprintf(out,
                "          pre(CR=0x%08" PRIx32 ",SR=0x%08" PRIx32 ") post(CR=0x%08" PRIx32 ",SR=0x%08" PRIx32 ")\n",
                entry->cr_before,
                entry->sr_before,
                entry->cr_after,
                entry->sr_after);
        if (entry->has_failure_regs) {
            fprintf(out, "          failure register snapshot:\n");
            tpu_soc_fp32_dump_regs(out, &entry->failure_regs);
        }
    }

    first_failure = tpu_soc_fp32_cdma_first_failure_label(trace);
    fprintf(out, "CDMA first failure = %s\n", (first_failure != NULL) ? first_failure : "none");
}

int tpu_soc_fp32_cdma_transfer_with_trace(tpu_soc_fp32_t *ctx,
                                          tpu_soc_fp32_cdma_trace_t *trace,
                                          const char *label,
                                          uint64_t src_addr,
                                          uint64_t dst_addr,
                                          size_t size_bytes,
                                          unsigned timeout_ms) {
    uint64_t deadline;
    uint32_t status = 0u;
    unsigned entry_index = 0u;
    unsigned spin_budget;
    unsigned sleep_us_value;
    tpu_soc_fp32_cdma_transfer_log_t *entry;

    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }

    if (timeout_ms == 0u) {
        timeout_ms = 10000u;
    }
    spin_budget = poll_spin_budget_default();
    sleep_us_value = poll_sleep_us_default();

    entry_index = (trace != NULL) ? trace->transfer_count : 0u;
    entry = cdma_trace_append(trace, label);
    if (entry != NULL) {
        entry->src_addr = src_addr;
        entry->dst_addr = dst_addr;
        entry->size_bytes = size_bytes;
        entry->timeout_ms = timeout_ms;
    }

    if (size_bytes == 0u || size_bytes > UINT32_MAX) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = EINVAL;
        }
        if (trace != NULL && trace->first_failure_index < 0) {
            trace->first_failure_index = (int)entry_index;
        }
        return failf(EINVAL, "invalid CDMA transfer size%s%s: %zu",
                     (label != NULL) ? " for " : "",
                     (label != NULL) ? label : "",
                     size_bytes);
    }

    if (cdma_read_cr_sr(ctx, entry != NULL ? &entry->cr_before : NULL, entry != NULL ? &entry->sr_before : NULL) != 0) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = errno;
        }
        if (trace != NULL && trace->first_failure_index < 0) {
            trace->first_failure_index = (int)entry_index;
        }
        return -1;
    }

    if (axil_mmio_write32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_SA, (uint32_t)(src_addr & 0xffffffffu)) != 0 ||
        axil_mmio_write32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_SA_MSB, (uint32_t)(src_addr >> 32)) != 0 ||
        axil_mmio_write32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_DA, (uint32_t)(dst_addr & 0xffffffffu)) != 0 ||
        axil_mmio_write32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_DA_MSB, (uint32_t)(dst_addr >> 32)) != 0 ||
        axil_mmio_write32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_BTT, (uint32_t)size_bytes) != 0) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = errno;
            cdma_trace_store_after_regs(ctx, entry, status);
        }
        cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
        return -1;
    }

    deadline = monotonic_ms() + (uint64_t)timeout_ms;
    if (entry != NULL) {
        entry->start_ns = monotonic_ns();
    }
    for (;;) {
        if (axil_mmio_read32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_SR, &status) != 0) {
            if (entry != NULL) {
                entry->rc = -1;
                entry->err = errno;
                cdma_trace_store_after_regs(ctx, entry, status);
            }
            cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
            return -1;
        }
        if (entry != NULL) {
            entry->poll_count++;
        }
        if ((status & TPU_SOC_FP32_CDMA_SR_IOC) != 0u) {
            if (entry != NULL) {
                entry->done_ns = monotonic_ns();
                if (entry->done_ns >= entry->start_ns) {
                    entry->duration_ns = entry->done_ns - entry->start_ns;
                }
            }
            break;
        }
        if (monotonic_ms() > deadline) {
            if (entry != NULL) {
                entry->rc = -1;
                entry->err = ETIMEDOUT;
                entry->timed_out = 1;
                cdma_trace_store_after_regs(ctx, entry, status);
            }
            cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
            return failf(ETIMEDOUT,
                         "CDMA transfer timed out%s%s%s (SR=0x%08x)",
                         (label != NULL) ? " [" : "",
                         (label != NULL) ? label : "",
                         (label != NULL) ? "]" : "",
                         status);
        }
        poll_wait_step(&spin_budget, sleep_us_value);
    }

    if ((status & TPU_SOC_FP32_CDMA_SR_ERR_MASK) != 0u) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = EIO;
            cdma_trace_store_after_regs(ctx, entry, status);
        }
        cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
        return failf(EIO,
                     "CDMA transfer failed%s%s%s (SR=0x%08x)",
                     (label != NULL) ? " [" : "",
                     (label != NULL) ? label : "",
                     (label != NULL) ? "]" : "",
                     status);
    }

    if (axil_mmio_write32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_SR, TPU_SOC_FP32_CDMA_SR_IOC) != 0) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = errno;
            cdma_trace_store_after_regs(ctx, entry, status);
        }
        cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
        return -1;
    }

    if (entry != NULL) {
        entry->rc = 0;
        entry->err = 0;
        cdma_trace_store_after_regs(ctx, entry, status);
    }
    return 0;
}

int tpu_soc_fp32_cdma_transfer_sg_chain_with_trace(tpu_soc_fp32_t *ctx,
                                                   tpu_soc_fp32_cdma_trace_t *trace,
                                                   const char *label,
                                                   const tpu_soc_fp32_cdma_segment_t *segments,
                                                   unsigned segment_count,
                                                   unsigned timeout_ms) {
    uint64_t deadline;
    uint64_t total_bytes = 0u;
    uint32_t status = 0u;
    uint32_t last_bd_status = 0u;
    uint32_t current_cr = 0u;
    unsigned entry_index = 0u;
    unsigned spin_budget;
    unsigned sleep_us_value;
    unsigned i;
    size_t desc_bytes;
    tpu_soc_fp32_cdma_transfer_log_t *entry;
    tpu_soc_fp32_buffer_t desc_buf;
    void *last_bd_base = NULL;

    if (ctx == NULL || segments == NULL || segment_count == 0u) {
        errno = EINVAL;
        return -1;
    }

    memset(&desc_buf, 0, sizeof(desc_buf));

    if (timeout_ms == 0u) {
        timeout_ms = 10000u;
    }
    spin_budget = poll_spin_budget_default();
    sleep_us_value = poll_sleep_us_default();

    entry_index = (trace != NULL) ? trace->transfer_count : 0u;
    entry = cdma_trace_append(trace, (label != NULL) ? label : "ABC-sg");
    if (entry != NULL) {
        entry->src_addr = segments[0].src_addr;
        entry->dst_addr = segments[0].dst_addr;
        entry->timeout_ms = timeout_ms;
    }

    for (i = 0u; i < segment_count; ++i) {
        if (segments[i].size_bytes == 0u || segments[i].size_bytes > TPU_SOC_FP32_CDMA_BD_CTRL_LEN_MASK) {
            if (entry != NULL) {
                entry->rc = -1;
                entry->err = EINVAL;
            }
            if (trace != NULL && trace->first_failure_index < 0) {
                trace->first_failure_index = (int)entry_index;
            }
            return failf(EINVAL,
                         "invalid SG CDMA segment size%s%s[%u]=%zu",
                         (label != NULL) ? " for " : "",
                         (label != NULL) ? label : "",
                         i,
                         segments[i].size_bytes);
        }
        if (validate_dma_source_addr(segments[i].src_addr, segments[i].size_bytes) != 0) {
            if (entry != NULL) {
                entry->rc = -1;
                entry->err = errno;
            }
            if (trace != NULL && trace->first_failure_index < 0) {
                trace->first_failure_index = (int)entry_index;
            }
            return -1;
        }
        if (segments[i].dst_addr > UINT32_MAX) {
            if (entry != NULL) {
                entry->rc = -1;
                entry->err = ERANGE;
            }
            if (trace != NULL && trace->first_failure_index < 0) {
                trace->first_failure_index = (int)entry_index;
            }
            return failf(ERANGE,
                         "SG CDMA destination%s%s[%u]=0x%016" PRIx64 " exceeds 32-bit address space",
                         (label != NULL) ? " for " : "",
                         (label != NULL) ? label : "",
                         i,
                         segments[i].dst_addr);
        }
        total_bytes += segments[i].size_bytes;
    }

    if (entry != NULL) {
        entry->size_bytes = (size_t)((total_bytes > SIZE_MAX) ? SIZE_MAX : total_bytes);
    }

    desc_bytes = (size_t)segment_count * TPU_SOC_FP32_CDMA_BD_BYTES;
    if (tpu_soc_fp32_alloc_buffer(ctx, &desc_buf, desc_bytes) != 0) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = errno;
        }
        if (trace != NULL && trace->first_failure_index < 0) {
            trace->first_failure_index = (int)entry_index;
        }
        return -1;
    }

    if (validate_dma_source_addr(desc_buf.device_addr, desc_bytes) != 0) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = errno;
        }
        cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
        tpu_soc_fp32_free_buffer(&desc_buf);
        return -1;
    }

    if (cdma_read_cr_sr(ctx,
                        entry != NULL ? &entry->cr_before : NULL,
                        entry != NULL ? &entry->sr_before : &status) != 0) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = errno;
        }
        if (trace != NULL && trace->first_failure_index < 0) {
            trace->first_failure_index = (int)entry_index;
        }
        tpu_soc_fp32_free_buffer(&desc_buf);
        return -1;
    }
    if (entry != NULL) {
        status = entry->sr_before;
    }
    if ((status & TPU_SOC_FP32_CDMA_SR_SGINCLD) == 0u) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = ENOTSUP;
            cdma_trace_store_after_regs(ctx, entry, status);
        }
        cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
        tpu_soc_fp32_free_buffer(&desc_buf);
        return failf(ENOTSUP, "CDMA scatter-gather mode is not enabled in hardware");
    }

    for (i = 0u; i < segment_count; ++i) {
        uint64_t bd_phys_addr = desc_buf.device_addr + (uint64_t)i * TPU_SOC_FP32_CDMA_BD_BYTES;
        uint64_t next_phys_addr = desc_buf.device_addr +
                                  (uint64_t)((i + 1u) % segment_count) * TPU_SOC_FP32_CDMA_BD_BYTES;
        void *bd_base = (uint8_t *)desc_buf.mapping + (size_t)i * TPU_SOC_FP32_CDMA_BD_BYTES;

        cdma_bd_prepare(bd_base, bd_phys_addr, next_phys_addr, &segments[i]);
    }
    last_bd_base = (uint8_t *)desc_buf.mapping + (size_t)(segment_count - 1u) * TPU_SOC_FP32_CDMA_BD_BYTES;

    if (tpu_soc_fp32_buffer_sync_to_device(&desc_buf, desc_bytes) != 0) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = errno;
        }
        cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
        tpu_soc_fp32_free_buffer(&desc_buf);
        return -1;
    }

    if (axil_mmio_read32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_CR, &current_cr) != 0) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = errno;
        }
        cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
        tpu_soc_fp32_free_buffer(&desc_buf);
        return -1;
    }
    current_cr |= TPU_SOC_FP32_CDMA_CR_SGMODE;
    if (axil_mmio_write32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_CR, current_cr) != 0 ||
        axil_mmio_write64(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_CDESC, desc_buf.device_addr) != 0) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = errno;
        }
        cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
        tpu_soc_fp32_free_buffer(&desc_buf);
        return -1;
    }

    deadline = monotonic_ms() + (uint64_t)timeout_ms;
    if (entry != NULL) {
        entry->start_ns = monotonic_ns();
    }
    if (axil_mmio_write64(&ctx->cdma_mmio,
                          TPU_SOC_FP32_CDMA_TDESC,
                          desc_buf.device_addr + (uint64_t)(segment_count - 1u) * TPU_SOC_FP32_CDMA_BD_BYTES) != 0) {
        if (entry != NULL) {
            entry->rc = -1;
            entry->err = errno;
        }
        cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
        tpu_soc_fp32_free_buffer(&desc_buf);
        return -1;
    }

    for (;;) {
        if (axil_mmio_read32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_SR, &status) != 0) {
            if (entry != NULL) {
                entry->rc = -1;
                entry->err = errno;
                cdma_trace_store_after_regs(ctx, entry, status);
            }
            cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
            tpu_soc_fp32_free_buffer(&desc_buf);
            return -1;
        }
        if (entry != NULL) {
            entry->poll_count++;
        }
        if ((status & (TPU_SOC_FP32_CDMA_SR_ERR_MASK | TPU_SOC_FP32_CDMA_SR_SG_ERR_MASK)) != 0u) {
            if (entry != NULL) {
                entry->rc = -1;
                entry->err = EIO;
                cdma_trace_store_after_regs(ctx, entry, status);
            }
            cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
            tpu_soc_fp32_free_buffer(&desc_buf);
            return failf(EIO,
                         "CDMA SG transfer failed%s%s%s (SR=0x%08x)",
                         (label != NULL) ? " [" : "",
                         (label != NULL) ? label : "",
                         (label != NULL) ? "]" : "",
                         status);
        }
        if ((status & TPU_SOC_FP32_CDMA_SR_IDLE) != 0u) {
            if (tpu_soc_fp32_buffer_sync_from_device(&desc_buf, desc_bytes) != 0) {
                if (entry != NULL) {
                    entry->rc = -1;
                    entry->err = errno;
                    cdma_trace_store_after_regs(ctx, entry, status);
                }
                cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
                tpu_soc_fp32_free_buffer(&desc_buf);
                return -1;
            }
            last_bd_status = *(uint32_t *)(void *)((uint8_t *)last_bd_base + TPU_SOC_FP32_CDMA_BD_STS_OFFSET);
            if ((last_bd_status & TPU_SOC_FP32_CDMA_BD_STS_ERR_MASK) != 0u) {
                if (entry != NULL) {
                    entry->rc = -1;
                    entry->err = EIO;
                    cdma_trace_store_after_regs(ctx, entry, status);
                }
                cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
                tpu_soc_fp32_free_buffer(&desc_buf);
                return failf(EIO,
                             "CDMA SG descriptor failed%s%s%s (BD_STS=0x%08x)",
                             (label != NULL) ? " [" : "",
                             (label != NULL) ? label : "",
                             (label != NULL) ? "]" : "",
                             last_bd_status);
            }
            if ((last_bd_status & TPU_SOC_FP32_CDMA_BD_STS_COMPLETE_MASK) != 0u) {
                if (entry != NULL) {
                    entry->done_ns = monotonic_ns();
                    if (entry->done_ns >= entry->start_ns) {
                        entry->duration_ns = entry->done_ns - entry->start_ns;
                    }
                }
                break;
            }
        }
        if (monotonic_ms() > deadline) {
            if (entry != NULL) {
                entry->rc = -1;
                entry->err = ETIMEDOUT;
                entry->timed_out = 1;
                cdma_trace_store_after_regs(ctx, entry, status);
            }
            cdma_trace_store_failure_regs(ctx, trace, entry, entry_index);
            tpu_soc_fp32_free_buffer(&desc_buf);
            return failf(ETIMEDOUT,
                         "CDMA SG transfer timed out%s%s%s (SR=0x%08x BD_STS=0x%08x)",
                         (label != NULL) ? " [" : "",
                         (label != NULL) ? label : "",
                         (label != NULL) ? "]" : "",
                         status,
                         last_bd_status);
        }
        poll_wait_step(&spin_budget, sleep_us_value);
    }

    if (axil_mmio_read32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_CR, &current_cr) == 0) {
        (void)axil_mmio_write32(&ctx->cdma_mmio,
                                TPU_SOC_FP32_CDMA_CR,
                                current_cr & ~TPU_SOC_FP32_CDMA_CR_SGMODE);
    }

    if (entry != NULL) {
        entry->rc = 0;
        entry->err = 0;
        cdma_trace_store_after_regs(ctx, entry, status);
    }
    tpu_soc_fp32_free_buffer(&desc_buf);
    return 0;
}

int tpu_soc_fp32_cdma_tail_fixup_with_trace(tpu_soc_fp32_t *ctx,
                                            tpu_soc_fp32_cdma_trace_t *trace,
                                            const char *label,
                                            uint64_t src_addr,
                                            uint64_t dst_addr,
                                            size_t size_bytes,
                                            unsigned timeout_ms) {
    /*
     * Current board bring-up shows the AXI write path can miss the final 64-bit
     * beat of a multi-beat CDMA transfer. Re-issuing a tiny tail transfer with
     * the logical last beat as the first beat forces the target to observe the
     * final matrix word without changing the main data layout.
     */
    size_t beat_bytes = 8u;
    size_t kick_bytes = 8u;

    if (ctx == NULL || size_bytes < beat_bytes) {
        return 0;
    }

    return tpu_soc_fp32_cdma_transfer_with_trace(ctx,
                                                 trace,
                                                 label,
                                                 src_addr + size_bytes - beat_bytes,
                                                 dst_addr + size_bytes - beat_bytes,
                                                 kick_bytes,
                                                 timeout_ms);
}

int tpu_soc_fp32_open(tpu_soc_fp32_t *ctx,
                      uint64_t csr_base,
                      uint64_t cdma_base,
                      uint64_t bram_base,
                      unsigned device_index,
                      unsigned mem_group,
                      const char *devmem_path) {
    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }

    memset(ctx, 0, sizeof(*ctx));
    ctx->csr_mmio.fd = -1;
    ctx->cdma_mmio.fd = -1;
    ctx->bram_mmio.fd = -1;
    ctx->device_index = device_index;
    ctx->mem_group = mem_group;
    tpu_soc_fp32_get_input_ddr_aperture(&ctx->input_ddr_base, &ctx->input_ddr_high);
    tpu_soc_fp32_get_output_ddr_aperture(&ctx->output_ddr_base, &ctx->output_ddr_high);
    tpu_soc_fp32_cdma_trace_clear(&ctx->last_cdma_trace);

    if (axil_mmio_open(&ctx->csr_mmio, csr_base, TPU_SOC_FP32_CSR_SPAN, devmem_path) != 0) {
        return -1;
    }
    if (axil_mmio_open(&ctx->cdma_mmio, cdma_base, TPU_SOC_FP32_CDMA_SPAN, devmem_path) != 0) {
        tpu_soc_fp32_close(ctx);
        return -1;
    }
    if (axil_mmio_open(&ctx->bram_mmio, bram_base, TPU_SOC_FP32_BRAM_SPAN, devmem_path) != 0) {
        tpu_soc_fp32_close(ctx);
        return -1;
    }

    ctx->xrt_device = xrt_cpp_device_open(device_index);
    if (ctx->xrt_device == NULL) {
        tpu_soc_fp32_close(ctx);
        return failf(EIO, "xrt_cpp_device_open(%u) failed: %s", device_index, xrt_cpp_last_error());
    }

    return 0;
}

void tpu_soc_fp32_close(tpu_soc_fp32_t *ctx) {
    if (ctx == NULL) {
        return;
    }

    if (ctx->xrt_device != NULL) {
        xrt_cpp_device_close(ctx->xrt_device);
        ctx->xrt_device = NULL;
    }

    axil_mmio_close(&ctx->bram_mmio);
    axil_mmio_close(&ctx->cdma_mmio);
    axil_mmio_close(&ctx->csr_mmio);
}

int tpu_soc_fp32_alloc_buffer_ex(tpu_soc_fp32_t *ctx, tpu_soc_fp32_buffer_t *buf, size_t size_bytes, int cacheable) {
    size_t alloc_bytes;

    if (ctx == NULL || buf == NULL || ctx->xrt_device == NULL || size_bytes == 0u) {
        errno = EINVAL;
        return -1;
    }

    memset(buf, 0, sizeof(*buf));

    alloc_bytes = (size_bytes < 8192u) ? 8192u : size_bytes;
    buf->handle = xrt_cpp_bo_alloc(ctx->xrt_device, alloc_bytes, ctx->mem_group, cacheable ? 1 : 0);
    if (buf->handle == NULL) {
        return failf(ENOMEM, "xrt_cpp_bo_alloc(%zu) failed: %s", alloc_bytes, xrt_cpp_last_error());
    }

    buf->mapping = xrt_cpp_bo_map(buf->handle);
    if (buf->mapping == NULL) {
        tpu_soc_fp32_free_buffer(buf);
        return failf(EIO, "xrt_cpp_bo_map failed: %s", xrt_cpp_last_error());
    }

    buf->device_addr = xrt_cpp_bo_address(buf->handle);
    if (buf->device_addr == UINT64_MAX || buf->device_addr == (uint64_t)LLONG_MAX) {
        tpu_soc_fp32_free_buffer(buf);
        return failf(EIO, "xrt_cpp_bo_address failed: %s", xrt_cpp_last_error());
    }

    buf->size = alloc_bytes;
    return 0;
}

int tpu_soc_fp32_alloc_buffer(tpu_soc_fp32_t *ctx, tpu_soc_fp32_buffer_t *buf, size_t size_bytes) {
    return tpu_soc_fp32_alloc_buffer_ex(ctx, buf, size_bytes, 0);
}

void tpu_soc_fp32_free_buffer(tpu_soc_fp32_buffer_t *buf) {
    if (buf == NULL) {
        return;
    }
    if (buf->handle != NULL) {
        xrt_cpp_bo_free(buf->handle);
    }
    memset(buf, 0, sizeof(*buf));
}

int tpu_soc_fp32_buffer_sync_to_device(tpu_soc_fp32_buffer_t *buf, size_t size_bytes) {
    if (buf == NULL || buf->handle == NULL || size_bytes > buf->size) {
        errno = EINVAL;
        return -1;
    }
    if (xrt_cpp_bo_sync_to_device(buf->handle, size_bytes, 0u) != 0) {
        return failf(EIO, "xrt_cpp_bo_sync_to_device failed: %s", xrt_cpp_last_error());
    }
    return 0;
}

int tpu_soc_fp32_buffer_sync_from_device(tpu_soc_fp32_buffer_t *buf, size_t size_bytes) {
    if (buf == NULL || buf->handle == NULL || size_bytes > buf->size) {
        errno = EINVAL;
        return -1;
    }
    if (xrt_cpp_bo_sync_from_device(buf->handle, size_bytes, 0u) != 0) {
        return failf(EIO, "xrt_cpp_bo_sync_from_device failed: %s", xrt_cpp_last_error());
    }
    return 0;
}

int tpu_soc_fp32_read_regs(tpu_soc_fp32_t *ctx, tpu_soc_fp32_regs_t *regs) {
    if (ctx == NULL || regs == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_CTRL, &regs->ctrl) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_STATUS, &regs->status) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_CONFIG, &regs->config) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_DONE, &regs->dbg_done) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_LAST_ADDR, &regs->dbg_last_addr) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_FLOW, &regs->dbg_flow) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_COUNTS, &regs->dbg_counts) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_COUNTS2, &regs->dbg_counts2) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_COUNTS3, &regs->dbg_counts3) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_COUNTS4, &regs->dbg_counts4) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_COUNTS5, &regs->dbg_counts5) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_COUNTS6, &regs->dbg_counts6) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_ACC_FIRST, &regs->dbg_acc_first) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_ACC_LAST, &regs->dbg_acc_last) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_CFG1, &regs->dbg_cfg1) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_CFG2, &regs->dbg_cfg2) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_CFG3, &regs->dbg_cfg3) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_ACC_CTRL0, &regs->dbg_acc_ctrl0) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_ACC_CTRL1, &regs->dbg_acc_ctrl1) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_ACC_CTRL2, &regs->dbg_acc_ctrl2) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_AXI_AW, &regs->dbg_axi_aw) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_AXI_W, &regs->dbg_axi_w) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_AXI_WLAST, &regs->dbg_axi_wlast) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_AXI_RAM_WR, &regs->dbg_axi_ram_wr) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_AXI_LAST, &regs->dbg_axi_last) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_AXI_AW_BEATS, &regs->dbg_axi_aw_beats) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_AXI_SHORT_BURST, &regs->dbg_axi_short_burst) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_AXI_SHORT_MISS, &regs->dbg_axi_short_miss) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_AXI_TOTALS, &regs->dbg_axi_totals) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_AXI_NONE, &regs->dbg_axi_none) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_ACC_PIPE0, &regs->dbg_acc_pipe0) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_ACC_PIPE1, &regs->dbg_acc_pipe1) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_ACC_PIPE2, &regs->dbg_acc_pipe2) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_STORE_TARGET_ADDR, &regs->store_target_addr) != 0 ||
        axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_INGRESS, &regs->dbg_ingress) != 0) {
        return -1;
    }
    return 0;
}

static void dump_packed_cfg(FILE *out, const char *label, uint32_t cfg_word) {
    uint32_t precision = (cfg_word >> 24) & 0xffu;
    uint32_t m = (cfg_word >> 16) & 0xffu;
    uint32_t n = (cfg_word >> 8) & 0xffu;
    uint32_t k = cfg_word & 0xffu;

    fprintf(out,
            "%-6s = 0x%08" PRIx32 " (precision=%" PRIu32 " M=%" PRIu32 " N=%" PRIu32 " K=%" PRIu32 ")\n",
            label,
            cfg_word,
            precision,
            m,
            n,
            k);
}

static void dump_acc_ctrl_dbg(FILE *out,
                              uint32_t ctrl0,
                              uint32_t ctrl1,
                              uint32_t ctrl2) {
    uint32_t flags = ctrl0 & 0xffu;
    uint32_t partial_phase = (ctrl0 >> 8) & 0xffu;
    uint32_t partials_per_output = (ctrl0 >> 16) & 0xffu;
    uint32_t precision = (ctrl0 >> 24) & 0x0fu;
    uint32_t valid_cycle_count = ctrl1 & 0xffffu;
    uint32_t emit_cycle_count = (ctrl1 >> 16) & 0xffffu;
    uint32_t valid_cycle_limit = ctrl2 & 0xffffu;
    uint32_t output_cycle_limit = (ctrl2 >> 16) & 0xffffu;

    fprintf(out,
            "ACCCTL = 0x%08" PRIx32 "/0x%08" PRIx32 "/0x%08" PRIx32
            " (precision=%" PRIu32
            " int=%u run=%u data_valid=%u accept=%u first=%u emit=%u done_raw=%u done_tail=%u"
            " partial_phase=%" PRIu32 " partials=%" PRIu32
            " valid_cnt=%" PRIu32 " emit_cnt=%" PRIu32
            " valid_limit=%" PRIu32 " output_limit=%" PRIu32 ")\n",
            ctrl0,
            ctrl1,
            ctrl2,
            precision,
            flags & 0x1u,
            (flags >> 1) & 0x1u,
            (flags >> 2) & 0x1u,
            (flags >> 3) & 0x1u,
            (flags >> 4) & 0x1u,
            (flags >> 5) & 0x1u,
            (flags >> 6) & 0x1u,
            (flags >> 7) & 0x1u,
            partial_phase,
            partials_per_output,
            valid_cycle_count,
            emit_cycle_count,
            valid_cycle_limit,
            output_cycle_limit);
}

static void dump_axi_triplet(FILE *out, const char *label, uint32_t packed) {
    fprintf(out,
            "%-6s = 0x%08" PRIx32 " (a=%" PRIu32 " b=%" PRIu32 " c=%" PRIu32 ")\n",
            label,
            packed,
            packed & 0x3ffu,
            (packed >> 10) & 0x3ffu,
            (packed >> 20) & 0x3ffu);
}

static void dump_axi_last_dbg(FILE *out, uint32_t packed) {
    fprintf(out,
            "AXILAST= 0x%08" PRIx32 " (last_aw_sel=%" PRIu32 " active_sel=%" PRIu32 " last_awlen=%" PRIu32 " last_wstrb=0x%02" PRIx32 ")\n",
            packed,
            packed & 0x3u,
            (packed >> 2) & 0x3u,
            (packed >> 8) & 0xffu,
            (packed >> 16) & 0xffu);
}

static void dump_axi_pair(FILE *out, const char *label, uint32_t packed) {
    fprintf(out,
            "%-6s = 0x%08" PRIx32 " (w=%" PRIu32 " ram=%" PRIu32 ")\n",
            label,
            packed,
            packed & 0xffffu,
            (packed >> 16) & 0xffffu);
}

static void dump_acc_pipe_dbg(FILE *out,
                              uint32_t pipe0,
                              uint32_t pipe1,
                              uint32_t pipe2) {
    fprintf(out,
            "ACCP0  = 0x%08" PRIx32
            " (lane0_demux=%" PRIu32 " lane0_ctrl_emit=%" PRIu32 " lane0_ctrl_done=%" PRIu32 " lane0_fp_req_tail=%" PRIu32 ")\n",
            pipe0,
            pipe0 & 0xffu,
            (pipe0 >> 8) & 0xffu,
            (pipe0 >> 16) & 0xffu,
            (pipe0 >> 24) & 0xffu);
    fprintf(out,
            "ACCP1  = 0x%08" PRIx32
            " (lane0_fp_emit_tail=%" PRIu32 " lane0_fp_done_tail=%" PRIu32 " lane0_fifo_wr=%" PRIu32 " lane0_done=%" PRIu32 ")\n",
            pipe1,
            pipe1 & 0xffu,
            (pipe1 >> 8) & 0xffu,
            (pipe1 >> 16) & 0xffu,
            (pipe1 >> 24) & 0xffu);
    fprintf(out,
            "ACCP2  = 0x%08" PRIx32
            " (lane_done_mask=0x%02" PRIx32 " array_done=%" PRIu32 " deferred_final=%" PRIu32 " any_lane_done=%" PRIu32 ")\n",
            pipe2,
            pipe2 & 0xffu,
            (pipe2 >> 8) & 0xffu,
            (pipe2 >> 16) & 0xffu,
            (pipe2 >> 24) & 0xffu);
}

void tpu_soc_fp32_dump_regs(FILE *stream, const tpu_soc_fp32_regs_t *regs) {
    FILE *out = (stream != NULL) ? stream : stdout;
    uint32_t precision;
    uint32_t m;
    uint32_t n;
    uint32_t k;

    if (regs == NULL) {
        return;
    }

    precision = (regs->config >> 24) & 0xffu;
    m = (regs->config >> 16) & 0xffu;
    n = (regs->config >> 8) & 0xffu;
    k = regs->config & 0xffu;

    fprintf(out, "CTRL   = 0x%08" PRIx32 " (enable=%u sparse=%u soft_reset=%u)\n",
            regs->ctrl,
            (regs->ctrl & TPU_VANILLA_CTRL_ENABLE) != 0u,
            (regs->ctrl & TPU_VANILLA_CTRL_SPARSE) != 0u,
            (regs->ctrl & TPU_VANILLA_CTRL_SOFT_RESET) != 0u);
    fprintf(out, "STATUS = 0x%08" PRIx32 " (ready=%u load=%u compute=%u store=%u done=%u reset_active=%u)\n",
            regs->status,
            (regs->status & TPU_VANILLA_STATUS_READY) != 0u,
            (regs->status & TPU_VANILLA_STATUS_LOAD_BUSY) != 0u,
            (regs->status & TPU_VANILLA_STATUS_COMPUTE_BUSY) != 0u,
            (regs->status & TPU_VANILLA_STATUS_STORE_BUSY) != 0u,
            (regs->status & TPU_VANILLA_STATUS_DONE) != 0u,
            (regs->status & TPU_VANILLA_STATUS_SOFT_RESET_ACTIVE) != 0u);
    fprintf(out,
            "         dbg(a_wr=%u b_wr=%u c_wr=%u a_rd=%u b_rd=%u c_wr_ready=%u d_wr_ready=%u d_rd=%u)\n",
            (regs->status & TPU_VANILLA_STATUS_DBG_A_WR_DONE) != 0u,
            (regs->status & TPU_VANILLA_STATUS_DBG_B_WR_DONE) != 0u,
            (regs->status & TPU_VANILLA_STATUS_DBG_C_WR_DONE) != 0u,
            (regs->status & TPU_VANILLA_STATUS_DBG_A_RD_READY) != 0u,
            (regs->status & TPU_VANILLA_STATUS_DBG_B_RD_READY) != 0u,
            (regs->status & TPU_VANILLA_STATUS_DBG_C_WR_READY) != 0u,
            (regs->status & TPU_VANILLA_STATUS_DBG_D_WR_READY) != 0u,
            (regs->status & TPU_VANILLA_STATUS_DBG_D_RD_READY) != 0u);
    fprintf(out, "CONFIG = 0x%08" PRIx32 " (precision=%" PRIu32 " M=%" PRIu32 " N=%" PRIu32 " K=%" PRIu32 ")\n",
            regs->config, precision, m, n, k);
    dump_packed_cfg(out, "CFG1", regs->dbg_cfg1);
    dump_packed_cfg(out, "CFG2", regs->dbg_cfg2);
    dump_packed_cfg(out, "CFG3", regs->dbg_cfg3);
    fprintf(out, "DBG    = 0x%08" PRIx32 " (a_seen=%u b_seen=%u c_seen=%u) last_awaddr=0x%08" PRIx32 "\n",
            regs->dbg_done,
            (regs->dbg_done & (1u << 0)) != 0u,
            (regs->dbg_done & (1u << 1)) != 0u,
            (regs->dbg_done & (1u << 2)) != 0u,
            regs->dbg_last_addr);
    fprintf(out,
            "FLOW   = 0x%08" PRIx32 " (rctf_done=%u systolic_done=%u accu_fifo_wr=%u fifo_c_rd=%u ram_d_wr=%u compute_done=%u transfer_done=%u systolic_valid=%u first_data=%u last_data=%u systolic_wr=%u accu_output_done=%u)\n",
            regs->dbg_flow,
            (regs->dbg_flow & (1u << 0)) != 0u,
            (regs->dbg_flow & (1u << 1)) != 0u,
            (regs->dbg_flow & (1u << 2)) != 0u,
            (regs->dbg_flow & (1u << 3)) != 0u,
            (regs->dbg_flow & (1u << 4)) != 0u,
            (regs->dbg_flow & (1u << 5)) != 0u,
            (regs->dbg_flow & (1u << 6)) != 0u,
            (regs->dbg_flow & (1u << 7)) != 0u,
            (regs->dbg_flow & (1u << 8)) != 0u,
            (regs->dbg_flow & (1u << 9)) != 0u,
            (regs->dbg_flow & (1u << 10)) != 0u,
            (regs->dbg_flow & (1u << 11)) != 0u);
    fprintf(out,
            "INGRESS= 0x%08" PRIx32 " (a_rd=%u a_wr=%u b_rd=%u b_wr=%u c_wr=%u d_wr=%u map_a=%u map_b=%u map_c=%u axi_wr_ready=%u ab_safe=%u bc_safe=%u compute_pending=%u store_pending=%u compute_can_go=%u store_can_go=%u)\n",
            regs->dbg_ingress,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_A_RD_READY) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_A_WR_READY) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_B_RD_READY) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_B_WR_READY) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_C_WR_READY) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_D_WR_READY) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_MAPPER_A_ACTIVE) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_MAPPER_B_ACTIVE) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_MAPPER_C_ACTIVE) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_AXI_WR_READY) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_AB_SAFE) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_BC_SAFE) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_COMPUTE_PENDING) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_STORE_PENDING) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_COMPUTE_CAN_GO) != 0u,
            (regs->dbg_ingress & TPU_VANILLA_DBG_INGRESS_STORE_CAN_GO) != 0u);
    fprintf(out,
            "COUNT  = 0x%08" PRIx32 " (accu_fifo_wr=%" PRIu32 " fifo_c_rd=%" PRIu32 " ram_d_wr=%" PRIu32 " last_ram_d_addr=%" PRIu32 ")\n",
            regs->dbg_counts,
            regs->dbg_counts & 0xffu,
            (regs->dbg_counts >> 8) & 0xffu,
            (regs->dbg_counts >> 16) & 0xffu,
            (regs->dbg_counts >> 24) & 0xffu);
    fprintf(out,
            "COUNT2 = 0x%08" PRIx32 " (systolic_valid=%" PRIu32 " first_data=%" PRIu32 " last_data=%" PRIu32 " systolic_wr=%" PRIu32 ")\n",
            regs->dbg_counts2,
            regs->dbg_counts2 & 0xffu,
            (regs->dbg_counts2 >> 8) & 0xffu,
            (regs->dbg_counts2 >> 16) & 0xffu,
            (regs->dbg_counts2 >> 24) & 0xffu);
    fprintf(out,
            "COUNT3 = 0x%08" PRIx32 " (accu_fifo_wr_full=%" PRIu32 " fifo_c_rd_full=%" PRIu32 ")\n",
            regs->dbg_counts3,
            regs->dbg_counts3 & 0xffffu,
            (regs->dbg_counts3 >> 16) & 0xffffu);
    fprintf(out,
            "COUNT4 = 0x%08" PRIx32 " (ram_d_wr_full=%" PRIu32 " systolic_valid_full=%" PRIu32 ")\n",
            regs->dbg_counts4,
            regs->dbg_counts4 & 0xffffu,
            (regs->dbg_counts4 >> 16) & 0xffffu);
    fprintf(out,
            "COUNT5 = 0x%08" PRIx32 " (systolic_wr_full=%" PRIu32 " first_data_cnt=%" PRIu32 " last_data_cnt=%" PRIu32 ")\n",
            regs->dbg_counts5,
            regs->dbg_counts5 & 0xffffu,
            (regs->dbg_counts5 >> 16) & 0xffu,
            (regs->dbg_counts5 >> 24) & 0xffu);
    fprintf(out,
            "COUNT6 = 0x%08" PRIx32 " (systolic_valid_or_full=%" PRIu32 " accu_demux_wr_full=%" PRIu32 ")\n",
            regs->dbg_counts6,
            regs->dbg_counts6 & 0xffffu,
            (regs->dbg_counts6 >> 16) & 0xffffu);
    dump_axi_triplet(out, "AXIAW", regs->dbg_axi_aw);
    dump_axi_triplet(out, "AXIAB", regs->dbg_axi_aw_beats);
    dump_axi_triplet(out, "AXIW", regs->dbg_axi_w);
    dump_axi_triplet(out, "AXIWL", regs->dbg_axi_wlast);
    dump_axi_triplet(out, "AXIRAM", regs->dbg_axi_ram_wr);
    dump_axi_triplet(out, "AXISB", regs->dbg_axi_short_burst);
    dump_axi_triplet(out, "AXISM", regs->dbg_axi_short_miss);
    dump_axi_pair(out, "AXITOT", regs->dbg_axi_totals);
    dump_axi_pair(out, "AXINON", regs->dbg_axi_none);
    dump_axi_last_dbg(out, regs->dbg_axi_last);
    fprintf(out, "DADDR  = 0x%08" PRIx32 "\n", regs->store_target_addr);
    fprintf(out,
            "ACC    = first=0x%08" PRIx32 " last=0x%08" PRIx32 "\n",
            regs->dbg_acc_first,
            regs->dbg_acc_last);
    dump_acc_ctrl_dbg(out, regs->dbg_acc_ctrl0, regs->dbg_acc_ctrl1, regs->dbg_acc_ctrl2);
    dump_acc_pipe_dbg(out, regs->dbg_acc_pipe0, regs->dbg_acc_pipe1, regs->dbg_acc_pipe2);
}

int tpu_soc_fp32_preflight(tpu_soc_fp32_t *ctx, tpu_soc_fp32_regs_t *regs_out) {
    tpu_soc_fp32_regs_t regs;

    if (tpu_soc_fp32_read_regs(ctx, &regs) != 0) {
        return -1;
    }
    if ((regs.status & TPU_VANILLA_STATUS_READY) == 0u) {
        return failf(EBUSY, "TPU is not ready (STATUS=0x%08" PRIx32 ")", regs.status);
    }

    if (regs_out != NULL) {
        *regs_out = regs;
    }
    return 0;
}

static int soft_reset_impl(tpu_soc_fp32_t *ctx,
                           unsigned timeout_ms,
                           int has_hold_override,
                           uint32_t hold_cycles,
                           int quiet_timeout,
                           int *saw_active_out,
                           uint32_t *last_status_out) {
    uint64_t deadline;
    uint32_t status = 0u;
    int saw_active = 0;
    unsigned spin_budget = 4096u;
    const uint32_t busy_mask = TPU_VANILLA_STATUS_LOAD_BUSY |
                               TPU_VANILLA_STATUS_COMPUTE_BUSY |
                               TPU_VANILLA_STATUS_STORE_BUSY |
                               TPU_VANILLA_STATUS_DONE;

    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (env_is_truthy(getenv("TPU_DISABLE_SOFT_RESET"))) {
        return 0;
    }
    if (timeout_ms == 0u) {
        timeout_ms = 10000u;
    }

    if (has_hold_override) {
        if (set_soft_reset_hold_cycles(ctx, hold_cycles) != 0) {
            return -1;
        }
    } else {
        if (configure_soft_reset_cycles_from_env(ctx) != 0) {
            return -1;
        }
    }

    if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_CTRL, TPU_VANILLA_CTRL_SOFT_RESET) != 0) {
        return -1;
    }

    deadline = monotonic_ms() + (uint64_t)timeout_ms;
    for (;;) {
        if (axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_STATUS, &status) != 0) {
            return -1;
        }
        if ((status & TPU_VANILLA_STATUS_SOFT_RESET_ACTIVE) != 0u) {
            saw_active = 1;
        }
        if (saw_active &&
            (status & TPU_VANILLA_STATUS_SOFT_RESET_ACTIVE) == 0u &&
            (status & busy_mask) == 0u &&
            (status & TPU_VANILLA_STATUS_READY) != 0u) {
            if (saw_active_out != NULL) {
                *saw_active_out = saw_active;
            }
            if (last_status_out != NULL) {
                *last_status_out = status;
            }
            return 0;
        }
        if (monotonic_ms() > deadline) {
            if (saw_active_out != NULL) {
                *saw_active_out = saw_active;
            }
            if (last_status_out != NULL) {
                *last_status_out = status;
            }
            if (quiet_timeout) {
                errno = ETIMEDOUT;
                return -1;
            }
            return failf(ETIMEDOUT,
                         "timeout waiting for TPU soft reset completion (saw_active=%d last_status=0x%08x)",
                         saw_active,
                         status);
        }
        if (spin_budget != 0u) {
            --spin_budget;
        } else {
            sleep_us(50u);
        }
    }
}

int tpu_soc_fp32_soft_reset(tpu_soc_fp32_t *ctx, unsigned timeout_ms) {
    return soft_reset_impl(ctx, timeout_ms, 0, 0u, 0, NULL, NULL);
}

int tpu_soc_fp32_soft_reset_with_hold_cycles(tpu_soc_fp32_t *ctx,
                                             unsigned timeout_ms,
                                             uint32_t hold_cycles) {
    return soft_reset_impl(ctx, timeout_ms, 1, hold_cycles, 0, NULL, NULL);
}

int tpu_soc_fp32_soft_reset_with_hold_cycles_probe(tpu_soc_fp32_t *ctx,
                                                   unsigned timeout_ms,
                                                   uint32_t hold_cycles,
                                                   int *saw_active_out,
                                                   uint32_t *last_status_out) {
    return soft_reset_impl(ctx,
                           timeout_ms,
                           1,
                           hold_cycles,
                           1,
                           saw_active_out,
                           last_status_out);
}

int tpu_soc_fp32_cdma_reset(tpu_soc_fp32_t *ctx) {
    return tpu_soc_fp32_cdma_reset_with_trace(ctx, NULL);
}

int tpu_soc_fp32_cdma_reset_with_trace(tpu_soc_fp32_t *ctx, tpu_soc_fp32_cdma_trace_t *trace) {
    tpu_soc_fp32_cdma_reset_log_t *reset_log = NULL;

    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }

    if (trace != NULL) {
        reset_log = &trace->reset;
        memset(reset_log, 0, sizeof(*reset_log));
        reset_log->attempted = 1;
        if (cdma_read_cr_sr(ctx, &reset_log->cr_before, &reset_log->sr_before) != 0) {
            reset_log->rc = -1;
            reset_log->err = errno;
            return -1;
        }
    }

    if (axil_mmio_write32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_CR, TPU_SOC_FP32_CDMA_CR_RESET) != 0 ||
        axil_mmio_write32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_CR, 0u) != 0 ||
        axil_mmio_write32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_SA_MSB, 0u) != 0 ||
        axil_mmio_write32(&ctx->cdma_mmio, TPU_SOC_FP32_CDMA_DA_MSB, 0u) != 0) {
        if (reset_log != NULL) {
            reset_log->rc = -1;
            reset_log->err = errno;
            (void)cdma_read_cr_sr(ctx, &reset_log->cr_after, &reset_log->sr_after);
        }
        return -1;
    }

    if (reset_log != NULL) {
        reset_log->rc = 0;
        reset_log->err = 0;
        (void)cdma_read_cr_sr(ctx, &reset_log->cr_after, &reset_log->sr_after);
    }
    return 0;
}

static int execute_precision_mode(tpu_soc_fp32_t *ctx,
                                  const tpu_soc_fp32_buffer_t *a_buf,
                                  const tpu_soc_fp32_buffer_t *b_buf,
                                  const tpu_soc_fp32_buffer_t *c_buf,
                                  uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  uint32_t precision_mode,
                                  size_t a_elem_bytes,
                                  size_t b_elem_bytes,
                                  size_t c_elem_bytes,
                                  size_t d_elem_bytes,
                                  int run_tail_fixups,
                                  int sparse,
                                  unsigned timeout_ms) {
    size_t a_bytes = 0u;
    size_t b_bytes = 0u;
    size_t c_bytes = 0u;
    size_t d_bytes = 0u;
    uint32_t config_word;
    uint32_t ctrl_word;
    unsigned post_start_delay_ms;
    unsigned between_cdma_delay_ms;
    unsigned post_cdma_delay_ms;

    if (ctx == NULL || a_buf == NULL || b_buf == NULL || c_buf == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (a_buf->mapping == NULL || b_buf->mapping == NULL || c_buf->mapping == NULL) {
        return failf(EINVAL, "all XRT buffers must be allocated before execute");
    }
    if (timeout_ms == 0u) {
        timeout_ms = 10000u;
    }
    tpu_soc_fp32_cdma_trace_clear(&ctx->last_cdma_trace);

    if (validate_matrix_sizes_with_bytes(m,
                                         n,
                                         k,
                                         a_elem_bytes,
                                         b_elem_bytes,
                                         c_elem_bytes,
                                         d_elem_bytes,
                                         &a_bytes,
                                         &b_bytes,
                                         &c_bytes,
                                         &d_bytes) != 0) {
        return -1;
    }
    (void)d_bytes;

    if (validate_dma_source_addr(a_buf->device_addr, a_bytes) != 0 ||
        validate_dma_source_addr(b_buf->device_addr, b_bytes) != 0 ||
        validate_dma_source_addr(c_buf->device_addr, c_bytes) != 0) {
        return -1;
    }

    if (tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)a_buf, a_bytes) != 0 ||
        tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)b_buf, b_bytes) != 0 ||
        tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)c_buf, c_bytes) != 0) {
        return -1;
    }

    if (tpu_soc_fp32_soft_reset(ctx, timeout_ms) != 0) {
        return -1;
    }

    if (tpu_soc_fp32_cdma_reset_with_trace(ctx, &ctx->last_cdma_trace) != 0) {
        return -1;
    }

    if (wait_status_mask(ctx, TPU_VANILLA_STATUS_READY, 0u, timeout_ms, NULL) != 0) {
        return -1;
    }

    config_word = (precision_mode << 24) | (m << 16) | (n << 8) | k;
    ctrl_word = sparse ? (TPU_VANILLA_CTRL_ENABLE | TPU_VANILLA_CTRL_SPARSE) : TPU_VANILLA_CTRL_ENABLE;
    if (program_config_then_start(ctx, config_word, ctrl_word, timeout_ms) != 0) {
        return -1;
    }

    /*
     * v2 is sensitive to the first few cycles after CTRL enable on real hardware,
     * especially for small-M / larger-K shapes. Keep a conservative 1 ms default
     * unless the caller explicitly overrides it.
     */
    post_start_delay_ms = env_u32_or_default("TPU_POST_START_DELAY_MS", 1u);
    between_cdma_delay_ms = env_u32_or_default("TPU_BETWEEN_CDMA_DELAY_MS", 0u);
    post_cdma_delay_ms = env_u32_or_default("TPU_POST_CDMA_DELAY_MS", 0u);

    if (post_start_delay_ms != 0u) {
        fprintf(stderr, "runtime delay: sleeping %u ms after CTRL enable before CDMA\n", post_start_delay_ms);
        sleep_ms(post_start_delay_ms);
    }

    if (tpu_soc_fp32_cdma_transfer_with_trace(ctx, &ctx->last_cdma_trace, "A-main", a_buf->device_addr, TPU_SOC_FP32_CDMA_MATRIX_A, a_bytes, timeout_ms) != 0) {
        return -1;
    }
    if (between_cdma_delay_ms != 0u) {
        fprintf(stderr, "runtime delay: sleeping %u ms between CDMA transfers\n", between_cdma_delay_ms);
        sleep_ms(between_cdma_delay_ms);
    }
    if (tpu_soc_fp32_cdma_transfer_with_trace(ctx, &ctx->last_cdma_trace, "B-main", b_buf->device_addr, TPU_SOC_FP32_CDMA_MATRIX_B, b_bytes, timeout_ms) != 0) {
        return -1;
    }
    if (between_cdma_delay_ms != 0u) {
        fprintf(stderr, "runtime delay: sleeping %u ms between CDMA transfers\n", between_cdma_delay_ms);
        sleep_ms(between_cdma_delay_ms);
    }
    if (tpu_soc_fp32_cdma_transfer_with_trace(ctx, &ctx->last_cdma_trace, "C-main", c_buf->device_addr, TPU_SOC_FP32_CDMA_MATRIX_C, c_bytes, timeout_ms) != 0) {
        return -1;
    }

    if (run_tail_fixups) {
        if (tpu_soc_fp32_cdma_tail_fixup_with_trace(ctx, &ctx->last_cdma_trace, "A-tail", a_buf->device_addr, TPU_SOC_FP32_CDMA_MATRIX_A, a_bytes, timeout_ms) != 0 ||
            tpu_soc_fp32_cdma_tail_fixup_with_trace(ctx, &ctx->last_cdma_trace, "B-tail", b_buf->device_addr, TPU_SOC_FP32_CDMA_MATRIX_B, b_bytes, timeout_ms) != 0 ||
            tpu_soc_fp32_cdma_tail_fixup_with_trace(ctx, &ctx->last_cdma_trace, "C-tail", c_buf->device_addr, TPU_SOC_FP32_CDMA_MATRIX_C, c_bytes, timeout_ms) != 0) {
            return -1;
        }
    }

    if (post_cdma_delay_ms != 0u) {
        fprintf(stderr, "runtime delay: sleeping %u ms after final CDMA transfer\n", post_cdma_delay_ms);
        sleep_ms(post_cdma_delay_ms);
    }

    if (wait_status_mask(ctx, TPU_VANILLA_STATUS_DONE, 0u, timeout_ms, NULL) != 0) {
        return -1;
    }
    if (wait_status_mask(ctx, 0u, TPU_VANILLA_STATUS_STORE_BUSY, timeout_ms, NULL) != 0) {
        return -1;
    }

    return 0;
}

int tpu_soc_fp32_execute(tpu_soc_fp32_t *ctx,
                         const tpu_soc_fp32_buffer_t *a_buf,
                         const tpu_soc_fp32_buffer_t *b_buf,
                         const tpu_soc_fp32_buffer_t *c_buf,
                         uint32_t m,
                         uint32_t n,
                         uint32_t k,
                         int sparse,
                         unsigned timeout_ms) {
    return execute_precision_mode(ctx,
                                  a_buf,
                                  b_buf,
                                  c_buf,
                                  m,
                                  n,
                                  k,
                                  TPU_VANILLA_PM_FP32_ALL,
                                  sizeof(float),
                                  sizeof(float),
                                  sizeof(float),
                                  sizeof(float),
                                  1,
                                  sparse,
                                  timeout_ms);
}

int tpu_soc_fp16_execute(tpu_soc_fp32_t *ctx,
                         const tpu_soc_fp32_buffer_t *a_buf,
                         const tpu_soc_fp32_buffer_t *b_buf,
                         const tpu_soc_fp32_buffer_t *c_buf,
                         uint32_t m,
                         uint32_t n,
                         uint32_t k,
                         int sparse,
                         unsigned timeout_ms) {
    return execute_precision_mode(ctx,
                                  a_buf,
                                  b_buf,
                                  c_buf,
                                  m,
                                  n,
                                  k,
                                  TPU_VANILLA_PM_FP16_ALL,
                                  2u,
                                  2u,
                                  2u,
                                  2u,
                                  0,
                                  sparse,
                                  timeout_ms);
}

int tpu_soc_fp16_mix_execute(tpu_soc_fp32_t *ctx,
                             const tpu_soc_fp32_buffer_t *a_buf,
                             const tpu_soc_fp32_buffer_t *b_buf,
                             const tpu_soc_fp32_buffer_t *c_buf,
                             uint32_t m,
                             uint32_t n,
                             uint32_t k,
                             int sparse,
                             unsigned timeout_ms) {
    return execute_precision_mode(ctx,
                                  a_buf,
                                  b_buf,
                                  c_buf,
                                  m,
                                  n,
                                  k,
                                  TPU_VANILLA_PM_FP16_MIX,
                                  2u,
                                  2u,
                                  sizeof(float),
                                  sizeof(float),
                                  0,
                                  sparse,
                                  timeout_ms);
}

int tpu_soc_fp32_read_result(tpu_soc_fp32_t *ctx, float *dst, size_t element_count) {
    size_t i;
    union {
        uint32_t u32;
        float f32;
    } word;

    if (ctx == NULL || dst == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (element_count * sizeof(float) > TPU_SOC_FP32_BRAM_SPAN) {
        return failf(ERANGE, "requested result read exceeds BRAM span");
    }

    for (i = 0; i < element_count; ++i) {
        if (axil_mmio_read32(&ctx->bram_mmio, (uint32_t)(i * sizeof(uint32_t)), &word.u32) != 0) {
            return -1;
        }
        dst[i] = word.f32;
    }
    return 0;
}

int tpu_soc_fp32_read_result_words64(tpu_soc_fp32_t *ctx, uint64_t *dst, size_t word_count) {
    size_t i;

    if (ctx == NULL || dst == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (word_count * sizeof(uint64_t) > TPU_SOC_FP32_BRAM_SPAN) {
        return failf(ERANGE, "requested raw result read exceeds BRAM span");
    }

    for (i = 0; i < word_count; ++i) {
        if (axil_mmio_read64(&ctx->bram_mmio, (uint32_t)(i * sizeof(uint64_t)), &dst[i]) != 0) {
            return -1;
        }
    }
    return 0;
}
