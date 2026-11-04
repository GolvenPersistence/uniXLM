#define _POSIX_C_SOURCE 200809L

#include "tpu_soc_v2_int8.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <strings.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
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

static int execute_report_finish(tpu_soc_v2_execute_report_t *report,
                                 uint64_t execute_entry_ns,
                                 int rc) {
    if (report != NULL) {
        report->execute_total_ns = monotonic_ns() - execute_entry_ns;
    }
    return rc;
}

static int env_is_truthy(const char *value);
static uint32_t hw_load_mask_default(void);
static unsigned post_start_delay_us_default(void);
static unsigned between_cdma_delay_us_ab_default(void);
static unsigned between_cdma_delay_us_bc_default(void);
static unsigned post_cdma_delay_us_default(void);

static int g_prev_d_shape_valid = 0;
static uint32_t g_prev_d_m = 0u;
static uint32_t g_prev_d_n = 0u;

static void cdma_trace_lazy_reset(tpu_soc_fp32_cdma_trace_t *trace) {
    if (trace == NULL) {
        return;
    }
    memset(&trace->reset, 0, sizeof(trace->reset));
    trace->transfer_count = 0u;
    trace->first_failure_index = -1;
}

static void ensure_v2_exec_cfg_cached(tpu_soc_fp32_t *ctx) {
    if (ctx == NULL || ctx->v2_exec_cfg_valid) {
        return;
    }

    ctx->v2_trace_cdma_path = env_is_truthy(getenv("TPU_TRACE_CDMA_PATH"));
    ctx->v2_tail_fixup = env_is_truthy(getenv("TPU_ENABLE_CDMA_TAIL_FIXUP"));
    ctx->v2_use_sg = env_is_truthy(getenv("TPU_CDMA_USE_SG"));
    ctx->v2_hw_ingress = env_is_truthy(getenv("TPU_HW_INGRESS"));
    ctx->v2_gemv_m1 = env_is_truthy(getenv("TPU_GEMV_M1"));
    ctx->v2_transfer_only = env_is_truthy(getenv("TPU_V2_TRANSFER_ONLY"));
    ctx->v2_hw_load_mask = hw_load_mask_default();
    ctx->v2_post_start_delay_us = post_start_delay_us_default();
    ctx->v2_between_cdma_delay_us_ab = between_cdma_delay_us_ab_default();
    ctx->v2_between_cdma_delay_us_bc = between_cdma_delay_us_bc_default();
    ctx->v2_post_cdma_delay_us = post_cdma_delay_us_default();
    ctx->v2_exec_cfg_valid = 1;
}

static int validate_dma_source_addr_cached(tpu_soc_fp32_t *ctx,
                                           uint64_t addr,
                                           size_t size_bytes) {
    uint64_t end_addr;

    if (ctx == NULL) {
        return tpu_soc_fp32_validate_input_ddr_addr(addr, size_bytes);
    }
    if (size_bytes == 0u) {
        return failf(EINVAL, "input buffer transfer size must be non-zero");
    }
    if (addr > UINT32_MAX) {
        return failf(ERANGE,
                     "input buffer device address 0x%016" PRIx64 " exceeds 32-bit AXI aperture",
                     addr);
    }

    end_addr = addr + size_bytes - 1u;
    if (end_addr < addr) {
        return failf(ERANGE,
                     "input buffer device range starting at 0x%016" PRIx64 " overflows",
                     addr);
    }
    if ((addr < ctx->input_ddr_base || end_addr > ctx->input_ddr_high) &&
        (!ctx->input_ddr_high_window_valid ||
         addr < ctx->input_ddr_high_window_base ||
         end_addr > ctx->input_ddr_high_window_high)) {
        if (ctx->input_ddr_high_window_valid) {
            return failf(ERANGE,
                         "input buffer device range [0x%08" PRIx64 ", 0x%08" PRIx64 "] is outside DDR apertures [0x%08x, 0x%08x] and [0x%08x, 0x%08x]",
                         addr,
                         end_addr,
                         ctx->input_ddr_base,
                         ctx->input_ddr_high,
                         ctx->input_ddr_high_window_base,
                         ctx->input_ddr_high_window_high);
        }
        return failf(ERANGE,
                     "input buffer device range [0x%08" PRIx64 ", 0x%08" PRIx64 "] is outside DDR aperture [0x%08x, 0x%08x]",
                     addr,
                     end_addr,
                     ctx->input_ddr_base,
                     ctx->input_ddr_high);
    }
    return 0;
}

static int resolve_v2_sizes_cached(tpu_soc_fp32_t *ctx,
                                   uint32_t m,
                                   uint32_t n,
                                   uint32_t k,
                                   size_t *a_bytes,
                                   size_t *b_bytes,
                                   size_t *c_bytes,
                                   size_t *d_bytes) {
    if (ctx != NULL &&
        ctx->v2_shape_cache_valid &&
        ctx->v2_cached_m == m &&
        ctx->v2_cached_n == n &&
        ctx->v2_cached_k == k) {
        if (a_bytes != NULL) {
            *a_bytes = ctx->v2_cached_a_bytes;
        }
        if (b_bytes != NULL) {
            *b_bytes = ctx->v2_cached_b_bytes;
        }
        if (c_bytes != NULL) {
            *c_bytes = ctx->v2_cached_c_bytes;
        }
        if (d_bytes != NULL) {
            *d_bytes = ctx->v2_cached_d_bytes;
        }
        return 0;
    }

    if (tpu_soc_v2_validate_sizes(m, n, k, a_bytes, b_bytes, c_bytes, d_bytes) != 0) {
        return -1;
    }

    if (ctx != NULL) {
        ctx->v2_shape_cache_valid = 1;
        ctx->v2_cached_m = m;
        ctx->v2_cached_n = n;
        ctx->v2_cached_k = k;
        ctx->v2_cached_a_bytes = (a_bytes != NULL) ? *a_bytes : 0u;
        ctx->v2_cached_b_bytes = (b_bytes != NULL) ? *b_bytes : 0u;
        ctx->v2_cached_c_bytes = (c_bytes != NULL) ? *c_bytes : 0u;
        ctx->v2_cached_d_bytes = (d_bytes != NULL) ? *d_bytes : 0u;
    }
    return 0;
}

static int ensure_cached_input_addr(tpu_soc_fp32_t *ctx,
                                    int *valid_flag,
                                    uint64_t *cached_addr,
                                    size_t *cached_size,
                                    uint64_t addr,
                                    size_t size_bytes) {
    if (ctx != NULL &&
        valid_flag != NULL &&
        cached_addr != NULL &&
        cached_size != NULL &&
        *valid_flag &&
        *cached_addr == addr &&
        *cached_size == size_bytes) {
        return 0;
    }

    if (validate_dma_source_addr_cached(ctx, addr, size_bytes) != 0) {
        return -1;
    }

    if (ctx != NULL && valid_flag != NULL && cached_addr != NULL && cached_size != NULL) {
        *valid_flag = 1;
        *cached_addr = addr;
        *cached_size = size_bytes;
    }
    return 0;
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

static uint32_t ceil_div_u32(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1u) / divisor;
}

static int is_power_of_two_u32(uint32_t value) {
    return value != 0u && ((value & (value - 1u)) == 0u);
}

static int env_is_truthy(const char *value) {
    if (value == NULL || *value == '\0') {
        return 0;
    }
    return strcmp(value, "0") != 0;
}

static int trace_cdma_path_enabled(void) {
    return env_is_truthy(getenv("TPU_TRACE_CDMA_PATH"));
}

static int hw_ingress_enabled(void) {
    return env_is_truthy(getenv("TPU_HW_INGRESS"));
}

static int gemv_m1_enabled(void) {
    return env_is_truthy(getenv("TPU_GEMV_M1"));
}

static uint32_t hw_load_mask_default(void) {
    const char *value = getenv("TPU_HW_LOAD_MASK");
    uint32_t mask = TPU_VANILLA_LOAD_CTRL_MASK_ALL;

    if (value == NULL || *value == '\0') {
        return mask;
    }
    if (strcasecmp(value, "a") == 0) {
        return TPU_VANILLA_LOAD_CTRL_MASK_A;
    }
    if (strcasecmp(value, "b") == 0) {
        return TPU_VANILLA_LOAD_CTRL_MASK_B;
    }
    if (strcasecmp(value, "c") == 0) {
        return TPU_VANILLA_LOAD_CTRL_MASK_C;
    }
    if (strcasecmp(value, "ab") == 0 || strcasecmp(value, "ba") == 0) {
        return TPU_VANILLA_LOAD_CTRL_MASK_A | TPU_VANILLA_LOAD_CTRL_MASK_B;
    }
    if (strcasecmp(value, "ac") == 0 || strcasecmp(value, "ca") == 0) {
        return TPU_VANILLA_LOAD_CTRL_MASK_A | TPU_VANILLA_LOAD_CTRL_MASK_C;
    }
    if (strcasecmp(value, "bc") == 0 || strcasecmp(value, "cb") == 0) {
        return TPU_VANILLA_LOAD_CTRL_MASK_B | TPU_VANILLA_LOAD_CTRL_MASK_C;
    }
    if (strcasecmp(value, "abc") == 0 || strcasecmp(value, "all") == 0) {
        return TPU_VANILLA_LOAD_CTRL_MASK_ALL;
    }

    {
        char *endptr = NULL;
        errno = 0;
        unsigned long parsed = strtoul(value, &endptr, 0);
        if (errno == 0 && endptr != value && endptr != NULL && *endptr == '\0') {
            mask = ((uint32_t)parsed << 1) & TPU_VANILLA_LOAD_CTRL_MASK_ALL;
            if (mask != 0u) {
                return mask;
            }
        }
    }

    return TPU_VANILLA_LOAD_CTRL_MASK_ALL;
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

static unsigned poll_spin_budget_default(void) {
    return env_u32_or_default("TPU_POLL_SPIN_BUDGET", 4096u);
}

static unsigned poll_sleep_us_default(void) {
    return env_u32_or_default("TPU_POLL_SLEEP_US", 1u);
}

static unsigned config_to_ctrl_delay_us_default(void) {
    return env_u32_or_default("TPU_CONFIG_TO_CTRL_DELAY_US", 0u);
}

static unsigned post_start_delay_us_default(void) {
    return env_u32_or_default("TPU_V2_POST_START_DELAY_US", 0u);
}

static unsigned ctrl_deassert_delay_us_default(void) {
    return env_u32_or_default("TPU_V2_CTRL_DEASSERT_DELAY_US", 1u);
}

static unsigned between_cdma_delay_us_default(void) {
    return env_u32_or_default("TPU_V2_BETWEEN_CDMA_DELAY_US", 44u);
}

static unsigned between_cdma_delay_us_ab_default(void) {
    return env_u32_or_default("TPU_V2_BETWEEN_CDMA_DELAY_US_AB",
                              between_cdma_delay_us_default());
}

static unsigned between_cdma_delay_us_bc_default(void) {
    return env_u32_or_default("TPU_V2_BETWEEN_CDMA_DELAY_US_BC",
                              between_cdma_delay_us_default());
}

static unsigned post_cdma_delay_us_default(void) {
    return env_u32_or_default("TPU_V2_POST_CDMA_DELAY_US", 0u);
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

static int tail_fixup_enabled(void) {
    return env_is_truthy(getenv("TPU_ENABLE_CDMA_TAIL_FIXUP"));
}

static int output_read_mode_is_memcpy(void) {
    const char *mode = getenv("TPU_V2_OUTPUT_READ_MODE");

    if (mode == NULL || *mode == '\0') {
        return 1;
    }
    if (strcasecmp(mode, "unpack64") == 0 || strcasecmp(mode, "legacy") == 0) {
        return 0;
    }
    return 1;
}

static void read_result_report_init(tpu_soc_v2_read_result_report_t *report,
                                    uint32_t elem_count,
                                    uint32_t word_count) {
    if (report == NULL) {
        return;
    }

    memset(report, 0, sizeof(*report));
    report->elem_count = elem_count;
    report->word_count = word_count;
    report->bytes = (uint64_t) word_count * sizeof(uint64_t);
}

static int trace_ingress_gaps_enabled(void) {
    return env_is_truthy(getenv("TPU_V2_TRACE_INGRESS_GAPS"));
}

static int wait_ingress_safe_enabled(void) {
    return env_is_truthy(getenv("TPU_V2_WAIT_INGRESS_SAFE"));
}

static int skip_clear_if_outbuf_enabled(void) {
    return env_is_truthy(getenv("TPU_V2_SKIP_CLEAR_IF_OUTBUF"));
}

static int transfer_only_enabled(void) {
    return env_is_truthy(getenv("TPU_V2_TRANSFER_ONLY"));
}

static int read_dbg_ingress(tpu_soc_fp32_t *ctx, uint32_t *ingress_out) {
    if (ctx == NULL || ingress_out == NULL) {
        errno = EINVAL;
        return -1;
    }
    return axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_DBG_INGRESS, ingress_out);
}

static unsigned preload_tail_quiet_us_default(void) {
    return env_u32_or_default("TPU_V2_PRELOAD_TAIL_QUIET_US", 0u);
}

static int wait_hw_ingress_tail_quiet(tpu_soc_fp32_t *ctx,
                                      unsigned quiet_us,
                                      uint32_t *last_load_status,
                                      uint32_t *last_ingress_status) {
    uint64_t deadline_ns;
    uint32_t load_status = 0u;
    uint32_t ingress = 0u;
    unsigned spin_budget;
    unsigned sleep_us_value;

    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (quiet_us == 0u) {
        if (last_load_status != NULL) {
            *last_load_status = 0u;
        }
        if (last_ingress_status != NULL) {
            *last_ingress_status = 0u;
        }
        return 0;
    }

    deadline_ns = monotonic_ns() + (uint64_t) quiet_us * 1000ull;
    spin_budget = poll_spin_budget_default();
    sleep_us_value = poll_sleep_us_default();

    for (;;) {
        if (axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_STATUS, &load_status) != 0) {
            return -1;
        }
        if (read_dbg_ingress(ctx, &ingress) != 0) {
            return -1;
        }
        if (last_load_status != NULL) {
            *last_load_status = load_status;
        }
        if (last_ingress_status != NULL) {
            *last_ingress_status = ingress;
        }

        if ((load_status & TPU_VANILLA_LOAD_STATUS_BUSY) == 0u &&
            (ingress & (TPU_VANILLA_DBG_INGRESS_MAPPER_A_ACTIVE |
                        TPU_VANILLA_DBG_INGRESS_MAPPER_B_ACTIVE |
                        TPU_VANILLA_DBG_INGRESS_MAPPER_C_ACTIVE |
                        TPU_VANILLA_DBG_INGRESS_COMPUTE_PENDING |
                        TPU_VANILLA_DBG_INGRESS_STORE_PENDING)) == 0u &&
            (ingress & (TPU_VANILLA_DBG_INGRESS_A_WR_READY |
                        TPU_VANILLA_DBG_INGRESS_B_WR_READY)) ==
                (TPU_VANILLA_DBG_INGRESS_A_WR_READY |
                 TPU_VANILLA_DBG_INGRESS_B_WR_READY)) {
            return 0;
        }

        if (monotonic_ns() > deadline_ns) {
            return failf(ETIMEDOUT,
                         "timeout waiting for hardware ingress tail quiet (LOAD_STATUS=0x%08x INGRESS=0x%08x quiet_us=%u)",
                         load_status,
                         ingress,
                         quiet_us);
        }
        poll_wait_step(&spin_budget, sleep_us_value);
    }
}

static void dump_ingress_bits(FILE *stream, uint32_t ingress) {
    if (stream == NULL) {
        return;
    }
    fprintf(stream,
            "raw=0x%08" PRIx32
            " a_rd=%u a_wr=%u b_rd=%u b_wr=%u c_wr=%u d_wr=%u"
            " mapA=%u mapB=%u mapC=%u axi_wr=%u"
            " ab_safe=%u bc_safe=%u comp_pend=%u store_pend=%u comp_go=%u store_go=%u",
            ingress,
            (ingress & TPU_VANILLA_DBG_INGRESS_A_RD_READY) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_A_WR_READY) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_B_RD_READY) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_B_WR_READY) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_C_WR_READY) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_D_WR_READY) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_MAPPER_A_ACTIVE) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_MAPPER_B_ACTIVE) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_MAPPER_C_ACTIVE) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_AXI_WR_READY) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_AB_SAFE) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_BC_SAFE) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_COMPUTE_PENDING) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_STORE_PENDING) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_COMPUTE_CAN_GO) != 0u,
            (ingress & TPU_VANILLA_DBG_INGRESS_STORE_CAN_GO) != 0u);
}

static void handle_ingress_gap(tpu_soc_fp32_t *ctx,
                               const char *label,
                               unsigned delay_us,
                               uint32_t safe_mask) {
    uint32_t start_ingress = 0u;
    uint32_t first_safe_ingress = 0u;
    uint32_t end_ingress = 0u;
    uint64_t start_ns;
    uint64_t deadline_ns;
    uint64_t first_safe_ns = 0u;
    unsigned poll_count = 0u;
    int saw_start = 0;
    int saw_safe = 0;
    int saw_end = 0;
    int trace_enabled;
    int wait_enabled;

    if (ctx == NULL || delay_us == 0u) {
        return;
    }
    trace_enabled = trace_ingress_gaps_enabled();
    wait_enabled = wait_ingress_safe_enabled();
    if (!trace_enabled && !wait_enabled) {
        sleep_us(delay_us);
        return;
    }

    if (read_dbg_ingress(ctx, &start_ingress) == 0) {
        saw_start = 1;
    }

    start_ns = monotonic_ns();
    deadline_ns = start_ns + (uint64_t)delay_us * 1000ull;
    while (monotonic_ns() < deadline_ns) {
        ++poll_count;
        if (read_dbg_ingress(ctx, &first_safe_ingress) == 0) {
            if (!saw_safe && (first_safe_ingress & safe_mask) != 0u) {
                saw_safe = 1;
                first_safe_ns = monotonic_ns();
                if (wait_enabled) {
                    break;
                }
            }
        }
    }

    if (read_dbg_ingress(ctx, &end_ingress) == 0) {
        saw_end = 1;
    }

    if (trace_enabled) {
        fprintf(stderr,
                "INGRESS_GAP %s delay_us=%u polls=%u mode=%s",
                label != NULL ? label : "?",
                delay_us,
                poll_count,
                wait_enabled ? "wait-safe" : "trace-only");
        if (saw_start) {
            fputs(" start{", stderr);
            dump_ingress_bits(stderr, start_ingress);
            fputc('}', stderr);
        }
        if (saw_safe) {
            fprintf(stderr,
                    " first_safe_us=%.3f safe{",
                    (double)(first_safe_ns - start_ns) / 1000.0);
            dump_ingress_bits(stderr, first_safe_ingress);
            fputc('}', stderr);
        } else {
            fputs(" first_safe_us=none", stderr);
        }
        if (saw_end) {
            fputs(" end{", stderr);
            dump_ingress_bits(stderr, end_ingress);
            fputc('}', stderr);
        }
        fputc('\n', stderr);
    }

}

int tpu_soc_v2_resolve_soft_reset_policy(tpu_soc_v2_soft_reset_policy_t *policy) {
    const char *explicit_cycles_env;
    const char *profile_env;
    char *endptr = NULL;
    unsigned long parsed;
    tpu_soc_v2_soft_reset_policy_t resolved;

    if (policy == NULL) {
        errno = EINVAL;
        return -1;
    }

    explicit_cycles_env = getenv("TPU_SOFT_RESET_HOLD_CYCLES");
    if (explicit_cycles_env != NULL && *explicit_cycles_env != '\0') {
        errno = 0;
        parsed = strtoul(explicit_cycles_env, &endptr, 0);
        if (errno != 0 || endptr == explicit_cycles_env || (endptr != NULL && *endptr != '\0') ||
            parsed > TPU_VANILLA_SOFT_RESET_HOLD_MASK) {
            return failf(EINVAL,
                         "invalid TPU_SOFT_RESET_HOLD_CYCLES=%s (expected 1..%u)",
                         explicit_cycles_env,
                         TPU_VANILLA_SOFT_RESET_HOLD_MASK);
        }
        resolved.hold_cycles = (uint32_t)parsed;
        if (resolved.hold_cycles == 0u) {
            resolved.hold_cycles = 1u;
        }
        resolved.fallback_hold_cycles = 0u;
        resolved.label = "explicit-cycles";
        *policy = resolved;
        return 0;
    }

    profile_env = getenv("TPU_V2_SOFT_RESET_PROFILE");
    if (profile_env == NULL || *profile_env == '\0') {
        resolved.hold_cycles = TPU_SOC_V2_SOFT_RESET_HOLD_CYCLES_FAST;
        resolved.fallback_hold_cycles = TPU_SOC_V2_SOFT_RESET_HOLD_CYCLES_LEGACY;
        resolved.label = "default-fast-fallback";
        *policy = resolved;
        return 0;
    }
    if (strcasecmp(profile_env, "fast") == 0 || strcasecmp(profile_env, "default") == 0) {
        resolved.hold_cycles = TPU_SOC_V2_SOFT_RESET_HOLD_CYCLES_FAST;
        resolved.fallback_hold_cycles = 0u;
        resolved.label = "profile-fast";
        *policy = resolved;
        return 0;
    }
    if (strcasecmp(profile_env, "legacy") == 0) {
        resolved.hold_cycles = TPU_SOC_V2_SOFT_RESET_HOLD_CYCLES_LEGACY;
        resolved.fallback_hold_cycles = 0u;
        resolved.label = "profile-legacy";
        *policy = resolved;
        return 0;
    }

    return failf(EINVAL,
                 "invalid TPU_V2_SOFT_RESET_PROFILE=%s (expected fast or legacy)",
                 profile_env);
}

int tpu_soc_v2_soft_reset(tpu_soc_fp32_t *ctx, unsigned timeout_ms) {
    tpu_soc_v2_soft_reset_policy_t policy;
    uint32_t last_status = 0u;
    int saw_active = 0;
    unsigned probe_timeout_ms;
    const uint32_t quiet_fallback_mask = TPU_VANILLA_STATUS_LOAD_BUSY |
                                         TPU_VANILLA_STATUS_COMPUTE_BUSY |
                                         TPU_VANILLA_STATUS_STORE_BUSY |
                                         TPU_VANILLA_STATUS_DONE |
                                         TPU_VANILLA_STATUS_SOFT_RESET_ACTIVE;

    if (tpu_soc_v2_resolve_soft_reset_policy(&policy) != 0) {
        return -1;
    }

    if (policy.fallback_hold_cycles == 0u) {
        return tpu_soc_fp32_soft_reset_with_hold_cycles(ctx, timeout_ms, policy.hold_cycles);
    }

    probe_timeout_ms = env_u32_or_default("TPU_V2_FAST_RESET_PROBE_TIMEOUT_MS", 1u);
    if (probe_timeout_ms == 0u) {
        probe_timeout_ms = 1u;
    }
    if (timeout_ms != 0u && probe_timeout_ms > timeout_ms) {
        probe_timeout_ms = timeout_ms;
    }

    if (tpu_soc_fp32_soft_reset_with_hold_cycles_probe(ctx,
                                                       probe_timeout_ms,
                                                       policy.hold_cycles,
                                                       &saw_active,
                                                       &last_status) == 0) {
        return 0;
    }

    if (errno == ETIMEDOUT &&
        saw_active == 0 &&
        (last_status & TPU_VANILLA_STATUS_READY) != 0u &&
        (last_status & quiet_fallback_mask) == 0u) {
        fprintf(stderr,
                "default v2 soft reset fell back from %" PRIu32 " to %" PRIu32 " cycles (last_status=0x%08" PRIx32 ")\n",
                policy.hold_cycles,
                policy.fallback_hold_cycles,
                last_status);
        return tpu_soc_fp32_soft_reset_with_hold_cycles(ctx, timeout_ms, policy.fallback_hold_cycles);
    }

    return -1;
}

static int validate_dma_source_addr(uint64_t addr, size_t size_bytes) {
    return tpu_soc_fp32_validate_input_ddr_addr(addr, size_bytes);
}

static int wait_status_mask(tpu_soc_fp32_t *ctx,
                            uint32_t set_mask,
                            uint32_t clear_mask,
                            unsigned timeout_ms,
                            uint32_t *last_status) {
    uint64_t deadline = monotonic_ms() + (uint64_t)timeout_ms;
    uint32_t status = 0u;
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
                         set_mask,
                         clear_mask,
                         status);
        }
        poll_wait_step(&spin_budget, sleep_us_value);
    }
}

static int wait_csr_mask(tpu_soc_fp32_t *ctx,
                         uint32_t reg_offset,
                         uint32_t set_mask,
                         uint32_t clear_mask,
                         unsigned timeout_ms,
                         uint32_t *last_value) {
    uint64_t deadline = monotonic_ms() + (uint64_t)timeout_ms;
    uint32_t value = 0u;
    unsigned spin_budget = poll_spin_budget_default();
    unsigned sleep_us_value = poll_sleep_us_default();

    for (;;) {
        if (axil_mmio_read32(&ctx->csr_mmio, reg_offset, &value) != 0) {
            return -1;
        }
        if ((value & set_mask) == set_mask && (value & clear_mask) == 0u) {
            if (last_value != NULL) {
                *last_value = value;
            }
            return 0;
        }
        if (monotonic_ms() > deadline) {
            return failf(ETIMEDOUT,
                         "timeout waiting for CSR[0x%03x] set=0x%08x clear=0x%08x (last=0x%08x)",
                         reg_offset,
                         set_mask,
                         clear_mask,
                         value);
        }
        poll_wait_step(&spin_budget, sleep_us_value);
    }
}

static int wait_hw_ingress_done_fresh(tpu_soc_fp32_t *ctx,
                                      unsigned timeout_ms,
                                      uint32_t *last_load_status,
                                      int *saw_compute_busy) {
    uint64_t deadline = monotonic_ms() + (uint64_t)timeout_ms;
    uint64_t accept_done_deadline = monotonic_ms() + 1u;
    uint32_t load_status = 0u;
    unsigned spin_budget = poll_spin_budget_default();
    unsigned sleep_us_value = poll_sleep_us_default();
    int saw_busy = 0;
    int saw_non_done = 0;

    if (saw_compute_busy != NULL) {
        *saw_compute_busy = 0;
    }

    for (;;) {
        if (axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_STATUS, &load_status) != 0) {
            return -1;
        }
        if ((load_status & TPU_VANILLA_LOAD_STATUS_ERROR) != 0u) {
            uint32_t ingress = 0u;
            (void) read_dbg_ingress(ctx, &ingress);
            return failf(EIO,
                         "hardware ingress completed with error (LOAD_STATUS=0x%08x INGRESS=0x%08x saw_busy=%d saw_non_done=%d)",
                         load_status,
                         ingress,
                         saw_busy,
                         saw_non_done);
        }
        if ((load_status & TPU_VANILLA_LOAD_STATUS_BUSY) != 0u) {
            saw_busy = 1;
        }
        if ((load_status & TPU_VANILLA_LOAD_STATUS_DONE) == 0u) {
            saw_non_done = 1;
        }
        if ((load_status & TPU_VANILLA_LOAD_STATUS_DONE) != 0u &&
            (load_status & TPU_VANILLA_LOAD_STATUS_BUSY) == 0u &&
            (saw_busy || saw_non_done || monotonic_ms() > accept_done_deadline)) {
            if (last_load_status != NULL) {
                *last_load_status = load_status;
            }
            return 0;
        }
        if (monotonic_ms() > deadline) {
            return failf(ETIMEDOUT,
                         "timeout waiting for fresh hardware ingress completion (last LOAD_STATUS=0x%08x)",
                         load_status);
        }
        poll_wait_step(&spin_budget, sleep_us_value);
    }
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
    unsigned ctrl_deassert_delay_us;

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
    ctrl_deassert_delay_us = ctrl_deassert_delay_us_default();

    if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_CTRL, 0u) != 0 ||
        axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_CONFIG, config_word) != 0) {
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
    if (ctrl_deassert_delay_us != 0u) {
        sleep_us(ctrl_deassert_delay_us);
    }
    if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_CTRL, 0u) != 0) {
        return -1;
    }

    return 0;
}

void tpu_soc_v2_run_args_init(tpu_soc_v2_run_args_t *args) {
    if (args == NULL) {
        return;
    }

    memset(args, 0, sizeof(*args));
    args->c_mode = TPU_SOC_V2_C_MODE_ZERO;
    args->d_mode = TPU_SOC_V2_D_MODE_READBACK;
    args->precision_mode = TPU_VANILLA_PM_INT8_INT32;
    args->csr_base = TPU_SOC_FP32_CSR_BASE_DEFAULT;
    args->cdma_base = TPU_SOC_FP32_CDMA_BASE_DEFAULT;
    args->bram_base = TPU_SOC_FP32_BRAM_BASE_DEFAULT;
    args->device_index = 0u;
    args->mem_group = 0u;
    args->timeout_ms = 10000u;
    args->devmem_path = "/dev/mem";
    args->run_preflight = 1;
}

void tpu_soc_v2_run_report_init(tpu_soc_v2_run_report_t *report) {
    if (report == NULL) {
        return;
    }

    memset(report, 0, sizeof(*report));
    tpu_soc_fp32_cdma_trace_clear(&report->cdma_trace);
}

uint32_t tpu_soc_v2_int8_word_count(uint32_t rows, uint32_t cols) {
    return ceil_div_u32(rows * cols, 8u);
}

uint32_t tpu_soc_v2_int32_word_count(uint32_t rows, uint32_t cols) {
    return ceil_div_u32(rows * cols, 2u);
}

int tpu_soc_v2_validate_sizes(uint32_t m,
                              uint32_t n,
                              uint32_t k,
                              size_t *a_bytes,
                              size_t *b_bytes,
                              size_t *c_bytes,
                              size_t *d_bytes) {
    uint32_t a_words;
    uint32_t b_words;
    uint32_t c_words;
    size_t a_sz;
    size_t b_sz;
    size_t c_sz;
    size_t d_sz;

    if (!is_power_of_two_u32(n) || !is_power_of_two_u32(k)) {
        return failf(EINVAL,
                     "v2 INT8 path currently requires power-of-two N/K dimensions, got M=%u N=%u K=%u",
                     m,
                     n,
                     k);
    }
    if (m == 0u || m > 32u || n < 8u || k < 8u || n > 255u || k > 255u) {
        return failf(EINVAL,
                     "v2 INT8 path currently supports M in 1..32 and N/K in power-of-two 8..255, got M=%u N=%u K=%u",
                     m,
                     n,
                     k);
    }

    a_words = tpu_soc_v2_int8_word_count(m, k);
    b_words = tpu_soc_v2_int8_word_count(k, n);
    c_words = tpu_soc_v2_int32_word_count(m, n);

    a_sz = (size_t)a_words * 8u;
    b_sz = (size_t)b_words * 8u;
    c_sz = (size_t)c_words * 8u;
    d_sz = (size_t)m * (size_t)n * sizeof(int32_t);

    if (a_sz > TPU_VANILLA_MATRIX_WINDOW_STRIDE) {
        return failf(ERANGE, "A matrix is too large for TPU input window: %zu bytes", a_sz);
    }
    if (b_sz > TPU_VANILLA_MATRIX_WINDOW_STRIDE) {
        return failf(ERANGE, "B matrix is too large for TPU input window: %zu bytes", b_sz);
    }
    if (c_sz > TPU_VANILLA_MATRIX_WINDOW_STRIDE) {
        return failf(ERANGE, "C matrix is too large for TPU input window: %zu bytes", c_sz);
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

void tpu_soc_v2_pack_a_int8_row_major(const int8_t *src_mk,
                                      uint32_t m,
                                      uint32_t k,
                                      uint64_t *dst_words) {
    uint32_t elem_count = m * k;
    uint32_t word_count = tpu_soc_v2_int8_word_count(m, k);
    uint32_t i;

    if (src_mk == NULL || dst_words == NULL) {
        return;
    }

    memset(dst_words, 0, (size_t)word_count * sizeof(uint64_t));
    for (i = 0; i < elem_count; ++i) {
        uint32_t word_idx = i / 8u;
        uint32_t lane = i % 8u;
        dst_words[word_idx] |= ((uint64_t)(uint8_t)src_mk[i]) << (lane * 8u);
    }
}

void tpu_soc_v2_pack_b_int8_nk_to_kn(const int8_t *src_nk,
                                     uint32_t n,
                                     uint32_t k,
                                     uint64_t *dst_words) {
    uint32_t word_count = tpu_soc_v2_int8_word_count(k, n);
    uint32_t kk;
    uint32_t nn;

    if (src_nk == NULL || dst_words == NULL) {
        return;
    }

    memset(dst_words, 0, (size_t)word_count * sizeof(uint64_t));
    for (kk = 0; kk < k; ++kk) {
        for (nn = 0; nn < n; ++nn) {
            uint32_t linear_idx = kk * n + nn;
            uint32_t word_idx = linear_idx / 8u;
            uint32_t lane = linear_idx % 8u;
            uint8_t value = (uint8_t)src_nk[nn * k + kk];
            dst_words[word_idx] |= ((uint64_t)value) << (lane * 8u);
        }
    }
}

void tpu_soc_v2_pack_b_int8_kn_raw(const int8_t *src_kn,
                                   uint32_t n,
                                   uint32_t k,
                                   uint64_t *dst_words) {
    uint32_t elem_count = k * n;
    uint32_t word_count = tpu_soc_v2_int8_word_count(k, n);
    uint32_t i;

    if (src_kn == NULL || dst_words == NULL) {
        return;
    }

    memset(dst_words, 0, (size_t)word_count * sizeof(uint64_t));
    for (i = 0; i < elem_count; ++i) {
        uint32_t word_idx = i / 8u;
        uint32_t lane = i % 8u;
        dst_words[word_idx] |= ((uint64_t)(uint8_t)src_kn[i]) << (lane * 8u);
    }
}

void tpu_soc_v2_pack_c_int32_row_major(const int32_t *src_mn,
                                       uint32_t m,
                                       uint32_t n,
                                       uint64_t *dst_words) {
    uint32_t elem_count = m * n;
    uint32_t word_count = tpu_soc_v2_int32_word_count(m, n);
    uint32_t i;

    if (src_mn == NULL || dst_words == NULL) {
        return;
    }

    memset(dst_words, 0, (size_t)word_count * sizeof(uint64_t));
    for (i = 0; i < elem_count; ++i) {
        uint32_t word_idx = i / 2u;
        uint32_t lane = i % 2u;
        dst_words[word_idx] |= ((uint64_t)(uint32_t)src_mn[i]) << (lane * 32u);
    }
}

void tpu_soc_v2_reference_gemm_int8_int32(const int8_t *a_mk,
                                          const int8_t *b_nk,
                                          const int32_t *c_mn,
                                          int32_t *d_mn,
                                          uint32_t m,
                                          uint32_t n,
                                          uint32_t k) {
    uint32_t row;
    uint32_t col;
    uint32_t kk;

    if (a_mk == NULL || b_nk == NULL || c_mn == NULL || d_mn == NULL) {
        return;
    }

    for (row = 0; row < m; ++row) {
        for (col = 0; col < n; ++col) {
            int32_t acc = c_mn[row * n + col];
            for (kk = 0; kk < k; ++kk) {
                acc += (int32_t)a_mk[row * k + kk] * (int32_t)b_nk[col * k + kk];
            }
            d_mn[row * n + col] = acc;
        }
    }
}

int tpu_soc_v2_clear_output_region(tpu_soc_fp32_t *ctx, uint32_t m, uint32_t n) {
    uint32_t word_count;
    uint32_t i;

    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }

    word_count = tpu_soc_v2_int32_word_count(m, n);
    if ((size_t)word_count * sizeof(uint64_t) > TPU_SOC_FP32_BRAM_SPAN) {
        return failf(ERANGE, "requested output clear exceeds BRAM span");
    }
    for (i = 0; i < word_count; ++i) {
        if (axil_mmio_write64(&ctx->bram_mmio, i * 8u, 0u) != 0) {
            return -1;
        }
    }
    return 0;
}

static void unpack_result_words_int32(const uint64_t *src_words, int32_t *dst, uint32_t elem_count) {
    uint32_t word_count;
    uint32_t word_idx;

    word_count = tpu_soc_v2_int32_word_count(elem_count, 1u);
    for (word_idx = 0; word_idx < word_count; ++word_idx) {
        uint32_t dst_index = word_idx * 2u;
        uint64_t word = src_words[word_idx];

        dst[dst_index] = (int32_t)(word & 0xffffffffu);
        if (dst_index + 1u < elem_count) {
            dst[dst_index + 1u] = (int32_t)((word >> 32) & 0xffffffffu);
        }
    }
}

int tpu_soc_v2_read_result_int32(tpu_soc_fp32_t *ctx, int32_t *dst, uint32_t elem_count) {
    return tpu_soc_v2_read_result_int32_with_report(ctx, dst, elem_count, NULL);
}

int tpu_soc_v2_read_result_int32_with_report(tpu_soc_fp32_t *ctx,
                                             int32_t *dst,
                                             uint32_t elem_count,
                                             tpu_soc_v2_read_result_report_t *report) {
    uint32_t word_count;
    uint32_t word_idx;
    const uint64_t total_start_ns = monotonic_ns();

    if (ctx == NULL || dst == NULL) {
        errno = EINVAL;
        return -1;
    }
    word_count = tpu_soc_v2_int32_word_count(elem_count, 1u);
    if ((size_t)word_count * 8u > TPU_SOC_FP32_BRAM_SPAN) {
        return failf(ERANGE, "requested result read exceeds BRAM span");
    }

    read_result_report_init(report, elem_count, word_count);
    if (report != NULL) {
        report->used_mmio = 1;
    }

    for (word_idx = 0; word_idx < word_count; ++word_idx) {
        const uint64_t read_start_ns = monotonic_ns();
        uint64_t word = 0;
        uint32_t dst_index = word_idx * 2u;

        if (axil_mmio_read64(&ctx->bram_mmio, word_idx * 8u, &word) != 0) {
            return -1;
        }
        if (report != NULL) {
            report->mmio_read_ns += monotonic_ns() - read_start_ns;
        }
        dst[dst_index] = (int32_t)(word & 0xffffffffu);
        if (dst_index + 1u < elem_count) {
            dst[dst_index + 1u] = (int32_t)((word >> 32) & 0xffffffffu);
        }
    }
    if (report != NULL) {
        report->total_ns = monotonic_ns() - total_start_ns;
    }
    return 0;
}

static void copy_int32_row_inline(int32_t *dst, const int32_t *src, uint32_t cols) {
    uint32_t col = 0;
    for (; col + 8u <= cols; col += 8u) {
        dst[col + 0u] = src[col + 0u];
        dst[col + 1u] = src[col + 1u];
        dst[col + 2u] = src[col + 2u];
        dst[col + 3u] = src[col + 3u];
        dst[col + 4u] = src[col + 4u];
        dst[col + 5u] = src[col + 5u];
        dst[col + 6u] = src[col + 6u];
        dst[col + 7u] = src[col + 7u];
    }
    for (; col < cols; ++col) {
        dst[col] = src[col];
    }
}

int tpu_soc_v2_read_result_int32_from_buffer(tpu_soc_fp32_buffer_t *buf,
                                             int32_t *dst,
                                             uint32_t elem_count) {
    return tpu_soc_v2_read_result_int32_from_buffer_with_report(buf, dst, elem_count, NULL);
}

int tpu_soc_v2_read_result_int32_from_buffer_with_rect(tpu_soc_fp32_buffer_t *buf,
                                                       int32_t *dst,
                                                       uint32_t packed_rows,
                                                       uint32_t packed_cols,
                                                       uint32_t copy_rows,
                                                       uint32_t copy_cols,
                                                       uint32_t dst_stride) {
    return tpu_soc_v2_read_result_int32_from_buffer_with_rect_report(
            buf, dst, packed_rows, packed_cols, copy_rows, copy_cols, dst_stride, NULL);
}

int tpu_soc_v2_read_result_int32_from_buffer_with_rect_report(tpu_soc_fp32_buffer_t *buf,
                                                              int32_t *dst,
                                                              uint32_t packed_rows,
                                                              uint32_t packed_cols,
                                                              uint32_t copy_rows,
                                                              uint32_t copy_cols,
                                                              uint32_t dst_stride,
                                                              tpu_soc_v2_read_result_report_t *report) {
    uint32_t word_count;
    const uint64_t total_start_ns = monotonic_ns();

    if (buf == NULL || dst == NULL || buf->mapping == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (packed_rows == 0u || packed_cols == 0u || copy_rows == 0u || copy_cols == 0u) {
        errno = EINVAL;
        return -1;
    }
    if (copy_rows > packed_rows || copy_cols > packed_cols || dst_stride < copy_cols) {
        errno = EINVAL;
        return -1;
    }

    word_count = tpu_soc_v2_int32_word_count(packed_rows, packed_cols);
    if ((size_t)word_count * sizeof(uint64_t) > buf->size) {
        return failf(ERANGE, "requested result read exceeds output buffer size");
    }

    read_result_report_init(report, copy_rows * copy_cols, word_count);
    if (report != NULL) {
        report->used_output_buffer = 1;
    }

    {
        const uint64_t sync_start_ns = monotonic_ns();
        if (tpu_soc_fp32_buffer_sync_from_device(buf, (size_t)word_count * sizeof(uint64_t)) != 0) {
            return -1;
        }
        if (report != NULL) {
            report->sync_ns = monotonic_ns() - sync_start_ns;
        }
    }

    if (output_read_mode_is_memcpy()) {
        const uint64_t copy_start_ns = monotonic_ns();
        const int32_t * src = (const int32_t *) buf->mapping;

        if (copy_rows == packed_rows && copy_cols == packed_cols && dst_stride == packed_cols) {
            memcpy(dst, src, (size_t) copy_rows * copy_cols * sizeof(int32_t));
        } else {
            const size_t packed_row_stride = (size_t) packed_cols;
            const size_t dst_row_stride = (size_t) dst_stride;
            const int32_t * src_row = src;
            int32_t * dst_row = dst;
            uint32_t row;
            for (row = 0; row < copy_rows; ++row) {
                copy_int32_row_inline(dst_row, src_row, copy_cols);
                src_row += packed_row_stride;
                dst_row += dst_row_stride;
            }
        }
        if (report != NULL) {
            report->copy_ns = monotonic_ns() - copy_start_ns;
            report->used_memcpy = 1;
            report->total_ns = monotonic_ns() - total_start_ns;
        }
        return 0;
    }

    {
        const uint64_t unpack_start_ns = monotonic_ns();
        uint32_t row;
        const uint64_t * src_words = (const uint64_t *) buf->mapping;
        const uint32_t packed_row_word_count = tpu_soc_v2_int32_word_count(packed_cols, 1u);
        for (row = 0; row < copy_rows; ++row) {
            unpack_result_words_int32(src_words + (size_t) row * packed_row_word_count,
                                      dst + (size_t) row * dst_stride,
                                      copy_cols);
        }
        if (report != NULL) {
            report->unpack_ns = monotonic_ns() - unpack_start_ns;
            report->used_unpack = 1;
            report->total_ns = monotonic_ns() - total_start_ns;
        }
    }
    return 0;
}

int tpu_soc_v2_read_result_int32_from_buffer_with_layout(tpu_soc_fp32_buffer_t *buf,
                                                         int32_t *dst,
                                                         uint32_t elem_count,
                                                         uint32_t rows,
                                                         uint32_t cols,
                                                         uint32_t dst_stride) {
    return tpu_soc_v2_read_result_int32_from_buffer_with_layout_report(
            buf, dst, elem_count, rows, cols, dst_stride, NULL);
}

int tpu_soc_v2_read_result_int32_from_buffer_with_layout_report(tpu_soc_fp32_buffer_t *buf,
                                                                int32_t *dst,
                                                                uint32_t elem_count,
                                                                uint32_t rows,
                                                                uint32_t cols,
                                                                uint32_t dst_stride,
                                                                tpu_soc_v2_read_result_report_t *report) {
    if ((uint64_t) rows * (uint64_t) cols != (uint64_t) elem_count) {
        errno = EINVAL;
        return -1;
    }
    return tpu_soc_v2_read_result_int32_from_buffer_with_rect_report(
            buf, dst, rows, cols, rows, cols, dst_stride, report);
}

int tpu_soc_v2_read_result_int32_from_buffer_with_report(tpu_soc_fp32_buffer_t *buf,
                                                         int32_t *dst,
                                                         uint32_t elem_count,
                                                         tpu_soc_v2_read_result_report_t *report) {
    return tpu_soc_v2_read_result_int32_from_buffer_with_rect_report(
            buf, dst, 1u, elem_count, 1u, elem_count, elem_count, report);
}

int tpu_soc_v2_dump_output_buffer_bytes(FILE *stream,
                                        tpu_soc_fp32_buffer_t *buf,
                                        uint32_t elem_count,
                                        uint32_t max_bytes) {
    uint32_t word_count;
    size_t total_bytes;
    size_t dump_bytes;
    const unsigned char *bytes;
    size_t offset;
    size_t lane;

    if (buf == NULL || buf->mapping == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (stream == NULL) {
        stream = stderr;
    }

    word_count = tpu_soc_v2_int32_word_count(elem_count, 1u);
    total_bytes = (size_t)word_count * sizeof(uint64_t);
    if (total_bytes > buf->size) {
        return failf(ERANGE, "requested result dump exceeds output buffer size");
    }
    if (tpu_soc_fp32_buffer_sync_from_device(buf, total_bytes) != 0) {
        return -1;
    }

    dump_bytes = total_bytes;
    if (max_bytes != 0u && dump_bytes > (size_t)max_bytes) {
        dump_bytes = (size_t)max_bytes;
    }

    bytes = (const unsigned char *)buf->mapping;
    fprintf(stream,
            "OUTPUTBUF raw bytes: total=%zu dump=%zu device_addr=0x%016" PRIx64 " mapping=%p\n",
            total_bytes,
            dump_bytes,
            buf->device_addr,
            buf->mapping);
    for (offset = 0; offset < dump_bytes; offset += 16u) {
        size_t line_bytes = dump_bytes - offset;
        if (line_bytes > 16u) {
            line_bytes = 16u;
        }
        fprintf(stream, "  %04zx :", offset);
        for (lane = 0; lane < line_bytes; ++lane) {
            fprintf(stream, " %02x", bytes[offset + lane]);
        }
        fputc('\n', stream);
    }
    if (dump_bytes < total_bytes) {
        fprintf(stream, "  ... truncated ...\n");
    }

    return 0;
}

int tpu_soc_v2_set_output_target_addr(tpu_soc_fp32_t *ctx, uint32_t addr) {
    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }
    return axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_STORE_TARGET_ADDR, addr);
}

int tpu_soc_v2_get_output_target_addr(tpu_soc_fp32_t *ctx, uint32_t *addr) {
    if (ctx == NULL || addr == NULL) {
        errno = EINVAL;
        return -1;
    }
    return axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_STORE_TARGET_ADDR, addr);
}

int tpu_soc_v2_set_reuse_mode(tpu_soc_fp32_t *ctx,
                              uint32_t c_mode,
                              uint32_t d_mode,
                              int clear_prev_d) {
    uint32_t mode_word;

    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (c_mode > TPU_SOC_V2_C_MODE_PREV_D) {
        return failf(EINVAL, "unsupported c_mode=%u", c_mode);
    }
    if (c_mode == TPU_SOC_V2_C_MODE_HOST) {
        return failf(ENOTSUP,
                     "HOST_C is disabled in the ZERO/PREV_D no-C-RAM build; use ZERO or PREV_D");
    }
    if (d_mode > TPU_SOC_V2_D_MODE_RETAIN) {
        return failf(EINVAL, "unsupported d_mode=%u", d_mode);
    }

    if (clear_prev_d) {
        if (axil_mmio_write32(&ctx->csr_mmio,
                              TPU_VANILLA_REG_CACHE_CTRL,
                              TPU_VANILLA_CACHE_CTRL_CLEAR_PREV_D) != 0 ||
            axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_CACHE_CTRL, 0u) != 0) {
            return -1;
        }
    }

    mode_word = ((c_mode << TPU_VANILLA_CACHE_MODE_C_MODE_SHIFT) & TPU_VANILLA_CACHE_MODE_C_MODE_MASK) |
                ((d_mode << TPU_VANILLA_CACHE_MODE_D_MODE_SHIFT) & TPU_VANILLA_CACHE_MODE_D_MODE_MASK);
    if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_CACHE_MODE, mode_word) != 0) {
        return -1;
    }

    ctx->v2_c_mode = c_mode;
    ctx->v2_d_mode = d_mode;
    if (clear_prev_d) {
        g_prev_d_shape_valid = 0;
        g_prev_d_m = 0u;
        g_prev_d_n = 0u;
        ctx->v2_prev_d_shape_valid = 0;
        ctx->v2_prev_d_m = 0u;
        ctx->v2_prev_d_n = 0u;
    }
    return 0;
}

int tpu_soc_v2_get_reuse_status(tpu_soc_fp32_t *ctx, uint32_t *status_out) {
    if (ctx == NULL || status_out == NULL) {
        errno = EINVAL;
        return -1;
    }
    return axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_CACHE_STATUS, status_out);
}

int tpu_soc_v2_clear_prev_d_slot(tpu_soc_fp32_t *ctx) {
    uint32_t c_mode = TPU_SOC_V2_C_MODE_ZERO;
    uint32_t d_mode = TPU_SOC_V2_D_MODE_READBACK;

    if (ctx != NULL) {
        c_mode = (ctx->v2_c_mode == TPU_SOC_V2_C_MODE_HOST) ? TPU_SOC_V2_C_MODE_ZERO : ctx->v2_c_mode;
        d_mode = ctx->v2_d_mode;
    }

    return tpu_soc_v2_set_reuse_mode(ctx,
                                     c_mode,
                                     d_mode,
                                     1);
}

int tpu_soc_v2_try_enable_output_buffer(tpu_soc_fp32_t *ctx,
                                        const tpu_soc_fp32_buffer_t *d_buf,
                                        int *enabled_out) {
    uint32_t observed = 0u;
    uint32_t target_addr;
    uint32_t aperture_base = 0u;
    uint32_t aperture_high = 0u;
    uint64_t end_addr = 0u;

    if (enabled_out == NULL) {
        errno = EINVAL;
        return -1;
    }
    *enabled_out = 0;

    if (ctx == NULL || d_buf == NULL || d_buf->handle == NULL || d_buf->mapping == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (d_buf->device_addr > UINT32_MAX) {
        return 0;
    }
    aperture_base = ctx->output_ddr_base;
    aperture_high = ctx->output_ddr_high;
    end_addr = d_buf->device_addr + (d_buf->size != 0u ? (uint64_t)d_buf->size - 1u : 0u);
    if (end_addr < d_buf->device_addr) {
        fprintf(stderr,
                "output buffer device range starting at 0x%016" PRIx64 " overflows; leaving output-buffer disabled\n",
                d_buf->device_addr);
        return 0;
    }
    if ((d_buf->device_addr < aperture_base || end_addr > aperture_high) &&
        (!ctx->output_ddr_high_window_valid ||
         d_buf->device_addr < ctx->output_ddr_high_window_base ||
         end_addr > ctx->output_ddr_high_window_high)) {
        if (ctx->output_ddr_high_window_valid) {
            fprintf(stderr,
                    "output buffer device range [0x%08" PRIx64 ", 0x%08" PRIx64 "] is outside output DDR apertures [0x%08x, 0x%08x] and [0x%08x, 0x%08x]; leaving output-buffer disabled\n",
                    d_buf->device_addr,
                    end_addr,
                    aperture_base,
                    aperture_high,
                    ctx->output_ddr_high_window_base,
                    ctx->output_ddr_high_window_high);
            return 0;
        }
        fprintf(stderr,
                "output buffer device range [0x%08" PRIx64 ", 0x%08" PRIx64 "] is outside output DDR aperture [0x%08x, 0x%08x]; leaving output-buffer disabled\n",
                d_buf->device_addr,
                end_addr,
                aperture_base,
                aperture_high);
        return 0;
    }

    target_addr = (uint32_t)d_buf->device_addr;
    if (tpu_soc_v2_set_output_target_addr(ctx, target_addr) != 0) {
        return -1;
    }
    if (tpu_soc_v2_get_output_target_addr(ctx, &observed) != 0) {
        return -1;
    }

    if (observed == target_addr) {
        *enabled_out = 1;
    }
    return 0;
}

int tpu_soc_v2_start_int8_int32_hw_ingress_no_wait(tpu_soc_fp32_t *ctx,
                                                   const tpu_soc_fp32_buffer_t *a_buf,
                                                   const tpu_soc_fp32_buffer_t *b_buf,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   int sparse,
                                                   int use_preloaded_ab,
                                                   unsigned timeout_ms,
                                                   tpu_soc_v2_execute_report_t *report) {
    size_t a_bytes = 0u;
    size_t b_bytes = 0u;
    size_t c_bytes = 0u;
    size_t d_bytes = 0u;
    uint32_t config_word;
    uint32_t ctrl_word;
    uint32_t hw_load_ctrl;
    uint32_t reuse_status = 0u;
    uint64_t execute_entry_ns = monotonic_ns();
    uint64_t t0;

    if (ctx == NULL || (!use_preloaded_ab && (a_buf == NULL || b_buf == NULL))) {
        errno = EINVAL;
        return -1;
    }
    if (report != NULL) {
        memset(report, 0, sizeof(*report));
        report->selected_path = use_preloaded_ab ? "hw_ingress_preloaded_ab" : "hw_ingress_no_wait";
    }
    ensure_v2_exec_cfg_cached(ctx);
    if (!ctx->v2_hw_ingress) {
        return execute_report_finish(report,
                                     execute_entry_ns,
                                     failf(ENOTSUP, "overlap start requires TPU_HW_INGRESS=1"));
    }
    if (ctx->v2_use_sg || ctx->v2_transfer_only) {
        return execute_report_finish(report,
                                     execute_entry_ns,
                                     failf(ENOTSUP, "overlap start supports only hw_ingress, not SG/transfer-only"));
    }
    if (ctx->v2_gemv_m1) {
        return execute_report_finish(report,
                                     execute_entry_ns,
                                     failf(ENOTSUP, "TPU_GEMV_M1 is disabled in the ZERO/PREV_D no-C-RAM build"));
    }
    if (ctx->v2_c_mode == TPU_SOC_V2_C_MODE_HOST) {
        return execute_report_finish(report,
                                     execute_entry_ns,
                                     failf(ENOTSUP, "HOST_C is disabled in the ZERO/PREV_D no-C-RAM build"));
    }
    if (sparse) {
        return execute_report_finish(report,
                                     execute_entry_ns,
                                     failf(ENOTSUP, "sparse mode is not enabled in the current v2 INT8 runtime path"));
    }
    if (timeout_ms == 0u) {
        timeout_ms = 10000u;
    }
    if (resolve_v2_sizes_cached(ctx, m, n, k, &a_bytes, &b_bytes, &c_bytes, &d_bytes) != 0) {
        return execute_report_finish(report, execute_entry_ns, -1);
    }
    (void)c_bytes;
    (void)d_bytes;

    if (!use_preloaded_ab) {
        if (a_buf->mapping == NULL || b_buf->mapping == NULL) {
            return execute_report_finish(report,
                                         execute_entry_ns,
                                         failf(EINVAL, "A/B XRT buffers must be allocated before execute"));
        }
        if (a_buf->device_addr > UINT32_MAX || b_buf->device_addr > UINT32_MAX) {
            return execute_report_finish(report,
                                         execute_entry_ns,
                                         failf(ERANGE, "TPU_HW_INGRESS currently requires 32-bit source device addresses"));
        }
        if (ensure_cached_input_addr(ctx,
                                     &ctx->v2_a_addr_cache_valid,
                                     &ctx->v2_cached_a_addr,
                                     &ctx->v2_cached_a_size,
                                     a_buf->device_addr,
                                     a_bytes) != 0 ||
            ensure_cached_input_addr(ctx,
                                     &ctx->v2_b_addr_cache_valid,
                                     &ctx->v2_cached_b_addr,
                                     &ctx->v2_cached_b_size,
                                     b_buf->device_addr,
                                     b_bytes) != 0) {
            return execute_report_finish(report, execute_entry_ns, -1);
        }

        t0 = monotonic_ns();
        if (tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)a_buf, a_bytes) != 0 ||
            tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)b_buf, b_bytes) != 0) {
            return execute_report_finish(report, execute_entry_ns, -1);
        }
        if (report != NULL) {
            report->sync_ns = monotonic_ns() - t0;
        }
    }

    t0 = monotonic_ns();
    if (wait_status_mask(ctx, TPU_VANILLA_STATUS_READY, 0u, timeout_ms, NULL) != 0) {
        return execute_report_finish(report, execute_entry_ns, -1);
    }
    if (report != NULL) {
        report->ready_wait_ns = monotonic_ns() - t0;
    }

    if (tpu_soc_v2_set_reuse_mode(ctx, ctx->v2_c_mode, ctx->v2_d_mode, 0) != 0) {
        return execute_report_finish(report, execute_entry_ns, -1);
    }
    if (ctx->v2_c_mode == TPU_SOC_V2_C_MODE_PREV_D) {
        if (tpu_soc_v2_get_reuse_status(ctx, &reuse_status) != 0) {
            return execute_report_finish(report, execute_entry_ns, -1);
        }
        if ((reuse_status & TPU_VANILLA_CACHE_STATUS_PREV_D_VALID) == 0u) {
            return execute_report_finish(report,
                                         execute_entry_ns,
                                         failf(ENOENT, "PREV_D requested but retained D slot is invalid"));
        }
    }

    config_word = (TPU_VANILLA_PM_INT8_INT32 << 24) | (m << 16) | (n << 8) | k;
    ctrl_word = TPU_VANILLA_CTRL_ENABLE |
                (use_preloaded_ab ? TPU_VANILLA_CTRL_USE_PRELOADED_AB : 0u);
    hw_load_ctrl = TPU_VANILLA_LOAD_CTRL_MODE_HW |
                   TPU_VANILLA_LOAD_CTRL_MASK_A |
                   TPU_VANILLA_LOAD_CTRL_MASK_B;

    t0 = monotonic_ns();
    if (!use_preloaded_ab) {
        if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_A_ADDR, (uint32_t)a_buf->device_addr) != 0 ||
            axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_B_ADDR, (uint32_t)b_buf->device_addr) != 0 ||
            axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_CTRL, hw_load_ctrl) != 0 ||
            axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_GEMV_CTRL, 0u) != 0) {
            return execute_report_finish(report, execute_entry_ns, -1);
        }
    } else {
        /* Clear stale PRELOAD_ONLY/INGRESS_START bits before consuming preloaded A/B. */
        if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_CTRL, hw_load_ctrl) != 0 ||
            axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_GEMV_CTRL, 0u) != 0) {
            return execute_report_finish(report, execute_entry_ns, -1);
        }
    }
    if (report != NULL) {
        report->path_setup_ns = monotonic_ns() - t0;
    }

    t0 = monotonic_ns();
    if (program_config_then_start(ctx, config_word, ctrl_word, timeout_ms) != 0) {
        return execute_report_finish(report, execute_entry_ns, -1);
    }
    if (report != NULL) {
        report->start_ns = monotonic_ns() - t0;
    }

    if (!use_preloaded_ab) {
        t0 = monotonic_ns();
        if (wait_hw_ingress_done_fresh(ctx, timeout_ms, NULL, NULL) != 0) {
            return execute_report_finish(report, execute_entry_ns, -1);
        }
        if (report != NULL) {
            report->transfer_ns = monotonic_ns() - t0;
        }
    } else if (report != NULL) {
        /*
         * STATUS.DONE is read-to-clear.  A USE_PRELOADED_AB run has no fresh
         * A/B ingress to wait for here, and polling STATUS just to observe
         * LOAD_BUSY can clear a very fast final READBACK completion before the
         * caller's active wait sees it.
         */
        report->transfer_ns = 0u;
    }

    return execute_report_finish(report, execute_entry_ns, 0);
}

int tpu_soc_v2_preload_ab_int8_int32_with_report(tpu_soc_fp32_t *ctx,
                                                 const tpu_soc_fp32_buffer_t *a_buf,
                                                 const tpu_soc_fp32_buffer_t *b_buf,
                                                 uint32_t m,
                                                 uint32_t n,
                                                 uint32_t k,
                                                 unsigned timeout_ms,
                                                 tpu_soc_v2_ab_preload_report_t *report) {
    size_t a_bytes = 0u;
    size_t b_bytes = 0u;
    size_t c_bytes = 0u;
    size_t d_bytes = 0u;
    uint32_t hw_load_ctrl;
    uint32_t quiet_load_status = 0u;
    uint32_t quiet_ingress_status = 0u;
    unsigned quiet_us;
    uint64_t t0;

    if (ctx == NULL || a_buf == NULL || b_buf == NULL ||
        a_buf->mapping == NULL || b_buf->mapping == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (report != NULL) {
        memset(report, 0, sizeof(*report));
    }
    ensure_v2_exec_cfg_cached(ctx);
    if (!ctx->v2_hw_ingress) {
        return failf(ENOTSUP, "A/B preload requires TPU_HW_INGRESS=1");
    }
    if (ctx->v2_gemv_m1 || ctx->v2_use_sg || ctx->v2_transfer_only) {
        return failf(ENOTSUP, "A/B preload supports only the normal hw_ingress path");
    }
    if (timeout_ms == 0u) {
        timeout_ms = 10000u;
    }
    if (resolve_v2_sizes_cached(ctx, m, n, k, &a_bytes, &b_bytes, &c_bytes, &d_bytes) != 0) {
        return -1;
    }
    (void)c_bytes;
    (void)d_bytes;
    if (a_buf->device_addr > UINT32_MAX || b_buf->device_addr > UINT32_MAX) {
        return failf(ERANGE, "TPU_HW_INGRESS currently requires 32-bit source device addresses");
    }
    if (validate_dma_source_addr_cached(ctx, a_buf->device_addr, a_bytes) != 0 ||
        validate_dma_source_addr_cached(ctx, b_buf->device_addr, b_bytes) != 0) {
        return -1;
    }

    t0 = monotonic_ns();
    if (tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)a_buf, a_bytes) != 0 ||
        tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)b_buf, b_bytes) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->sync_ns = monotonic_ns() - t0;
    }

    quiet_us = preload_tail_quiet_us_default();
    t0 = monotonic_ns();
    if (wait_hw_ingress_tail_quiet(ctx, quiet_us, &quiet_load_status, &quiet_ingress_status) != 0) {
        return -1;
    }

    t0 = monotonic_ns();
    if (wait_csr_mask(ctx,
                      TPU_VANILLA_REG_DBG_INGRESS,
                      TPU_VANILLA_DBG_INGRESS_A_WR_READY |
                      TPU_VANILLA_DBG_INGRESS_B_WR_READY,
                      0u,
                      timeout_ms,
                      NULL) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->write_ready_wait_ns = monotonic_ns() - t0;
    }
    if (trace_ingress_gaps_enabled()) {
        fprintf(stderr,
                "PRELOAD_QUIET quiet_us=%u load_status=0x%08x ingress={",
                quiet_us,
                quiet_load_status);
        dump_ingress_bits(stderr, quiet_ingress_status);
        fprintf(stderr, "}\n");
    }

    hw_load_ctrl = TPU_VANILLA_LOAD_CTRL_MODE_HW |
                   TPU_VANILLA_LOAD_CTRL_MASK_A |
                   TPU_VANILLA_LOAD_CTRL_MASK_B |
                   TPU_VANILLA_LOAD_CTRL_PRELOAD_ONLY |
                   TPU_VANILLA_LOAD_CTRL_INGRESS_START;

    t0 = monotonic_ns();
    if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_A_ADDR, (uint32_t)a_buf->device_addr) != 0 ||
        axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_B_ADDR, (uint32_t)b_buf->device_addr) != 0 ||
        axil_mmio_write32(&ctx->csr_mmio,
                          TPU_VANILLA_REG_LOAD_CTRL,
                          TPU_VANILLA_LOAD_CTRL_MODE_HW |
                          TPU_VANILLA_LOAD_CTRL_MASK_A |
                          TPU_VANILLA_LOAD_CTRL_MASK_B) != 0 ||
        axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_CTRL, hw_load_ctrl) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->path_setup_ns = monotonic_ns() - t0;
    }

    t0 = monotonic_ns();
    if (wait_hw_ingress_done_fresh(ctx,
                                   timeout_ms,
                                   NULL,
                                   report != NULL ? &report->saw_compute_busy : NULL) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->transfer_ns = monotonic_ns() - t0;
    }
    return 0;
}

static int tpu_soc_v2_preload_ab_int8_int32_common(tpu_soc_fp32_t *ctx,
                                                   const tpu_soc_fp32_buffer_t *a_buf,
                                                   const tpu_soc_fp32_buffer_t *b_buf,
                                                   uint32_t m,
                                                   uint32_t n,
                                                   uint32_t k,
                                                   unsigned timeout_ms,
                                                   int inputs_presynced,
                                                   int wait_for_done,
                                                   tpu_soc_v2_ab_preload_report_t *report) {
    size_t a_bytes = 0u;
    size_t b_bytes = 0u;
    size_t c_bytes = 0u;
    size_t d_bytes = 0u;
    uint32_t hw_load_ctrl;
    uint64_t t0;

    if (ctx == NULL || a_buf == NULL || b_buf == NULL ||
        a_buf->mapping == NULL || b_buf->mapping == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (report != NULL) {
        memset(report, 0, sizeof(*report));
    }
    ensure_v2_exec_cfg_cached(ctx);
    if (!ctx->v2_hw_ingress) {
        return failf(ENOTSUP, "A/B preload requires TPU_HW_INGRESS=1");
    }
    if (ctx->v2_gemv_m1 || ctx->v2_use_sg || ctx->v2_transfer_only) {
        return failf(ENOTSUP, "A/B preload supports only the normal hw_ingress path");
    }
    if (timeout_ms == 0u) {
        timeout_ms = 10000u;
    }
    if (resolve_v2_sizes_cached(ctx, m, n, k, &a_bytes, &b_bytes, &c_bytes, &d_bytes) != 0) {
        return -1;
    }
    (void)c_bytes;
    (void)d_bytes;
    if (a_buf->device_addr > UINT32_MAX || b_buf->device_addr > UINT32_MAX) {
        return failf(ERANGE, "TPU_HW_INGRESS currently requires 32-bit source device addresses");
    }
    if (validate_dma_source_addr_cached(ctx, a_buf->device_addr, a_bytes) != 0 ||
        validate_dma_source_addr_cached(ctx, b_buf->device_addr, b_bytes) != 0) {
        return -1;
    }

    if (!inputs_presynced) {
        t0 = monotonic_ns();
        if (tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)a_buf, a_bytes) != 0 ||
            tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)b_buf, b_bytes) != 0) {
            return -1;
        }
        if (report != NULL) {
            report->sync_ns = monotonic_ns() - t0;
        }
    }

    t0 = monotonic_ns();
    if (wait_csr_mask(ctx,
                      TPU_VANILLA_REG_DBG_INGRESS,
                      TPU_VANILLA_DBG_INGRESS_A_WR_READY |
                      TPU_VANILLA_DBG_INGRESS_B_WR_READY,
                      0u,
                      timeout_ms,
                      NULL) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->write_ready_wait_ns = monotonic_ns() - t0;
    }

    hw_load_ctrl = TPU_VANILLA_LOAD_CTRL_MODE_HW |
                   TPU_VANILLA_LOAD_CTRL_MASK_A |
                   TPU_VANILLA_LOAD_CTRL_MASK_B |
                   TPU_VANILLA_LOAD_CTRL_PRELOAD_ONLY |
                   TPU_VANILLA_LOAD_CTRL_INGRESS_START;

    t0 = monotonic_ns();
    if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_A_ADDR, (uint32_t)a_buf->device_addr) != 0 ||
        axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_B_ADDR, (uint32_t)b_buf->device_addr) != 0 ||
        axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_CTRL, hw_load_ctrl) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->path_setup_ns = monotonic_ns() - t0;
    }

    if (wait_for_done) {
        t0 = monotonic_ns();
        if (wait_hw_ingress_done_fresh(ctx,
                                       timeout_ms,
                                       NULL,
                                       report != NULL ? &report->saw_compute_busy : NULL) != 0) {
            return -1;
        }
        if (report != NULL) {
            report->transfer_ns = monotonic_ns() - t0;
        }
    } else if (report != NULL) {
        report->transfer_ns = 0u;
    }
    return 0;
}

int tpu_soc_v2_preload_ab_int8_int32_presynced_with_report(tpu_soc_fp32_t *ctx,
                                                           const tpu_soc_fp32_buffer_t *a_buf,
                                                           const tpu_soc_fp32_buffer_t *b_buf,
                                                           uint32_t m,
                                                           uint32_t n,
                                                           uint32_t k,
                                                           unsigned timeout_ms,
                                                           tpu_soc_v2_ab_preload_report_t *report) {
    return tpu_soc_v2_preload_ab_int8_int32_common(ctx,
                                                   a_buf,
                                                   b_buf,
                                                   m,
                                                   n,
                                                   k,
                                                   timeout_ms,
                                                   1,
                                                   1,
                                                   report);
}

int tpu_soc_v2_preload_ab_int8_int32_start_with_report(tpu_soc_fp32_t *ctx,
                                                       const tpu_soc_fp32_buffer_t *a_buf,
                                                       const tpu_soc_fp32_buffer_t *b_buf,
                                                       uint32_t m,
                                                       uint32_t n,
                                                       uint32_t k,
                                                       unsigned timeout_ms,
                                                       tpu_soc_v2_ab_preload_report_t *report) {
    return tpu_soc_v2_preload_ab_int8_int32_common(ctx,
                                                   a_buf,
                                                   b_buf,
                                                   m,
                                                   n,
                                                   k,
                                                   timeout_ms,
                                                   0,
                                                   0,
                                                   report);
}

int tpu_soc_v2_preload_ab_int8_int32_presynced_start_with_report(tpu_soc_fp32_t *ctx,
                                                                 const tpu_soc_fp32_buffer_t *a_buf,
                                                                 const tpu_soc_fp32_buffer_t *b_buf,
                                                                 uint32_t m,
                                                                 uint32_t n,
                                                                 uint32_t k,
                                                                 unsigned timeout_ms,
                                                                 tpu_soc_v2_ab_preload_report_t *report) {
    return tpu_soc_v2_preload_ab_int8_int32_common(ctx,
                                                   a_buf,
                                                   b_buf,
                                                   m,
                                                   n,
                                                   k,
                                                   timeout_ms,
                                                   1,
                                                   0,
                                                   report);
}

static int tpu_soc_v2_program_next_ab_preload_int8_int32_common(tpu_soc_fp32_t *ctx,
                                                                const tpu_soc_fp32_buffer_t *a_buf,
                                                                const tpu_soc_fp32_buffer_t *b_buf,
                                                                uint32_t m,
                                                                uint32_t n,
                                                                uint32_t k,
                                                                int inputs_presynced,
                                                                tpu_soc_v2_ab_preload_report_t *report) {
    size_t a_bytes = 0u;
    size_t b_bytes = 0u;
    size_t c_bytes = 0u;
    size_t d_bytes = 0u;
    uint32_t next_load_ctrl;
    uint64_t t0;

    if (ctx == NULL || a_buf == NULL || b_buf == NULL ||
        a_buf->mapping == NULL || b_buf->mapping == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (report != NULL) {
        memset(report, 0, sizeof(*report));
    }
    ensure_v2_exec_cfg_cached(ctx);
    if (!ctx->v2_hw_ingress) {
        return failf(ENOTSUP, "next A/B descriptor requires TPU_HW_INGRESS=1");
    }
    if (ctx->v2_gemv_m1 || ctx->v2_use_sg || ctx->v2_transfer_only) {
        return failf(ENOTSUP, "next A/B descriptor supports only the normal hw_ingress path");
    }
    if (resolve_v2_sizes_cached(ctx, m, n, k, &a_bytes, &b_bytes, &c_bytes, &d_bytes) != 0) {
        return -1;
    }
    (void)c_bytes;
    (void)d_bytes;
    if (a_buf->device_addr > UINT32_MAX || b_buf->device_addr > UINT32_MAX) {
        return failf(ERANGE, "TPU_HW_INGRESS currently requires 32-bit source device addresses");
    }
    if (validate_dma_source_addr_cached(ctx, a_buf->device_addr, a_bytes) != 0 ||
        validate_dma_source_addr_cached(ctx, b_buf->device_addr, b_bytes) != 0) {
        return -1;
    }

    if (!inputs_presynced) {
        t0 = monotonic_ns();
        if (tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)a_buf, a_bytes) != 0 ||
            tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)b_buf, b_bytes) != 0) {
            return -1;
        }
        if (report != NULL) {
            report->sync_ns = monotonic_ns() - t0;
        }
    }

    next_load_ctrl = TPU_VANILLA_LOAD_CTRL_MODE_HW |
                     TPU_VANILLA_LOAD_CTRL_MASK_A |
                     TPU_VANILLA_LOAD_CTRL_MASK_B |
                     TPU_VANILLA_LOAD_CTRL_PRELOAD_ONLY |
                     TPU_VANILLA_LOAD_CTRL_INGRESS_START;

    t0 = monotonic_ns();
    if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_NEXT_LOAD_A_ADDR, (uint32_t)a_buf->device_addr) != 0 ||
        axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_NEXT_LOAD_B_ADDR, (uint32_t)b_buf->device_addr) != 0 ||
        axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_NEXT_LOAD_CTRL, next_load_ctrl) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->path_setup_ns = monotonic_ns() - t0;
        report->transfer_ns = 0u;
    }
    return 0;
}

int tpu_soc_v2_program_next_ab_preload_int8_int32_with_report(tpu_soc_fp32_t *ctx,
                                                              const tpu_soc_fp32_buffer_t *a_buf,
                                                              const tpu_soc_fp32_buffer_t *b_buf,
                                                              uint32_t m,
                                                              uint32_t n,
                                                              uint32_t k,
                                                              tpu_soc_v2_ab_preload_report_t *report) {
    return tpu_soc_v2_program_next_ab_preload_int8_int32_common(ctx,
                                                                a_buf,
                                                                b_buf,
                                                                m,
                                                                n,
                                                                k,
                                                                0,
                                                                report);
}

int tpu_soc_v2_program_next_ab_preload_int8_int32_presynced_with_report(tpu_soc_fp32_t *ctx,
                                                                        const tpu_soc_fp32_buffer_t *a_buf,
                                                                        const tpu_soc_fp32_buffer_t *b_buf,
                                                                        uint32_t m,
                                                                        uint32_t n,
                                                                        uint32_t k,
                                                                        tpu_soc_v2_ab_preload_report_t *report) {
    return tpu_soc_v2_program_next_ab_preload_int8_int32_common(ctx,
                                                                a_buf,
                                                                b_buf,
                                                                m,
                                                                n,
                                                                k,
                                                                1,
                                                                report);
}

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
                                                                       tpu_soc_v2_sequence_report_t *report) {
    size_t a_bytes = 0u;
    size_t b_bytes = 0u;
    size_t c_bytes = 0u;
    size_t d_bytes = 0u;
    uint32_t config_word;
    uint32_t seq_ctrl;
    uint32_t seq_status = 0u;
    uint32_t reuse_status = 0u;
    uint64_t deadline;
    uint64_t execute_entry_ns = monotonic_ns();
    uint64_t t0;
    unsigned sleep_us_value;
    uint32_t i;

    if (ctx == NULL || a_bufs == NULL || b_bufs == NULL ||
        run_count < 1u || run_count > TPU_VANILLA_SEQ_MAX_RUNS) {
        errno = EINVAL;
        return -1;
    }
    if (report != NULL) {
        memset(report, 0, sizeof(*report));
    }
    ensure_v2_exec_cfg_cached(ctx);
    if (!ctx->v2_hw_ingress) {
        return failf(ENOTSUP, "sequence controller requires TPU_HW_INGRESS=1 and a bitstream with SEQ CSR registers");
    }
    if (ctx->v2_gemv_m1 || ctx->v2_use_sg || ctx->v2_transfer_only) {
        return failf(ENOTSUP, "sequence controller supports only the normal hw_ingress path");
    }
    if (timeout_ms == 0u) {
        timeout_ms = 10000u;
    }
    if (resolve_v2_sizes_cached(ctx, m, n, k, &a_bytes, &b_bytes, &c_bytes, &d_bytes) != 0) {
        return -1;
    }
    (void)c_bytes;
    (void)d_bytes;

    for (i = 0u; i < run_count; ++i) {
        if (a_bufs[i].mapping == NULL || b_bufs[i].mapping == NULL ||
            a_bufs[i].device_addr > UINT32_MAX || b_bufs[i].device_addr > UINT32_MAX) {
            return failf(EINVAL, "sequence A/B buffers must be mapped and use 32-bit device addresses");
        }
        if (validate_dma_source_addr_cached(ctx, a_bufs[i].device_addr, a_bytes) != 0 ||
            validate_dma_source_addr_cached(ctx, b_bufs[i].device_addr, b_bytes) != 0) {
            return -1;
        }
    }

    t0 = monotonic_ns();
    if (wait_status_mask(ctx, TPU_VANILLA_STATUS_READY, 0u, timeout_ms, NULL) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->ready_wait_ns = monotonic_ns() - t0;
    }

    config_word = (TPU_VANILLA_PM_INT8_INT32 << 24) | (m << 16) | (n << 8) | k;
    seq_ctrl = TPU_VANILLA_SEQ_CTRL_START |
               ((run_count << TPU_VANILLA_SEQ_CTRL_RUN_COUNT_SHIFT) &
                TPU_VANILLA_SEQ_CTRL_RUN_COUNT_MASK);
    if (start_prev_d) {
        seq_ctrl |= TPU_VANILLA_SEQ_CTRL_START_PREV_D;
    }
    if (final_retain) {
        seq_ctrl |= TPU_VANILLA_SEQ_CTRL_FINAL_RETAIN;
    }

    t0 = monotonic_ns();
    if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_CONFIG, config_word) != 0 ||
        axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_GEMV_CTRL, 0u) != 0) {
        return -1;
    }
    for (i = 0u; i < run_count; ++i) {
        if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_SEQ_DESC_INDEX, i) != 0 ||
            axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_SEQ_DESC_A_ADDR, (uint32_t)a_bufs[i].device_addr) != 0 ||
            axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_SEQ_DESC_B_ADDR, (uint32_t)b_bufs[i].device_addr) != 0) {
            return -1;
        }
    }
    if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_SEQ_CTRL, seq_ctrl) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->program_ns = monotonic_ns() - t0;
    }

    sleep_us_value = poll_sleep_us_default();
    deadline = monotonic_ms() + timeout_ms;
    t0 = monotonic_ns();
    while (monotonic_ms() <= deadline) {
        if (axil_mmio_read32(&ctx->csr_mmio, TPU_VANILLA_REG_SEQ_STATUS, &seq_status) != 0) {
            return -1;
        }
        if ((seq_status & TPU_VANILLA_SEQ_STATUS_ERROR) != 0u) {
            return failf(EIO, "hardware sequence controller reported error (SEQ_STATUS=0x%08x)", seq_status);
        }
        if ((seq_status & TPU_VANILLA_SEQ_STATUS_DONE) != 0u &&
            (seq_status & TPU_VANILLA_SEQ_STATUS_ACTIVE) == 0u) {
            if (report != NULL) {
                uint32_t preload01 = 0u;
                uint32_t preload23 = 0u;
                uint32_t preload_status = 0u;
                report->wait_done_ns = monotonic_ns() - t0;
                report->seq_status = seq_status;
                report->execute_total_ns = monotonic_ns() - execute_entry_ns;
                if (axil_mmio_read32(&ctx->csr_mmio,
                                      TPU_VANILLA_REG_PERF_PRELOAD01_CYCLES,
                                      &preload01) != 0 ||
                    axil_mmio_read32(&ctx->csr_mmio,
                                      TPU_VANILLA_REG_PERF_PRELOAD23_CYCLES,
                                      &preload23) != 0 ||
                    axil_mmio_read32(&ctx->csr_mmio,
                                      TPU_VANILLA_REG_PERF_PRELOAD_STATUS,
                                      &preload_status) != 0) {
                    return -1;
                }
                report->preload_hw_cycles[0] = preload01 & 0xffffu;
                report->preload_hw_cycles[1] = (preload01 >> 16) & 0xffffu;
                report->preload_hw_cycles[2] = preload23 & 0xffffu;
                report->preload_hw_cycles[3] = (preload23 >> 16) & 0xffffu;
                report->preload_hw_count = (preload_status >> 16) & 0x7u;
                report->preload_hw_status = preload_status;
            }
            if (tpu_soc_v2_get_reuse_status(ctx, &reuse_status) != 0) {
                return -1;
            }
            if ((reuse_status & TPU_VANILLA_CACHE_STATUS_MODE_ERROR) != 0u) {
                return failf(EIO, "hardware reuse path reported mode error after sequence (CACHE_STATUS=0x%08x)",
                             reuse_status);
            }
            return 0;
        }
        if (sleep_us_value != 0u) {
            sleep_us(sleep_us_value);
        }
    }
    return failf(ETIMEDOUT, "timeout waiting for sequence done (last SEQ_STATUS=0x%08x)", seq_status);
}

int tpu_soc_v2_run_prev_d_sequence_prepacked_int8_int32_with_report(tpu_soc_fp32_t *ctx,
                                                                    const tpu_soc_fp32_buffer_t *a_bufs,
                                                                    const tpu_soc_fp32_buffer_t *b_bufs,
                                                                    uint32_t run_count,
                                                                    uint32_t m,
                                                                    uint32_t n,
                                                                    uint32_t k,
                                                                    unsigned timeout_ms,
                                                                    tpu_soc_v2_sequence_report_t *report) {
    return tpu_soc_v2_run_prev_d_sequence_prepacked_int8_int32_ex_with_report(ctx,
                                                                              a_bufs,
                                                                              b_bufs,
                                                                              run_count,
                                                                              m,
                                                                              n,
                                                                              k,
                                                                              0,
                                                                              0,
                                                                              timeout_ms,
                                                                              report);
}

int tpu_soc_v2_wait_int8_int32_active(tpu_soc_fp32_t *ctx,
                                      uint32_t c_mode,
                                      uint32_t d_mode,
                                      unsigned timeout_ms,
                                      tpu_soc_v2_execute_report_t *report) {
    uint32_t reuse_status = 0u;
    uint64_t t0;

    if (ctx == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (timeout_ms == 0u) {
        timeout_ms = 10000u;
    }

    t0 = monotonic_ns();
    if (d_mode == TPU_SOC_V2_D_MODE_READBACK) {
        if (wait_status_mask(ctx, TPU_VANILLA_STATUS_DONE, 0u, timeout_ms, NULL) != 0 ||
            wait_status_mask(ctx, 0u, TPU_VANILLA_STATUS_STORE_BUSY, timeout_ms, NULL) != 0) {
            return -1;
        }
    } else {
        const uint32_t busy_mask = TPU_VANILLA_STATUS_LOAD_BUSY |
                                   TPU_VANILLA_STATUS_COMPUTE_BUSY |
                                   TPU_VANILLA_STATUS_STORE_BUSY;
        if (wait_status_mask(ctx, TPU_VANILLA_STATUS_READY, busy_mask, timeout_ms, NULL) != 0) {
            return -1;
        }
    }
    if (c_mode == TPU_SOC_V2_C_MODE_PREV_D || d_mode == TPU_SOC_V2_D_MODE_RETAIN) {
        if (tpu_soc_v2_get_reuse_status(ctx, &reuse_status) != 0) {
            return -1;
        }
        if ((reuse_status & TPU_VANILLA_CACHE_STATUS_MODE_ERROR) != 0u) {
            return failf(EIO, "hardware reuse path reported mode error (CACHE_STATUS=0x%08x)", reuse_status);
        }
    }
    if (report != NULL) {
        report->wait_done_ns += monotonic_ns() - t0;
    }
    return 0;
}

int tpu_soc_v2_execute_int8_int32_with_report(tpu_soc_fp32_t *ctx,
                                              const tpu_soc_fp32_buffer_t *a_buf,
                                              const tpu_soc_fp32_buffer_t *b_buf,
                                              const tpu_soc_fp32_buffer_t *c_buf,
                                              uint32_t m,
                                              uint32_t n,
                                              uint32_t k,
                                              int sparse,
                                              unsigned timeout_ms,
                                              tpu_soc_v2_execute_report_t *report) {
    size_t a_bytes = 0u;
    size_t b_bytes = 0u;
    size_t c_bytes = 0u;
    size_t d_bytes = 0u;
    uint32_t config_word;
    uint32_t ctrl_word;
    int do_tail_fixup;
    int use_sg;
    int use_hw_ingress;
    int use_gemv_m1;
    int c_source_uses_host;
    int d_mode_readback;
    uint32_t hw_load_ctrl;
    uint32_t reuse_status = 0u;
    unsigned sg_segment_count = 2u;
    unsigned post_start_delay_us;
    unsigned between_cdma_delay_us_ab;
    unsigned between_cdma_delay_us_bc;
    unsigned post_cdma_delay_us;
    tpu_soc_fp32_cdma_segment_t sg_segments[3];
    uint64_t t0;
    uint64_t execute_entry_ns = monotonic_ns();

    if (ctx == NULL || a_buf == NULL || b_buf == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (report != NULL) {
        memset(report, 0, sizeof(*report));
        report->selected_path = "simple";
    }
    ensure_v2_exec_cfg_cached(ctx);
    c_source_uses_host = (ctx->v2_c_mode == TPU_SOC_V2_C_MODE_HOST);
    d_mode_readback = (ctx->v2_d_mode == TPU_SOC_V2_D_MODE_READBACK);
    if (c_source_uses_host) {
        return failf(ENOTSUP,
                     "HOST_C is disabled in the ZERO/PREV_D no-C-RAM build; use ZERO or PREV_D");
    }
    if (a_buf->mapping == NULL || b_buf->mapping == NULL) {
        return failf(EINVAL, "A/B XRT buffers must be allocated before execute");
    }
    if (c_source_uses_host && (c_buf == NULL || c_buf->mapping == NULL)) {
        return failf(EINVAL, "HOST C mode requires an allocated C buffer");
    }
    if (sparse) {
        return failf(ENOTSUP, "sparse mode is not enabled in the current v2 INT8 runtime path");
    }
    if (timeout_ms == 0u) {
        timeout_ms = 10000u;
    }
    do_tail_fixup = ctx->v2_tail_fixup;
    use_sg = ctx->v2_use_sg;
    use_hw_ingress = ctx->v2_hw_ingress;
    use_gemv_m1 = ctx->v2_gemv_m1;
    if (use_gemv_m1) {
        return failf(ENOTSUP,
                     "TPU_GEMV_M1 is disabled in the ZERO/PREV_D no-C-RAM build");
    }
    if (use_hw_ingress && !ctx->v2_trace_cdma_path) {
        cdma_trace_lazy_reset(&ctx->last_cdma_trace);
    } else {
        tpu_soc_fp32_cdma_trace_clear(&ctx->last_cdma_trace);
    }

    if (use_hw_ingress && use_sg) {
        return failf(ENOTSUP, "TPU_HW_INGRESS and TPU_CDMA_USE_SG cannot be enabled together");
    }
    if (use_hw_ingress && ctx->v2_transfer_only) {
        return failf(ENOTSUP, "TPU_V2_TRANSFER_ONLY is not supported with TPU_HW_INGRESS");
    }

    if (resolve_v2_sizes_cached(ctx, m, n, k, &a_bytes, &b_bytes, &c_bytes, &d_bytes) != 0) {
        return -1;
    }
    (void)d_bytes;

    hw_load_ctrl = TPU_VANILLA_LOAD_CTRL_MASK_ALL;
    if (use_hw_ingress) {
        hw_load_ctrl = TPU_VANILLA_LOAD_CTRL_MODE_HW | ctx->v2_hw_load_mask;
    }
    if (!c_source_uses_host) {
        hw_load_ctrl &= ~TPU_VANILLA_LOAD_CTRL_MASK_C;
    }

    if (((!use_hw_ingress) || ((hw_load_ctrl & TPU_VANILLA_LOAD_CTRL_MASK_A) != 0u)) &&
        ensure_cached_input_addr(ctx,
                                 &ctx->v2_a_addr_cache_valid,
                                 &ctx->v2_cached_a_addr,
                                 &ctx->v2_cached_a_size,
                                 a_buf->device_addr,
                                 a_bytes) != 0) {
        return -1;
    }
    if (((!use_hw_ingress) || ((hw_load_ctrl & TPU_VANILLA_LOAD_CTRL_MASK_B) != 0u)) &&
        ensure_cached_input_addr(ctx,
                                 &ctx->v2_b_addr_cache_valid,
                                 &ctx->v2_cached_b_addr,
                                 &ctx->v2_cached_b_size,
                                 b_buf->device_addr,
                                 b_bytes) != 0) {
        return -1;
    }
    if (c_source_uses_host &&
        (((!use_hw_ingress) || ((hw_load_ctrl & TPU_VANILLA_LOAD_CTRL_MASK_C) != 0u))) &&
        ensure_cached_input_addr(ctx,
                                 &ctx->v2_c_addr_cache_valid,
                                 &ctx->v2_cached_c_addr,
                                 &ctx->v2_cached_c_size,
                                 c_buf->device_addr,
                                 c_bytes) != 0) {
        return -1;
    }

    t0 = monotonic_ns();
    if (tpu_soc_v2_soft_reset(ctx, timeout_ms) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->soft_reset_ns = monotonic_ns() - t0;
    }

    /*
     * Keep the TPU-side reset ahead of the XRT BO flushes on the public
     * run()/execute() path. On board, very short soft-reset holds are stable
     * in the hot-stage path but can get lost here if we place the BO syncs
     * immediately before the reset write.
     */
    t0 = monotonic_ns();
    if (((!use_hw_ingress) || ((hw_load_ctrl & TPU_VANILLA_LOAD_CTRL_MASK_A) != 0u)) &&
        tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)a_buf, a_bytes) != 0) {
        return -1;
    }
    if (((!use_hw_ingress) || ((hw_load_ctrl & TPU_VANILLA_LOAD_CTRL_MASK_B) != 0u)) &&
        tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)b_buf, b_bytes) != 0) {
        return -1;
    }
    if (c_source_uses_host &&
        (((!use_hw_ingress) || ((hw_load_ctrl & TPU_VANILLA_LOAD_CTRL_MASK_C) != 0u))) &&
        tpu_soc_fp32_buffer_sync_to_device((tpu_soc_fp32_buffer_t *)c_buf, c_bytes) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->sync_ns = monotonic_ns() - t0;
    }

    t0 = monotonic_ns();
    if (tpu_soc_fp32_cdma_reset_with_trace(ctx, &ctx->last_cdma_trace) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->cdma_reset_ns = monotonic_ns() - t0;
    }
    t0 = monotonic_ns();
    if (wait_status_mask(ctx, TPU_VANILLA_STATUS_READY, 0u, timeout_ms, NULL) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->ready_wait_ns = monotonic_ns() - t0;
    }
    if (tpu_soc_v2_set_reuse_mode(ctx, ctx->v2_c_mode, ctx->v2_d_mode, 0) != 0) {
        return -1;
    }
    if (ctx->v2_c_mode == TPU_SOC_V2_C_MODE_PREV_D) {
        if (tpu_soc_v2_get_reuse_status(ctx, &reuse_status) != 0) {
            return -1;
        }
        if ((reuse_status & TPU_VANILLA_CACHE_STATUS_PREV_D_VALID) == 0u) {
            return failf(ENOENT, "PREV_D requested but retained D slot is invalid");
        }
    }

    config_word = (TPU_VANILLA_PM_INT8_INT32 << 24) | (m << 16) | (n << 8) | k;
    ctrl_word = TPU_VANILLA_CTRL_ENABLE;
    post_start_delay_us = ctx->v2_post_start_delay_us;
    between_cdma_delay_us_ab = ctx->v2_between_cdma_delay_us_ab;
    between_cdma_delay_us_bc = ctx->v2_between_cdma_delay_us_bc;
    post_cdma_delay_us = ctx->v2_post_cdma_delay_us;
    t0 = monotonic_ns();
    if (use_hw_ingress) {
        if (a_buf->device_addr > UINT32_MAX ||
            b_buf->device_addr > UINT32_MAX ||
            (c_source_uses_host && c_buf->device_addr > UINT32_MAX)) {
            return failf(ERANGE, "TPU_HW_INGRESS currently requires 32-bit source device addresses");
        }
        if (axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_A_ADDR, (uint32_t)a_buf->device_addr) != 0 ||
            axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_B_ADDR, (uint32_t)b_buf->device_addr) != 0 ||
            axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_CTRL, hw_load_ctrl) != 0 ||
            axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_GEMV_CTRL,
                              use_gemv_m1 ? TPU_VANILLA_GEMV_CTRL_M1_EN : 0u) != 0) {
            return -1;
        }
        if (c_source_uses_host &&
            axil_mmio_write32(&ctx->csr_mmio, TPU_VANILLA_REG_LOAD_C_ADDR, (uint32_t)c_buf->device_addr) != 0) {
            return -1;
        }
    }
    if (report != NULL) {
        report->path_setup_ns = monotonic_ns() - t0;
    }

    t0 = monotonic_ns();
    if (program_config_then_start(ctx, config_word, ctrl_word, timeout_ms) != 0) {
        return -1;
    }
    if (report != NULL) {
        report->start_ns = monotonic_ns() - t0;
    }
    if (post_start_delay_us != 0u) {
        t0 = monotonic_ns();
        sleep_us(post_start_delay_us);
        if (report != NULL) {
            report->post_start_delay_ns = monotonic_ns() - t0;
        }
    }

    if (ctx->v2_trace_cdma_path) {
        fprintf(stderr,
                "CDMA_PATH use_sg=%d use_hw_ingress=%d gemv_m1=%d tail_fixup=%d delays(ab=%u,bc=%u,post=%u)\n",
                use_sg,
                use_hw_ingress,
                use_gemv_m1,
                do_tail_fixup,
                between_cdma_delay_us_ab,
                between_cdma_delay_us_bc,
                post_cdma_delay_us);
    }

    if (use_hw_ingress) {
        uint32_t load_status = 0u;

        if (ctx->v2_trace_cdma_path) {
            fprintf(stderr, "CDMA_PATH selected=hw-ingress%s\n",
                    use_gemv_m1 ? "+gemv-m1" : "");
        }
        if (report != NULL) {
            report->selected_path = use_gemv_m1 ? "hw_ingress+gemv_m1" : "hw_ingress";
        }
        t0 = monotonic_ns();
        if (wait_csr_mask(ctx,
                          TPU_VANILLA_REG_LOAD_STATUS,
                          TPU_VANILLA_LOAD_STATUS_DONE,
                          TPU_VANILLA_LOAD_STATUS_BUSY,
                          timeout_ms,
                          &load_status) != 0) {
            return -1;
        }
        if ((load_status & TPU_VANILLA_LOAD_STATUS_ERROR) != 0u) {
            return failf(EIO, "hardware ingress completed with error (LOAD_STATUS=0x%08x)", load_status);
        }
        if (report != NULL) {
            report->transfer_ns = monotonic_ns() - t0;
        }
    } else if (use_sg) {
        if (ctx->v2_trace_cdma_path) {
            fprintf(stderr, "CDMA_PATH selected=sg\n");
        }
        if (report != NULL) {
            report->selected_path = "sg";
        }
        if (do_tail_fixup) {
        return execute_report_finish(report,
                                     execute_entry_ns,
                                     failf(ENOTSUP,
                                           "TPU_CDMA_USE_SG does not support CDMA tail fixups"));
        }
        if (between_cdma_delay_us_ab != 0u || between_cdma_delay_us_bc != 0u || post_cdma_delay_us != 0u) {
            fprintf(stderr,
                    "TPU_CDMA_USE_SG=1 ignores staged ingress/post-CDMA delays (ab=%u us, bc=%u us, post=%u us)\n",
                    between_cdma_delay_us_ab,
                    between_cdma_delay_us_bc,
                    post_cdma_delay_us);
        }

        sg_segments[0].label = "A-main";
        sg_segments[0].src_addr = a_buf->device_addr;
        sg_segments[0].dst_addr = TPU_SOC_FP32_CDMA_MATRIX_A;
        sg_segments[0].size_bytes = a_bytes;
        sg_segments[1].label = "B-main";
        sg_segments[1].src_addr = b_buf->device_addr;
        sg_segments[1].dst_addr = TPU_SOC_FP32_CDMA_MATRIX_B;
        sg_segments[1].size_bytes = b_bytes;
        if (c_source_uses_host) {
            sg_segments[2].label = "C-main";
            sg_segments[2].src_addr = c_buf->device_addr;
            sg_segments[2].dst_addr = TPU_SOC_FP32_CDMA_MATRIX_C;
            sg_segments[2].size_bytes = c_bytes;
            sg_segment_count = 3u;
        }

        t0 = monotonic_ns();
        if (tpu_soc_fp32_cdma_transfer_sg_chain_with_trace(ctx,
                                                           &ctx->last_cdma_trace,
                                                           c_source_uses_host ? "ABC-sg" : "AB-sg",
                                                           sg_segments,
                                                           sg_segment_count,
                                                           timeout_ms) != 0) {
            return -1;
        }
        if (report != NULL) {
            report->transfer_ns = monotonic_ns() - t0;
        }
    } else {
        if (ctx->v2_trace_cdma_path) {
            fprintf(stderr, "CDMA_PATH selected=simple\n");
        }
        if (report != NULL) {
            report->selected_path = "simple";
        }
        t0 = monotonic_ns();
        if (tpu_soc_fp32_cdma_transfer_with_trace(ctx,
                                                  &ctx->last_cdma_trace,
                                                  "A-main",
                                                  a_buf->device_addr,
                                                  TPU_SOC_FP32_CDMA_MATRIX_A,
                                                  a_bytes,
                                                  timeout_ms) != 0) {
            return -1;
        }
        if (between_cdma_delay_us_ab != 0u) {
            handle_ingress_gap(ctx,
                               "A->B",
                               between_cdma_delay_us_ab,
                               TPU_VANILLA_DBG_INGRESS_AB_SAFE);
        }
        if (tpu_soc_fp32_cdma_transfer_with_trace(ctx,
                                                  &ctx->last_cdma_trace,
                                                  "B-main",
                                                  b_buf->device_addr,
                                                  TPU_SOC_FP32_CDMA_MATRIX_B,
                                                  b_bytes,
                                                  timeout_ms) != 0) {
            return -1;
        }
        if (between_cdma_delay_us_bc != 0u) {
            handle_ingress_gap(ctx,
                               "B->C",
                               between_cdma_delay_us_bc,
                               TPU_VANILLA_DBG_INGRESS_BC_SAFE);
        }
        if (c_source_uses_host) {
            if (tpu_soc_fp32_cdma_transfer_with_trace(ctx,
                                                      &ctx->last_cdma_trace,
                                                      "C-main",
                                                      c_buf->device_addr,
                                                      TPU_SOC_FP32_CDMA_MATRIX_C,
                                                      c_bytes,
                                                      timeout_ms) != 0) {
                return -1;
            }
        }
        if (post_cdma_delay_us != 0u) {
            sleep_us(post_cdma_delay_us);
        }
        if (report != NULL) {
            report->transfer_ns = monotonic_ns() - t0;
        }
    }

    if (!use_hw_ingress && do_tail_fixup) {
        t0 = monotonic_ns();
        if (tpu_soc_fp32_cdma_tail_fixup_with_trace(ctx,
                                                    &ctx->last_cdma_trace,
                                                    "A-tail",
                                                    a_buf->device_addr,
                                                    TPU_SOC_FP32_CDMA_MATRIX_A,
                                                    a_bytes,
                                                    timeout_ms) != 0 ||
            tpu_soc_fp32_cdma_tail_fixup_with_trace(ctx,
                                                    &ctx->last_cdma_trace,
                                                    "B-tail",
                                                    b_buf->device_addr,
                                                    TPU_SOC_FP32_CDMA_MATRIX_B,
                                                    b_bytes,
                                                    timeout_ms) != 0 ||
            (c_source_uses_host &&
             tpu_soc_fp32_cdma_tail_fixup_with_trace(ctx,
                                                     &ctx->last_cdma_trace,
                                                     "C-tail",
                                                     c_buf->device_addr,
                                                     TPU_SOC_FP32_CDMA_MATRIX_C,
                                                     c_bytes,
                                                     timeout_ms) != 0)) {
            return -1;
        }
        if (report != NULL) {
            report->tail_fixup_ns = monotonic_ns() - t0;
        }
    } else {
        if (ctx->v2_trace_cdma_path) {
            fprintf(stderr,
                    "Skipping CDMA tail fixups (default off; set TPU_ENABLE_CDMA_TAIL_FIXUP=1 to re-enable the historical workaround)\n");
        }
    }

    if (ctx->v2_transfer_only) {
        fprintf(stderr,
                "Transfer-only mode: stopping after A/B/C transfers and skipping wait_done.\n");
        return execute_report_finish(report, execute_entry_ns, 0);
    }

    t0 = monotonic_ns();
    if (d_mode_readback) {
        if (wait_status_mask(ctx, TPU_VANILLA_STATUS_DONE, 0u, timeout_ms, NULL) != 0) {
            return -1;
        }
        if (wait_status_mask(ctx, 0u, TPU_VANILLA_STATUS_STORE_BUSY, timeout_ms, NULL) != 0) {
            return -1;
        }
    } else {
        const uint32_t ready_mask = TPU_VANILLA_STATUS_READY;
        const uint32_t busy_mask = TPU_VANILLA_STATUS_LOAD_BUSY |
                                   TPU_VANILLA_STATUS_COMPUTE_BUSY |
                                   TPU_VANILLA_STATUS_STORE_BUSY;
        if (wait_status_mask(ctx, ready_mask, busy_mask, timeout_ms, NULL) != 0) {
            return -1;
        }
    }
    if (ctx->v2_c_mode == TPU_SOC_V2_C_MODE_PREV_D || ctx->v2_d_mode == TPU_SOC_V2_D_MODE_RETAIN) {
        if (tpu_soc_v2_get_reuse_status(ctx, &reuse_status) != 0) {
            return -1;
        }
        if ((reuse_status & TPU_VANILLA_CACHE_STATUS_MODE_ERROR) != 0u) {
            return failf(EIO, "hardware reuse path reported mode error (CACHE_STATUS=0x%08x)", reuse_status);
        }
    }
    if (report != NULL) {
        report->wait_done_ns = monotonic_ns() - t0;
    }

    return execute_report_finish(report, execute_entry_ns, 0);
}

int tpu_soc_v2_execute_int8_int32(tpu_soc_fp32_t *ctx,
                                  const tpu_soc_fp32_buffer_t *a_buf,
                                  const tpu_soc_fp32_buffer_t *b_buf,
                                  const tpu_soc_fp32_buffer_t *c_buf,
                                  uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  int sparse,
                                  unsigned timeout_ms) {
    return tpu_soc_v2_execute_int8_int32_with_report(ctx,
                                                     a_buf,
                                                     b_buf,
                                                     c_buf,
                                                     m,
                                                     n,
                                                     k,
                                                     sparse,
                                                     timeout_ms,
                                                     NULL);
}

int tpu_soc_v2_run(const tpu_soc_v2_run_args_t *args, tpu_soc_v2_run_report_t *report) {
    tpu_soc_fp32_t ctx;
    tpu_soc_fp32_buffer_t a_buf;
    tpu_soc_fp32_buffer_t b_buf;
    tpu_soc_fp32_buffer_t c_buf;
    tpu_soc_fp32_buffer_t d_buf;
    uint64_t *a_words = NULL;
    uint64_t *b_words = NULL;
    uint64_t *c_words = NULL;
    size_t a_bytes = 0u;
    size_t b_bytes = 0u;
    size_t c_bytes = 0u;
    size_t d_bytes = 0u;
    uint32_t a_word_count;
    uint32_t b_word_count;
    uint32_t c_word_count;
    size_t d_transfer_bytes = 0u;
    int rc = -1;
    int ctx_open = 0;
    int output_buffer_enabled = 0;
    int skip_clear_if_outbuf = skip_clear_if_outbuf_enabled();

    if (args == NULL) {
        errno = EINVAL;
        return -1;
    }
    if (args->c_mode > TPU_SOC_V2_C_MODE_PREV_D) {
        return failf(EINVAL, "unsupported c_mode=%u", args->c_mode);
    }
    if (args->c_mode == TPU_SOC_V2_C_MODE_HOST) {
        return failf(ENOTSUP,
                     "HOST_C is disabled in the ZERO/PREV_D no-C-RAM build; use ZERO or PREV_D");
    }
    if (args->d_mode > TPU_SOC_V2_D_MODE_RETAIN) {
        return failf(EINVAL, "unsupported d_mode=%u", args->d_mode);
    }
    if (args->a_addr == NULL || args->b_addr == NULL) {
        return failf(EINVAL, "a_addr and b_addr must be non-null");
    }
    if (args->c_mode == TPU_SOC_V2_C_MODE_HOST && args->c_addr == NULL) {
        return failf(EINVAL, "HOST c_mode requires non-null c_addr");
    }
    if (args->d_mode == TPU_SOC_V2_D_MODE_READBACK && args->d_addr == NULL) {
        return failf(EINVAL, "READBACK d_mode requires non-null d_addr");
    }
    if (args->precision_mode != TPU_VANILLA_PM_INT8_INT32) {
        return failf(ENOTSUP,
                     "precision mode %u is not supported by the current signed-off v2 runtime; only TPU_VANILLA_PM_INT8_INT32 is supported",
                     args->precision_mode);
    }

    if (report != NULL) {
        tpu_soc_v2_run_report_init(report);
    }

    if (tpu_soc_v2_validate_sizes(args->m, args->n, args->k, &a_bytes, &b_bytes, &c_bytes, &d_bytes) != 0) {
        return -1;
    }

    a_word_count = tpu_soc_v2_int8_word_count(args->m, args->k);
    b_word_count = tpu_soc_v2_int8_word_count(args->k, args->n);
    c_word_count = tpu_soc_v2_int32_word_count(args->m, args->n);
    d_transfer_bytes = (size_t)c_word_count * sizeof(uint64_t);

    a_words = (uint64_t *)calloc(a_word_count, sizeof(uint64_t));
    b_words = (uint64_t *)calloc(b_word_count, sizeof(uint64_t));
    if (args->c_mode == TPU_SOC_V2_C_MODE_HOST) {
        c_words = (uint64_t *)calloc(c_word_count, sizeof(uint64_t));
    }
    if (a_words == NULL || b_words == NULL || (args->c_mode == TPU_SOC_V2_C_MODE_HOST && c_words == NULL)) {
        errno = ENOMEM;
        goto out;
    }

    tpu_soc_v2_pack_a_int8_row_major((const int8_t *)args->a_addr, args->m, args->k, a_words);
    tpu_soc_v2_pack_b_int8_nk_to_kn((const int8_t *)args->b_addr, args->n, args->k, b_words);
    if (args->c_mode == TPU_SOC_V2_C_MODE_HOST) {
        tpu_soc_v2_pack_c_int32_row_major((const int32_t *)args->c_addr, args->m, args->n, c_words);
    }

    memset(&ctx, 0, sizeof(ctx));
    memset(&a_buf, 0, sizeof(a_buf));
    memset(&b_buf, 0, sizeof(b_buf));
    memset(&c_buf, 0, sizeof(c_buf));
    memset(&d_buf, 0, sizeof(d_buf));

    if (tpu_soc_fp32_open(&ctx,
                          args->csr_base,
                          args->cdma_base,
                          args->bram_base,
                          args->device_index,
                          args->mem_group,
                          (args->devmem_path != NULL) ? args->devmem_path : "/dev/mem") != 0) {
        goto out;
    }
    ctx_open = 1;

    if (report != NULL && tpu_soc_fp32_read_regs(&ctx, &report->initial_regs) == 0) {
        report->have_initial_regs = 1;
    }

    if (args->run_preflight) {
        if (tpu_soc_fp32_preflight(&ctx, NULL) != 0) {
            goto out;
        }
    }

    ctx.v2_c_mode = args->c_mode;
    ctx.v2_d_mode = args->d_mode;
    if (args->clear_prev_d) {
        g_prev_d_shape_valid = 0;
        g_prev_d_m = 0u;
        g_prev_d_n = 0u;
    }
    if (args->c_mode == TPU_SOC_V2_C_MODE_PREV_D) {
        if (!g_prev_d_shape_valid) {
            rc = failf(ENOENT, "PREV_D requested but no retained D metadata is present");
            goto out;
        }
        if (g_prev_d_m != args->m || g_prev_d_n != args->n) {
            rc = failf(EINVAL,
                       "PREV_D shape mismatch: retained M/N=%ux%u, requested M/N=%ux%u",
                       g_prev_d_m,
                       g_prev_d_n,
                       args->m,
                       args->n);
            goto out;
        }
    }

    if (tpu_soc_fp32_alloc_buffer(&ctx, &a_buf, a_bytes) != 0 ||
        tpu_soc_fp32_alloc_buffer(&ctx, &b_buf, b_bytes) != 0) {
        goto out;
    }
    if (args->c_mode == TPU_SOC_V2_C_MODE_HOST &&
        tpu_soc_fp32_alloc_buffer(&ctx, &c_buf, c_bytes) != 0) {
        goto out;
    }
    if (args->d_mode == TPU_SOC_V2_D_MODE_READBACK &&
        tpu_soc_fp32_alloc_buffer(&ctx, &d_buf, d_transfer_bytes) == 0) {
        if (tpu_soc_v2_try_enable_output_buffer(&ctx, &d_buf, &output_buffer_enabled) != 0) {
            goto out;
        }
    }

    memcpy(a_buf.mapping, a_words, a_bytes);
    memcpy(b_buf.mapping, b_words, b_bytes);
    if (args->c_mode == TPU_SOC_V2_C_MODE_HOST) {
        memcpy(c_buf.mapping, c_words, c_bytes);
    }

    if (args->d_mode == TPU_SOC_V2_D_MODE_READBACK &&
        !(skip_clear_if_outbuf && output_buffer_enabled)) {
        if (tpu_soc_v2_clear_output_region(&ctx, args->m, args->n) != 0) {
            goto out;
        }
    }

    if (tpu_soc_v2_execute_int8_int32(&ctx,
                                      &a_buf,
                                      &b_buf,
                                      (args->c_mode == TPU_SOC_V2_C_MODE_HOST) ? &c_buf : NULL,
                                      args->m,
                                      args->n,
                                      args->k,
                                      args->sparse,
                                      args->timeout_ms) != 0) {
        goto out;
    }

    if (args->d_mode == TPU_SOC_V2_D_MODE_READBACK) {
        if (output_buffer_enabled) {
            if (tpu_soc_v2_read_result_int32_from_buffer(&d_buf, (int32_t *)args->d_addr, args->m * args->n) != 0) {
                goto out;
            }
        } else {
            if (tpu_soc_v2_read_result_int32(&ctx, (int32_t *)args->d_addr, args->m * args->n) != 0) {
                goto out;
            }
        }
    }

    if (report != NULL) {
        report->d_bytes = d_bytes;
        report->cdma_trace = ctx.last_cdma_trace;
        if (tpu_soc_fp32_read_regs(&ctx, &report->final_regs) == 0) {
            report->have_final_regs = 1;
        }
    }

    if (args->d_mode == TPU_SOC_V2_D_MODE_RETAIN) {
        g_prev_d_shape_valid = 1;
        g_prev_d_m = args->m;
        g_prev_d_n = args->n;
    } else {
        g_prev_d_shape_valid = 0;
        g_prev_d_m = 0u;
        g_prev_d_n = 0u;
    }

    rc = 0;

out:
    if (report != NULL && ctx_open) {
        report->d_bytes = d_bytes;
        report->cdma_trace = ctx.last_cdma_trace;
        if (!report->have_final_regs && tpu_soc_fp32_read_regs(&ctx, &report->final_regs) == 0) {
            report->have_final_regs = 1;
        }
    }
    tpu_soc_fp32_free_buffer(&a_buf);
    tpu_soc_fp32_free_buffer(&b_buf);
    tpu_soc_fp32_free_buffer(&c_buf);
    tpu_soc_fp32_free_buffer(&d_buf);
    if (ctx_open) {
        tpu_soc_fp32_close(&ctx);
    }
    free(a_words);
    free(b_words);
    free(c_words);
    return rc;
}
