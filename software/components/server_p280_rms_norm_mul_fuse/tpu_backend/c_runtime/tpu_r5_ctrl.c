#define _POSIX_C_SOURCE 200809L

#include "tpu_r5_ctrl.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

#define R5_TCM_BASE_DEFAULT 0xffe00000u
#define R5_TCM_SIZE         0x10000u
#define R5_SHM_OFF          0x8000u

#define CTRL_MAGIC          0x00u
#define CTRL_SEQ            0x04u
#define CTRL_DONE           0x08u
#define CTRL_REPS           0x0cu
#define CTRL_CYCLES         0x10u
#define CTRL_CHECKSUM       0x14u
#define CTRL_STATUS         0x18u
#define CTRL_CMD            0x1cu
#define CTRL_ARG0           0x20u
#define CTRL_ARG1           0x24u
#define CTRL_ARG2           0x28u
#define CTRL_ARG3           0x2cu
#define CTRL_ERROR          0x30u
#define CTRL_ARG4           0x34u
#define CTRL_ARG5           0x38u

#define MAGIC_R5C1          0x52354331u
#define CMD_WAIT_LIMIT      4u
#define CMD_WAIT_PAIR_LIMIT 5u
#define CMD_SEQ_RUN         6u
#define CMD_SEQ_BATCH_RUN   7u
#define CMD_MATMUL_TASK     8u
#define ERR_WAIT_TIMEOUT    2u
#define ERR_BAD_ARG         3u
#define ERR_HW_ERROR        4u

#define CTRL_SEQ_STATUS_RESULT    0x80u
#define CTRL_CACHE_STATUS_RESULT  0x84u
#define CTRL_PRELOAD01_RESULT     0x88u
#define CTRL_PRELOAD23_RESULT     0x8cu
#define CTRL_PRELOAD_STATUS_RESULT 0x90u

#define CTRL_SEQ_DESC_BASE   0x100u
#define CTRL_SEQ_DESC_STRIDE 0x08u
#define CTRL_SEQ_MAX_RUNS    TPU_R5_SEQ_MAX_RUNS

#define CTRL_SEQ_BATCH_BASE        0x200u
#define CTRL_SEQ_BATCH_STRIDE      0x80u
#define CTRL_SEQ_BATCH_CONFIG      0x00u
#define CTRL_SEQ_BATCH_SEQ_CTRL    0x04u
#define CTRL_SEQ_BATCH_RUN_COUNT   0x08u
#define CTRL_SEQ_BATCH_ERROR       0x0cu
#define CTRL_SEQ_BATCH_ADDR_BASE   0x10u
#define CTRL_SEQ_BATCH_RESULT_BASE 0x50u
#define CTRL_SEQ_BATCH_SEQ_STATUS  0x50u
#define CTRL_SEQ_BATCH_CACHE_STATUS 0x54u
#define CTRL_SEQ_BATCH_PRELOAD01   0x58u
#define CTRL_SEQ_BATCH_PRELOAD23   0x5cu
#define CTRL_SEQ_BATCH_PRELOAD_STATUS 0x60u
#define CTRL_SEQ_BATCH_MAX_COMMANDS TPU_R5_SEQ_BATCH_MAX_COMMANDS

#define CTRL_MATMUL_TASK_BASE           0x3000u
#define CTRL_MATMUL_TASK_MAGIC          0x00u
#define CTRL_MATMUL_TASK_VERSION        0x04u
#define CTRL_MATMUL_TASK_FLAGS          0x08u
#define CTRL_MATMUL_TASK_FENCE_ID       0x0cu
#define CTRL_MATMUL_TASK_A_BASE         0x10u
#define CTRL_MATMUL_TASK_B_PACKED_BASE  0x14u
#define CTRL_MATMUL_TASK_C_BASE         0x18u
#define CTRL_MATMUL_TASK_M              0x1cu
#define CTRL_MATMUL_TASK_N              0x20u
#define CTRL_MATMUL_TASK_K              0x24u
#define CTRL_MATMUL_TASK_LDA_BYTES      0x28u
#define CTRL_MATMUL_TASK_LDC_BYTES      0x2cu
#define CTRL_MATMUL_TASK_TILE_M         0x30u
#define CTRL_MATMUL_TASK_TILE_N         0x34u
#define CTRL_MATMUL_TASK_TILE_K         0x38u
#define CTRL_MATMUL_TASK_B_TILE_BYTES   0x3cu
#define CTRL_MATMUL_TASK_B_N_TILES      0x40u
#define CTRL_MATMUL_TASK_B_K_TILES      0x44u
#define CTRL_MATMUL_TASK_STATUS         0x48u
#define CTRL_MATMUL_TASK_ERROR          0x4cu
#define CTRL_MATMUL_TASK_COMPLETED      0x50u
#define CTRL_MATMUL_TASK_TOTAL_TILES    0x54u
#define CTRL_MATMUL_TASK_DONE_FENCE     0x58u
#define CTRL_MATMUL_TASK_R5_CYCLES      0x5cu
#define CTRL_MATMUL_TASK_FPGA_WAIT      0x60u
#define CTRL_MATMUL_TASK_LAST_HW_STATUS 0x64u
#define CTRL_MATMUL_TASK_SCRATCH_M0     0x68u
#define CTRL_MATMUL_TASK_SCRATCH_N0     0x6cu
#define CTRL_MATMUL_TASK_SCRATCH_K0     0x70u
#define CTRL_MATMUL_TASK_SCRATCH_NT     0x74u
#define CTRL_MATMUL_TASK_SCRATCH_KT     0x78u

static int g_fd = -1;
static volatile uint8_t *g_base;
static uint32_t g_seq;
static int g_warned_unavailable;
static int g_stats_registered;
static uint64_t g_wait_calls;
static uint64_t g_wait_success;
static uint64_t g_wait_pair_calls;
static uint64_t g_wait_chunk_timeouts;
static uint64_t g_wait_errors;
static uint64_t g_wait_total_ns;
static uint64_t g_seq_calls;
static uint64_t g_seq_success;
static uint64_t g_seq_errors;
static uint64_t g_seq_total_ns;
static uint64_t g_seq_batch_calls;
static uint64_t g_seq_batch_success;
static uint64_t g_seq_batch_errors;
static uint64_t g_seq_batch_total_ns;
static uint64_t g_matmul_task_calls;
static uint64_t g_matmul_task_success;
static uint64_t g_matmul_task_errors;
static uint64_t g_matmul_task_total_ns;
static uint32_t g_logged_wait_reg_offset = UINT32_MAX;

static uint64_t now_ns(void) {
    struct timespec ts;

    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

static int env_truthy(const char *name) {
    const char *value = getenv(name);

    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static uint32_t env_u32_or_default(const char *name, uint32_t default_value) {
    const char *value = getenv(name);
    char *endptr = NULL;
    unsigned long parsed;

    if (value == NULL || value[0] == '\0') {
        return default_value;
    }

    errno = 0;
    parsed = strtoul(value, &endptr, 0);
    if (errno != 0 || endptr == value || (endptr != NULL && *endptr != '\0') || parsed > UINT32_MAX) {
        return default_value;
    }
    return (uint32_t) parsed;
}

static uint32_t load32(uint32_t offset) {
    volatile uint32_t *p = (volatile uint32_t *) (void *) (g_base + R5_SHM_OFF + offset);

    return *p;
}

static void store32(uint32_t offset, uint32_t value) {
    volatile uint32_t *p = (volatile uint32_t *) (void *) (g_base + R5_SHM_OFF + offset);

    *p = value;
    __sync_synchronize();
}

static void store32_relaxed(uint32_t offset, uint32_t value) {
    volatile uint32_t *p = (volatile uint32_t *) (void *) (g_base + R5_SHM_OFF + offset);

    *p = value;
}

static void warn_unavailable_once(const char *reason) {
    if (!g_warned_unavailable && env_truthy("TPU_R5_WAIT_LOG")) {
        fprintf(stderr, "[FPGA/R5] status-wait unavailable, falling back to A53 polling: %s\n", reason);
        g_warned_unavailable = 1;
    }
}

static void log_wait_reg_offset_once(uint32_t reg_offset) {
    if (!env_truthy("TPU_R5_WAIT_LOG") || g_logged_wait_reg_offset == reg_offset) {
        return;
    }
    fprintf(stderr, "[FPGA/R5] wait poll register offset=0x%08" PRIx32 "\n", reg_offset);
    g_logged_wait_reg_offset = reg_offset;
}

static void print_stats(void) {
    if (env_truthy("TPU_R5_WAIT_LOG")) {
        fprintf(stderr,
                "[FPGA/R5] wait stats: calls=%" PRIu64 " success=%" PRIu64
                " pair_calls=%" PRIu64 " chunk_timeouts=%" PRIu64
                " errors=%" PRIu64 " total=%.3f ms avg=%.3f us\n",
                g_wait_calls,
                g_wait_success,
                g_wait_pair_calls,
                g_wait_chunk_timeouts,
                g_wait_errors,
                (double) g_wait_total_ns / 1000000.0,
                g_wait_calls != 0u ? (double) g_wait_total_ns / (double) g_wait_calls / 1000.0 : 0.0);
        if (g_seq_calls != 0u) {
            fprintf(stderr,
                    "[FPGA/R5] sequence stats: calls=%" PRIu64 " success=%" PRIu64
                    " errors=%" PRIu64 " total=%.3f ms avg=%.3f us\n",
                    g_seq_calls,
                    g_seq_success,
                    g_seq_errors,
                    (double) g_seq_total_ns / 1000000.0,
                    (double) g_seq_total_ns / (double) g_seq_calls / 1000.0);
        }
        if (g_seq_batch_calls != 0u) {
            fprintf(stderr,
                    "[FPGA/R5] sequence batch stats: calls=%" PRIu64 " success=%" PRIu64
                    " errors=%" PRIu64 " total=%.3f ms avg=%.3f us\n",
                    g_seq_batch_calls,
                    g_seq_batch_success,
                    g_seq_batch_errors,
                    (double) g_seq_batch_total_ns / 1000000.0,
                    (double) g_seq_batch_total_ns / (double) g_seq_batch_calls / 1000.0);
        }
        if (g_matmul_task_calls != 0u) {
            fprintf(stderr,
                    "[FPGA/R5] matmul task stats: calls=%" PRIu64 " success=%" PRIu64
                    " errors=%" PRIu64 " total=%.3f ms avg=%.3f us\n",
                    g_matmul_task_calls,
                    g_matmul_task_success,
                    g_matmul_task_errors,
                    (double) g_matmul_task_total_ns / 1000000.0,
                    (double) g_matmul_task_total_ns / (double) g_matmul_task_calls / 1000.0);
        }
    }
}

int tpu_r5_ctrl_enabled(void) {
    return env_truthy("TPU_R5_WAIT_STATUS");
}

int tpu_r5_ctrl_unavailable_errno(int err) {
    return err == ENOSYS || err == ENODEV || err == ENOENT || err == EACCES;
}

uint32_t tpu_r5_ctrl_status_reg_offset(uint32_t default_offset) {
    return env_u32_or_default("TPU_R5_STATUS_REG_OFFSET", default_offset);
}

static int tpu_r5_ctrl_init(void) {
    uint32_t tcm_base;
    long page_size;
    uint64_t page_mask;
    uint64_t map_base;
    uint64_t page_off;
    uint32_t magic;
    uint32_t status;

    if (g_base != NULL) {
        return 0;
    }

    if (!tpu_r5_ctrl_enabled()) {
        errno = ENOSYS;
        return -1;
    }

    tcm_base = env_u32_or_default("TPU_R5_TCM_BASE", R5_TCM_BASE_DEFAULT);
    page_size = sysconf(_SC_PAGESIZE);
    if (page_size <= 0) {
        errno = EINVAL;
        return -1;
    }
    page_mask = (uint64_t) page_size - 1u;
    map_base = (uint64_t) tcm_base & ~page_mask;
    page_off = (uint64_t) tcm_base - map_base;

    g_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (g_fd < 0) {
        warn_unavailable_once("open(/dev/mem) failed");
        return -1;
    }

    g_base = (volatile uint8_t *) mmap(NULL,
                                       R5_TCM_SIZE + (size_t) page_off,
                                       PROT_READ | PROT_WRITE,
                                       MAP_SHARED,
                                       g_fd,
                                       (off_t) map_base);
    if (g_base == MAP_FAILED) {
        g_base = NULL;
        close(g_fd);
        g_fd = -1;
        warn_unavailable_once("mmap(TCM) failed");
        return -1;
    }
    g_base += page_off;

    magic = load32(CTRL_MAGIC);
    status = load32(CTRL_STATUS);
    if (magic != MAGIC_R5C1 || status != 1u) {
        tpu_r5_ctrl_close();
        errno = ENODEV;
        warn_unavailable_once("R5 firmware magic/status not ready");
        return -1;
    }
    if (env_truthy("TPU_R5_WAIT_LOG")) {
        fprintf(stderr, "[FPGA/R5] status-wait ready: tcm=0x%08x\n", tcm_base);
    }
    /*
     * R5 keeps the last observed CTRL_SEQ in a private register.  A fresh
     * host process must continue from the shared CTRL_SEQ value instead of
     * starting at 1 again, otherwise R5 will see no edge and keep polling.
     */
    g_seq = load32(CTRL_SEQ);
    if (g_seq == 0u) {
        g_seq = load32(CTRL_DONE);
    }
    if (!g_stats_registered) {
        atexit(print_stats);
        g_stats_registered = 1;
    }

    return 0;
}

void tpu_r5_ctrl_close(void) {
    uint32_t tcm_base;
    long page_size;
    uint64_t page_mask;
    uint64_t page_off;

    if (g_base != NULL) {
        tcm_base = env_u32_or_default("TPU_R5_TCM_BASE", R5_TCM_BASE_DEFAULT);
        page_size = sysconf(_SC_PAGESIZE);
        page_mask = page_size > 0 ? (uint64_t) page_size - 1u : 0u;
        page_off = (uint64_t) tcm_base & page_mask;
        munmap((void *) (g_base - page_off), R5_TCM_SIZE + (size_t) page_off);
        g_base = NULL;
    }
    if (g_fd >= 0) {
        close(g_fd);
        g_fd = -1;
    }
}

int tpu_r5_ctrl_wait_mask(uint32_t csr_base,
                          uint32_t reg_offset,
                          uint32_t set_mask,
                          uint32_t clear_mask,
                          unsigned timeout_ms,
                          uint32_t *last_value) {
    const uint32_t chunk_polls = env_u32_or_default("TPU_R5_WAIT_CHUNK_POLLS", 10000u);
    const uint64_t timeout_ns = (uint64_t) (timeout_ms == 0u ? 10000u : timeout_ms) * 1000000ull;
    const uint64_t call_start = now_ns();
    const uint64_t deadline = now_ns() + timeout_ns;

    ++g_wait_calls;
    if (tpu_r5_ctrl_init() != 0) {
        ++g_wait_errors;
        return -1;
    }
    log_wait_reg_offset_once(reg_offset);

    for (;;) {
        uint32_t seq;
        uint32_t err;
        uint32_t value;

        store32(CTRL_ARG0, csr_base);
        store32(CTRL_ARG1, reg_offset);
        store32(CTRL_ARG2, set_mask);
        store32(CTRL_ARG3, clear_mask);
        store32(CTRL_REPS, chunk_polls == 0u ? 1u : chunk_polls);
        store32(CTRL_CMD, CMD_WAIT_LIMIT);
        store32(CTRL_ERROR, 0u);
        store32(CTRL_DONE, 0u);

        seq = __sync_add_and_fetch(&g_seq, 1u);
        if (seq == 0u) {
            seq = __sync_add_and_fetch(&g_seq, 1u);
        }
        store32(CTRL_SEQ, seq);

        while (load32(CTRL_DONE) != seq) {
            if (now_ns() > deadline) {
                errno = ETIMEDOUT;
                ++g_wait_errors;
                g_wait_total_ns += now_ns() - call_start;
                return -1;
            }
        }

        value = load32(CTRL_CHECKSUM);
        err = load32(CTRL_ERROR);
        if (last_value != NULL) {
            *last_value = value;
        }
        if (err == 0u) {
            ++g_wait_success;
            g_wait_total_ns += now_ns() - call_start;
            return 0;
        }
        if (err != ERR_WAIT_TIMEOUT) {
            errno = EIO;
            ++g_wait_errors;
            g_wait_total_ns += now_ns() - call_start;
            return -1;
        }
        ++g_wait_chunk_timeouts;
        if (now_ns() > deadline) {
            errno = ETIMEDOUT;
            ++g_wait_errors;
            g_wait_total_ns += now_ns() - call_start;
            return -1;
        }
    }
}

int tpu_r5_ctrl_wait_pair_mask(uint32_t csr_base,
                               uint32_t reg_offset,
                               uint32_t set_mask_0,
                               uint32_t clear_mask_0,
                               uint32_t set_mask_1,
                               uint32_t clear_mask_1,
                               unsigned timeout_ms,
                               uint32_t *last_value) {
    const uint32_t chunk_polls = env_u32_or_default("TPU_R5_WAIT_CHUNK_POLLS", 10000u);
    const uint64_t timeout_ns = (uint64_t) (timeout_ms == 0u ? 10000u : timeout_ms) * 1000000ull;
    const uint64_t call_start = now_ns();
    const uint64_t deadline = call_start + timeout_ns;
    uint32_t seq;
    uint32_t err;
    uint32_t value;

    ++g_wait_calls;
    ++g_wait_pair_calls;
    if (tpu_r5_ctrl_init() != 0) {
        ++g_wait_errors;
        return -1;
    }
    log_wait_reg_offset_once(reg_offset);

    store32(CTRL_ARG0, csr_base);
    store32(CTRL_ARG1, reg_offset);
    store32(CTRL_ARG2, set_mask_0);
    store32(CTRL_ARG3, clear_mask_0);
    store32(CTRL_ARG4, set_mask_1);
    store32(CTRL_ARG5, clear_mask_1);
    store32(CTRL_REPS, chunk_polls == 0u ? 1u : chunk_polls);
    store32(CTRL_CMD, CMD_WAIT_PAIR_LIMIT);
    store32(CTRL_ERROR, 0u);
    store32(CTRL_DONE, 0u);

    seq = __sync_add_and_fetch(&g_seq, 1u);
    if (seq == 0u) {
        seq = __sync_add_and_fetch(&g_seq, 1u);
    }
    store32(CTRL_SEQ, seq);

    while (load32(CTRL_DONE) != seq) {
        if (now_ns() > deadline) {
            errno = ETIMEDOUT;
            ++g_wait_errors;
            g_wait_total_ns += now_ns() - call_start;
            return -1;
        }
    }

    value = load32(CTRL_CHECKSUM);
    err = load32(CTRL_ERROR);
    if (last_value != NULL) {
        *last_value = value;
    }
    if (err == 0u) {
        ++g_wait_success;
        g_wait_total_ns += now_ns() - call_start;
        return 0;
    }
    if (err == ERR_WAIT_TIMEOUT) {
        errno = ETIMEDOUT;
        ++g_wait_chunk_timeouts;
    } else {
        errno = EIO;
    }
    ++g_wait_errors;
    g_wait_total_ns += now_ns() - call_start;
    return -1;
}

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
                              uint32_t *r5_cycles) {
    const uint32_t poll_limit = env_u32_or_default("TPU_R5_SEQ_POLL_LIMIT", 50000000u);
    const uint64_t timeout_ns = (uint64_t) (timeout_ms == 0u ? 10000u : timeout_ms) * 1000000ull;
    const uint64_t call_start = now_ns();
    const uint64_t deadline = call_start + timeout_ns;
    uint32_t seq;
    uint32_t err;
    uint32_t value;
    uint32_t i;

    ++g_seq_calls;
    if (run_count < 1u || run_count > CTRL_SEQ_MAX_RUNS || a_addrs == NULL || b_addrs == NULL) {
        errno = EINVAL;
        ++g_seq_errors;
        return -1;
    }
    if (tpu_r5_ctrl_init() != 0) {
        ++g_seq_errors;
        return -1;
    }

    for (i = 0u; i < run_count; ++i) {
        const uint32_t desc_off = CTRL_SEQ_DESC_BASE + i * CTRL_SEQ_DESC_STRIDE;
        store32(desc_off + 0u, a_addrs[i]);
        store32(desc_off + 4u, b_addrs[i]);
    }
    for (; i < CTRL_SEQ_MAX_RUNS; ++i) {
        const uint32_t desc_off = CTRL_SEQ_DESC_BASE + i * CTRL_SEQ_DESC_STRIDE;
        store32(desc_off + 0u, 0u);
        store32(desc_off + 4u, 0u);
    }

    store32(CTRL_SEQ_STATUS_RESULT, 0u);
    store32(CTRL_CACHE_STATUS_RESULT, 0u);
    store32(CTRL_PRELOAD01_RESULT, 0u);
    store32(CTRL_PRELOAD23_RESULT, 0u);
    store32(CTRL_PRELOAD_STATUS_RESULT, 0u);
    store32(CTRL_ARG0, csr_base);
    store32(CTRL_ARG1, config_word);
    store32(CTRL_ARG2, seq_ctrl);
    store32(CTRL_ARG3, run_count);
    store32(CTRL_ARG4, poll_limit == 0u ? 1u : poll_limit);
    store32(CTRL_ARG5, 0u);
    store32(CTRL_REPS, poll_limit == 0u ? 1u : poll_limit);
    store32(CTRL_CMD, CMD_SEQ_RUN);
    store32(CTRL_ERROR, 0u);
    store32(CTRL_DONE, 0u);

    seq = __sync_add_and_fetch(&g_seq, 1u);
    if (seq == 0u) {
        seq = __sync_add_and_fetch(&g_seq, 1u);
    }
    store32(CTRL_SEQ, seq);

    while (load32(CTRL_DONE) != seq) {
        if (now_ns() > deadline) {
            errno = ETIMEDOUT;
            ++g_seq_errors;
            g_seq_total_ns += now_ns() - call_start;
            return -1;
        }
    }

    value = load32(CTRL_CHECKSUM);
    err = load32(CTRL_ERROR);
    if (seq_status != NULL) {
        *seq_status = load32(CTRL_SEQ_STATUS_RESULT);
        if (*seq_status == 0u) {
            *seq_status = value;
        }
    }
    if (cache_status != NULL) {
        *cache_status = load32(CTRL_CACHE_STATUS_RESULT);
    }
    if (preload01 != NULL) {
        *preload01 = load32(CTRL_PRELOAD01_RESULT);
    }
    if (preload23 != NULL) {
        *preload23 = load32(CTRL_PRELOAD23_RESULT);
    }
    if (preload_status != NULL) {
        *preload_status = load32(CTRL_PRELOAD_STATUS_RESULT);
    }
    if (r5_cycles != NULL) {
        *r5_cycles = load32(CTRL_CYCLES);
    }

    if (err == 0u) {
        ++g_seq_success;
        g_seq_total_ns += now_ns() - call_start;
        return 0;
    }

    if (err == ERR_WAIT_TIMEOUT) {
        errno = ETIMEDOUT;
    } else if (err == ERR_BAD_ARG) {
        errno = EINVAL;
    } else if (err == ERR_HW_ERROR) {
        errno = EIO;
    } else {
        errno = EIO;
    }
    ++g_seq_errors;
    g_seq_total_ns += now_ns() - call_start;
    return -1;
}

int tpu_r5_ctrl_run_sequence_batch(uint32_t csr_base,
                                   const tpu_r5_sequence_cmd_t *cmds,
                                   uint32_t command_count,
                                   unsigned timeout_ms,
                                   tpu_r5_sequence_result_t *results,
                                   uint32_t *completed_count,
                                   uint32_t *r5_cycles) {
    const uint32_t poll_limit = env_u32_or_default("TPU_R5_SEQ_POLL_LIMIT", 50000000u);
    const uint64_t timeout_ns = (uint64_t) (timeout_ms == 0u ? 10000u : timeout_ms) * 1000000ull;
    const uint64_t call_start = now_ns();
    const uint64_t deadline = call_start + timeout_ns;
    uint32_t seq;
    uint32_t err;
    uint32_t completed;
    uint32_t ci;

    ++g_seq_batch_calls;
    if (cmds == NULL || command_count < 1u || command_count > CTRL_SEQ_BATCH_MAX_COMMANDS) {
        errno = EINVAL;
        ++g_seq_batch_errors;
        return -1;
    }
    for (ci = 0u; ci < command_count; ++ci) {
        if (cmds[ci].run_count < 1u || cmds[ci].run_count > CTRL_SEQ_MAX_RUNS) {
            errno = EINVAL;
            ++g_seq_batch_errors;
            return -1;
        }
    }
    if (tpu_r5_ctrl_init() != 0) {
        ++g_seq_batch_errors;
        return -1;
    }

    for (ci = 0u; ci < command_count; ++ci) {
        const tpu_r5_sequence_cmd_t *cmd = &cmds[ci];
        const uint32_t base = CTRL_SEQ_BATCH_BASE + ci * CTRL_SEQ_BATCH_STRIDE;
        uint32_t ri;

        store32_relaxed(base + CTRL_SEQ_BATCH_CONFIG, cmd->config_word);
        store32_relaxed(base + CTRL_SEQ_BATCH_SEQ_CTRL, cmd->seq_ctrl);
        store32_relaxed(base + CTRL_SEQ_BATCH_RUN_COUNT, cmd->run_count);
        store32_relaxed(base + CTRL_SEQ_BATCH_ERROR, 0u);
        for (ri = 0u; ri < CTRL_SEQ_MAX_RUNS; ++ri) {
            const uint32_t addr_base = base + CTRL_SEQ_BATCH_ADDR_BASE + ri * CTRL_SEQ_DESC_STRIDE;
            store32_relaxed(addr_base + 0u, ri < cmd->run_count ? cmd->a_addrs[ri] : 0u);
            store32_relaxed(addr_base + 4u, ri < cmd->run_count ? cmd->b_addrs[ri] : 0u);
        }
        store32_relaxed(base + CTRL_SEQ_BATCH_SEQ_STATUS, 0u);
        store32_relaxed(base + CTRL_SEQ_BATCH_CACHE_STATUS, 0u);
        store32_relaxed(base + CTRL_SEQ_BATCH_PRELOAD01, 0u);
        store32_relaxed(base + CTRL_SEQ_BATCH_PRELOAD23, 0u);
        store32_relaxed(base + CTRL_SEQ_BATCH_PRELOAD_STATUS, 0u);
    }
    __sync_synchronize();

    store32(CTRL_ARG0, csr_base);
    store32(CTRL_ARG1, command_count);
    store32(CTRL_ARG2, 0u);
    store32(CTRL_ARG3, 0u);
    store32(CTRL_ARG4, poll_limit == 0u ? 1u : poll_limit);
    store32(CTRL_ARG5, 0u);
    store32(CTRL_REPS, poll_limit == 0u ? 1u : poll_limit);
    store32(CTRL_CMD, CMD_SEQ_BATCH_RUN);
    store32(CTRL_ERROR, 0u);
    store32(CTRL_DONE, 0u);

    seq = __sync_add_and_fetch(&g_seq, 1u);
    if (seq == 0u) {
        seq = __sync_add_and_fetch(&g_seq, 1u);
    }
    store32(CTRL_SEQ, seq);

    while (load32(CTRL_DONE) != seq) {
        if (now_ns() > deadline) {
            errno = ETIMEDOUT;
            ++g_seq_batch_errors;
            g_seq_batch_total_ns += now_ns() - call_start;
            return -1;
        }
    }

    completed = load32(CTRL_CHECKSUM);
    err = load32(CTRL_ERROR);
    if (completed_count != NULL) {
        *completed_count = completed;
    }
    if (r5_cycles != NULL) {
        *r5_cycles = load32(CTRL_CYCLES);
    }
    if (results != NULL) {
        for (ci = 0u; ci < command_count; ++ci) {
            const uint32_t base = CTRL_SEQ_BATCH_BASE + ci * CTRL_SEQ_BATCH_STRIDE;

            results[ci].seq_status = load32(base + CTRL_SEQ_BATCH_SEQ_STATUS);
            results[ci].cache_status = load32(base + CTRL_SEQ_BATCH_CACHE_STATUS);
            results[ci].preload01 = load32(base + CTRL_SEQ_BATCH_PRELOAD01);
            results[ci].preload23 = load32(base + CTRL_SEQ_BATCH_PRELOAD23);
            results[ci].preload_status = load32(base + CTRL_SEQ_BATCH_PRELOAD_STATUS);
            results[ci].error = load32(base + CTRL_SEQ_BATCH_ERROR);
        }
    }

    if (err == 0u) {
        ++g_seq_batch_success;
        g_seq_batch_total_ns += now_ns() - call_start;
        return 0;
    }

    if (err == ERR_WAIT_TIMEOUT) {
        errno = ETIMEDOUT;
    } else if (err == ERR_BAD_ARG) {
        errno = EINVAL;
    } else if (err == ERR_HW_ERROR) {
        errno = EIO;
    } else {
        errno = EIO;
    }
    ++g_seq_batch_errors;
    g_seq_batch_total_ns += now_ns() - call_start;
    return -1;
}

static void store_matmul_task(const tpu_r5_matmul_task_t *task) {
    const uint32_t base = CTRL_MATMUL_TASK_BASE;

    store32_relaxed(base + CTRL_MATMUL_TASK_MAGIC, task->magic);
    store32_relaxed(base + CTRL_MATMUL_TASK_VERSION, task->version);
    store32_relaxed(base + CTRL_MATMUL_TASK_FLAGS, task->flags);
    store32_relaxed(base + CTRL_MATMUL_TASK_FENCE_ID, task->fence_id);
    store32_relaxed(base + CTRL_MATMUL_TASK_A_BASE, task->a_base);
    store32_relaxed(base + CTRL_MATMUL_TASK_B_PACKED_BASE, task->b_packed_base);
    store32_relaxed(base + CTRL_MATMUL_TASK_C_BASE, task->c_base);
    store32_relaxed(base + CTRL_MATMUL_TASK_M, task->m);
    store32_relaxed(base + CTRL_MATMUL_TASK_N, task->n);
    store32_relaxed(base + CTRL_MATMUL_TASK_K, task->k);
    store32_relaxed(base + CTRL_MATMUL_TASK_LDA_BYTES, task->lda_bytes);
    store32_relaxed(base + CTRL_MATMUL_TASK_LDC_BYTES, task->ldc_bytes);
    store32_relaxed(base + CTRL_MATMUL_TASK_TILE_M, task->tile_m);
    store32_relaxed(base + CTRL_MATMUL_TASK_TILE_N, task->tile_n);
    store32_relaxed(base + CTRL_MATMUL_TASK_TILE_K, task->tile_k);
    store32_relaxed(base + CTRL_MATMUL_TASK_B_TILE_BYTES, task->b_tile_bytes);
    store32_relaxed(base + CTRL_MATMUL_TASK_B_N_TILES, task->b_n_tiles);
    store32_relaxed(base + CTRL_MATMUL_TASK_B_K_TILES, task->b_k_tiles);
    store32_relaxed(base + CTRL_MATMUL_TASK_STATUS, TPU_R5_MATMUL_STATUS_SUBMITTED);
    store32_relaxed(base + CTRL_MATMUL_TASK_ERROR, TPU_R5_MATMUL_ERR_NONE);
    store32_relaxed(base + CTRL_MATMUL_TASK_COMPLETED, 0u);
    store32_relaxed(base + CTRL_MATMUL_TASK_TOTAL_TILES, task->total_tiles);
    store32_relaxed(base + CTRL_MATMUL_TASK_DONE_FENCE, 0u);
    store32_relaxed(base + CTRL_MATMUL_TASK_R5_CYCLES, 0u);
    store32_relaxed(base + CTRL_MATMUL_TASK_FPGA_WAIT, 0u);
    store32_relaxed(base + CTRL_MATMUL_TASK_LAST_HW_STATUS, 0u);
    store32_relaxed(base + CTRL_MATMUL_TASK_SCRATCH_M0, 0u);
    store32_relaxed(base + CTRL_MATMUL_TASK_SCRATCH_N0, 0u);
    store32_relaxed(base + CTRL_MATMUL_TASK_SCRATCH_K0, 0u);
    store32_relaxed(base + CTRL_MATMUL_TASK_SCRATCH_NT, 0u);
    store32_relaxed(base + CTRL_MATMUL_TASK_SCRATCH_KT, 0u);
    __sync_synchronize();
}

static void load_matmul_result(tpu_r5_matmul_result_t *result) {
    const uint32_t base = CTRL_MATMUL_TASK_BASE;

    if (result == NULL) {
        return;
    }

    result->status = load32(base + CTRL_MATMUL_TASK_STATUS);
    result->error = load32(base + CTRL_MATMUL_TASK_ERROR);
    result->completed_tiles = load32(base + CTRL_MATMUL_TASK_COMPLETED);
    result->total_tiles = load32(base + CTRL_MATMUL_TASK_TOTAL_TILES);
    result->done_fence = load32(base + CTRL_MATMUL_TASK_DONE_FENCE);
    result->r5_cycles = load32(base + CTRL_MATMUL_TASK_R5_CYCLES);
    result->fpga_wait_cycles = load32(base + CTRL_MATMUL_TASK_FPGA_WAIT);
    result->last_hw_status = load32(base + CTRL_MATMUL_TASK_LAST_HW_STATUS);
}

int tpu_r5_ctrl_run_matmul_task(uint32_t csr_base,
                                const tpu_r5_matmul_task_t *task,
                                unsigned timeout_ms,
                                tpu_r5_matmul_result_t *result) {
    const uint32_t poll_limit = env_u32_or_default("TPU_R5_MATMUL_POLL_LIMIT",
                                                   env_u32_or_default("TPU_R5_SEQ_POLL_LIMIT", 50000000u));
    const uint64_t timeout_ns = (uint64_t) (timeout_ms == 0u ? 10000u : timeout_ms) * 1000000ull;
    const uint64_t call_start = now_ns();
    const uint64_t deadline = call_start + timeout_ns;
    uint32_t seq;
    uint32_t err;

    ++g_matmul_task_calls;
    if (task == NULL ||
        task->magic != TPU_R5_MATMUL_TASK_MAGIC ||
        task->version != TPU_R5_MATMUL_TASK_VERSION ||
        task->m == 0u ||
        task->n == 0u ||
        task->k == 0u ||
        task->tile_m == 0u ||
        task->tile_n == 0u ||
        task->tile_k == 0u ||
        task->b_tile_bytes == 0u ||
        task->b_n_tiles == 0u ||
        task->b_k_tiles == 0u) {
        errno = EINVAL;
        ++g_matmul_task_errors;
        return -1;
    }
    if (tpu_r5_ctrl_init() != 0) {
        ++g_matmul_task_errors;
        return -1;
    }

    store_matmul_task(task);
    store32(CTRL_ARG0, csr_base);
    store32(CTRL_ARG1, poll_limit == 0u ? 1u : poll_limit);
    store32(CTRL_ARG2, 0u);
    store32(CTRL_ARG3, 0u);
    store32(CTRL_ARG4, 0u);
    store32(CTRL_ARG5, 0u);
    store32(CTRL_REPS, poll_limit == 0u ? 1u : poll_limit);
    store32(CTRL_CMD, CMD_MATMUL_TASK);
    store32(CTRL_ERROR, 0u);
    store32(CTRL_DONE, 0u);

    seq = __sync_add_and_fetch(&g_seq, 1u);
    if (seq == 0u) {
        seq = __sync_add_and_fetch(&g_seq, 1u);
    }
    store32(CTRL_SEQ, seq);

    while (load32(CTRL_DONE) != seq) {
        if (now_ns() > deadline) {
            load_matmul_result(result);
            errno = ETIMEDOUT;
            ++g_matmul_task_errors;
            g_matmul_task_total_ns += now_ns() - call_start;
            return -1;
        }
    }

    load_matmul_result(result);
    err = load32(CTRL_ERROR);
    if (result != NULL && result->r5_cycles == 0u) {
        result->r5_cycles = load32(CTRL_CYCLES);
    }
    if (env_truthy("TPU_R5_MATMUL_LOG")) {
        if (result != NULL) {
            fprintf(stderr,
                    "[R5] matmul task done ctrl_error=%" PRIu32
                    " status=%" PRIu32 " error=%" PRIu32
                    " completed=%" PRIu32 "/%" PRIu32
                    " done_fence=%" PRIu32
                    " r5_cycles=%" PRIu32
                    " fpga_wait_cycles=%" PRIu32
                    " last_hw_status=0x%08" PRIx32 "\n",
                    err,
                    result->status,
                    result->error,
                    result->completed_tiles,
                    result->total_tiles,
                    result->done_fence,
                    result->r5_cycles,
                    result->fpga_wait_cycles,
                    result->last_hw_status);
        } else {
            fprintf(stderr, "[R5] matmul task done ctrl_error=%" PRIu32 " result=NULL\n", err);
        }
    }

    if (err == 0u && (result == NULL || result->status == TPU_R5_MATMUL_STATUS_DONE)) {
        ++g_matmul_task_success;
        g_matmul_task_total_ns += now_ns() - call_start;
        return 0;
    }

    if (err == ERR_WAIT_TIMEOUT || (result != NULL && result->error == TPU_R5_MATMUL_ERR_TIMEOUT)) {
        errno = ETIMEDOUT;
    } else if (err == ERR_BAD_ARG || (result != NULL && result->error == TPU_R5_MATMUL_ERR_BAD_ARG)) {
        errno = EINVAL;
    } else {
        errno = EIO;
    }
    ++g_matmul_task_errors;
    g_matmul_task_total_ns += now_ns() - call_start;
    return -1;
}
