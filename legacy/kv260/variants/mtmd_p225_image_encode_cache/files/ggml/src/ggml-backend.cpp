// Note: porting this file to C++ is a work in progress

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#   define NOMINMAX
#endif
#include <windows.h>
#endif

#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-alloc.h"
#include "ggml-impl.h"

#include <assert.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define GGML_FPGA_FFN_STREAM_HAVE_NEON 1
#else
#define GGML_FPGA_FFN_STREAM_HAVE_NEON 0
#endif

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

struct ggml_fpga_ffn_stream_entry {
    int64_t rows = 0;
    int64_t cols = 0;
    std::vector<int8_t> data;
    std::vector<float> scales;
    std::vector<float> inv_scales;
    std::vector<uint8_t> row_ready;
    std::atomic<int64_t> ready_rows { 0 };
    bool scale_reuse_allowed = false;
};

struct ggml_fpga_ffn_stream_stats {
    std::atomic<uint64_t> target_scan_calls { 0 };
    std::atomic<uint64_t> target_scan_phase_skips { 0 };
    std::atomic<uint64_t> target_scan_fpga_splits { 0 };
    std::atomic<uint64_t> target_scan_inputs { 0 };
    std::atomic<uint64_t> target_scan_can_prepare { 0 };
    std::atomic<uint64_t> target_scan_mul_mat_consumers { 0 };
    std::atomic<uint64_t> target_marks { 0 };
    std::atomic<uint64_t> target_aliases { 0 };
    std::atomic<uint64_t> target_queries { 0 };
    std::atomic<uint64_t> target_query_hits { 0 };
    std::atomic<uint64_t> prepare_row_calls { 0 };
    std::atomic<uint64_t> prepare_row_rejects { 0 };
    std::atomic<uint64_t> prepared_tensors { 0 };
    std::atomic<uint64_t> prepared_rows { 0 };
    std::atomic<uint64_t> prepared_bytes { 0 };
    std::atomic<uint64_t> prepare_us { 0 };
    std::atomic<uint64_t> cache_hits { 0 };
    std::atomic<uint64_t> cache_misses { 0 };
    std::atomic<uint64_t> consumed_tensors { 0 };
    std::atomic<uint64_t> consumed_bytes { 0 };
};

static std::mutex g_ggml_fpga_ffn_stream_mutex;
static std::unordered_map<const ggml_tensor *, std::shared_ptr<ggml_fpga_ffn_stream_entry>> g_ggml_fpga_ffn_stream_cache;
static std::unordered_set<const ggml_tensor *> g_ggml_fpga_ffn_stream_prepare_targets;
static std::unordered_map<const ggml_tensor *, const ggml_tensor *> g_ggml_fpga_ffn_stream_aliases;
static ggml_fpga_ffn_stream_stats g_ggml_fpga_ffn_stream_stats;
static thread_local ggml_backend_fpga_phase g_ggml_fpga_phase = GGML_BACKEND_FPGA_PHASE_UNKNOWN;
static std::atomic<int> g_ggml_fpga_compute_phase { GGML_BACKEND_FPGA_PHASE_UNKNOWN };

void ggml_backend_fpga_set_phase(enum ggml_backend_fpga_phase phase) {
    g_ggml_fpga_phase = phase >= GGML_BACKEND_FPGA_PHASE_UNKNOWN && phase < GGML_BACKEND_FPGA_PHASE_COUNT ?
        phase : GGML_BACKEND_FPGA_PHASE_UNKNOWN;
    g_ggml_fpga_compute_phase.store((int) g_ggml_fpga_phase, std::memory_order_relaxed);
}

enum ggml_backend_fpga_phase ggml_backend_fpga_get_phase(void) {
    return g_ggml_fpga_phase;
}

enum ggml_backend_fpga_phase ggml_backend_fpga_get_compute_phase(void) {
    const int phase = g_ggml_fpga_compute_phase.load(std::memory_order_relaxed);
    return phase >= GGML_BACKEND_FPGA_PHASE_UNKNOWN && phase < GGML_BACKEND_FPGA_PHASE_COUNT ?
        (enum ggml_backend_fpga_phase) phase : GGML_BACKEND_FPGA_PHASE_UNKNOWN;
}

const char * ggml_backend_fpga_phase_name(enum ggml_backend_fpga_phase phase) {
    switch (phase) {
        case GGML_BACKEND_FPGA_PHASE_ENCODER: return "ViT/Encoder";
        case GGML_BACKEND_FPGA_PHASE_PREFILL: return "Prefill";
        case GGML_BACKEND_FPGA_PHASE_DECODE:  return "Decode";
        case GGML_BACKEND_FPGA_PHASE_UNKNOWN:
        case GGML_BACKEND_FPGA_PHASE_COUNT:   return "Unknown";
    }
    return "Unknown";
}

static int64_t ggml_backend_fpga_ffn_stream_now_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now().time_since_epoch()).count();
}

static bool ggml_backend_fpga_ffn_stream_env_enabled(const char * name) {
    const char * value = getenv(name);
    return value != NULL && *value != '\0' && strcmp(value, "0") != 0;
}

static int64_t ggml_backend_fpga_ffn_stream_env_i64(const char * name, int64_t fallback) {
    const char * value = getenv(name);
    if (value == NULL || *value == '\0') {
        return fallback;
    }
    char * end = NULL;
    const long long parsed = strtoll(value, &end, 0);
    if (end == value || (end != NULL && *end != '\0')) {
        return fallback;
    }
    return (int64_t) parsed;
}

bool ggml_backend_fpga_ffn_streaming_enabled(void) {
    return ggml_backend_fpga_ffn_stream_env_enabled("GGML_FPGA_FFN_STREAMING_CPU");
}

static bool ggml_backend_fpga_ffn_streaming_m1_decode_enabled(void) {
    return ggml_backend_fpga_ffn_stream_env_enabled("GGML_FPGA_FFN_STREAMING_M1_DECODE");
}

static bool ggml_backend_fpga_ffn_streaming_allows_small_tensor(int64_t rows, int64_t cols) {
    (void) cols;
    return rows == 1 &&
        ggml_backend_fpga_get_compute_phase() == GGML_BACKEND_FPGA_PHASE_DECODE &&
        ggml_backend_fpga_ffn_streaming_m1_decode_enabled();
}

bool ggml_backend_fpga_ffn_streaming_should_prepare(const struct ggml_tensor * tensor) {
    if (tensor == NULL || !ggml_backend_fpga_ffn_streaming_enabled()) {
        return false;
    }
    std::lock_guard<std::mutex> lock(g_ggml_fpga_ffn_stream_mutex);
    g_ggml_fpga_ffn_stream_stats.target_queries.fetch_add(1, std::memory_order_relaxed);
    const bool hit = g_ggml_fpga_ffn_stream_prepare_targets.find(tensor) != g_ggml_fpga_ffn_stream_prepare_targets.end();
    if (hit) {
        g_ggml_fpga_ffn_stream_stats.target_query_hits.fetch_add(1, std::memory_order_relaxed);
    }
    return hit;
}

void ggml_backend_fpga_ffn_streaming_mark_prepare_target(const struct ggml_tensor * tensor) {
    if (tensor == NULL || !ggml_backend_fpga_ffn_streaming_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_ggml_fpga_ffn_stream_mutex);
    if (g_ggml_fpga_ffn_stream_prepare_targets.insert(tensor).second) {
        g_ggml_fpga_ffn_stream_stats.target_marks.fetch_add(1, std::memory_order_relaxed);
    }
}

void ggml_backend_fpga_ffn_streaming_mark_prepare_alias(
        const struct ggml_tensor * tensor,
        const struct ggml_tensor * alias) {
    if (tensor == NULL || alias == NULL || tensor == alias || !ggml_backend_fpga_ffn_streaming_enabled()) {
        return;
    }
    std::lock_guard<std::mutex> lock(g_ggml_fpga_ffn_stream_mutex);
    g_ggml_fpga_ffn_stream_aliases[alias] = tensor;
    g_ggml_fpga_ffn_stream_stats.target_aliases.fetch_add(1, std::memory_order_relaxed);
}

void ggml_backend_fpga_ffn_streaming_clear_prepare_targets(void) {
    std::lock_guard<std::mutex> lock(g_ggml_fpga_ffn_stream_mutex);
    g_ggml_fpga_ffn_stream_prepare_targets.clear();
    g_ggml_fpga_ffn_stream_aliases.clear();
}

static int64_t ggml_backend_fpga_ffn_streaming_min_elems(void) {
    const int64_t value = ggml_backend_fpga_ffn_stream_env_i64("GGML_FPGA_FFN_STREAMING_MIN_ELEMS", 262144);
    return value < 0 ? 0 : value;
}

static int8_t ggml_backend_fpga_ffn_stream_quantize_value(float value, float inv_scale) {
    const float scaled = std::round(value * inv_scale);
    const float clamped = std::max(-127.0f, std::min(127.0f, scaled));
    return (int8_t) clamped;
}

static float ggml_backend_fpga_ffn_stream_row_max_abs_scalar(const float * src, int64_t cols) {
    float max_abs = 0.0f;
    for (int64_t col = 0; col < cols; ++col) {
        max_abs = std::max(max_abs, std::fabs(src[col]));
    }
    return max_abs;
}

#if GGML_FPGA_FFN_STREAM_HAVE_NEON
static float ggml_backend_fpga_ffn_stream_row_max_abs_neon(const float * src, int64_t cols) {
    float32x4_t max_v = vdupq_n_f32(0.0f);
    int64_t col = 0;
    for (; col + 4 <= cols; col += 4) {
        max_v = vmaxnmq_f32(max_v, vabsq_f32(vld1q_f32(src + col)));
    }
    float lanes[4];
    vst1q_f32(lanes, max_v);
    float max_abs = std::max(std::max(lanes[0], lanes[1]), std::max(lanes[2], lanes[3]));
    for (; col < cols; ++col) {
        max_abs = std::max(max_abs, std::fabs(src[col]));
    }
    return max_abs;
}

static void ggml_backend_fpga_ffn_stream_quantize_row_neon(
        const float * src,
        int64_t cols,
        float inv_scale,
        int8_t * dst) {
    const float32x4_t scale_v = vdupq_n_f32(inv_scale);
    const int32x4_t min_v = vdupq_n_s32(-127);
    const int32x4_t max_v = vdupq_n_s32(127);
    int64_t col = 0;
    for (; col + 8 <= cols; col += 8) {
        int32x4_t q0 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + col), scale_v));
        int32x4_t q1 = vcvtaq_s32_f32(vmulq_f32(vld1q_f32(src + col + 4), scale_v));
        q0 = vmaxq_s32(min_v, vminq_s32(max_v, q0));
        q1 = vmaxq_s32(min_v, vminq_s32(max_v, q1));
        const int16x8_t q16 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
        const int8x8_t q8 = vqmovn_s16(q16);
        vst1_s8(dst + col, q8);
    }
    for (; col < cols; ++col) {
        dst[col] = ggml_backend_fpga_ffn_stream_quantize_value(src[col], inv_scale);
    }
}
#endif

static void ggml_backend_fpga_ffn_stream_quantize_row(
        const float * src,
        int64_t cols,
        float inv_scale,
        int8_t * dst) {
#if GGML_FPGA_FFN_STREAM_HAVE_NEON
    if (ggml_backend_fpga_ffn_stream_env_enabled("GGML_FPGA_ACT_QUANT_NEON")) {
        ggml_backend_fpga_ffn_stream_quantize_row_neon(src, cols, inv_scale, dst);
        return;
    }
#endif
    for (int64_t col = 0; col < cols; ++col) {
        dst[col] = ggml_backend_fpga_ffn_stream_quantize_value(src[col], inv_scale);
    }
}

static void ggml_backend_fpga_ffn_streaming_prepare_row_impl(
        const struct ggml_tensor * tensor,
        int64_t row,
        const float * values,
        int64_t cols,
        bool scale_reuse_allowed,
        bool force_prepare) {
    g_ggml_fpga_ffn_stream_stats.prepare_row_calls.fetch_add(1, std::memory_order_relaxed);
    if (!ggml_backend_fpga_ffn_streaming_enabled() || tensor == NULL || values == NULL || tensor->type != GGML_TYPE_F32) {
        g_ggml_fpga_ffn_stream_stats.prepare_row_rejects.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    if (row < 0 || cols <= 0 || tensor->ne[0] != cols || tensor->nb[0] != (int64_t) sizeof(float)) {
        g_ggml_fpga_ffn_stream_stats.prepare_row_rejects.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    const int64_t rows = ggml_nrows(tensor);
    const bool allow_small_tensor = ggml_backend_fpga_ffn_streaming_allows_small_tensor(rows, cols);
    if (row >= rows || (!force_prepare && !allow_small_tensor && rows * cols < ggml_backend_fpga_ffn_streaming_min_elems())) {
        g_ggml_fpga_ffn_stream_stats.prepare_row_rejects.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    std::shared_ptr<ggml_fpga_ffn_stream_entry> entry;
    {
        std::lock_guard<std::mutex> lock(g_ggml_fpga_ffn_stream_mutex);
        auto it = g_ggml_fpga_ffn_stream_cache.find(tensor);
        if (it == g_ggml_fpga_ffn_stream_cache.end()) {
            entry = std::make_shared<ggml_fpga_ffn_stream_entry>();
            entry->rows = rows;
            entry->cols = cols;
            entry->data.resize((size_t) rows * (size_t) cols);
            entry->scales.resize((size_t) rows);
            entry->inv_scales.resize((size_t) rows);
            entry->row_ready.assign((size_t) rows, 0);
            entry->scale_reuse_allowed = scale_reuse_allowed;
            g_ggml_fpga_ffn_stream_cache.emplace(tensor, entry);
            g_ggml_fpga_ffn_stream_stats.prepared_tensors.fetch_add(1, std::memory_order_relaxed);
            g_ggml_fpga_ffn_stream_stats.prepared_bytes.fetch_add(
                    (uint64_t) rows * (uint64_t) cols,
                    std::memory_order_relaxed);
        } else {
            entry = it->second;
            entry->scale_reuse_allowed = entry->scale_reuse_allowed || scale_reuse_allowed;
        }
    }

    if (entry->rows != rows || entry->cols != cols || (!force_prepare && entry->row_ready[(size_t) row])) {
        return;
    }

    const int64_t start_us = ggml_backend_fpga_ffn_stream_now_us();
#if GGML_FPGA_FFN_STREAM_HAVE_NEON
    const float max_abs = ggml_backend_fpga_ffn_stream_env_enabled("GGML_FPGA_ACT_QUANT_NEON") ?
        ggml_backend_fpga_ffn_stream_row_max_abs_neon(values, cols) :
        ggml_backend_fpga_ffn_stream_row_max_abs_scalar(values, cols);
#else
    const float max_abs = ggml_backend_fpga_ffn_stream_row_max_abs_scalar(values, cols);
#endif
    const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
    const float inv_scale = max_abs > 0.0f ? 127.0f / max_abs : 0.0f;
    entry->scales[(size_t) row] = scale;
    entry->inv_scales[(size_t) row] = inv_scale;
    int8_t * row_q = entry->data.data() + (size_t) row * (size_t) cols;
    ggml_backend_fpga_ffn_stream_quantize_row(values, cols, inv_scale, row_q);
    if (!entry->row_ready[(size_t) row]) {
        entry->row_ready[(size_t) row] = 1;
        entry->ready_rows.fetch_add(1, std::memory_order_relaxed);
    }
    g_ggml_fpga_ffn_stream_stats.prepared_rows.fetch_add(1, std::memory_order_relaxed);
    g_ggml_fpga_ffn_stream_stats.prepare_us.fetch_add(
            (uint64_t) (ggml_backend_fpga_ffn_stream_now_us() - start_us),
            std::memory_order_relaxed);
}

void ggml_backend_fpga_ffn_streaming_prepare_row(
        const struct ggml_tensor * tensor,
        int64_t row,
        const float * values,
        int64_t cols) {
    ggml_backend_fpga_ffn_streaming_prepare_row_impl(tensor, row, values, cols, false, false);
}

void ggml_backend_fpga_ffn_streaming_prepare_row_force(
        const struct ggml_tensor * tensor,
        int64_t row,
        const float * values,
        int64_t cols) {
    ggml_backend_fpga_ffn_streaming_prepare_row_impl(tensor, row, values, cols, false, true);
}

void ggml_backend_fpga_ffn_streaming_prepare_marked_row(
        const struct ggml_tensor * tensor,
        int64_t row,
        const float * values,
        int64_t cols) {
    ggml_backend_fpga_ffn_streaming_prepare_row_impl(tensor, row, values, cols, true, false);
}

bool ggml_backend_fpga_ffn_streaming_get(
        const struct ggml_tensor * tensor,
        int64_t rows,
        int64_t cols,
        struct ggml_fpga_prepared_activation * out) {
    if (!ggml_backend_fpga_ffn_streaming_enabled() || tensor == NULL || out == NULL) {
        return false;
    }
    std::shared_ptr<ggml_fpga_ffn_stream_entry> entry;
    {
        std::lock_guard<std::mutex> lock(g_ggml_fpga_ffn_stream_mutex);
        auto it = g_ggml_fpga_ffn_stream_cache.find(tensor);
        if (it == g_ggml_fpga_ffn_stream_cache.end()) {
            auto alias_it = g_ggml_fpga_ffn_stream_aliases.find(tensor);
            if (alias_it != g_ggml_fpga_ffn_stream_aliases.end()) {
                it = g_ggml_fpga_ffn_stream_cache.find(alias_it->second);
            }
        }
        if (it == g_ggml_fpga_ffn_stream_cache.end()) {
            g_ggml_fpga_ffn_stream_stats.cache_misses.fetch_add(1, std::memory_order_relaxed);
            return false;
        }
        entry = it->second;
    }
    if (entry->rows != rows || entry->cols != cols || entry->ready_rows.load(std::memory_order_relaxed) != rows) {
        g_ggml_fpga_ffn_stream_stats.cache_misses.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    out->data = entry->data.data();
    out->scales = entry->scales.data();
    out->inv_scales = entry->inv_scales.data();
    out->rows = rows;
    out->cols = cols;
    out->scale_reuse_allowed = entry->scale_reuse_allowed;
    g_ggml_fpga_ffn_stream_stats.cache_hits.fetch_add(1, std::memory_order_relaxed);
    return true;
}

void ggml_backend_fpga_ffn_streaming_consume(const struct ggml_tensor * tensor) {
    if (tensor == NULL) {
        return;
    }
    std::shared_ptr<ggml_fpga_ffn_stream_entry> entry;
    {
        std::lock_guard<std::mutex> lock(g_ggml_fpga_ffn_stream_mutex);
        auto it = g_ggml_fpga_ffn_stream_cache.find(tensor);
        if (it == g_ggml_fpga_ffn_stream_cache.end()) {
            auto alias_it = g_ggml_fpga_ffn_stream_aliases.find(tensor);
            if (alias_it != g_ggml_fpga_ffn_stream_aliases.end()) {
                it = g_ggml_fpga_ffn_stream_cache.find(alias_it->second);
            }
        }
        if (it == g_ggml_fpga_ffn_stream_cache.end()) {
            return;
        }
        entry = it->second;
        g_ggml_fpga_ffn_stream_cache.erase(it);
    }
    g_ggml_fpga_ffn_stream_stats.consumed_tensors.fetch_add(1, std::memory_order_relaxed);
    g_ggml_fpga_ffn_stream_stats.consumed_bytes.fetch_add(
            (uint64_t) entry->rows * (uint64_t) entry->cols,
            std::memory_order_relaxed);
}

void ggml_backend_fpga_ffn_streaming_log_summary(void) {
    if (!ggml_backend_fpga_ffn_streaming_enabled()) {
        return;
    }
    GGML_LOG_INFO("ggml_backend_sched: P23b FFN streaming sidecache: prepared_tensors=%" PRIu64
            " prepared_rows=%" PRIu64 " prepared_bytes=%.3f MiB prepare=%.3f ms hits=%" PRIu64
            " misses=%" PRIu64 " consumed_tensors=%" PRIu64 " consumed_bytes=%.3f MiB resident=%zu"
            " target_scan_calls=%" PRIu64 " target_phase_skips=%" PRIu64
            " target_fpga_splits=%" PRIu64 " target_inputs=%" PRIu64
            " target_can_prepare=%" PRIu64 " target_mul_mat_consumers=%" PRIu64
            " target_marks=%" PRIu64 " target_aliases=%" PRIu64 " target_queries=%" PRIu64
            " target_query_hits=%" PRIu64 " prepare_row_calls=%" PRIu64 " prepare_row_rejects=%" PRIu64 "\n",
            g_ggml_fpga_ffn_stream_stats.prepared_tensors.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.prepared_rows.load(std::memory_order_relaxed),
            (double) g_ggml_fpga_ffn_stream_stats.prepared_bytes.load(std::memory_order_relaxed) / (1024.0 * 1024.0),
            (double) g_ggml_fpga_ffn_stream_stats.prepare_us.load(std::memory_order_relaxed) / 1000.0,
            g_ggml_fpga_ffn_stream_stats.cache_hits.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.cache_misses.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.consumed_tensors.load(std::memory_order_relaxed),
            (double) g_ggml_fpga_ffn_stream_stats.consumed_bytes.load(std::memory_order_relaxed) / (1024.0 * 1024.0),
            g_ggml_fpga_ffn_stream_cache.size(),
            g_ggml_fpga_ffn_stream_stats.target_scan_calls.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.target_scan_phase_skips.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.target_scan_fpga_splits.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.target_scan_inputs.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.target_scan_can_prepare.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.target_scan_mul_mat_consumers.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.target_marks.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.target_aliases.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.target_queries.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.target_query_hits.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.prepare_row_calls.load(std::memory_order_relaxed),
            g_ggml_fpga_ffn_stream_stats.prepare_row_rejects.load(std::memory_order_relaxed));
}

// backend buffer type

const char * ggml_backend_buft_name(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_name(buft);
}

ggml_backend_buffer_t ggml_backend_buft_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    if (size == 0) {
        // return a dummy buffer for zero-sized allocations
        return ggml_backend_buffer_init(buft, {}, NULL, 0);
    }

    GGML_ASSERT(buft);
    return buft->iface.alloc_buffer(buft, size);
}

size_t ggml_backend_buft_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->iface.get_alignment(buft);
}

size_t ggml_backend_buft_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    // get_max_size is optional, defaults to SIZE_MAX
    if (buft->iface.get_max_size) {
        return buft->iface.get_max_size(buft);
    }
    return SIZE_MAX;
}

size_t ggml_backend_buft_get_alloc_size(ggml_backend_buffer_type_t buft, const struct ggml_tensor * tensor) {
    GGML_ASSERT(buft);
    // get_alloc_size is optional, defaults to ggml_nbytes
    if (buft->iface.get_alloc_size) {
        size_t size = buft->iface.get_alloc_size(buft, tensor);
        assert(size >= ggml_nbytes(tensor));
        return size;
    }
    return ggml_nbytes(tensor);
}

bool ggml_backend_buft_is_host(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    if (buft->iface.is_host) {
        return buft->iface.is_host(buft);
    }
    return false;
}

ggml_backend_dev_t ggml_backend_buft_get_device(ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(buft);
    return buft->device;
}

// backend buffer

ggml_backend_buffer_t ggml_backend_buffer_init(
               ggml_backend_buffer_type_t buft,
        struct ggml_backend_buffer_i      iface,
               void *                     context,
               size_t                     size) {
    ggml_backend_buffer_t buffer = new ggml_backend_buffer {
        /* .interface = */ iface,
        /* .buft      = */ buft,
        /* .context   = */ context,
        /* .size      = */ size,
        /* .usage     = */ GGML_BACKEND_BUFFER_USAGE_ANY
    };

    return buffer;
}

const char * ggml_backend_buffer_name(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_name(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_free(ggml_backend_buffer_t buffer) {
    if (buffer == NULL) {
        return;
    }

    if (buffer->iface.free_buffer != NULL) {
        buffer->iface.free_buffer(buffer);
    }
    delete buffer;
}

size_t ggml_backend_buffer_get_size(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->size;
}

void * ggml_backend_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    // get_base is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return NULL;
    }

    void * base = buffer->iface.get_base(buffer);

    GGML_ASSERT(base != NULL && "backend buffer base cannot be NULL");

    return base;
}

enum ggml_status ggml_backend_buffer_init_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor) {
    GGML_ASSERT(buffer);
    // init_tensor is optional
    if (buffer->iface.init_tensor) {
        return buffer->iface.init_tensor(buffer, tensor);
    }
    return GGML_STATUS_SUCCESS;
}

void ggml_backend_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    // clear is optional if the buffer is zero-sized
    if (buffer->size == 0) {
        return;
    }

    buffer->iface.clear(buffer, value);
}

size_t ggml_backend_buffer_get_alignment(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_alignment(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_max_size(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_get_max_size(ggml_backend_buffer_get_type(buffer));
}

size_t ggml_backend_buffer_get_alloc_size(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor) {
    return ggml_backend_buft_get_alloc_size(ggml_backend_buffer_get_type(buffer), tensor);
}

bool ggml_backend_buffer_is_host(ggml_backend_buffer_t buffer) {
    return ggml_backend_buft_is_host(ggml_backend_buffer_get_type(buffer));
}

void ggml_backend_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    buffer->usage = usage;

    // FIXME: add a generic callback to the buffer interface
    if (ggml_backend_buffer_is_multi_buffer(buffer)) {
        ggml_backend_multi_buffer_set_usage(buffer, usage);
    }
}

enum ggml_backend_buffer_usage ggml_backend_buffer_get_usage(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->usage;
}

ggml_backend_buffer_type_t ggml_backend_buffer_get_type(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->buft;
}

void ggml_backend_buffer_reset(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    if (buffer->iface.reset) {
        buffer->iface.reset(buffer);
    }
}

bool ggml_backend_buffer_copy_tensor(const struct ggml_tensor * src, struct ggml_tensor * dst) {
    ggml_backend_buffer_t dst_buf = dst->view_src ? dst->view_src->buffer : dst->buffer;
    if (dst_buf->iface.cpy_tensor) {
        return dst_buf->iface.cpy_tensor(dst_buf, src, dst);
    }
    return false;
}

// backend

ggml_guid_t ggml_backend_guid(ggml_backend_t backend) {
    if (backend == NULL) {
        return NULL;
    }
    return backend->guid;
}

const char * ggml_backend_name(ggml_backend_t backend) {
    if (backend == NULL) {
        return "NULL";
    }
    return backend->iface.get_name(backend);
}

void ggml_backend_free(ggml_backend_t backend) {
    if (backend == NULL) {
        return;
    }

    backend->iface.free(backend);
}

ggml_backend_buffer_type_t ggml_backend_get_default_buffer_type(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_buffer_type(backend->device);
}

ggml_backend_buffer_t ggml_backend_alloc_buffer(ggml_backend_t backend, size_t size) {
    return ggml_backend_buft_alloc_buffer(ggml_backend_get_default_buffer_type(backend), size);
}

size_t ggml_backend_get_alignment(ggml_backend_t backend) {
    return ggml_backend_buft_get_alignment(ggml_backend_get_default_buffer_type(backend));
}

size_t ggml_backend_get_max_size(ggml_backend_t backend) {
    return ggml_backend_buft_get_max_size(ggml_backend_get_default_buffer_type(backend));
}

void ggml_backend_tensor_set_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    if (backend->iface.set_tensor_async == NULL) {
        ggml_backend_tensor_set(tensor, data, offset, size);
    } else {
        backend->iface.set_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_get_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(backend);
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    if (backend->iface.get_tensor_async == NULL) {
        ggml_backend_tensor_get(tensor, data, offset, size);
    } else {
        backend->iface.get_tensor_async(backend, tensor, data, offset, size);
    }
}

void ggml_backend_tensor_set(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");

    buf->iface.set_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_get(const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor read out of bounds");

    buf->iface.get_tensor(buf, tensor, data, offset, size);
}

void ggml_backend_tensor_memset(struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    if (size == 0) {
        return;
    }

    GGML_ASSERT(buf != NULL && "tensor buffer not set");
    GGML_ASSERT(tensor->data != NULL && "tensor not allocated");
    GGML_ASSERT(offset + size <= ggml_nbytes(tensor) && "tensor write out of bounds");
    GGML_ASSERT(buf->iface.memset_tensor != NULL && "memset not implemented by backend buffer");

    buf->iface.memset_tensor(buf, tensor, value, offset, size);
}

void ggml_backend_synchronize(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    if (backend->iface.synchronize == NULL) {
        return;
    }

    backend->iface.synchronize(backend);
}

ggml_backend_graph_plan_t ggml_backend_graph_plan_create(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_create != NULL);

    return backend->iface.graph_plan_create(backend, cgraph);
}

void ggml_backend_graph_plan_free(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_free != NULL);

    backend->iface.graph_plan_free(backend, plan);
}

enum ggml_status ggml_backend_graph_plan_compute(ggml_backend_t backend, ggml_backend_graph_plan_t plan) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.graph_plan_compute != NULL);

    return backend->iface.graph_plan_compute(backend, plan);
}

enum ggml_status ggml_backend_graph_compute(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    enum ggml_status err = ggml_backend_graph_compute_async(backend, cgraph);
    ggml_backend_synchronize(backend);
    return err;
}

enum ggml_status ggml_backend_graph_compute_async(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    return backend->iface.graph_compute(backend, cgraph);
}

bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_op(backend->device, op);
}

bool ggml_backend_supports_buft(ggml_backend_t backend, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_supports_buft(backend->device, buft);
}

bool ggml_backend_offload_op(ggml_backend_t backend, const struct ggml_tensor * op) {
    GGML_ASSERT(backend);
    return ggml_backend_dev_offload_op(backend->device, op);
}

ggml_backend_dev_t ggml_backend_get_device(ggml_backend_t backend) {
    GGML_ASSERT(backend);
    return backend->device;
}

// backend copy

void ggml_backend_tensor_copy(struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        ggml_backend_tensor_set(dst, src->data, 0, ggml_nbytes(src));
    } else if (ggml_backend_buffer_is_host(dst->buffer)) {
        ggml_backend_tensor_get(src, dst->data, 0, ggml_nbytes(src));
    } else if (!ggml_backend_buffer_copy_tensor(src, dst)) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: warning: slow copy from %s to %s\n", __func__, ggml_backend_buffer_name(src->buffer), ggml_backend_buffer_name(dst->buffer));
#endif
        size_t nbytes = ggml_nbytes(src);
        void * data = malloc(nbytes);
        ggml_backend_tensor_get(src, data, 0, nbytes);
        ggml_backend_tensor_set(dst, data, 0, nbytes);
        free(data);
    }
}

void ggml_backend_tensor_copy_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_layout(src, dst) && "cannot copy tensors with different layouts");

    if (src == dst) {
        return;
    }

    GGML_ASSERT(backend_dst);
    if (backend_dst->iface.cpy_tensor_async != NULL) {
        if (backend_dst->iface.cpy_tensor_async(backend_src, backend_dst, src, dst)) {
            return;
        }
    }

    // an async copy would normally happen after all the queued operations on both backends are completed
    // to simulate the same behavior, we need to synchronize both backends first, and do a blocking copy
    ggml_backend_synchronize(backend_src);
    ggml_backend_synchronize(backend_dst);
    ggml_backend_tensor_copy(src, dst);
}

// events

ggml_backend_event_t ggml_backend_event_new(ggml_backend_dev_t device) {
    // null device is allowed for the transition period to the device interface
    if (device == NULL || device->iface.event_new == NULL) {
        return NULL;
    }
    return device->iface.event_new(device);
}

void ggml_backend_event_free(ggml_backend_event_t event) {
    if (event == NULL) {
        return;
    }
    event->device->iface.event_free(event->device, event);
}

void ggml_backend_event_record(ggml_backend_event_t event, ggml_backend_t backend) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_record != NULL);

    backend->iface.event_record(backend, event);
}

void ggml_backend_event_synchronize(ggml_backend_event_t event) {
    GGML_ASSERT(event);
    GGML_ASSERT(event->device->iface.event_synchronize);

    event->device->iface.event_synchronize(event->device, event);
}

void ggml_backend_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    GGML_ASSERT(backend);
    GGML_ASSERT(backend->iface.event_wait != NULL);

    backend->iface.event_wait(backend, event);
}

static void ggml_backend_graph_optimize(ggml_backend_t backend, struct ggml_cgraph * cgraph) {
    GGML_ASSERT(backend);
    if (backend->iface.graph_optimize != NULL) {
        backend->iface.graph_optimize(backend, cgraph);
    }
}

// Backend device

const char * ggml_backend_dev_name(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_name(device);
}

const char * ggml_backend_dev_description(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_description(device);
}

void ggml_backend_dev_memory(ggml_backend_dev_t device, size_t * free, size_t * total) {
    GGML_ASSERT(device);
    device->iface.get_memory(device, free, total);
}

enum ggml_backend_dev_type ggml_backend_dev_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_type(device);
}

void ggml_backend_dev_get_props(ggml_backend_dev_t device, struct ggml_backend_dev_props * props) {
    memset(props, 0, sizeof(*props));
    device->iface.get_props(device, props);
}

ggml_backend_reg_t ggml_backend_dev_backend_reg(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->reg;
}

ggml_backend_t ggml_backend_dev_init(ggml_backend_dev_t device, const char * params) {
    GGML_ASSERT(device);
    return device->iface.init_backend(device, params);
}

ggml_backend_buffer_type_t ggml_backend_dev_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    return device->iface.get_buffer_type(device);
}

ggml_backend_buffer_type_t ggml_backend_dev_host_buffer_type(ggml_backend_dev_t device) {
    GGML_ASSERT(device);
    if (device->iface.get_host_buffer_type == NULL) {
        return NULL;
    }

    return device->iface.get_host_buffer_type(device);
}

ggml_backend_buffer_t ggml_backend_dev_buffer_from_host_ptr(ggml_backend_dev_t device, void * ptr, size_t size, size_t max_tensor_size) {
    GGML_ASSERT(device);
    return device->iface.buffer_from_host_ptr(device, ptr, size, max_tensor_size);
}

bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    return device->iface.supports_op(device, op);
}

bool ggml_backend_dev_supports_buft(ggml_backend_dev_t device, ggml_backend_buffer_type_t buft) {
    GGML_ASSERT(device);
    return device->iface.supports_buft(device, buft);
}

bool ggml_backend_dev_offload_op(ggml_backend_dev_t device, const struct ggml_tensor * op) {
    GGML_ASSERT(device);
    if (device->iface.offload_op != NULL) {
        return device->iface.offload_op(device, op);
    }

    return false;
}

// Backend (reg)

const char * ggml_backend_reg_name(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_name(reg);
}

size_t ggml_backend_reg_dev_count(ggml_backend_reg_t reg) {
    GGML_ASSERT(reg);
    return reg->iface.get_device_count(reg);
}

ggml_backend_dev_t ggml_backend_reg_dev_get(ggml_backend_reg_t reg, size_t index) {
    GGML_ASSERT(reg);
    return reg->iface.get_device(reg, index);
}

void * ggml_backend_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_ASSERT(reg);
    if (!reg->iface.get_proc_address) {
        return NULL;
    }
    return reg->iface.get_proc_address(reg, name);
}

// multi-buffer buffer

struct ggml_backend_multi_buffer_context {
    ggml_backend_buffer_t * buffers;
    size_t n_buffers;
};

static void ggml_backend_multi_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_free(ctx->buffers[i]);
    }

    free(ctx->buffers);
    free(ctx);
}

static void ggml_backend_multi_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_clear(ctx->buffers[i], value);
    }
}

static const struct ggml_backend_buffer_i ggml_backend_multi_buffer_i = {
    /* .free_buffer     = */ ggml_backend_multi_buffer_free_buffer,
    /* .get_base        = */ NULL,
    /* .init_tensor     = */ NULL,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ NULL,
    /* .get_tensor      = */ NULL,
    /* .cpy_tensor      = */ NULL,
    /* .clear           = */ ggml_backend_multi_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_multi_buffer_alloc_buffer(ggml_backend_buffer_t * buffers, size_t n_buffers) {
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) malloc(sizeof(struct ggml_backend_multi_buffer_context));
    ctx->n_buffers = n_buffers;
    ctx->buffers = (ggml_backend_buffer_t *) malloc(n_buffers * sizeof(ggml_backend_buffer_t));

    GGML_ASSERT(ctx->buffers != NULL);

    size_t total_size = 0;
    for (size_t i = 0; i < n_buffers; i++) {
        ctx->buffers[i] = buffers[i];
        total_size += ggml_backend_buffer_get_size(buffers[i]);
    }

    return ggml_backend_buffer_init(buffers[0]->buft, ggml_backend_multi_buffer_i, ctx, total_size);
}

bool ggml_backend_buffer_is_multi_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    return buffer->iface.free_buffer == ggml_backend_multi_buffer_free_buffer;
}

void ggml_backend_multi_buffer_set_usage(ggml_backend_buffer_t buffer, enum ggml_backend_buffer_usage usage) {
    GGML_ASSERT(buffer);
    GGML_ASSERT(ggml_backend_buffer_is_multi_buffer(buffer));
    ggml_backend_multi_buffer_context * ctx = (ggml_backend_multi_buffer_context *) buffer->context;
    for (size_t i = 0; i < ctx->n_buffers; i++) {
        ggml_backend_buffer_set_usage(ctx->buffers[i], usage);
    }
}

// creates a copy of the tensor with the same memory layout
static struct ggml_tensor * ggml_dup_tensor_layout(struct ggml_context * ctx, const struct ggml_tensor * tensor) {
    struct ggml_tensor * dup = ggml_dup_tensor(ctx, tensor);
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        dup->nb[i] = tensor->nb[i];
    }
    return dup;
}

static bool ggml_is_view_op(enum ggml_op op) {
    return op == GGML_OP_VIEW || op == GGML_OP_RESHAPE || op == GGML_OP_PERMUTE || op == GGML_OP_TRANSPOSE;
}

// scheduler

#ifndef GGML_SCHED_MAX_BACKENDS
#define GGML_SCHED_MAX_BACKENDS 16
#endif

#ifndef GGML_SCHED_MAX_SPLIT_INPUTS
#define GGML_SCHED_MAX_SPLIT_INPUTS 30
#endif

#ifndef GGML_SCHED_MAX_COPIES
#define GGML_SCHED_MAX_COPIES 4
#endif

struct ggml_backend_sched_split {
    int backend_id;
    int i_start;
    int i_end;
    struct ggml_tensor * inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_inputs;
    // graph view of this split
    struct ggml_cgraph graph;
};

struct ggml_backend_sched {
    bool is_reset; // true if the scheduler has been reset since the last graph split
    bool is_alloc;

    int n_backends;

    ggml_backend_t backends[GGML_SCHED_MAX_BACKENDS];
    ggml_backend_buffer_type_t bufts[GGML_SCHED_MAX_BACKENDS];
    ggml_gallocr_t galloc;

    // hash map of the nodes in the graph
    struct ggml_hash_set  hash_set;
    int                 * hv_tensor_backend_ids; // [hash_set.size]
    struct ggml_tensor ** hv_tensor_copies;      // [hash_set.size][n_backends][n_copies]

    int * node_backend_ids; // [graph_size]
    int * leaf_backend_ids; // [graph_size]

    int * prev_node_backend_ids; // [graph_size]
    int * prev_leaf_backend_ids; // [graph_size]

    // copy of the graph with modified inputs
    struct ggml_cgraph graph;

    // graph splits
    struct ggml_backend_sched_split * splits;
    int n_splits;
    int splits_capacity;

    // pipeline parallelism support
    int n_copies;
    int cur_copy;
    int next_copy;
    ggml_backend_event_t events[GGML_SCHED_MAX_BACKENDS][GGML_SCHED_MAX_COPIES];
    struct ggml_tensor * graph_inputs[GGML_SCHED_MAX_SPLIT_INPUTS];
    int n_graph_inputs;

    struct ggml_context * ctx;

    ggml_backend_sched_eval_callback callback_eval;
    void * callback_eval_user_data;

    char * context_buffer;
    size_t context_buffer_size;

    bool op_offload;

    int debug;
};

#define hash_id(tensor) ggml_hash_find_or_insert(&sched->hash_set, tensor)
#define tensor_backend_id(tensor) sched->hv_tensor_backend_ids[hash_id(tensor)]
#define tensor_id_copy(id, backend_id, copy_id) sched->hv_tensor_copies[(id) * sched->n_backends * sched->n_copies + (backend_id) * sched->n_copies + (copy_id)]
#define tensor_copy(tensor, backend_id, copy_id) tensor_id_copy(hash_id(tensor), backend_id, copy_id)

// returns the priority of the backend, lower id is higher priority
static int ggml_backend_sched_backend_id(ggml_backend_sched_t sched, ggml_backend_t backend) {
    for (int i = 0; i < sched->n_backends; i++) {
        if (sched->backends[i] == backend) {
            return i;
        }
    }
    return -1;
}

static int ggml_backend_sched_backend_from_buffer(ggml_backend_sched_t sched, const struct ggml_tensor * tensor, const struct ggml_tensor * op) {
    ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
    if (buffer == NULL) {
        return -1;
    }

    // find highest prio backend that supports the buffer type and the op
    for (int i = 0; i < sched->n_backends; i++) {
        if (ggml_backend_supports_buft(sched->backends[i], buffer->buft) &&
            ggml_backend_supports_op(sched->backends[i], op)) {
            return i;
        }
    }

#ifndef NDEBUG
    GGML_LOG_DEBUG("%s: warning: no backend supports op %s with a weight with buffer type %s used in tensor %s, the weight will need to be copied\n",
        __func__, ggml_op_desc(tensor), ggml_backend_buffer_name(buffer), tensor->name);
#endif

    return -1;
}

#if 0
#define GGML_SCHED_MAX_SPLITS_DEBUG 4096
static char causes[GGML_DEFAULT_GRAPH_SIZE*16 + GGML_SCHED_MAX_SPLITS_DEBUG*GGML_SCHED_MAX_SPLIT_INPUTS][128]; // debug only
#define SET_CAUSE(node, ...) sprintf(causes[hash_id(node)], __VA_ARGS__)
#define GET_CAUSE(node) causes[hash_id(node)]
#else
#define SET_CAUSE(node, ...)
#define GET_CAUSE(node) ""
#endif

// returns the backend that should be used for the node based on the current locations
static int ggml_backend_sched_backend_id_from_cur(ggml_backend_sched_t sched, struct ggml_tensor * tensor) {
    // assign pre-allocated nodes to their backend
    int cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor, tensor);
    if (cur_backend_id != -1) {
        SET_CAUSE(tensor, "1.dst");
        return cur_backend_id;
    }

    // view_src
    if (tensor->view_src != NULL) {
        cur_backend_id = ggml_backend_sched_backend_from_buffer(sched, tensor->view_src, tensor);
        if (cur_backend_id != -1) {
            SET_CAUSE(tensor, "1.vsrc");
            return cur_backend_id;
        }
    }

    if (tensor->buffer || (tensor->view_src && tensor->view_src->buffer)) {
        // since the tensor is pre-allocated, it cannot be moved to another backend
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        GGML_ABORT("pre-allocated tensor (%s) in a buffer (%s) that cannot run the operation (%s)", tensor->name, ggml_backend_buffer_name(buffer), ggml_op_name(tensor->op));
    }

    // graph input
    if (tensor->flags & GGML_TENSOR_FLAG_INPUT) {
        cur_backend_id = sched->n_backends - 1; // last backend (assumed CPU)
        SET_CAUSE(tensor, "1.inp");
        return cur_backend_id;
    }

    // operations with weights are preferably run on the same backend as the weights
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        const struct ggml_tensor * src = tensor->src[i];
        if (src == NULL) {
            continue;
        }
        // skip ROPE since the rope freqs tensor is too small to choose a backend based on it
        // not an ideal solution
        if (tensor->op != GGML_OP_ROPE && src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
            int src_backend_id = ggml_backend_sched_backend_from_buffer(sched, src, tensor);
            // check if a backend with higher prio wants to offload the op
            if (sched->op_offload && src_backend_id == sched->n_backends - 1 && ggml_backend_buffer_is_host(src->buffer)) {
                for (int b = 0; b < src_backend_id; b++) {
                    if (ggml_backend_supports_op(sched->backends[b], tensor) && ggml_backend_offload_op(sched->backends[b], tensor)) {
                        SET_CAUSE(tensor, "1.off");
                        return b;
                    }
                }
            }
            SET_CAUSE(tensor, "1.wgt%d", i);
            return src_backend_id;
        }
    }

    return -1;
}

static char * fmt_size(size_t size) {
    static char buffer[128];
    if (size >= 1024*1024) {
        snprintf(buffer, sizeof(buffer), "%zuM", size/1024/1024);
    } else {
        snprintf(buffer, sizeof(buffer), "%zuK", size/1024);
    }
    return buffer;
}

static void ggml_backend_sched_print_assignments(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    int cur_split = 0;
    for (int i = 0; i < graph->n_nodes; i++) {
        if (cur_split < sched->n_splits && i == sched->splits[cur_split].i_start) {
            ggml_backend_t split_backend = sched->backends[sched->splits[cur_split].backend_id];
            GGML_LOG_DEBUG("\n## SPLIT #%d: %s # %d inputs", cur_split, ggml_backend_name(split_backend),
                sched->splits[cur_split].n_inputs);
            for (int j = 0; j < sched->splits[cur_split].n_inputs; j++) {
                if (j == 0) {
                    GGML_LOG_DEBUG(": ");
                }
                GGML_LOG_DEBUG("[%s (%5.5s)] ", sched->splits[cur_split].inputs[j]->name,
                    fmt_size(ggml_nbytes(sched->splits[cur_split].inputs[j])));
            }
            GGML_LOG_DEBUG("\n");
            cur_split++;
        }
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        if (sched->debug > 1) {
            ggml_backend_t tensor_backend = ggml_backend_sched_get_tensor_backend(sched, node);
            GGML_LOG_DEBUG("node #%3d (%10.10s): %20.20s (%5.5s) [%5.5s %8.8s] use=%d:", i, ggml_op_name(node->op), node->name,
                fmt_size(ggml_nbytes(node)), tensor_backend ? ggml_backend_name(tensor_backend) : "NULL", GET_CAUSE(node),
                graph->use_counts[ggml_hash_find(&graph->visited_hash_set, node)]);
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                ggml_backend_t src_backend = ggml_backend_sched_get_tensor_backend(sched, src);
                GGML_LOG_DEBUG(" %20.20s (%5.5s) [%5.5s %8.8s]", src->name,
                    fmt_size(ggml_nbytes(src)), src_backend ? ggml_backend_name(src_backend) : "NULL", GET_CAUSE(src));
            }
            GGML_LOG_DEBUG("\n");
        }
    }
}

static bool ggml_backend_sched_buffer_supported(ggml_backend_sched_t sched, struct ggml_tensor * t, int backend_id) {
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    ggml_backend_buffer_type_t buft = NULL;

    if (buf) {
        // the tensor is already allocated
        buft = buf->buft;
    } else {
        // see if the tensor already has a backend assigned, and use the buffer type of that backend
        int tensor_backend_id = tensor_backend_id(t);
        if (tensor_backend_id == -1 && t->view_src) {
            tensor_backend_id = tensor_backend_id(t->view_src);
        }
        if (tensor_backend_id != -1) {
            buft = sched->bufts[tensor_backend_id];
        }
    }

    return buft != NULL && ggml_backend_supports_buft(sched->backends[backend_id], buft);
}

static void ggml_backend_sched_set_if_supported(ggml_backend_sched_t sched, struct ggml_tensor * node, int cur_backend_id, int * node_backend_id) {
    if (ggml_backend_supports_op(sched->backends[cur_backend_id], node)) {
        *node_backend_id = cur_backend_id;
        SET_CAUSE(node, "2.sup");
    }
}

// assigns backends to ops and splits the graph into subgraphs that can be computed on the same backend
void ggml_backend_sched_split_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    // reset splits
    sched->n_splits = 0;
    sched->n_graph_inputs = 0;
    sched->is_reset = false;

    struct ggml_init_params params = {
        /* .mem_size =   */ sched->context_buffer_size,
        /* .mem_buffer = */ sched->context_buffer,
        /* .no_alloc =   */ true
    };

    ggml_free(sched->ctx);

    sched->ctx = ggml_init(params);
    if (sched->ctx == NULL) {
        GGML_ABORT("%s: failed to initialize context\n", __func__);
    }

    // pass 1: assign backends to ops with pre-allocated inputs
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        int * leaf_backend_id = &tensor_backend_id(leaf);
        // do not overwrite user assignments
        if (*leaf_backend_id == -1) {
            *leaf_backend_id = ggml_backend_sched_backend_id_from_cur(sched, leaf);
        }
    }

    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * node_backend_id = &tensor_backend_id(node);
        // do not overwrite user assignments
        if (*node_backend_id == -1) {
            *node_backend_id = ggml_backend_sched_backend_id_from_cur(sched, node);

#if 0
            // src
            if (node->op == GGML_OP_NONE) {
                continue;
            }

            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }
                int * src_backend_id = &tensor_backend_id(src);
                if (*src_backend_id == -1) {
                    *src_backend_id = ggml_backend_sched_backend_id_from_cur(sched, src);
                }
            }
#endif
        }
    }

    // pass 2: expand current backend assignments
    // assign the same backend to adjacent nodes
    // expand gpu backends (i.e. non last prio) up and down, ignoring cpu (the lowest priority backend)
    // thus, cpu will never be used unless weights are on cpu, or there are no gpu ops between cpu ops
    // ops unsupported by the backend being expanded will be left unassigned so that they can be assigned later when the locations of its inputs are known
    // expand gpu down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand gpu up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                if (*node_backend_id == sched->n_backends - 1) {
                    // skip cpu (lowest prio backend)
                    cur_backend_id = -1;
                } else {
                    cur_backend_id = *node_backend_id;
                }
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest down
    {
        int cur_backend_id = -1;
        for (int i = 0; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }
    // expand rest up
    {
        int cur_backend_id = -1;
        for (int i = graph->n_nodes - 1; i >= 0; i--) {
            struct ggml_tensor * node = graph->nodes[i];
            if (ggml_is_view_op(node->op)) {
                continue;
            }
            int * node_backend_id = &tensor_backend_id(node);
            if (*node_backend_id != -1) {
                cur_backend_id = *node_backend_id;
            } else if (cur_backend_id != -1) {
                ggml_backend_sched_set_if_supported(sched, node, cur_backend_id, node_backend_id);
            }
        }
    }

    // pass 3: upgrade nodes to higher prio backends with compatible buffer types
    // if the tensor is already in the same buffer type (*) as another higher priority backend, we should move it there
    // however, we also need to verify that the sources are in compatible buffer types
    // (*) the actual requirement is more relaxed, the buffer type of the backend should be supported by all the users of this tensor further down the graph
    // however, this is slow to verify, so we have a more strict requirement that the buffer type is the same
    // this is not uncommon since multiple backends can use host memory, with the same buffer type (eg. BLAS and CPU)
    // additionally, set remaining unassigned nodes to the backend with the most supported inputs
    // only nodes that could not be assigned during expansion due to the backend not supporting the op should be unassigned at this point
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        if (ggml_is_view_op(node->op)) {
            continue;
        }
        int * node_backend_id = &tensor_backend_id(node);
        if (*node_backend_id == -1) {
            // unassigned node: find the backend with the most supported inputs
            int n_supported_best = -1;
            for (int b = 0; b < sched->n_backends; b++) {
                if (ggml_backend_supports_op(sched->backends[b], node)) {
                    int n_supported = 0;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if ((tensor_backend_id(src) != -1 || tensor_backend_id(src->view_src) != -1) && ggml_backend_sched_buffer_supported(sched, src, b)) {
                            n_supported++;
                        }
                    }
                    if (n_supported > n_supported_best) {
                        n_supported_best = n_supported;
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.best");
                    }
                }
            }
        } else {
            // assigned node: upgrade to higher prio backend if possible
            for (int b = 0; b < *node_backend_id; b++) {
                if (sched->bufts[b] == sched->bufts[*node_backend_id] && ggml_backend_supports_op(sched->backends[b], node)) {
                    bool supported = true;
                    for (int j = 0; j < GGML_MAX_SRC; j++) {
                        struct ggml_tensor * src = node->src[j];
                        if (src == NULL) {
                            continue;
                        }
                        if (!ggml_backend_sched_buffer_supported(sched, src, b)) {
                            supported = false;
                            break;
                        }
                    }
                    if (supported) {
                        *node_backend_id = b;
                        SET_CAUSE(node, "3.upg");
                        break;
                    }
                }
            }
        }
    }

    // pass 4: assign backends to remaining src from dst and view_src
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        int * cur_backend_id = &tensor_backend_id(node);
        if (node->view_src != NULL && *cur_backend_id == -1) {
            *cur_backend_id = tensor_backend_id(node->view_src);
            SET_CAUSE(node, "4.vsrc");
        }
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            struct ggml_tensor * src = node->src[j];
            if (src == NULL) {
                continue;
            }
            int * src_backend_id = &tensor_backend_id(src);
            if (*src_backend_id == -1) {
                if (src->view_src != NULL) {
                    // views are always on the same backend as the source
                    *src_backend_id = tensor_backend_id(src->view_src);
                    SET_CAUSE(src, "4.vsrc");
                } else {
                    *src_backend_id = *cur_backend_id;
                    SET_CAUSE(src, "4.cur");
                }
            }
        }
        // if the node is still unassigned, assign it to the first backend that supports it
        for (int b = 0; b < sched->n_backends && *cur_backend_id == -1; b++) {
            ggml_backend_sched_set_if_supported(sched, node, b, cur_backend_id);
        }
        GGML_ASSERT(*cur_backend_id != -1);
    }

    // pass 5: split graph, find tensors that need to be copied
    {
        int i_split = 0;
        struct ggml_backend_sched_split * split = &sched->splits[0];
        // find the backend of the first split, skipping view ops
        int i = 0;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];
            if (!ggml_is_view_op(node->op)) {
                split->backend_id = tensor_backend_id(node);
                break;
            }
        }
        split->i_start = 0;
        split->n_inputs = 0;
        int cur_backend_id = split->backend_id;
        for (; i < graph->n_nodes; i++) {
            struct ggml_tensor * node = graph->nodes[i];

            if (ggml_is_view_op(node->op)) {
                continue;
            }

            const int node_backend_id = tensor_backend_id(node);

            GGML_ASSERT(node_backend_id != -1); // all nodes should be assigned by now, this can happen if there is no CPU fallback

            // check if we should start a new split based on the sources of the current node
            bool need_new_split = false;
            if (node_backend_id == cur_backend_id && split->n_inputs > 0) {
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    struct ggml_tensor * src = node->src[j];
                    if (src == NULL) {
                        continue;
                    }
                    // check if a weight is on a different and incompatible backend
                    // by starting a new split, the memory of the previously offloaded weights can be reused
                    if (src->buffer != NULL && src->buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
                        int src_backend_id = tensor_backend_id(src);
                        if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                            need_new_split = true;
                            break;
                        }
                    }
                    // check if the split has too many inputs
                    // FIXME: count the number of inputs instead of only checking when full
                    if (split->n_inputs == GGML_SCHED_MAX_SPLIT_INPUTS) {
                        const size_t id = hash_id(src);
                        int src_backend_id = sched->hv_tensor_backend_ids[id];
                        bool supported = ggml_backend_sched_buffer_supported(sched, src, cur_backend_id);
                        if (src_backend_id != cur_backend_id && tensor_id_copy(id, cur_backend_id, 0) == NULL && !supported) {
                            need_new_split = true;
                            break;
                        }
                    }
                }
            }

            if (node_backend_id != cur_backend_id || need_new_split) {
                split->i_end = i;
                i_split++;
                if (i_split >= sched->splits_capacity) {
                    sched->splits_capacity *= 2;
                    sched->splits = (ggml_backend_sched_split *)
                        realloc(sched->splits, sched->splits_capacity * sizeof(struct ggml_backend_sched_split));
                    GGML_ASSERT(sched->splits != NULL);
                }
                split = &sched->splits[i_split];
                split->backend_id = node_backend_id;
                split->i_start = i;
                split->n_inputs = 0;
                cur_backend_id = node_backend_id;
            }

            // find inputs that are not on the same backend
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                struct ggml_tensor * src = node->src[j];
                if (src == NULL) {
                    continue;
                }

                size_t src_id = hash_id(src);
                const int src_backend_id = sched->hv_tensor_backend_ids[src_id];
                GGML_ASSERT(src_backend_id != -1); // all inputs should be assigned by now

                if (src->flags & GGML_TENSOR_FLAG_INPUT && sched->n_copies > 1) {
                    if (tensor_id_copy(src_id, src_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[src_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy;
                            if (c == sched->cur_copy) {
                                tensor_copy = src; // use the original tensor as the current copy
                            } else {
                                tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                                ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            }
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, src_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_graph_inputs = sched->n_graph_inputs++;
                        GGML_ASSERT(n_graph_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        sched->graph_inputs[n_graph_inputs] = src;
                    }
                }

                if (src_backend_id != cur_backend_id && !ggml_backend_sched_buffer_supported(sched, src, cur_backend_id)) {
                    // create a copy of the input in the split's backend
                    if (tensor_id_copy(src_id, cur_backend_id, 0) == NULL) {
                        ggml_backend_t backend = sched->backends[cur_backend_id];
                        for (int c = 0; c < sched->n_copies; c++) {
                            struct ggml_tensor * tensor_copy = ggml_dup_tensor_layout(sched->ctx, src);
                            ggml_format_name(tensor_copy, "%s#%s#%d", ggml_backend_name(backend), src->name, c);
                            if (sched->n_copies > 1) {
                                ggml_set_input(tensor_copy);
                                ggml_set_output(tensor_copy); // prevent ggml-alloc from overwriting the tensor
                            }
                            tensor_id_copy(src_id, cur_backend_id, c) = tensor_copy;
                            SET_CAUSE(tensor_copy, "4.cpy");
                        }
                        int n_inputs = split->n_inputs++;
                        GGML_ASSERT(n_inputs < GGML_SCHED_MAX_SPLIT_INPUTS);
                        split->inputs[n_inputs] = src;
                    }
                    node->src[j] = tensor_id_copy(src_id, cur_backend_id, sched->cur_copy);
                }
            }
        }
        split->i_end = graph->n_nodes;
        sched->n_splits = i_split + 1;
    }

    if (sched->debug) {
        ggml_backend_sched_print_assignments(sched, graph);
    }

    // swap node_backend_ids and leaf _backend_ids with prevs
    {
        int * tmp = sched->node_backend_ids;
        sched->node_backend_ids = sched->prev_node_backend_ids;
        sched->prev_node_backend_ids = tmp;

        tmp = sched->leaf_backend_ids;
        sched->leaf_backend_ids = sched->prev_leaf_backend_ids;
        sched->prev_leaf_backend_ids = tmp;
    }

    int graph_size = std::max(graph->n_nodes, graph->n_leafs) + sched->n_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sched->n_copies;
    if (sched->graph.size < graph_size) {
        sched->graph.size = graph_size;
        sched->graph.nodes = (ggml_tensor **) realloc(sched->graph.nodes, graph_size * sizeof(struct ggml_tensor *));
        sched->graph.leafs = (ggml_tensor **) realloc(sched->graph.leafs, graph_size * sizeof(struct ggml_tensor *));
        GGML_ASSERT(sched->graph.nodes != NULL);
        GGML_ASSERT(sched->graph.leafs != NULL);
    }
    sched->graph.n_nodes = 0;
    sched->graph.n_leafs = 0;

    struct ggml_cgraph * graph_copy = &sched->graph;

    for (int i = 0; i < sched->n_splits; i++) {
        struct ggml_backend_sched_split * split = &sched->splits[i];
        split->graph = ggml_graph_view(graph, split->i_start, split->i_end);

        // Optimize this split of the graph. This needs to happen before we make graph_copy,
        // so they are in sync.
        ggml_backend_graph_optimize(sched->backends[split->backend_id], &split->graph);

        // add inputs to the graph copy so that they are allocated by ggml-alloc at the start of the split
        for (int j = 0; j < split->n_inputs; j++) {
            assert(graph_copy->size > (graph_copy->n_nodes + 1));

            struct ggml_tensor * input = split->inputs[j];
            const size_t input_id = hash_id(input);
            struct ggml_tensor * input_cpy = tensor_id_copy(input_id, split->backend_id, sched->cur_copy);

            // add a dependency to the input source so that it is not freed before the copy is done
            struct ggml_tensor * input_dep = ggml_view_tensor(sched->ctx, input);
            input_dep->src[0] = input;
            sched->node_backend_ids[graph_copy->n_nodes] = sched->hv_tensor_backend_ids[input_id];
            graph_copy->nodes[graph_copy->n_nodes++] = input_dep;

            // add a dependency to the input copy so that it is allocated at the start of the split
            sched->node_backend_ids[graph_copy->n_nodes] = split->backend_id;
            graph_copy->nodes[graph_copy->n_nodes++] = input_cpy;
        }

        for (int j = split->i_start; j < split->i_end; j++) {
            assert(graph_copy->size > graph_copy->n_nodes);
            sched->node_backend_ids[graph_copy->n_nodes] = tensor_backend_id(graph->nodes[j]);
            graph_copy->nodes[graph_copy->n_nodes++] = graph->nodes[j];
        }
    }

    if (sched->n_copies > 1) {
        // add input copies as leafs so that they are allocated first
        for (int i = 0; i < sched->n_graph_inputs; i++) {
            struct ggml_tensor * input = sched->graph_inputs[i];
            size_t id = hash_id(input);
            int backend_id = tensor_backend_id(input);
            for (int c = 0; c < sched->n_copies; c++) {
                struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                assert(graph_copy->size > graph_copy->n_leafs);
                graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
            }
        }

        for (int i = 0; i < sched->n_splits; i++) {
            struct ggml_backend_sched_split * split = &sched->splits[i];
            int backend_id = split->backend_id;
            for (int j = 0; j < split->n_inputs; j++) {
                struct ggml_tensor * input = split->inputs[j];
                size_t id = hash_id(input);
                for (int c = 0; c < sched->n_copies; c++) {
                    struct ggml_tensor * input_cpy = tensor_id_copy(id, backend_id, c);
                    sched->leaf_backend_ids[graph_copy->n_leafs] = backend_id;
                    assert(graph_copy->size > graph_copy->n_leafs);
                    graph_copy->leafs[graph_copy->n_leafs++] = input_cpy;
                }
            }
        }
    }

    // add leafs from the original graph
    for (int i = 0; i < graph->n_leafs; i++) {
        struct ggml_tensor * leaf = graph->leafs[i];
        sched->leaf_backend_ids[graph_copy->n_leafs] = tensor_backend_id(leaf);
        assert(graph_copy->size > graph_copy->n_leafs);
        graph_copy->leafs[graph_copy->n_leafs++] = leaf;
    }
}

static bool ggml_backend_sched_alloc_splits(ggml_backend_sched_t sched) {
    bool backend_ids_changed = false;
    for (int i = 0; i < sched->graph.n_nodes; i++) {
        if (sched->node_backend_ids[i] != sched->prev_node_backend_ids[i] &&
            sched->bufts[sched->node_backend_ids[i]] != sched->bufts[sched->prev_node_backend_ids[i]]) {
            backend_ids_changed = true;
            break;
        }
    }
    if (!backend_ids_changed) {
        for (int i = 0; i < sched->graph.n_leafs; i++) {
            if (sched->leaf_backend_ids[i] != sched->prev_leaf_backend_ids[i] &&
                sched->bufts[sched->leaf_backend_ids[i]] != sched->bufts[sched->prev_leaf_backend_ids[i]]) {
                backend_ids_changed = true;
                break;
            }
        }
    }

    // allocate graph
    if (backend_ids_changed || !ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
        // the re-allocation may cause the split inputs to be moved to a different address
        // synchronize without ggml_backend_sched_synchronize to avoid changing cur_copy
        for (int i = 0; i < sched->n_backends; i++) {
            ggml_backend_synchronize(sched->backends[i]);
        }
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: failed to allocate graph, reserving (backend_ids_changed = %d)\n", __func__, backend_ids_changed);
#endif
        ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids);
        if (!ggml_gallocr_alloc_graph(sched->galloc, &sched->graph)) {
            GGML_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            return false;
        }
    }

    return true;
}

struct ggml_backend_sched_split_agg_counts {
    uint64_t mul_mat = 0;
    uint64_t r5_like_mul_mat = 0;
    uint64_t flash_attn = 0;
    uint64_t glu = 0;
    uint64_t add = 0;
    uint64_t rms_norm = 0;
    uint64_t mul = 0;
    uint64_t rope = 0;
    uint64_t set_rows = 0;
    uint64_t get_rows = 0;
    uint64_t reshape = 0;
    uint64_t view = 0;
    uint64_t permute = 0;
    uint64_t transpose = 0;
    uint64_t cont = 0;
};

struct ggml_backend_sched_split_agg_backend {
    uint64_t splits = 0;
    uint64_t nodes = 0;
    uint64_t copy_us = 0;
    uint64_t compute_us = 0;
    uint64_t mul_mat = 0;
    uint64_t r5_like_mul_mat = 0;
    uint64_t cpu_post_ops = 0;
    uint64_t matmul_splits = 0;
    uint64_t non_matmul_splits = 0;
    uint64_t mixed_splits = 0;
    uint64_t max_nodes_per_split = 0;
};

struct ggml_backend_sched_tensor_flow_item {
    uint64_t outputs = 0;
    uint64_t bytes = 0;
};

struct ggml_backend_sched_consumer_agg {
    uint64_t fpga_outputs = 0;
    uint64_t fpga_output_bytes = 0;
    uint64_t consumer_edges = 0;
    uint64_t cpu_edges = 0;
    uint64_t fpga_edges = 0;
    uint64_t other_edges = 0;
    uint64_t same_split_edges = 0;
    uint64_t cross_split_edges = 0;
    uint64_t multi_consumer_outputs = 0;
    uint64_t graph_output_or_cross_graph = 0;
    uint64_t no_consumer_non_output = 0;
    uint64_t first_cpu = 0;
    uint64_t first_fpga = 0;
    uint64_t first_other = 0;
    uint64_t first_graph_output = 0;
    uint64_t first_cpu_bytes = 0;
    uint64_t first_fpga_bytes = 0;
    uint64_t first_other_bytes = 0;
    uint64_t first_graph_output_bytes = 0;
    uint64_t no_consumer_non_output_bytes = 0;
    uint64_t multi_consumer_output_bytes = 0;
    uint64_t immediate_first = 0;
    uint64_t delayed_first = 0;
    uint64_t immediate_first_bytes = 0;
    uint64_t delayed_first_bytes = 0;
    uint64_t first_distance_sum = 0;
    uint64_t first_distance_max = 0;
    uint64_t consumer_op_mul_mat = 0;
    uint64_t consumer_op_flash = 0;
    uint64_t consumer_op_glu = 0;
    uint64_t consumer_op_rms_norm = 0;
    uint64_t consumer_op_add = 0;
    uint64_t consumer_op_mul = 0;
    uint64_t consumer_op_rope = 0;
    uint64_t consumer_op_get_rows = 0;
    uint64_t consumer_op_set_rows = 0;
    uint64_t consumer_op_layout = 0;
    uint64_t consumer_op_other = 0;
    uint64_t consumer_bytes_mul_mat = 0;
    uint64_t consumer_bytes_flash = 0;
    uint64_t consumer_bytes_glu = 0;
    uint64_t consumer_bytes_rms_norm = 0;
    uint64_t consumer_bytes_add = 0;
    uint64_t consumer_bytes_mul = 0;
    uint64_t consumer_bytes_rope = 0;
    uint64_t consumer_bytes_get_rows = 0;
    uint64_t consumer_bytes_set_rows = 0;
    uint64_t consumer_bytes_layout = 0;
    uint64_t consumer_bytes_other = 0;
    std::unordered_map<std::string, ggml_backend_sched_tensor_flow_item> first_consumer_tensor_flows;
    std::unordered_map<std::string, ggml_backend_sched_tensor_flow_item> no_consumer_tensors;
};

enum ggml_backend_sched_op_bucket {
    GGML_BACKEND_SCHED_OP_BUCKET_MUL_MAT = 0,
    GGML_BACKEND_SCHED_OP_BUCKET_FLASH,
    GGML_BACKEND_SCHED_OP_BUCKET_GLU,
    GGML_BACKEND_SCHED_OP_BUCKET_RMS_NORM,
    GGML_BACKEND_SCHED_OP_BUCKET_ADD,
    GGML_BACKEND_SCHED_OP_BUCKET_MUL,
    GGML_BACKEND_SCHED_OP_BUCKET_ROPE,
    GGML_BACKEND_SCHED_OP_BUCKET_GET_ROWS,
    GGML_BACKEND_SCHED_OP_BUCKET_SET_ROWS,
    GGML_BACKEND_SCHED_OP_BUCKET_LAYOUT,
    GGML_BACKEND_SCHED_OP_BUCKET_OTHER,
    GGML_BACKEND_SCHED_OP_BUCKET_COUNT,
};

struct ggml_backend_sched_readback_edge_agg {
    uint64_t copies = 0;
    uint64_t bytes = 0;
    uint64_t copy_us = 0;
    uint64_t producer_mul_mat = 0;
    uint64_t producer_mul_mat_bytes = 0;
    uint64_t producer_mul_mat_copy_us = 0;
    uint64_t producer_other = 0;
    uint64_t producer_other_bytes = 0;
    uint64_t producer_other_copy_us = 0;
    uint64_t immediate = 0;
    uint64_t delayed = 0;
    uint64_t no_consumer = 0;
    uint64_t multi_consumer = 0;
    uint64_t first_index_sum = 0;
    uint64_t first_index_max = 0;
    uint64_t consumer_count[GGML_BACKEND_SCHED_OP_BUCKET_COUNT] = {};
    uint64_t consumer_bytes[GGML_BACKEND_SCHED_OP_BUCKET_COUNT] = {};
    uint64_t consumer_copy_us[GGML_BACKEND_SCHED_OP_BUCKET_COUNT] = {};
};

struct ggml_backend_sched_consumer_chain_item {
    uint64_t outputs = 0;
    uint64_t bytes = 0;
};

struct ggml_backend_sched_consumer_chain_agg {
    std::unordered_map<std::string, ggml_backend_sched_consumer_chain_item> paths;
    uint64_t terminal_graph_output = 0;
    uint64_t terminal_no_consumer = 0;
    uint64_t terminal_depth_limit = 0;
    uint64_t branch_points = 0;
};

struct ggml_backend_sched_top_split_item {
    std::string backend;
    std::string nodes;
    uint64_t copy_us = 0;
    uint64_t compute_us = 0;
    uint64_t n_nodes = 0;
    ggml_backend_sched_split_agg_counts counts = {};
};

struct ggml_backend_sched_cpu_attn_mm_item {
    uint64_t splits = 0;
    uint64_t ops = 0;
    uint64_t kq_ops = 0;
    uint64_t kqv_ops = 0;
    uint64_t copy_us = 0;
    uint64_t compute_us = 0;
    uint64_t bytes = 0;
};

struct ggml_backend_sched_split_agg_phase {
    uint64_t graphs = 0;
    uint64_t graph_wall_us = 0;
    uint64_t split_loop_wall_us = 0;
    uint64_t split_input_copy_loop_us = 0;
    uint64_t split_compute_loop_us = 0;
    uint64_t split_event_record_us = 0;
    uint64_t split_stats_bookkeeping_us = 0;
    uint64_t consumer_analyze_us = 0;
    ggml_backend_sched_split_agg_backend cpu;
    ggml_backend_sched_split_agg_backend fpga;
    ggml_backend_sched_split_agg_backend other;
    uint64_t cpu_class_mul_mat_us = 0;
    uint64_t cpu_class_flash_us = 0;
    uint64_t cpu_class_glu_us = 0;
    uint64_t cpu_class_rms_norm_us = 0;
    uint64_t cpu_class_add_us = 0;
    uint64_t cpu_class_mul_us = 0;
    uint64_t cpu_class_rope_us = 0;
    uint64_t cpu_class_get_rows_us = 0;
    uint64_t cpu_class_set_rows_us = 0;
    uint64_t cpu_class_layout_us = 0;
    uint64_t cpu_class_other_us = 0;
    uint64_t fpga_class_mul_mat_us = 0;
    uint64_t fpga_class_non_mul_mat_us = 0;
    uint64_t cpu_ops_flash = 0;
    uint64_t cpu_ops_glu = 0;
    uint64_t cpu_ops_rms_norm = 0;
    uint64_t cpu_ops_add = 0;
    uint64_t cpu_ops_mul = 0;
    uint64_t cpu_ops_rope = 0;
    uint64_t cpu_ops_get_rows = 0;
    uint64_t cpu_ops_set_rows = 0;
    uint64_t cpu_ops_layout = 0;
    uint64_t cpu_ops_other = 0;
    uint64_t fpga_ops_mul_mat = 0;
    uint64_t fpga_ops_non_mul_mat = 0;
    ggml_backend_sched_consumer_agg fpga_output_consumers;
    ggml_backend_sched_readback_edge_agg fpga_to_cpu_readbacks;
    ggml_backend_sched_consumer_chain_agg fpga_output_chains;
    std::vector<ggml_backend_sched_top_split_item> cpu_top_splits;
    std::unordered_map<std::string, ggml_backend_sched_cpu_attn_mm_item> cpu_attn_mm_shapes;
};

static std::mutex g_ggml_backend_sched_split_agg_mutex;
static ggml_backend_sched_split_agg_phase g_ggml_backend_sched_split_agg[GGML_BACKEND_FPGA_PHASE_COUNT];

static bool ggml_backend_sched_split_agg_enabled() {
    const char * value = getenv("GGML_SCHED_SPLIT_AGG_STATS");
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool ggml_backend_sched_cpu_top_splits_enabled() {
    const char * value = getenv("GGML_SCHED_CPU_TOP_SPLITS");
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool ggml_backend_sched_cpu_attn_mm_shapes_enabled() {
    const char * value = getenv("GGML_SCHED_CPU_ATTN_MATMUL_SHAPES");
    return value != NULL && value[0] != '\0' && strcmp(value, "0") != 0;
}

static bool ggml_backend_sched_cpu_top_splits_phase_enabled(enum ggml_backend_fpga_phase phase) {
    const char * value = getenv("GGML_SCHED_CPU_TOP_SPLITS");
    if (value == NULL || value[0] == '\0' || strcmp(value, "0") == 0) {
        return false;
    }
    if (strcmp(value, "1") == 0 || strcmp(value, "all") == 0 || strcmp(value, "ALL") == 0) {
        return true;
    }
    if ((strcmp(value, "encoder") == 0 || strcmp(value, "ENCODER") == 0) &&
            phase == GGML_BACKEND_FPGA_PHASE_ENCODER) {
        return true;
    }
    if ((strcmp(value, "prefill") == 0 || strcmp(value, "PREFILL") == 0) &&
            phase == GGML_BACKEND_FPGA_PHASE_PREFILL) {
        return true;
    }
    if ((strcmp(value, "decode") == 0 || strcmp(value, "DECODE") == 0) &&
            phase == GGML_BACKEND_FPGA_PHASE_DECODE) {
        return true;
    }
    return false;
}

static bool ggml_backend_sched_cpu_attn_mm_shapes_phase_enabled(enum ggml_backend_fpga_phase phase) {
    const char * value = getenv("GGML_SCHED_CPU_ATTN_MATMUL_SHAPES");
    if (value == NULL || value[0] == '\0' || strcmp(value, "0") == 0) {
        return false;
    }
    if (strcmp(value, "1") == 0 || strcmp(value, "all") == 0 || strcmp(value, "ALL") == 0) {
        return true;
    }
    if ((strcmp(value, "encoder") == 0 || strcmp(value, "ENCODER") == 0) &&
            phase == GGML_BACKEND_FPGA_PHASE_ENCODER) {
        return true;
    }
    if ((strcmp(value, "prefill") == 0 || strcmp(value, "PREFILL") == 0) &&
            phase == GGML_BACKEND_FPGA_PHASE_PREFILL) {
        return true;
    }
    if ((strcmp(value, "decode") == 0 || strcmp(value, "DECODE") == 0) &&
            phase == GGML_BACKEND_FPGA_PHASE_DECODE) {
        return true;
    }
    return false;
}

static int64_t ggml_backend_sched_now_us() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration_cast<std::chrono::microseconds>(clock::now().time_since_epoch()).count();
}

static bool ggml_backend_sched_backend_name_has(const std::string & backend, const char * needle) {
    return strstr(backend.c_str(), needle) != NULL;
}

static bool ggml_backend_sched_is_cpu_backend_name(const std::string & backend) {
    return ggml_backend_sched_backend_name_has(backend, "CPU");
}

static bool ggml_backend_sched_is_fpga_backend_name(const std::string & backend) {
    return ggml_backend_sched_backend_name_has(backend, "FPGA");
}

static ggml_backend_sched_split_agg_backend & ggml_backend_sched_split_agg_select_backend(
        ggml_backend_sched_split_agg_phase & stats,
        const std::string & backend) {
    if (ggml_backend_sched_is_cpu_backend_name(backend)) {
        return stats.cpu;
    }
    if (ggml_backend_sched_is_fpga_backend_name(backend)) {
        return stats.fpga;
    }
    return stats.other;
}

static void ggml_backend_sched_split_agg_add_backend(
        ggml_backend_sched_split_agg_backend & dst,
        const ggml_backend_sched_split_agg_backend & src) {
    dst.splits          += src.splits;
    dst.nodes           += src.nodes;
    dst.copy_us         += src.copy_us;
    dst.compute_us      += src.compute_us;
    dst.mul_mat         += src.mul_mat;
    dst.r5_like_mul_mat += src.r5_like_mul_mat;
    dst.cpu_post_ops    += src.cpu_post_ops;
    dst.matmul_splits   += src.matmul_splits;
    dst.non_matmul_splits += src.non_matmul_splits;
    dst.mixed_splits    += src.mixed_splits;
    dst.max_nodes_per_split = std::max(dst.max_nodes_per_split, src.max_nodes_per_split);
}

static void ggml_backend_sched_cpu_attn_mm_shapes_add(
        std::unordered_map<std::string, ggml_backend_sched_cpu_attn_mm_item> & dst,
        const std::unordered_map<std::string, ggml_backend_sched_cpu_attn_mm_item> & src) {
    for (std::unordered_map<std::string, ggml_backend_sched_cpu_attn_mm_item>::const_iterator it =
            src.begin(); it != src.end(); ++it) {
        ggml_backend_sched_cpu_attn_mm_item & item = dst[it->first];
        item.splits += it->second.splits;
        item.ops += it->second.ops;
        item.kq_ops += it->second.kq_ops;
        item.kqv_ops += it->second.kqv_ops;
        item.copy_us += it->second.copy_us;
        item.compute_us += it->second.compute_us;
        item.bytes += it->second.bytes;
    }
}

static void ggml_backend_sched_consumer_agg_add(
        ggml_backend_sched_consumer_agg & dst,
        const ggml_backend_sched_consumer_agg & src) {
    dst.fpga_outputs += src.fpga_outputs;
    dst.fpga_output_bytes += src.fpga_output_bytes;
    dst.consumer_edges += src.consumer_edges;
    dst.cpu_edges += src.cpu_edges;
    dst.fpga_edges += src.fpga_edges;
    dst.other_edges += src.other_edges;
    dst.same_split_edges += src.same_split_edges;
    dst.cross_split_edges += src.cross_split_edges;
    dst.multi_consumer_outputs += src.multi_consumer_outputs;
    dst.graph_output_or_cross_graph += src.graph_output_or_cross_graph;
    dst.no_consumer_non_output += src.no_consumer_non_output;
    dst.first_cpu += src.first_cpu;
    dst.first_fpga += src.first_fpga;
    dst.first_other += src.first_other;
    dst.first_graph_output += src.first_graph_output;
    dst.first_cpu_bytes += src.first_cpu_bytes;
    dst.first_fpga_bytes += src.first_fpga_bytes;
    dst.first_other_bytes += src.first_other_bytes;
    dst.first_graph_output_bytes += src.first_graph_output_bytes;
    dst.no_consumer_non_output_bytes += src.no_consumer_non_output_bytes;
    dst.multi_consumer_output_bytes += src.multi_consumer_output_bytes;
    dst.immediate_first += src.immediate_first;
    dst.delayed_first += src.delayed_first;
    dst.immediate_first_bytes += src.immediate_first_bytes;
    dst.delayed_first_bytes += src.delayed_first_bytes;
    dst.first_distance_sum += src.first_distance_sum;
    dst.first_distance_max = std::max(dst.first_distance_max, src.first_distance_max);
    dst.consumer_op_mul_mat += src.consumer_op_mul_mat;
    dst.consumer_op_flash += src.consumer_op_flash;
    dst.consumer_op_glu += src.consumer_op_glu;
    dst.consumer_op_rms_norm += src.consumer_op_rms_norm;
    dst.consumer_op_add += src.consumer_op_add;
    dst.consumer_op_mul += src.consumer_op_mul;
    dst.consumer_op_rope += src.consumer_op_rope;
    dst.consumer_op_get_rows += src.consumer_op_get_rows;
    dst.consumer_op_set_rows += src.consumer_op_set_rows;
    dst.consumer_op_layout += src.consumer_op_layout;
    dst.consumer_op_other += src.consumer_op_other;
    dst.consumer_bytes_mul_mat += src.consumer_bytes_mul_mat;
    dst.consumer_bytes_flash += src.consumer_bytes_flash;
    dst.consumer_bytes_glu += src.consumer_bytes_glu;
    dst.consumer_bytes_rms_norm += src.consumer_bytes_rms_norm;
    dst.consumer_bytes_add += src.consumer_bytes_add;
    dst.consumer_bytes_mul += src.consumer_bytes_mul;
    dst.consumer_bytes_rope += src.consumer_bytes_rope;
    dst.consumer_bytes_get_rows += src.consumer_bytes_get_rows;
    dst.consumer_bytes_set_rows += src.consumer_bytes_set_rows;
    dst.consumer_bytes_layout += src.consumer_bytes_layout;
    dst.consumer_bytes_other += src.consumer_bytes_other;
    for (std::unordered_map<std::string, ggml_backend_sched_tensor_flow_item>::const_iterator it =
            src.first_consumer_tensor_flows.begin(); it != src.first_consumer_tensor_flows.end(); ++it) {
        ggml_backend_sched_tensor_flow_item & dst_item = dst.first_consumer_tensor_flows[it->first];
        dst_item.outputs += it->second.outputs;
        dst_item.bytes += it->second.bytes;
    }
    for (std::unordered_map<std::string, ggml_backend_sched_tensor_flow_item>::const_iterator it =
            src.no_consumer_tensors.begin(); it != src.no_consumer_tensors.end(); ++it) {
        ggml_backend_sched_tensor_flow_item & dst_item = dst.no_consumer_tensors[it->first];
        dst_item.outputs += it->second.outputs;
        dst_item.bytes += it->second.bytes;
    }
}

static void ggml_backend_sched_readback_edge_agg_add(
        ggml_backend_sched_readback_edge_agg & dst,
        const ggml_backend_sched_readback_edge_agg & src) {
    dst.copies += src.copies;
    dst.bytes += src.bytes;
    dst.copy_us += src.copy_us;
    dst.producer_mul_mat += src.producer_mul_mat;
    dst.producer_mul_mat_bytes += src.producer_mul_mat_bytes;
    dst.producer_mul_mat_copy_us += src.producer_mul_mat_copy_us;
    dst.producer_other += src.producer_other;
    dst.producer_other_bytes += src.producer_other_bytes;
    dst.producer_other_copy_us += src.producer_other_copy_us;
    dst.immediate += src.immediate;
    dst.delayed += src.delayed;
    dst.no_consumer += src.no_consumer;
    dst.multi_consumer += src.multi_consumer;
    dst.first_index_sum += src.first_index_sum;
    dst.first_index_max = std::max(dst.first_index_max, src.first_index_max);
    for (int i = 0; i < GGML_BACKEND_SCHED_OP_BUCKET_COUNT; ++i) {
        dst.consumer_count[i] += src.consumer_count[i];
        dst.consumer_bytes[i] += src.consumer_bytes[i];
        dst.consumer_copy_us[i] += src.consumer_copy_us[i];
    }
}

static void ggml_backend_sched_consumer_chain_agg_add(
        ggml_backend_sched_consumer_chain_agg & dst,
        const ggml_backend_sched_consumer_chain_agg & src) {
    dst.terminal_graph_output += src.terminal_graph_output;
    dst.terminal_no_consumer += src.terminal_no_consumer;
    dst.terminal_depth_limit += src.terminal_depth_limit;
    dst.branch_points += src.branch_points;
    for (std::unordered_map<std::string, ggml_backend_sched_consumer_chain_item>::const_iterator it = src.paths.begin();
            it != src.paths.end(); ++it) {
        ggml_backend_sched_consumer_chain_item & dst_item = dst.paths[it->first];
        dst_item.outputs += it->second.outputs;
        dst_item.bytes += it->second.bytes;
    }
}

static uint64_t ggml_backend_sched_u64_add_i64(int64_t value) {
    return value > 0 ? (uint64_t) value : 0;
}

static uint64_t ggml_backend_sched_u64_sub_floor(uint64_t lhs, uint64_t rhs) {
    return lhs > rhs ? lhs - rhs : 0;
}

static uint64_t ggml_backend_sched_split_agg_layout_ops(
        const ggml_backend_sched_split_agg_counts & counts) {
    return counts.reshape + counts.view + counts.permute + counts.transpose + counts.cont;
}

static bool ggml_backend_sched_split_agg_is_layout_op(enum ggml_op op) {
    return op == GGML_OP_RESHAPE ||
           op == GGML_OP_VIEW ||
           op == GGML_OP_PERMUTE ||
           op == GGML_OP_TRANSPOSE ||
           op == GGML_OP_CONT;
}

static ggml_backend_sched_op_bucket ggml_backend_sched_op_bucket_for_op(enum ggml_op op) {
    switch (op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:     return GGML_BACKEND_SCHED_OP_BUCKET_MUL_MAT;
        case GGML_OP_FLASH_ATTN_EXT: return GGML_BACKEND_SCHED_OP_BUCKET_FLASH;
        case GGML_OP_GLU:            return GGML_BACKEND_SCHED_OP_BUCKET_GLU;
        case GGML_OP_RMS_NORM:       return GGML_BACKEND_SCHED_OP_BUCKET_RMS_NORM;
        case GGML_OP_ADD:            return GGML_BACKEND_SCHED_OP_BUCKET_ADD;
        case GGML_OP_MUL:            return GGML_BACKEND_SCHED_OP_BUCKET_MUL;
        case GGML_OP_ROPE:           return GGML_BACKEND_SCHED_OP_BUCKET_ROPE;
        case GGML_OP_GET_ROWS:       return GGML_BACKEND_SCHED_OP_BUCKET_GET_ROWS;
        case GGML_OP_SET_ROWS:       return GGML_BACKEND_SCHED_OP_BUCKET_SET_ROWS;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:           return GGML_BACKEND_SCHED_OP_BUCKET_LAYOUT;
        default:                     return GGML_BACKEND_SCHED_OP_BUCKET_OTHER;
    }
}

static bool ggml_backend_sched_split_agg_is_fpga_output_candidate(const ggml_tensor * node) {
    return node != NULL && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID);
}

static const char * ggml_backend_sched_backend_tag(
        ggml_backend_sched_t sched,
        int backend_id) {
    if (sched == NULL || backend_id < 0 || backend_id >= sched->n_backends) {
        return "NONE";
    }
    const char * backend_name_c = ggml_backend_name(sched->backends[backend_id]);
    const std::string backend_name = backend_name_c != NULL ? backend_name_c : "unknown";
    if (ggml_backend_sched_is_cpu_backend_name(backend_name)) {
        return "CPU";
    }
    if (ggml_backend_sched_is_fpga_backend_name(backend_name)) {
        return "FPGA";
    }
    return "OTHER";
}

static std::string ggml_backend_sched_chain_op_label(
        ggml_backend_sched_t sched,
        const ggml_tensor * node,
        int backend_id) {
    const char * op_name = node != NULL ? ggml_op_name(node->op) : "NULL";
    std::string label = op_name != NULL ? op_name : "UNKNOWN";
    label += "[";
    label += ggml_backend_sched_backend_tag(sched, backend_id);
    label += "]";
    return label;
}

static std::string ggml_backend_sched_tensor_flow_label(
        ggml_backend_sched_t sched,
        const ggml_tensor * node,
        int backend_id) {
    std::string label = ggml_backend_sched_chain_op_label(sched, node, backend_id);
    label += ":";
    if (node != NULL && node->name[0] != '\0') {
        label += node->name;
    } else {
        label += "<unnamed>";
    }
    if (node != NULL) {
        char shape[128];
        snprintf(shape, sizeof(shape), "(%" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 ")",
                node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
        label += shape;
    }
    return label;
}

static std::string ggml_backend_sched_split_agg_node_label(const ggml_tensor * node) {
    if (node == NULL) {
        return "NULL";
    }

    std::string label = ggml_op_name(node->op);
    label += ":";
    label += node->name[0] != '\0' ? node->name : "<unnamed>";

    char shape[192];
    snprintf(shape, sizeof(shape),
            "(%s %" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64 " %.3fMiB)",
            ggml_type_name(node->type),
            node->ne[0], node->ne[1], node->ne[2], node->ne[3],
            ggml_nbytes(node) / (1024.0 * 1024.0));
    label += shape;
    return label;
}

static bool ggml_backend_sched_tensor_name_starts_with(
        const ggml_tensor * node,
        const char * prefix) {
    if (node == NULL || prefix == NULL) {
        return false;
    }
    const size_t n = strlen(prefix);
    return strncmp(node->name, prefix, n) == 0;
}

static std::string ggml_backend_sched_attn_mm_shape_label(const ggml_tensor * node) {
    if (node == NULL) {
        return "NULL";
    }

    const char * kind = "attn_mm";
    if (ggml_backend_sched_tensor_name_starts_with(node, "kqv")) {
        kind = "kqv";
    } else if (ggml_backend_sched_tensor_name_starts_with(node, "kq")) {
        kind = "kq";
    }

    char shape[256];
    snprintf(shape, sizeof(shape),
            "%s dst=%s %" PRId64 "x%" PRId64 "x%" PRId64 "x%" PRId64,
            kind,
            ggml_type_name(node->type),
            node->ne[0], node->ne[1], node->ne[2], node->ne[3]);
    return shape;
}

static std::string ggml_backend_sched_split_agg_node_list(
        const ggml_backend_sched_split * split,
        int node_limit) {
    std::string out;
    if (split == NULL || node_limit <= 0) {
        return out;
    }

    const int n = std::min(split->graph.n_nodes, node_limit);
    for (int i = 0; i < n; ++i) {
        if (!out.empty()) {
            out += " | ";
        }
        out += ggml_backend_sched_split_agg_node_label(split->graph.nodes[i]);
    }
    if (split->graph.n_nodes > node_limit) {
        char omitted[64];
        snprintf(omitted, sizeof(omitted), " | ... +%d nodes",
                split->graph.n_nodes - node_limit);
        out += omitted;
    }
    return out;
}

static void ggml_backend_sched_tensor_flow_agg_add(
        std::unordered_map<std::string, ggml_backend_sched_tensor_flow_item> & flows,
        const std::string & key,
        uint64_t bytes) {
    ggml_backend_sched_tensor_flow_item & item = flows[key];
    item.outputs++;
    item.bytes += bytes;
}

static void ggml_backend_sched_consumer_agg_add_op(
        ggml_backend_sched_consumer_agg & stats,
        enum ggml_op op,
        uint64_t bytes) {
    switch (op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:        stats.consumer_op_mul_mat++; stats.consumer_bytes_mul_mat += bytes; break;
        case GGML_OP_FLASH_ATTN_EXT:    stats.consumer_op_flash++; stats.consumer_bytes_flash += bytes; break;
        case GGML_OP_GLU:               stats.consumer_op_glu++; stats.consumer_bytes_glu += bytes; break;
        case GGML_OP_RMS_NORM:          stats.consumer_op_rms_norm++; stats.consumer_bytes_rms_norm += bytes; break;
        case GGML_OP_ADD:               stats.consumer_op_add++; stats.consumer_bytes_add += bytes; break;
        case GGML_OP_MUL:               stats.consumer_op_mul++; stats.consumer_bytes_mul += bytes; break;
        case GGML_OP_ROPE:              stats.consumer_op_rope++; stats.consumer_bytes_rope += bytes; break;
        case GGML_OP_GET_ROWS:          stats.consumer_op_get_rows++; stats.consumer_bytes_get_rows += bytes; break;
        case GGML_OP_SET_ROWS:          stats.consumer_op_set_rows++; stats.consumer_bytes_set_rows += bytes; break;
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_CONT:              stats.consumer_op_layout++; stats.consumer_bytes_layout += bytes; break;
        default:                        stats.consumer_op_other++; stats.consumer_bytes_other += bytes; break;
    }
}

static uint64_t ggml_backend_sched_split_agg_class_count(
        const ggml_backend_sched_split_agg_counts & counts) {
    uint64_t classes = 0;
    classes += counts.mul_mat > 0 ? 1 : 0;
    classes += counts.flash_attn > 0 ? 1 : 0;
    classes += counts.glu > 0 ? 1 : 0;
    classes += counts.rms_norm > 0 ? 1 : 0;
    classes += counts.add > 0 ? 1 : 0;
    classes += counts.mul > 0 ? 1 : 0;
    classes += counts.rope > 0 ? 1 : 0;
    classes += counts.get_rows > 0 ? 1 : 0;
    classes += counts.set_rows > 0 ? 1 : 0;
    classes += ggml_backend_sched_split_agg_layout_ops(counts) > 0 ? 1 : 0;
    return classes;
}

static ggml_backend_sched_split_agg_counts ggml_backend_sched_split_agg_count_ops(
        const ggml_backend_sched_split * split,
        const std::string & backend) {
    ggml_backend_sched_split_agg_counts counts = {};
    for (int i = 0; i < split->graph.n_nodes; ++i) {
        const ggml_tensor * node = split->graph.nodes[i];
        if (node == NULL) {
            continue;
        }
        switch (node->op) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
                counts.mul_mat++;
                if (ggml_backend_sched_is_fpga_backend_name(backend)) {
                    counts.r5_like_mul_mat++;
                }
                break;
            case GGML_OP_FLASH_ATTN_EXT: counts.flash_attn++; break;
            case GGML_OP_GLU:            counts.glu++; break;
            case GGML_OP_ADD:            counts.add++; break;
            case GGML_OP_RMS_NORM:       counts.rms_norm++; break;
            case GGML_OP_MUL:            counts.mul++; break;
            case GGML_OP_ROPE:           counts.rope++; break;
            case GGML_OP_SET_ROWS:       counts.set_rows++; break;
            case GGML_OP_GET_ROWS:       counts.get_rows++; break;
            case GGML_OP_RESHAPE:        counts.reshape++; break;
            case GGML_OP_VIEW:           counts.view++; break;
            case GGML_OP_PERMUTE:        counts.permute++; break;
            case GGML_OP_TRANSPOSE:      counts.transpose++; break;
            case GGML_OP_CONT:           counts.cont++; break;
            default: break;
        }
    }
    return counts;
}

static void ggml_backend_sched_find_real_consumers(
        const std::vector<ggml_tensor *> & nodes,
        const std::vector<std::vector<int>> & direct_consumers,
        int producer_index,
        int current_index,
        std::unordered_set<int> & visited,
        std::vector<int> & real_consumers) {
    if (current_index < 0 || current_index >= (int) nodes.size()) {
        return;
    }
    if (!visited.insert(current_index).second) {
        return;
    }
    for (const int consumer_index : direct_consumers[(size_t) current_index]) {
        if (consumer_index <= producer_index || consumer_index < 0 || consumer_index >= (int) nodes.size()) {
            continue;
        }
        const ggml_tensor * consumer = nodes[(size_t) consumer_index];
        if (consumer == NULL) {
            continue;
        }
        if (ggml_backend_sched_split_agg_is_layout_op(consumer->op)) {
            ggml_backend_sched_find_real_consumers(
                    nodes,
                    direct_consumers,
                    producer_index,
                    consumer_index,
                    visited,
                    real_consumers);
        } else {
            real_consumers.push_back(consumer_index);
        }
    }
}

static void ggml_backend_sched_find_next_real_consumers(
        const std::vector<ggml_tensor *> & nodes,
        const std::vector<std::vector<int>> & direct_consumers,
        int current_index,
        std::unordered_set<int> & visited_layout,
        std::vector<int> & real_consumers) {
    if (current_index < 0 || current_index >= (int) nodes.size()) {
        return;
    }
    if (!visited_layout.insert(current_index).second) {
        return;
    }
    for (const int consumer_index : direct_consumers[(size_t) current_index]) {
        if (consumer_index <= current_index || consumer_index < 0 || consumer_index >= (int) nodes.size()) {
            continue;
        }
        const ggml_tensor * consumer = nodes[(size_t) consumer_index];
        if (consumer == NULL) {
            continue;
        }
        if (ggml_backend_sched_split_agg_is_layout_op(consumer->op)) {
            ggml_backend_sched_find_next_real_consumers(
                    nodes,
                    direct_consumers,
                    consumer_index,
                    visited_layout,
                    real_consumers);
        } else {
            real_consumers.push_back(consumer_index);
        }
    }
}

static void ggml_backend_sched_consumer_chain_record_path(
        ggml_backend_sched_consumer_chain_agg & chain_stats,
        const std::string & path,
        uint64_t producer_bytes) {
    ggml_backend_sched_consumer_chain_item & item = chain_stats.paths[path];
    item.outputs++;
    item.bytes += producer_bytes;
}

static void ggml_backend_sched_split_agg_top_insert(
        std::vector<ggml_backend_sched_top_split_item> & top,
        const ggml_backend_sched_top_split_item & item,
        size_t limit) {
    if (limit == 0 || item.compute_us == 0) {
        return;
    }
    top.push_back(item);
    std::sort(top.begin(), top.end(),
            [](const ggml_backend_sched_top_split_item & a,
               const ggml_backend_sched_top_split_item & b) {
                if (a.compute_us != b.compute_us) {
                    return a.compute_us > b.compute_us;
                }
                if (a.copy_us != b.copy_us) {
                    return a.copy_us > b.copy_us;
                }
                return a.nodes < b.nodes;
            });
    if (top.size() > limit) {
        top.resize(limit);
    }
}

static void ggml_backend_sched_split_agg_top_merge(
        std::vector<ggml_backend_sched_top_split_item> & dst,
        const std::vector<ggml_backend_sched_top_split_item> & src,
        size_t limit) {
    for (const ggml_backend_sched_top_split_item & item : src) {
        ggml_backend_sched_split_agg_top_insert(dst, item, limit);
    }
}

static void ggml_backend_sched_split_agg_record_cpu_attn_mm_shapes(
        ggml_backend_sched_split_agg_phase & graph_stats,
        const ggml_backend_sched_split * split,
        const std::string & backend,
        int64_t copy_us,
        int64_t compute_us) {
    if (!ggml_backend_sched_is_cpu_backend_name(backend) || split == NULL) {
        return;
    }
    if (!ggml_backend_sched_cpu_attn_mm_shapes_phase_enabled(ggml_backend_fpga_get_compute_phase())) {
        return;
    }

    uint64_t ops = 0;
    uint64_t kq_ops = 0;
    uint64_t kqv_ops = 0;
    uint64_t bytes = 0;
    std::string label;

    for (int i = 0; i < split->graph.n_nodes; ++i) {
        const ggml_tensor * node = split->graph.nodes[i];
        if (node == NULL || node->op != GGML_OP_MUL_MAT) {
            continue;
        }

        const bool is_kqv = ggml_backend_sched_tensor_name_starts_with(node, "kqv");
        const bool is_kq = !is_kqv && ggml_backend_sched_tensor_name_starts_with(node, "kq");
        if (!is_kq && !is_kqv) {
            continue;
        }

        if (!label.empty()) {
            label += " + ";
        }
        label += ggml_backend_sched_attn_mm_shape_label(node);
        ops++;
        kq_ops += is_kq ? 1 : 0;
        kqv_ops += is_kqv ? 1 : 0;
        bytes += ggml_nbytes(node);
    }

    if (ops == 0) {
        return;
    }

    ggml_backend_sched_cpu_attn_mm_item & item = graph_stats.cpu_attn_mm_shapes[label];
    item.splits++;
    item.ops += ops;
    item.kq_ops += kq_ops;
    item.kqv_ops += kqv_ops;
    item.copy_us += ggml_backend_sched_u64_add_i64(copy_us);
    item.compute_us += ggml_backend_sched_u64_add_i64(compute_us);
    item.bytes += bytes;
}

static std::string ggml_backend_sched_split_agg_counts_label(
        const ggml_backend_sched_split_agg_counts & counts,
        uint64_t n_nodes) {
    const uint64_t layout_ops = ggml_backend_sched_split_agg_layout_ops(counts);
    const uint64_t counted_ops =
        counts.mul_mat +
        counts.flash_attn +
        counts.glu +
        counts.rms_norm +
        counts.add +
        counts.mul +
        counts.rope +
        counts.get_rows +
        counts.set_rows +
        layout_ops;
    const uint64_t other_ops = n_nodes > counted_ops ? n_nodes - counted_ops : 0;

    char buf[384];
    snprintf(buf, sizeof(buf),
            "mm=%" PRIu64 " flash=%" PRIu64 " glu=%" PRIu64
            " rms=%" PRIu64 " add=%" PRIu64 " mul=%" PRIu64
            " rope=%" PRIu64 " get_rows=%" PRIu64 " set_rows=%" PRIu64
            " layout=%" PRIu64 " other=%" PRIu64,
            counts.mul_mat,
            counts.flash_attn,
            counts.glu,
            counts.rms_norm,
            counts.add,
            counts.mul,
            counts.rope,
            counts.get_rows,
            counts.set_rows,
            layout_ops,
            other_ops);
    return buf;
}

static void ggml_backend_sched_split_agg_record_top_cpu_split(
        ggml_backend_sched_split_agg_phase & graph_stats,
        const ggml_backend_sched_split * split,
        const std::string & backend,
        int64_t copy_us,
        int64_t compute_us,
        const ggml_backend_sched_split_agg_counts & counts) {
    if (!ggml_backend_sched_is_cpu_backend_name(backend) || split == NULL) {
        return;
    }

    const uint64_t compute_u = ggml_backend_sched_u64_add_i64(compute_us);
    if (compute_u == 0) {
        return;
    }

    if (!ggml_backend_sched_cpu_top_splits_phase_enabled(ggml_backend_fpga_get_compute_phase()) || compute_u < 5000) {
        return;
    }

    // Keep this path cheap: only stringify a split if it can enter top-N.
    if (graph_stats.cpu_top_splits.size() >= 32 &&
            compute_u <= graph_stats.cpu_top_splits.back().compute_us) {
        return;
    }

    ggml_backend_sched_top_split_item item;
    item.backend = backend;
    item.copy_us = ggml_backend_sched_u64_add_i64(copy_us);
    item.compute_us = compute_u;
    item.n_nodes = split->graph.n_nodes > 0 ? (uint64_t) split->graph.n_nodes : 0;
    item.counts = counts;
    item.nodes = ggml_backend_sched_split_agg_node_list(split, 16);
    ggml_backend_sched_split_agg_top_insert(graph_stats.cpu_top_splits, item, 32);
}

static void ggml_backend_sched_trace_consumer_chain_dfs(
        ggml_backend_sched_t sched,
        const std::vector<ggml_tensor *> & nodes,
        const std::vector<int> & node_backend_ids,
        const std::vector<std::vector<int>> & direct_consumers,
        int current_index,
        int depth,
        int max_depth,
        const std::string & path,
        uint64_t producer_bytes,
        std::unordered_set<int> & active,
        ggml_backend_sched_consumer_chain_agg & chain_stats) {
    if (depth >= max_depth) {
        chain_stats.terminal_depth_limit++;
        ggml_backend_sched_consumer_chain_record_path(
                chain_stats,
                path.empty() ? "DEPTH_LIMIT" : path + "->DEPTH_LIMIT",
                producer_bytes);
        return;
    }
    if (!active.insert(current_index).second) {
        ggml_backend_sched_consumer_chain_record_path(
                chain_stats,
                path.empty() ? "CYCLE" : path + "->CYCLE",
                producer_bytes);
        return;
    }

    std::unordered_set<int> visited_layout;
    std::vector<int> next_consumers;
    ggml_backend_sched_find_next_real_consumers(
            nodes,
            direct_consumers,
            current_index,
            visited_layout,
            next_consumers);
    std::sort(next_consumers.begin(), next_consumers.end());
    next_consumers.erase(std::unique(next_consumers.begin(), next_consumers.end()), next_consumers.end());

    if (next_consumers.empty()) {
        const ggml_tensor * current = current_index >= 0 && current_index < (int) nodes.size() ?
            nodes[(size_t) current_index] : NULL;
        const bool graph_output = current != NULL && (current->flags & GGML_TENSOR_FLAG_OUTPUT);
        if (graph_output) {
            chain_stats.terminal_graph_output++;
            ggml_backend_sched_consumer_chain_record_path(
                    chain_stats,
                    path.empty() ? "GRAPH_OUTPUT" : path + "->GRAPH_OUTPUT",
                    producer_bytes);
        } else {
            chain_stats.terminal_no_consumer++;
            ggml_backend_sched_consumer_chain_record_path(
                    chain_stats,
                    path.empty() ? "NO_CONSUMER" : path + "->NO_CONSUMER",
                    producer_bytes);
        }
        active.erase(current_index);
        return;
    }

    if (next_consumers.size() > 1) {
        chain_stats.branch_points++;
    }

    for (const int consumer_index : next_consumers) {
        const ggml_tensor * consumer = nodes[(size_t) consumer_index];
        const int backend_id = node_backend_ids[(size_t) consumer_index];
        const std::string label = ggml_backend_sched_chain_op_label(sched, consumer, backend_id);
        const std::string next_path = path.empty() ? label : path + "->" + label;
        ggml_backend_sched_trace_consumer_chain_dfs(
                sched,
                nodes,
                node_backend_ids,
                direct_consumers,
                consumer_index,
                depth + 1,
                max_depth,
                next_path,
                producer_bytes,
                active,
                chain_stats);
    }
    active.erase(current_index);
}

static void ggml_backend_sched_trace_consumer_chain(
        ggml_backend_sched_t sched,
        const std::vector<ggml_tensor *> & nodes,
        const std::vector<int> & node_backend_ids,
        const std::vector<std::vector<int>> & direct_consumers,
        int producer_index,
        uint64_t producer_bytes,
        ggml_backend_sched_consumer_chain_agg & chain_stats) {
    std::unordered_set<int> active;
    ggml_backend_sched_trace_consumer_chain_dfs(
            sched,
            nodes,
            node_backend_ids,
            direct_consumers,
            producer_index,
            0,
            10,
            "",
            producer_bytes,
            active,
            chain_stats);
}

static bool ggml_backend_sched_node_consumes_tensor(
        const ggml_tensor * node,
        const ggml_tensor * tensor) {
    if (node == NULL || tensor == NULL) {
        return false;
    }
    for (int src_id = 0; src_id < GGML_MAX_SRC; ++src_id) {
        const ggml_tensor * src = node->src[src_id];
        if (src == tensor || (src != NULL && src->view_src == tensor)) {
            return true;
        }
    }
    return node->view_src == tensor;
}

static void ggml_backend_sched_find_real_consumers_in_split(
        const ggml_backend_sched_split * split,
        const ggml_tensor * tensor,
        std::unordered_set<const ggml_tensor *> & visited,
        std::vector<int> & real_consumers) {
    if (split == NULL || tensor == NULL || !visited.insert(tensor).second) {
        return;
    }
    for (int i = 0; i < split->graph.n_nodes; ++i) {
        const ggml_tensor * node = split->graph.nodes[i];
        if (!ggml_backend_sched_node_consumes_tensor(node, tensor)) {
            continue;
        }
        if (ggml_backend_sched_split_agg_is_layout_op(node->op)) {
            ggml_backend_sched_find_real_consumers_in_split(split, node, visited, real_consumers);
        } else {
            real_consumers.push_back(i);
        }
    }
}

static void ggml_backend_sched_readback_edge_agg_record(
        ggml_backend_sched_readback_edge_agg & stats,
        const ggml_tensor * producer,
        const ggml_backend_sched_split * consumer_split,
        const ggml_tensor * copied_tensor,
        uint64_t copy_us) {
    if (producer == NULL || consumer_split == NULL || copied_tensor == NULL) {
        return;
    }

    const uint64_t bytes = ggml_nbytes(producer);
    stats.copies++;
    stats.bytes += bytes;
    stats.copy_us += copy_us;
    if (ggml_backend_sched_split_agg_is_fpga_output_candidate(producer)) {
        stats.producer_mul_mat++;
        stats.producer_mul_mat_bytes += bytes;
        stats.producer_mul_mat_copy_us += copy_us;
    } else {
        stats.producer_other++;
        stats.producer_other_bytes += bytes;
        stats.producer_other_copy_us += copy_us;
    }

    std::unordered_set<const ggml_tensor *> visited;
    std::vector<int> real_consumers;
    ggml_backend_sched_find_real_consumers_in_split(
            consumer_split,
            copied_tensor,
            visited,
            real_consumers);
    std::sort(real_consumers.begin(), real_consumers.end());
    real_consumers.erase(std::unique(real_consumers.begin(), real_consumers.end()), real_consumers.end());

    if (real_consumers.empty()) {
        stats.no_consumer++;
        return;
    }
    if (real_consumers.size() > 1) {
        stats.multi_consumer++;
    }
    const int first_index = real_consumers.front();
    const uint64_t first_index_u = first_index >= 0 ? (uint64_t) first_index : 0;
    stats.first_index_sum += first_index_u;
    stats.first_index_max = std::max(stats.first_index_max, first_index_u);
    if (first_index == 0) {
        stats.immediate++;
    } else {
        stats.delayed++;
    }

    const ggml_tensor * consumer = consumer_split->graph.nodes[first_index];
    const ggml_backend_sched_op_bucket bucket = ggml_backend_sched_op_bucket_for_op(consumer->op);
    stats.consumer_count[bucket]++;
    stats.consumer_bytes[bucket] += bytes;
    stats.consumer_copy_us[bucket] += copy_us;
}

static bool ggml_backend_sched_vit_add_sidecache_enabled() {
    const char * value = getenv("GGML_FPGA_VIT_ADD_SIDECACHE");
    return value != NULL && value[0] != '\0' &&
           strcmp(value, "0") != 0 &&
           strcmp(value, "false") != 0 &&
           strcmp(value, "FALSE") != 0 &&
           strcmp(value, "off") != 0 &&
           strcmp(value, "OFF") != 0;
}

static bool ggml_backend_sched_prepared_activation_allows_m(int64_t m) {
    if (m > 1) {
        return true;
    }
    return m == 1 &&
        ggml_backend_fpga_get_compute_phase() == GGML_BACKEND_FPGA_PHASE_DECODE &&
        ggml_backend_fpga_ffn_streaming_m1_decode_enabled();
}

static bool ggml_backend_sched_is_prepared_activation_mul_mat_consumer(
        const ggml_tensor * node,
        const ggml_tensor * input_cpy) {
    if (node == NULL || input_cpy == NULL || node->op != GGML_OP_MUL_MAT || node->src[1] == NULL) {
        return false;
    }
    const bool consumes_input_copy =
        node->src[1] == input_cpy ||
        node->src[1]->view_src == input_cpy;
    if (!consumes_input_copy) {
        return false;
    }
    const ggml_tensor * src0 = node->src[0];
    const ggml_tensor * src1 = node->src[1];
    if (src0 == NULL || src1 == NULL || src1->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t m = src1->ne[1];
    const int64_t gemm_instances = node->ne[2] * node->ne[3];
    return gemm_instances == 1 && ggml_backend_sched_prepared_activation_allows_m(m);
}

static bool ggml_backend_sched_is_prepared_activation_mul_mat_node(const ggml_tensor * node) {
    if (node == NULL || node->op != GGML_OP_MUL_MAT || node->src[1] == NULL) {
        return false;
    }
    const ggml_tensor * src1 = node->src[1];
    if (src1->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t m = src1->ne[1];
    const int64_t gemm_instances = node->ne[2] * node->ne[3];
    return gemm_instances == 1 && ggml_backend_sched_prepared_activation_allows_m(m);
}

static bool ggml_backend_sched_can_prepare_cpu_activation(const ggml_tensor * input) {
    if (input == NULL || input->type != GGML_TYPE_F32 || input->nb[0] != (int64_t) sizeof(float)) {
        return false;
    }

    switch (input->op) {
        case GGML_OP_ADD:
        case GGML_OP_MUL:
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_GLU:
        case GGML_OP_UNARY:
            return true;
        default:
            return false;
    }
}

static const ggml_tensor * ggml_backend_sched_find_prepare_source(
        const ggml_tensor * tensor,
        const ggml_tensor ** alias) {
    if (alias != NULL) {
        *alias = NULL;
    }
    if (tensor == NULL) {
        return NULL;
    }
    if (ggml_backend_sched_can_prepare_cpu_activation(tensor)) {
        return tensor;
    }
    if (tensor->view_src != NULL && ggml_backend_sched_can_prepare_cpu_activation(tensor->view_src)) {
        if (alias != NULL) {
            *alias = tensor;
        }
        return tensor->view_src;
    }
    return NULL;
}

static void ggml_backend_sched_mark_prepare_source(
        const ggml_tensor * source,
        const ggml_tensor * alias) {
    if (source == NULL || !ggml_backend_sched_can_prepare_cpu_activation(source)) {
        return;
    }
    g_ggml_fpga_ffn_stream_stats.target_scan_can_prepare.fetch_add(1, std::memory_order_relaxed);
    ggml_backend_fpga_ffn_streaming_mark_prepare_target(source);
    if (alias != NULL && alias != source) {
        ggml_backend_fpga_ffn_streaming_mark_prepare_alias(source, alias);
    }
}

static void ggml_backend_sched_mark_vit_activation_sidecache_target(
        const ggml_backend_sched_split * split,
        const ggml_tensor * input,
        const ggml_tensor * input_cpy) {
    if (!ggml_backend_sched_vit_add_sidecache_enabled() ||
            split == NULL ||
            input == NULL ||
            input_cpy == NULL ||
            !ggml_backend_sched_can_prepare_cpu_activation(input)) {
        return;
    }
    g_ggml_fpga_ffn_stream_stats.target_scan_can_prepare.fetch_add(1, std::memory_order_relaxed);
    for (int i = 0; i < split->graph.n_nodes; ++i) {
        const ggml_tensor * node = split->graph.nodes[i];
        if (ggml_backend_sched_is_prepared_activation_mul_mat_consumer(node, input_cpy)) {
            g_ggml_fpga_ffn_stream_stats.target_scan_mul_mat_consumers.fetch_add(1, std::memory_order_relaxed);
            ggml_backend_sched_mark_prepare_source(input, input_cpy);
            return;
        }
    }
}

static void ggml_backend_sched_mark_vit_activation_sidecache_targets_for_graph(ggml_backend_sched_t sched) {
    g_ggml_fpga_ffn_stream_stats.target_scan_calls.fetch_add(1, std::memory_order_relaxed);
    if (sched == NULL || !ggml_backend_sched_vit_add_sidecache_enabled()) {
        return;
    }
    if (ggml_backend_fpga_get_compute_phase() != GGML_BACKEND_FPGA_PHASE_ENCODER) {
        g_ggml_fpga_ffn_stream_stats.target_scan_phase_skips.fetch_add(1, std::memory_order_relaxed);
        return;
    }

    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        const ggml_backend_sched_split * split = &sched->splits[split_id];
        const int split_backend_id = split->backend_id;
        if (split_backend_id < 0 || split_backend_id >= sched->n_backends) {
            continue;
        }

        const char * split_backend_name_c = ggml_backend_name(sched->backends[split_backend_id]);
        const std::string split_backend_name = split_backend_name_c != NULL ? split_backend_name_c : "unknown";
        if (!ggml_backend_sched_is_fpga_backend_name(split_backend_name)) {
            continue;
        }
        g_ggml_fpga_ffn_stream_stats.target_scan_fpga_splits.fetch_add(1, std::memory_order_relaxed);

        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            g_ggml_fpga_ffn_stream_stats.target_scan_inputs.fetch_add(1, std::memory_order_relaxed);
            ggml_tensor * input = split->inputs[input_id];
            ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
            ggml_backend_sched_mark_vit_activation_sidecache_target(split, input, input_cpy);
        }

        for (int i = 0; i < split->graph.n_nodes; ++i) {
            const ggml_tensor * node = split->graph.nodes[i];
            if (!ggml_backend_sched_is_prepared_activation_mul_mat_node(node)) {
                continue;
            }
            const ggml_tensor * alias = NULL;
            const ggml_tensor * source = ggml_backend_sched_find_prepare_source(node->src[1], &alias);
            if (source == NULL) {
                continue;
            }
            g_ggml_fpga_ffn_stream_stats.target_scan_mul_mat_consumers.fetch_add(1, std::memory_order_relaxed);
            ggml_backend_sched_mark_prepare_source(source, alias != NULL ? alias : node->src[1]);
        }
    }
}

static ggml_backend_sched_consumer_agg ggml_backend_sched_analyze_fpga_output_consumers(
        ggml_backend_sched_t sched,
        ggml_backend_sched_consumer_chain_agg * chain_stats) {
    ggml_backend_sched_consumer_agg stats = {};
    if (sched == NULL || sched->n_splits <= 0) {
        return stats;
    }

    std::vector<ggml_tensor *> nodes;
    std::vector<int> node_backend_ids;
    std::vector<int> node_split_ids;
    nodes.reserve((size_t) sched->graph.n_nodes + (size_t) sched->n_splits * GGML_SCHED_MAX_SPLIT_INPUTS);
    node_backend_ids.reserve(nodes.capacity());
    node_split_ids.reserve(nodes.capacity());

    std::unordered_map<const ggml_tensor *, int> node_index;
    node_index.reserve((size_t) sched->graph.n_nodes * 2);

    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        const ggml_backend_sched_split * split = &sched->splits[split_id];
        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            ggml_tensor * input = split->inputs[input_id];
            if (input != NULL && node_index.find(input) == node_index.end()) {
                const int idx = (int) nodes.size();
                node_index[input] = idx;
                nodes.push_back(input);
                node_backend_ids.push_back(tensor_backend_id(input));
                node_split_ids.push_back(-1);
            }
        }
        for (int i = split->i_start; i < split->i_end; ++i) {
            ggml_tensor * node = sched->graph.nodes[i];
            if (node != NULL && node_index.find(node) == node_index.end()) {
                const int idx = (int) nodes.size();
                node_index[node] = idx;
                nodes.push_back(node);
                node_backend_ids.push_back(tensor_backend_id(node));
                node_split_ids.push_back(split_id);
            }
        }
    }

    if (nodes.empty()) {
        return stats;
    }

    std::vector<std::vector<int>> direct_consumers(nodes.size());
    for (int split_id = 0; split_id < sched->n_splits; ++split_id) {
        const ggml_backend_sched_split * split = &sched->splits[split_id];
        for (int input_id = 0; input_id < split->n_inputs; ++input_id) {
            ggml_tensor * input = split->inputs[input_id];
            auto input_it = node_index.find(input);
            if (input_it == node_index.end()) {
                continue;
            }
            ggml_tensor * input_cpy = tensor_copy(input, split->backend_id, sched->cur_copy);
            for (int j = split->i_start; j < split->i_end; ++j) {
                ggml_tensor * node = sched->graph.nodes[j];
                if (node == NULL) {
                    continue;
                }
                bool consumes_copy = false;
                for (int src_id = 0; src_id < GGML_MAX_SRC; ++src_id) {
                    if (node->src[src_id] == input_cpy ||
                            (node->src[src_id] != NULL && node->src[src_id]->view_src == input_cpy)) {
                        consumes_copy = true;
                        break;
                    }
                }
                if (node->view_src == input_cpy) {
                    consumes_copy = true;
                }
                if (consumes_copy) {
                    auto consumer_it = node_index.find(node);
                    if (consumer_it != node_index.end()) {
                        direct_consumers[(size_t) input_it->second].push_back(consumer_it->second);
                    }
                }
            }
        }
    }

    for (int i = 0; i < (int) nodes.size(); ++i) {
        const ggml_tensor * node = nodes[(size_t) i];
        if (node == NULL) {
            continue;
        }
        for (int src_id = 0; src_id < GGML_MAX_SRC; ++src_id) {
            const ggml_tensor * src = node->src[src_id];
            if (src == NULL) {
                continue;
            }
            auto it = node_index.find(src);
            if (it != node_index.end()) {
                direct_consumers[(size_t) it->second].push_back(i);
            }
            if (src->view_src != NULL) {
                auto view_it = node_index.find(src->view_src);
                if (view_it != node_index.end()) {
                    direct_consumers[(size_t) view_it->second].push_back(i);
                }
            }
        }
        if (node->view_src != NULL) {
            auto view_it = node_index.find(node->view_src);
            if (view_it != node_index.end()) {
                direct_consumers[(size_t) view_it->second].push_back(i);
            }
        }
    }

    for (int producer_index = 0; producer_index < (int) nodes.size(); ++producer_index) {
        ggml_tensor * producer = nodes[(size_t) producer_index];
        if (!ggml_backend_sched_split_agg_is_fpga_output_candidate(producer)) {
            continue;
        }
        const int producer_backend_id = node_backend_ids[(size_t) producer_index];
        if (producer_backend_id < 0 || producer_backend_id >= sched->n_backends) {
            continue;
        }
        const char * producer_backend_name = ggml_backend_name(sched->backends[producer_backend_id]);
        const std::string producer_backend = producer_backend_name != NULL ? producer_backend_name : "unknown";
        if (!ggml_backend_sched_is_fpga_backend_name(producer_backend)) {
            continue;
        }

        const uint64_t producer_bytes = ggml_nbytes(producer);
        stats.fpga_outputs++;
        stats.fpga_output_bytes += producer_bytes;
        if (chain_stats != NULL) {
            ggml_backend_sched_trace_consumer_chain(
                    sched,
                    nodes,
                    node_backend_ids,
                    direct_consumers,
                    producer_index,
                    producer_bytes,
                    *chain_stats);
        }

        std::unordered_set<int> visited;
        std::vector<int> real_consumers;
        ggml_backend_sched_find_real_consumers(
                nodes,
                direct_consumers,
                producer_index,
                producer_index,
                visited,
                real_consumers);

        std::sort(real_consumers.begin(), real_consumers.end());
        real_consumers.erase(std::unique(real_consumers.begin(), real_consumers.end()), real_consumers.end());

        if (real_consumers.empty()) {
            if (producer->flags & GGML_TENSOR_FLAG_OUTPUT) {
                stats.graph_output_or_cross_graph++;
                stats.first_graph_output++;
                stats.first_graph_output_bytes += producer_bytes;
            } else {
                stats.no_consumer_non_output++;
                stats.no_consumer_non_output_bytes += producer_bytes;
            }
            ggml_backend_sched_tensor_flow_agg_add(
                    stats.no_consumer_tensors,
                    ggml_backend_sched_tensor_flow_label(sched, producer, producer_backend_id),
                    producer_bytes);
            continue;
        }
        if (real_consumers.size() > 1) {
            stats.multi_consumer_outputs++;
            stats.multi_consumer_output_bytes += producer_bytes;
        }

        const int producer_split_id = node_split_ids[(size_t) producer_index];
        int first_consumer_index = -1;
        for (const int consumer_index : real_consumers) {
            ggml_tensor * consumer = nodes[(size_t) consumer_index];
            const int consumer_backend_id = node_backend_ids[(size_t) consumer_index];
            if (consumer_backend_id < 0 || consumer_backend_id >= sched->n_backends) {
                continue;
            }
            const char * consumer_backend_name = ggml_backend_name(sched->backends[consumer_backend_id]);
            const std::string consumer_backend = consumer_backend_name != NULL ? consumer_backend_name : "unknown";

            stats.consumer_edges++;
            if (ggml_backend_sched_is_cpu_backend_name(consumer_backend)) {
                stats.cpu_edges++;
            } else if (ggml_backend_sched_is_fpga_backend_name(consumer_backend)) {
                stats.fpga_edges++;
            } else {
                stats.other_edges++;
            }

            const int consumer_split_id = node_split_ids[(size_t) consumer_index];
            if (consumer_split_id == producer_split_id) {
                stats.same_split_edges++;
            } else {
                stats.cross_split_edges++;
            }
            ggml_backend_sched_consumer_agg_add_op(stats, consumer->op, producer_bytes);
            if (first_consumer_index == -1 || consumer_index < first_consumer_index) {
                first_consumer_index = consumer_index;
            }
        }

        if (first_consumer_index != -1) {
            const int first_backend_id = node_backend_ids[(size_t) first_consumer_index];
            if (first_backend_id >= 0 && first_backend_id < sched->n_backends) {
                const char * first_backend_name = ggml_backend_name(sched->backends[first_backend_id]);
                const std::string first_backend = first_backend_name != NULL ? first_backend_name : "unknown";
                if (ggml_backend_sched_is_cpu_backend_name(first_backend)) {
                    stats.first_cpu++;
                    stats.first_cpu_bytes += producer_bytes;
                } else if (ggml_backend_sched_is_fpga_backend_name(first_backend)) {
                    stats.first_fpga++;
                    stats.first_fpga_bytes += producer_bytes;
                } else {
                    stats.first_other++;
                    stats.first_other_bytes += producer_bytes;
                }
                const ggml_tensor * first_consumer = nodes[(size_t) first_consumer_index];
                const std::string flow =
                    ggml_backend_sched_tensor_flow_label(sched, producer, producer_backend_id) +
                    " -> " +
                    ggml_backend_sched_tensor_flow_label(sched, first_consumer, first_backend_id);
                ggml_backend_sched_tensor_flow_agg_add(
                        stats.first_consumer_tensor_flows,
                        flow,
                        producer_bytes);
            }
            const uint64_t distance = first_consumer_index > producer_index ?
                (uint64_t) (first_consumer_index - producer_index) : 0;
            stats.first_distance_sum += distance;
            stats.first_distance_max = std::max(stats.first_distance_max, distance);
            if (distance == 1) {
                stats.immediate_first++;
                stats.immediate_first_bytes += producer_bytes;
            } else {
                stats.delayed_first++;
                stats.delayed_first_bytes += producer_bytes;
            }
        }
    }
    return stats;
}

static void ggml_backend_sched_split_agg_record_split(
        ggml_backend_sched_split_agg_phase & graph_stats,
        const std::string & backend,
        int n_nodes,
        int64_t copy_us,
        int64_t compute_us,
        const ggml_backend_sched_split_agg_counts & counts) {
    ggml_backend_sched_split_agg_backend & backend_stats =
        ggml_backend_sched_split_agg_select_backend(graph_stats, backend);

    backend_stats.splits++;
    backend_stats.nodes += n_nodes > 0 ? (uint64_t) n_nodes : 0;
    backend_stats.copy_us += ggml_backend_sched_u64_add_i64(copy_us);
    backend_stats.compute_us += ggml_backend_sched_u64_add_i64(compute_us);
    backend_stats.mul_mat += counts.mul_mat;
    backend_stats.r5_like_mul_mat += counts.r5_like_mul_mat;
    backend_stats.max_nodes_per_split = std::max(
            backend_stats.max_nodes_per_split,
            n_nodes > 0 ? (uint64_t) n_nodes : 0);
    if (counts.mul_mat > 0) {
        backend_stats.matmul_splits++;
    } else {
        backend_stats.non_matmul_splits++;
    }
    if (ggml_backend_sched_split_agg_class_count(counts) > 1) {
        backend_stats.mixed_splits++;
    }

    const uint64_t cpu_post_ops =
        counts.glu +
        counts.add +
        counts.rms_norm +
        counts.mul +
        counts.rope +
        counts.set_rows +
        counts.flash_attn +
        counts.get_rows;
    backend_stats.cpu_post_ops += cpu_post_ops;

    const uint64_t compute_u = ggml_backend_sched_u64_add_i64(compute_us);
    const uint64_t layout_ops = ggml_backend_sched_split_agg_layout_ops(counts);
    const uint64_t counted_ops =
        counts.mul_mat +
        counts.flash_attn +
        counts.glu +
        counts.rms_norm +
        counts.add +
        counts.mul +
        counts.rope +
        counts.get_rows +
        counts.set_rows +
        layout_ops;
    const uint64_t other_ops =
        n_nodes > 0 && (uint64_t) n_nodes > counted_ops ? (uint64_t) n_nodes - counted_ops : 0;

    if (ggml_backend_sched_is_fpga_backend_name(backend)) {
        graph_stats.fpga_ops_mul_mat += counts.mul_mat;
        graph_stats.fpga_ops_non_mul_mat +=
            counts.flash_attn +
            counts.glu +
            counts.rms_norm +
            counts.add +
            counts.mul +
            counts.rope +
            counts.get_rows +
            counts.set_rows +
            layout_ops +
            other_ops;
        if (counts.mul_mat > 0) {
            graph_stats.fpga_class_mul_mat_us += compute_u;
        } else {
            graph_stats.fpga_class_non_mul_mat_us += compute_u;
        }
        return;
    }

    if (!ggml_backend_sched_is_cpu_backend_name(backend)) {
        return;
    }

    graph_stats.cpu_ops_flash += counts.flash_attn;
    graph_stats.cpu_ops_glu += counts.glu;
    graph_stats.cpu_ops_rms_norm += counts.rms_norm;
    graph_stats.cpu_ops_add += counts.add;
    graph_stats.cpu_ops_mul += counts.mul;
    graph_stats.cpu_ops_rope += counts.rope;
    graph_stats.cpu_ops_get_rows += counts.get_rows;
    graph_stats.cpu_ops_set_rows += counts.set_rows;
    graph_stats.cpu_ops_layout += layout_ops;
    graph_stats.cpu_ops_other += other_ops;

    if (counts.mul_mat > 0) {
        graph_stats.cpu_class_mul_mat_us += compute_u;
    } else if (counts.flash_attn > 0) {
        graph_stats.cpu_class_flash_us += compute_u;
    } else if (counts.glu > 0) {
        graph_stats.cpu_class_glu_us += compute_u;
    } else if (counts.rms_norm > 0) {
        graph_stats.cpu_class_rms_norm_us += compute_u;
    } else if (counts.add > 0) {
        graph_stats.cpu_class_add_us += compute_u;
    } else if (counts.mul > 0) {
        graph_stats.cpu_class_mul_us += compute_u;
    } else if (counts.rope > 0) {
        graph_stats.cpu_class_rope_us += compute_u;
    } else if (counts.get_rows > 0) {
        graph_stats.cpu_class_get_rows_us += compute_u;
    } else if (counts.set_rows > 0) {
        graph_stats.cpu_class_set_rows_us += compute_u;
    } else if (layout_ops > 0) {
        graph_stats.cpu_class_layout_us += compute_u;
    } else {
        graph_stats.cpu_class_other_us += compute_u;
    }
}

static void ggml_backend_sched_split_agg_record_graph(
        enum ggml_backend_fpga_phase phase,
        int64_t graph_wall_us,
        const ggml_backend_sched_split_agg_phase & graph_stats) {
    const int phase_index = phase >= GGML_BACKEND_FPGA_PHASE_UNKNOWN && phase < GGML_BACKEND_FPGA_PHASE_COUNT ?
        (int) phase : (int) GGML_BACKEND_FPGA_PHASE_UNKNOWN;
    std::lock_guard<std::mutex> lock(g_ggml_backend_sched_split_agg_mutex);
    ggml_backend_sched_split_agg_phase & dst = g_ggml_backend_sched_split_agg[phase_index];
    dst.graphs++;
    dst.graph_wall_us += ggml_backend_sched_u64_add_i64(graph_wall_us);
    dst.split_loop_wall_us += graph_stats.split_loop_wall_us;
    dst.split_input_copy_loop_us += graph_stats.split_input_copy_loop_us;
    dst.split_compute_loop_us += graph_stats.split_compute_loop_us;
    dst.split_event_record_us += graph_stats.split_event_record_us;
    dst.split_stats_bookkeeping_us += graph_stats.split_stats_bookkeeping_us;
    dst.consumer_analyze_us += graph_stats.consumer_analyze_us;
    ggml_backend_sched_split_agg_add_backend(dst.cpu, graph_stats.cpu);
    ggml_backend_sched_split_agg_add_backend(dst.fpga, graph_stats.fpga);
    ggml_backend_sched_split_agg_add_backend(dst.other, graph_stats.other);
    dst.cpu_class_mul_mat_us   += graph_stats.cpu_class_mul_mat_us;
    dst.cpu_class_flash_us     += graph_stats.cpu_class_flash_us;
    dst.cpu_class_glu_us       += graph_stats.cpu_class_glu_us;
    dst.cpu_class_rms_norm_us  += graph_stats.cpu_class_rms_norm_us;
    dst.cpu_class_add_us       += graph_stats.cpu_class_add_us;
    dst.cpu_class_mul_us       += graph_stats.cpu_class_mul_us;
    dst.cpu_class_rope_us      += graph_stats.cpu_class_rope_us;
    dst.cpu_class_get_rows_us  += graph_stats.cpu_class_get_rows_us;
    dst.cpu_class_set_rows_us  += graph_stats.cpu_class_set_rows_us;
    dst.cpu_class_layout_us    += graph_stats.cpu_class_layout_us;
    dst.cpu_class_other_us     += graph_stats.cpu_class_other_us;
    dst.fpga_class_mul_mat_us     += graph_stats.fpga_class_mul_mat_us;
    dst.fpga_class_non_mul_mat_us += graph_stats.fpga_class_non_mul_mat_us;
    dst.cpu_ops_flash     += graph_stats.cpu_ops_flash;
    dst.cpu_ops_glu       += graph_stats.cpu_ops_glu;
    dst.cpu_ops_rms_norm  += graph_stats.cpu_ops_rms_norm;
    dst.cpu_ops_add       += graph_stats.cpu_ops_add;
    dst.cpu_ops_mul       += graph_stats.cpu_ops_mul;
    dst.cpu_ops_rope      += graph_stats.cpu_ops_rope;
    dst.cpu_ops_get_rows  += graph_stats.cpu_ops_get_rows;
    dst.cpu_ops_set_rows  += graph_stats.cpu_ops_set_rows;
    dst.cpu_ops_layout    += graph_stats.cpu_ops_layout;
    dst.cpu_ops_other     += graph_stats.cpu_ops_other;
    dst.fpga_ops_mul_mat     += graph_stats.fpga_ops_mul_mat;
    dst.fpga_ops_non_mul_mat += graph_stats.fpga_ops_non_mul_mat;
    ggml_backend_sched_consumer_agg_add(
            dst.fpga_output_consumers,
            graph_stats.fpga_output_consumers);
    ggml_backend_sched_readback_edge_agg_add(
            dst.fpga_to_cpu_readbacks,
            graph_stats.fpga_to_cpu_readbacks);
    ggml_backend_sched_consumer_chain_agg_add(
            dst.fpga_output_chains,
            graph_stats.fpga_output_chains);
    ggml_backend_sched_cpu_attn_mm_shapes_add(
            dst.cpu_attn_mm_shapes,
            graph_stats.cpu_attn_mm_shapes);
    ggml_backend_sched_split_agg_top_merge(
            dst.cpu_top_splits,
            graph_stats.cpu_top_splits,
            32);
}

static void ggml_backend_sched_split_agg_log_backend(
        const char * label,
        const ggml_backend_sched_split_agg_backend & stats) {
    GGML_LOG_INFO("    %-5s splits=%" PRIu64 " nodes=%" PRIu64
            " copy=%.3f ms compute=%.3f ms mm=%" PRIu64
            " r5like=%" PRIu64 " cpu_post_ops=%" PRIu64
            " matmul_splits=%" PRIu64 " non_matmul_splits=%" PRIu64
            " mixed_splits=%" PRIu64 " max_nodes_per_split=%" PRIu64 "\n",
            label,
            stats.splits,
            stats.nodes,
            stats.copy_us / 1000.0,
            stats.compute_us / 1000.0,
            stats.mul_mat,
            stats.r5_like_mul_mat,
            stats.cpu_post_ops,
            stats.matmul_splits,
            stats.non_matmul_splits,
            stats.mixed_splits,
            stats.max_nodes_per_split);
}

static void ggml_backend_sched_split_agg_log_top_cpu_splits(
        const std::vector<ggml_backend_sched_top_split_item> & top) {
    GGML_LOG_INFO("    CPU top split detail: count=%zu\n", top.size());
    const size_t limit = std::min<size_t>(top.size(), 24);
    for (size_t i = 0; i < limit; ++i) {
        const ggml_backend_sched_top_split_item & item = top[i];
        const std::string counts =
            ggml_backend_sched_split_agg_counts_label(item.counts, item.n_nodes);
        GGML_LOG_INFO("      cpu_top[%02zu] compute=%.3f ms copy=%.3f ms"
                " nodes=%" PRIu64 " backend=%s counts={%s}\n",
                i,
                item.compute_us / 1000.0,
                item.copy_us / 1000.0,
                item.n_nodes,
                item.backend.c_str(),
                counts.c_str());
        GGML_LOG_INFO("        nodes: %s\n", item.nodes.c_str());
    }
    if (top.size() > limit) {
        GGML_LOG_INFO("      ... %" PRIu64 " additional CPU top splits omitted\n",
                (uint64_t) (top.size() - limit));
    }
}

static void ggml_backend_sched_split_agg_log_cpu_attn_mm_shapes(
        const std::unordered_map<std::string, ggml_backend_sched_cpu_attn_mm_item> & shapes) {
    if (shapes.empty()) {
        return;
    }

    std::vector<std::pair<std::string, ggml_backend_sched_cpu_attn_mm_item>> ordered;
    ordered.reserve(shapes.size());
    for (std::unordered_map<std::string, ggml_backend_sched_cpu_attn_mm_item>::const_iterator it =
            shapes.begin(); it != shapes.end(); ++it) {
        ordered.push_back(*it);
    }
    std::sort(ordered.begin(), ordered.end(),
            [](const std::pair<std::string, ggml_backend_sched_cpu_attn_mm_item> & a,
               const std::pair<std::string, ggml_backend_sched_cpu_attn_mm_item> & b) {
                if (a.second.compute_us != b.second.compute_us) {
                    return a.second.compute_us > b.second.compute_us;
                }
                if (a.second.splits != b.second.splits) {
                    return a.second.splits > b.second.splits;
                }
                return a.first < b.first;
            });

    uint64_t total_splits = 0;
    uint64_t total_ops = 0;
    uint64_t total_kq_ops = 0;
    uint64_t total_kqv_ops = 0;
    uint64_t total_compute_us = 0;
    uint64_t total_copy_us = 0;
    uint64_t total_bytes = 0;
    for (const auto & item : ordered) {
        total_splits += item.second.splits;
        total_ops += item.second.ops;
        total_kq_ops += item.second.kq_ops;
        total_kqv_ops += item.second.kqv_ops;
        total_compute_us += item.second.compute_us;
        total_copy_us += item.second.copy_us;
        total_bytes += item.second.bytes;
    }

    GGML_LOG_INFO("    CPU attention MUL_MAT shape aggregate: unique=%zu"
            " splits=%" PRIu64 " ops=%" PRIu64 " kq=%" PRIu64 " kqv=%" PRIu64
            " compute=%.3f ms copy=%.3f ms bytes=%.3f MiB\n",
            ordered.size(),
            total_splits,
            total_ops,
            total_kq_ops,
            total_kqv_ops,
            total_compute_us / 1000.0,
            total_copy_us / 1000.0,
            total_bytes / (1024.0 * 1024.0));

    const size_t limit = std::min<size_t>(ordered.size(), 48);
    for (size_t i = 0; i < limit; ++i) {
        const ggml_backend_sched_cpu_attn_mm_item & item = ordered[i].second;
        GGML_LOG_INFO("      attn_mm_shape[%02zu] compute=%.3f ms copy=%.3f ms"
                " splits=%" PRIu64 " ops=%" PRIu64 " kq=%" PRIu64 " kqv=%" PRIu64
                " bytes=%.3f MiB | %s\n",
                i,
                item.compute_us / 1000.0,
                item.copy_us / 1000.0,
                item.splits,
                item.ops,
                item.kq_ops,
                item.kqv_ops,
                item.bytes / (1024.0 * 1024.0),
                ordered[i].first.c_str());
    }
    if (ordered.size() > limit) {
        GGML_LOG_INFO("      ... %" PRIu64 " additional attention MUL_MAT shapes omitted\n",
                (uint64_t) (ordered.size() - limit));
    }
}

static void ggml_backend_sched_split_agg_log_readback_bucket(
        const char * label,
        const ggml_backend_sched_readback_edge_agg & stats,
        ggml_backend_sched_op_bucket bucket) {
    GGML_LOG_INFO("      %-8s count=%" PRIu64 " bytes=%.3f MiB copy=%.3f ms\n",
            label,
            stats.consumer_count[bucket],
            stats.consumer_bytes[bucket] / (1024.0 * 1024.0),
            stats.consumer_copy_us[bucket] / 1000.0);
}

static void ggml_backend_sched_split_agg_log_consumer_chains(
        const ggml_backend_sched_consumer_chain_agg & stats) {
    GGML_LOG_INFO("    FPGA MUL_MAT output full consumer chains: unique_paths=%zu"
            " branch_points=%" PRIu64 " terminal(graph/no_consumer/depth_limit)=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
            stats.paths.size(),
            stats.branch_points,
            stats.terminal_graph_output,
            stats.terminal_no_consumer,
            stats.terminal_depth_limit);
    if (stats.paths.empty()) {
        return;
    }

    std::vector<std::pair<std::string, ggml_backend_sched_consumer_chain_item>> ordered;
    ordered.reserve(stats.paths.size());
    for (std::unordered_map<std::string, ggml_backend_sched_consumer_chain_item>::const_iterator it = stats.paths.begin();
            it != stats.paths.end(); ++it) {
        ordered.push_back(*it);
    }
    std::sort(ordered.begin(), ordered.end(),
            [](const std::pair<std::string, ggml_backend_sched_consumer_chain_item> & a,
               const std::pair<std::string, ggml_backend_sched_consumer_chain_item> & b) {
                if (a.second.bytes != b.second.bytes) {
                    return a.second.bytes > b.second.bytes;
                }
                if (a.second.outputs != b.second.outputs) {
                    return a.second.outputs > b.second.outputs;
                }
                return a.first < b.first;
            });

    const size_t limit = std::min<size_t>(ordered.size(), 64);
    for (size_t i = 0; i < limit; ++i) {
        GGML_LOG_INFO("      chain[%02zu] outputs=%" PRIu64 " bytes=%.3f MiB path=%s\n",
                i,
                ordered[i].second.outputs,
                ordered[i].second.bytes / (1024.0 * 1024.0),
                ordered[i].first.c_str());
    }
    if (ordered.size() > limit) {
        GGML_LOG_INFO("      ... %" PRIu64 " additional chain paths omitted\n",
                (uint64_t) (ordered.size() - limit));
    }
}

static void ggml_backend_sched_split_agg_log_tensor_flows(
        const char * title,
        const std::unordered_map<std::string, ggml_backend_sched_tensor_flow_item> & flows,
        size_t limit) {
    GGML_LOG_INFO("    %s: unique=%zu\n", title, flows.size());
    if (flows.empty()) {
        return;
    }

    std::vector<std::pair<std::string, ggml_backend_sched_tensor_flow_item>> ordered;
    ordered.reserve(flows.size());
    for (std::unordered_map<std::string, ggml_backend_sched_tensor_flow_item>::const_iterator it = flows.begin();
            it != flows.end(); ++it) {
        ordered.push_back(*it);
    }
    std::sort(ordered.begin(), ordered.end(),
            [](const std::pair<std::string, ggml_backend_sched_tensor_flow_item> & a,
               const std::pair<std::string, ggml_backend_sched_tensor_flow_item> & b) {
                if (a.second.bytes != b.second.bytes) {
                    return a.second.bytes > b.second.bytes;
                }
                if (a.second.outputs != b.second.outputs) {
                    return a.second.outputs > b.second.outputs;
                }
                return a.first < b.first;
            });

    const size_t n = std::min(limit, ordered.size());
    for (size_t i = 0; i < n; ++i) {
        GGML_LOG_INFO("      flow[%02zu] outputs=%" PRIu64 " bytes=%.3f MiB %s\n",
                i,
                ordered[i].second.outputs,
                ordered[i].second.bytes / (1024.0 * 1024.0),
                ordered[i].first.c_str());
    }
    if (ordered.size() > n) {
        GGML_LOG_INFO("      ... %" PRIu64 " additional tensor flows omitted\n",
                (uint64_t) (ordered.size() - n));
    }
}

static void ggml_backend_sched_split_agg_log() {
    if (!ggml_backend_sched_split_agg_enabled() &&
            !ggml_backend_sched_cpu_top_splits_enabled() &&
            !ggml_backend_sched_cpu_attn_mm_shapes_enabled()) {
        return;
    }

    static std::atomic<bool> logged { false };
    bool expected = false;
    if (!logged.compare_exchange_strong(expected, true)) {
        return;
    }

    std::lock_guard<std::mutex> lock(g_ggml_backend_sched_split_agg_mutex);
    GGML_LOG_INFO("ggml_backend_sched: split aggregate stats (same graph_compute phase, low-overhead)\n");
    for (int phase = 0; phase < (int) GGML_BACKEND_FPGA_PHASE_COUNT; ++phase) {
        const ggml_backend_sched_split_agg_phase & stats = g_ggml_backend_sched_split_agg[phase];
        if (stats.graphs == 0) {
            continue;
        }
        GGML_LOG_INFO("  [%s] graphs=%" PRIu64 " graph_wall=%.3f ms\n",
                ggml_backend_fpga_phase_name((ggml_backend_fpga_phase) phase),
                stats.graphs,
                stats.graph_wall_us / 1000.0);
        ggml_backend_sched_split_agg_log_backend("CPU", stats.cpu);
        ggml_backend_sched_split_agg_log_backend("FPGA", stats.fpga);
        ggml_backend_sched_split_agg_log_backend("other", stats.other);
        const uint64_t backend_copy_us =
            stats.cpu.copy_us + stats.fpga.copy_us + stats.other.copy_us;
        const uint64_t backend_compute_us =
            stats.cpu.compute_us + stats.fpga.compute_us + stats.other.compute_us;
        const uint64_t backend_accounted_us = backend_copy_us + backend_compute_us;
        const uint64_t graph_overhead_us =
            ggml_backend_sched_u64_sub_floor(stats.graph_wall_us, backend_accounted_us);
        const uint64_t split_loop_overhead_us =
            ggml_backend_sched_u64_sub_floor(stats.split_loop_wall_us, backend_accounted_us);
        const uint64_t split_known_overhead_us =
            stats.split_event_record_us + stats.split_stats_bookkeeping_us;
        const uint64_t split_loop_residual_us =
            ggml_backend_sched_u64_sub_floor(split_loop_overhead_us, split_known_overhead_us);
        const uint64_t graph_known_overhead_us =
            split_loop_overhead_us + stats.consumer_analyze_us;
        const uint64_t graph_residual_us =
            ggml_backend_sched_u64_sub_floor(graph_overhead_us, graph_known_overhead_us);
        GGML_LOG_INFO("    scheduler wall accounting: backend_copy=%.3f ms"
                " backend_compute=%.3f ms backend_accounted=%.3f ms"
                " graph_overhead=%.3f ms\n",
                backend_copy_us / 1000.0,
                backend_compute_us / 1000.0,
                backend_accounted_us / 1000.0,
                graph_overhead_us / 1000.0);
        GGML_LOG_INFO("    scheduler overhead detail: split_loop_wall=%.3f ms"
                " split_input_copy_loop=%.3f ms split_compute_loop=%.3f ms"
                " split_loop_overhead=%.3f ms event_record=%.3f ms"
                " stats_bookkeeping=%.3f ms split_loop_residual=%.3f ms"
                " consumer_analyze=%.3f ms graph_residual=%.3f ms\n",
                stats.split_loop_wall_us / 1000.0,
                stats.split_input_copy_loop_us / 1000.0,
                stats.split_compute_loop_us / 1000.0,
                split_loop_overhead_us / 1000.0,
                stats.split_event_record_us / 1000.0,
                stats.split_stats_bookkeeping_us / 1000.0,
                split_loop_residual_us / 1000.0,
                stats.consumer_analyze_us / 1000.0,
                graph_residual_us / 1000.0);
        GGML_LOG_INFO("    CPU classes: mul_mat=%.3f ms flash=%.3f ms glu=%.3f ms rms_norm=%.3f ms"
                " add=%.3f ms mul=%.3f ms rope=%.3f ms get_rows=%.3f ms"
                " set_rows=%.3f ms layout=%.3f ms other=%.3f ms\n",
                stats.cpu_class_mul_mat_us / 1000.0,
                stats.cpu_class_flash_us / 1000.0,
                stats.cpu_class_glu_us / 1000.0,
                stats.cpu_class_rms_norm_us / 1000.0,
                stats.cpu_class_add_us / 1000.0,
                stats.cpu_class_mul_us / 1000.0,
                stats.cpu_class_rope_us / 1000.0,
                stats.cpu_class_get_rows_us / 1000.0,
                stats.cpu_class_set_rows_us / 1000.0,
                stats.cpu_class_layout_us / 1000.0,
                stats.cpu_class_other_us / 1000.0);
        GGML_LOG_INFO("    CPU op counts: mul_mat=%" PRIu64 " flash=%" PRIu64 " glu=%" PRIu64
                " rms_norm=%" PRIu64 " add=%" PRIu64 " mul=%" PRIu64
                " rope=%" PRIu64 " get_rows=%" PRIu64 " set_rows=%" PRIu64
                " layout=%" PRIu64 " other=%" PRIu64 "\n",
                stats.cpu.mul_mat,
                stats.cpu_ops_flash,
                stats.cpu_ops_glu,
                stats.cpu_ops_rms_norm,
                stats.cpu_ops_add,
                stats.cpu_ops_mul,
                stats.cpu_ops_rope,
                stats.cpu_ops_get_rows,
                stats.cpu_ops_set_rows,
                stats.cpu_ops_layout,
                stats.cpu_ops_other);
        ggml_backend_sched_split_agg_log_cpu_attn_mm_shapes(stats.cpu_attn_mm_shapes);
        ggml_backend_sched_split_agg_log_top_cpu_splits(stats.cpu_top_splits);
        GGML_LOG_INFO("    FPGA classes: mul_mat=%.3f ms non_mul_mat=%.3f ms\n",
                stats.fpga_class_mul_mat_us / 1000.0,
                stats.fpga_class_non_mul_mat_us / 1000.0);
        GGML_LOG_INFO("    FPGA op counts: mul_mat=%" PRIu64 " non_mul_mat=%" PRIu64 "\n",
                stats.fpga_ops_mul_mat,
                stats.fpga_ops_non_mul_mat);
        const ggml_backend_sched_consumer_agg & consumers = stats.fpga_output_consumers;
        const uint64_t first_consumed =
            consumers.first_cpu + consumers.first_fpga + consumers.first_other;
        const double avg_first_distance = first_consumed > 0 ?
            (double) consumers.first_distance_sum / (double) first_consumed : 0.0;
        GGML_LOG_INFO("    FPGA MUL_MAT output consumers: outputs=%" PRIu64
                " bytes=%.3f MiB edges=%" PRIu64
                " cpu_edges=%" PRIu64 " fpga_edges=%" PRIu64 " other_edges=%" PRIu64
                " same_split=%" PRIu64 " cross_split=%" PRIu64 "\n",
                consumers.fpga_outputs,
                consumers.fpga_output_bytes / (1024.0 * 1024.0),
                consumers.consumer_edges,
                consumers.cpu_edges,
                consumers.fpga_edges,
                consumers.other_edges,
                consumers.same_split_edges,
                consumers.cross_split_edges);
        GGML_LOG_INFO("    FPGA MUL_MAT output first-consumer: cpu=%" PRIu64
                " fpga=%" PRIu64 " other=%" PRIu64
                " graph_output_or_cross_graph=%" PRIu64
                " no_consumer_non_output=%" PRIu64
                " multi_consumer_outputs=%" PRIu64
                " immediate=%" PRIu64 " delayed=%" PRIu64
                " avg_distance=%.2f max_distance=%" PRIu64 "\n",
                consumers.first_cpu,
                consumers.first_fpga,
                consumers.first_other,
                consumers.graph_output_or_cross_graph,
                consumers.no_consumer_non_output,
                consumers.multi_consumer_outputs,
                consumers.immediate_first,
                consumers.delayed_first,
                avg_first_distance,
                consumers.first_distance_max);
        GGML_LOG_INFO("    FPGA MUL_MAT output first-consumer bytes: cpu=%.3f MiB"
                " fpga=%.3f MiB other=%.3f MiB"
                " graph_output_or_cross_graph=%.3f MiB"
                " no_consumer_non_output=%.3f MiB"
                " multi_consumer_outputs=%.3f MiB"
                " immediate=%.3f MiB delayed=%.3f MiB\n",
                consumers.first_cpu_bytes / (1024.0 * 1024.0),
                consumers.first_fpga_bytes / (1024.0 * 1024.0),
                consumers.first_other_bytes / (1024.0 * 1024.0),
                consumers.first_graph_output_bytes / (1024.0 * 1024.0),
                consumers.no_consumer_non_output_bytes / (1024.0 * 1024.0),
                consumers.multi_consumer_output_bytes / (1024.0 * 1024.0),
                consumers.immediate_first_bytes / (1024.0 * 1024.0),
                consumers.delayed_first_bytes / (1024.0 * 1024.0));
        GGML_LOG_INFO("    FPGA MUL_MAT consumer op counts: mul_mat=%" PRIu64
                " flash=%" PRIu64 " glu=%" PRIu64 " rms_norm=%" PRIu64
                " add=%" PRIu64 " mul=%" PRIu64 " rope=%" PRIu64
                " get_rows=%" PRIu64 " set_rows=%" PRIu64
                " layout=%" PRIu64 " other=%" PRIu64 "\n",
                consumers.consumer_op_mul_mat,
                consumers.consumer_op_flash,
                consumers.consumer_op_glu,
                consumers.consumer_op_rms_norm,
                consumers.consumer_op_add,
                consumers.consumer_op_mul,
                consumers.consumer_op_rope,
                consumers.consumer_op_get_rows,
                consumers.consumer_op_set_rows,
                consumers.consumer_op_layout,
                consumers.consumer_op_other);
        GGML_LOG_INFO("    FPGA MUL_MAT consumer op bytes: mul_mat=%.3f MiB"
                " flash=%.3f MiB glu=%.3f MiB rms_norm=%.3f MiB"
                " add=%.3f MiB mul=%.3f MiB rope=%.3f MiB"
                " get_rows=%.3f MiB set_rows=%.3f MiB"
                " layout=%.3f MiB other=%.3f MiB\n",
                consumers.consumer_bytes_mul_mat / (1024.0 * 1024.0),
                consumers.consumer_bytes_flash / (1024.0 * 1024.0),
                consumers.consumer_bytes_glu / (1024.0 * 1024.0),
                consumers.consumer_bytes_rms_norm / (1024.0 * 1024.0),
                consumers.consumer_bytes_add / (1024.0 * 1024.0),
                consumers.consumer_bytes_mul / (1024.0 * 1024.0),
                consumers.consumer_bytes_rope / (1024.0 * 1024.0),
                consumers.consumer_bytes_get_rows / (1024.0 * 1024.0),
                consumers.consumer_bytes_set_rows / (1024.0 * 1024.0),
                consumers.consumer_bytes_layout / (1024.0 * 1024.0),
                consumers.consumer_bytes_other / (1024.0 * 1024.0));
        ggml_backend_sched_split_agg_log_tensor_flows(
                "FPGA MUL_MAT first-consumer tensor flows",
                consumers.first_consumer_tensor_flows,
                256);
        ggml_backend_sched_split_agg_log_tensor_flows(
                "FPGA MUL_MAT no-consumer tensors",
                consumers.no_consumer_tensors,
                32);
        ggml_backend_sched_split_agg_log_consumer_chains(stats.fpga_output_chains);
        const ggml_backend_sched_readback_edge_agg & rb = stats.fpga_to_cpu_readbacks;
        const uint64_t first_consumers = rb.immediate + rb.delayed;
        const double avg_first_index = first_consumers > 0 ?
            (double) rb.first_index_sum / (double) first_consumers : 0.0;
        GGML_LOG_INFO("    FPGA->CPU boundary readbacks: copies=%" PRIu64
                " bytes=%.3f MiB copy=%.3f ms"
                " producer_mul_mat=%" PRIu64 " / %.3f MiB / %.3f ms"
                " producer_other=%" PRIu64 " / %.3f MiB / %.3f ms\n",
                rb.copies,
                rb.bytes / (1024.0 * 1024.0),
                rb.copy_us / 1000.0,
                rb.producer_mul_mat,
                rb.producer_mul_mat_bytes / (1024.0 * 1024.0),
                rb.producer_mul_mat_copy_us / 1000.0,
                rb.producer_other,
                rb.producer_other_bytes / (1024.0 * 1024.0),
                rb.producer_other_copy_us / 1000.0);
        GGML_LOG_INFO("    FPGA->CPU boundary first consumer: immediate=%" PRIu64
                " delayed=%" PRIu64 " no_consumer=%" PRIu64
                " multi_consumer=%" PRIu64 " avg_first_index=%.2f max_first_index=%" PRIu64 "\n",
                rb.immediate,
                rb.delayed,
                rb.no_consumer,
                rb.multi_consumer,
                avg_first_index,
                rb.first_index_max);
        GGML_LOG_INFO("    FPGA->CPU boundary consumer buckets:\n");
        ggml_backend_sched_split_agg_log_readback_bucket("mul_mat",  rb, GGML_BACKEND_SCHED_OP_BUCKET_MUL_MAT);
        ggml_backend_sched_split_agg_log_readback_bucket("flash",    rb, GGML_BACKEND_SCHED_OP_BUCKET_FLASH);
        ggml_backend_sched_split_agg_log_readback_bucket("glu",      rb, GGML_BACKEND_SCHED_OP_BUCKET_GLU);
        ggml_backend_sched_split_agg_log_readback_bucket("rms_norm", rb, GGML_BACKEND_SCHED_OP_BUCKET_RMS_NORM);
        ggml_backend_sched_split_agg_log_readback_bucket("add",      rb, GGML_BACKEND_SCHED_OP_BUCKET_ADD);
        ggml_backend_sched_split_agg_log_readback_bucket("mul",      rb, GGML_BACKEND_SCHED_OP_BUCKET_MUL);
        ggml_backend_sched_split_agg_log_readback_bucket("rope",     rb, GGML_BACKEND_SCHED_OP_BUCKET_ROPE);
        ggml_backend_sched_split_agg_log_readback_bucket("get_rows", rb, GGML_BACKEND_SCHED_OP_BUCKET_GET_ROWS);
        ggml_backend_sched_split_agg_log_readback_bucket("set_rows", rb, GGML_BACKEND_SCHED_OP_BUCKET_SET_ROWS);
        ggml_backend_sched_split_agg_log_readback_bucket("layout",   rb, GGML_BACKEND_SCHED_OP_BUCKET_LAYOUT);
        ggml_backend_sched_split_agg_log_readback_bucket("other",    rb, GGML_BACKEND_SCHED_OP_BUCKET_OTHER);
    }
}

static enum ggml_status ggml_backend_sched_compute_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    struct ggml_backend_sched_split * splits = sched->splits;

    ggml_tensor * prev_ids_tensor = nullptr;
    std::vector<int32_t> ids;
    std::vector<ggml_bitset_t> used_ids;

    const enum ggml_backend_fpga_phase split_phase = ggml_backend_fpga_get_compute_phase();
    const bool split_agg_full_stats = ggml_backend_sched_split_agg_enabled();
    const bool split_cpu_top_stats = ggml_backend_sched_cpu_top_splits_phase_enabled(split_phase);
    const bool split_cpu_attn_mm_shapes_stats =
        ggml_backend_sched_cpu_attn_mm_shapes_phase_enabled(split_phase);
    const bool split_agg_stats =
        split_agg_full_stats || split_cpu_top_stats || split_cpu_attn_mm_shapes_stats;
    const int64_t graph_wall_start_us = split_agg_stats ? ggml_backend_sched_now_us() : 0;
    ggml_backend_sched_split_agg_phase split_graph_agg = {};
    const int64_t split_loop_start_us = split_agg_stats ? ggml_backend_sched_now_us() : 0;

    ggml_backend_fpga_ffn_streaming_clear_prepare_targets();
    ggml_backend_sched_mark_vit_activation_sidecache_targets_for_graph(sched);
    for (int split_id = 0; split_id < sched->n_splits; split_id++) {
        struct ggml_backend_sched_split * split = &splits[split_id];
        int split_backend_id = split->backend_id;
        ggml_backend_t split_backend = sched->backends[split_backend_id];

        // copy the input tensors to the split backend
        int64_t split_copy_us = 0;
        const int64_t split_copy_start_us = split_agg_stats ? ggml_backend_sched_now_us() : 0;
        for (int input_id = 0; input_id < split->n_inputs; input_id++) {
            ggml_backend_t input_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[input_id]);
            struct ggml_tensor * input = split->inputs[input_id];
            struct ggml_tensor * input_cpy = tensor_copy(input, split_backend_id, sched->cur_copy);
            bool track_fpga_to_cpu_readback = false;
            if (split_agg_full_stats) {
                const char * input_backend_name_c = input_backend != NULL ? ggml_backend_name(input_backend) : NULL;
                const char * split_backend_name_c = ggml_backend_name(split_backend);
                const std::string input_backend_name = input_backend_name_c != NULL ? input_backend_name_c : "unknown";
                const std::string split_backend_name = split_backend_name_c != NULL ? split_backend_name_c : "unknown";
                track_fpga_to_cpu_readback =
                    ggml_backend_sched_is_fpga_backend_name(input_backend_name) &&
                    ggml_backend_sched_is_cpu_backend_name(split_backend_name);
            }
            int64_t input_copy_start_us = 0;
            int64_t input_copy_us = 0;
            bool input_copy_tracked = false;

            if (input->flags & GGML_TENSOR_FLAG_INPUT) {
                // inputs from the user must be copied immediately to prevent the user overwriting the data before the copy is done
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }
                input_copy_start_us = track_fpga_to_cpu_readback ? ggml_backend_sched_now_us() : 0;
                ggml_backend_tensor_copy(input, input_cpy);
                if (track_fpga_to_cpu_readback) {
                    input_copy_us = ggml_backend_sched_now_us() - input_copy_start_us;
                    input_copy_tracked = true;
                }
            } else {
                // wait for the split backend to finish using the input before overwriting it
                if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                    ggml_backend_event_wait(split_backend, sched->events[split_backend_id][sched->cur_copy]);
                } else {
                    ggml_backend_synchronize(split_backend);
                }

                // when offloading MoE weights, we can reduce the amount of data copied by copying only the experts that are used
                ggml_tensor * node = split->graph.nodes[0];
                if (split->graph.n_nodes > 0 &&
                    ggml_backend_buffer_get_usage(input->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
                    ggml_backend_buffer_is_host(input->buffer) && (
                    (node->src[0] == input_cpy && node->op == GGML_OP_MUL_MAT_ID)
                    //|| (node->src[1] == input_cpy && node->op == GGML_OP_ADD_ID) /* GGML_OP_ADD_ID weights are small and not worth splitting */
                    )) {

                    const int64_t n_expert   = node->op == GGML_OP_MUL_MAT_ID ? input->ne[2] : input->ne[1];
                    const size_t expert_size = node->op == GGML_OP_MUL_MAT_ID ? input->nb[2] : input->nb[1];

                    ggml_backend_synchronize(input_backend);

                    // get the ids
                    ggml_tensor * ids_tensor = node->src[2];
                    ggml_backend_t ids_backend = split_backend;

                    // if the ids tensor is also an input of the split, it may not have been copied yet to the split backend
                    // in that case, we use the original ids tensor
                    for (int i = input_id + 1; i < split->n_inputs; i++) {
                        if (ids_tensor == tensor_copy(split->inputs[i], split_backend_id, sched->cur_copy)) {
                            ids_tensor = split->inputs[i];
                            ids_backend = ggml_backend_sched_get_tensor_backend(sched, split->inputs[i]);
                            break;
                        }
                    }

                    if (ids_tensor != prev_ids_tensor) {
                        ids.resize(ggml_nbytes(ids_tensor) / sizeof(int32_t));
                        ggml_backend_tensor_get_async(ids_backend, ids_tensor, ids.data(), 0, ggml_nbytes(ids_tensor));
                        ggml_backend_synchronize(ids_backend);

                        // find the used experts
                        used_ids.clear();
                        used_ids.resize(ggml_bitset_size(n_expert));
                        for (int64_t i1 = 0; i1 < ids_tensor->ne[1]; i1++) {
                            for (int64_t i0 = 0; i0 < ids_tensor->ne[0]; i0++) {
                                int32_t id = ids[i1 * ids_tensor->nb[1]/sizeof(int32_t) + i0 * ids_tensor->nb[0]/sizeof(int32_t)];
                                GGML_ASSERT(id >= 0 && id < n_expert);
                                ggml_bitset_set(used_ids.data(), id);
                            }
                        }

                        prev_ids_tensor = ids_tensor;
                    }

                    // group consecutive experts and copy them together
                    auto copy_experts = [&](int32_t first_id, int32_t last_id) {
                        const size_t expert_offset = first_id * expert_size;
                        const size_t expert_size_copy =  (last_id - first_id + 1) * expert_size;
                        const size_t padding = std::min<size_t>(expert_size, 512);
                        const size_t padding_end = last_id < n_expert - 1 ? padding : 0;

                        ggml_backend_tensor_set_async(split_backend,
                            input_cpy,
                            (const uint8_t *)input->data + expert_offset, expert_offset,
                            // copy a bit extra at the to ensure there are no NaNs in the padding of the last expert
                            // this is necessary for MMQ in the CUDA backend
                            expert_size_copy + padding_end);
                    };

                    int id = 0;
                    while (!ggml_bitset_get(used_ids.data(), id)) {
                        id++;
                    }
                    int32_t first_id = id;
                    int32_t last_id = first_id;

                    for (++id; id < n_expert; ++id) {
                        if (!ggml_bitset_get(used_ids.data(), id)) {
                            continue;
                        }

                        if (id == last_id + 1) {
                            last_id = id;
                            continue;
                        }

                        copy_experts(first_id, last_id);

                        first_id = id;
                        last_id = id;
                    }
                    copy_experts(first_id, last_id);
                } else {
                    // try async copy, but if not possible, we can still use a sync copy without synchronizing the dst backend, since we handle the synchronization here with multiple copies and events
                    // TODO: add public function to facilitate this, since applications do not have direct access to the backend interface
                    input_copy_start_us = track_fpga_to_cpu_readback ? ggml_backend_sched_now_us() : 0;
                    if (!split_backend->iface.cpy_tensor_async || !split_backend->iface.cpy_tensor_async(input_backend, split_backend, input, input_cpy)) {
                        ggml_backend_synchronize(input_backend);
                        if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                            ggml_backend_event_synchronize(sched->events[split_backend_id][sched->cur_copy]);
                        } else {
                            ggml_backend_synchronize(split_backend);
                        }
                        ggml_backend_tensor_copy(input, input_cpy);
                    }
                    if (track_fpga_to_cpu_readback) {
                        input_copy_us = ggml_backend_sched_now_us() - input_copy_start_us;
                        input_copy_tracked = true;
                    }
                }
            }
            if (input_copy_tracked) {
                ggml_backend_sched_readback_edge_agg_record(
                        split_graph_agg.fpga_to_cpu_readbacks,
                        input,
                        split,
                        input_cpy,
                        ggml_backend_sched_u64_add_i64(input_copy_us));
            }
        }
        if (split_agg_stats) {
            split_copy_us = ggml_backend_sched_now_us() - split_copy_start_us;
            split_graph_agg.split_input_copy_loop_us += ggml_backend_sched_u64_add_i64(split_copy_us);
        }

        int64_t split_compute_us = 0;
        const int64_t split_compute_start_us = split_agg_stats ? ggml_backend_sched_now_us() : 0;
        if (!sched->callback_eval) {
            enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &split->graph);
            if (ec != GGML_STATUS_SUCCESS) {
                return ec;
            }
        } else {
            // similar to ggml_backend_compare_graph_backend
            for (int j0 = 0; j0 < split->graph.n_nodes; j0++) {
                struct ggml_tensor * t = split->graph.nodes[j0];

                // check if the user needs data from this node
                bool need = sched->callback_eval(t, true, sched->callback_eval_user_data);

                int j1 = j0;

                // determine the range [j0, j1] of nodes that can be computed together
                while (!need && j1 < split->graph.n_nodes - 1) {
                    t = split->graph.nodes[++j1];
                    need = sched->callback_eval(t, true, sched->callback_eval_user_data);
                }

                struct ggml_cgraph gv = ggml_graph_view(&split->graph, j0, j1 + 1);

                enum ggml_status ec = ggml_backend_graph_compute_async(split_backend, &gv);
                if (ec != GGML_STATUS_SUCCESS) {
                    return ec;
                }

                // TODO: pass backend to the callback, then the user can decide if they want to synchronize
                ggml_backend_synchronize(split_backend);

                if (need && !sched->callback_eval(t, false, sched->callback_eval_user_data)) {
                    break;
                }

                j0 = j1;
            }
        }
        if (split_agg_stats) {
            split_compute_us = ggml_backend_sched_now_us() - split_compute_start_us;
            split_graph_agg.split_compute_loop_us += ggml_backend_sched_u64_add_i64(split_compute_us);
        }

        // record the event of this copy
        const int64_t split_event_record_start_us = split_agg_stats ? ggml_backend_sched_now_us() : 0;
        if (split->n_inputs > 0) {
            if (sched->events[split_backend_id][sched->cur_copy] != NULL) {
                ggml_backend_event_record(sched->events[split_backend_id][sched->cur_copy], split_backend);
            }
        }
        if (split_agg_stats) {
            split_graph_agg.split_event_record_us +=
                ggml_backend_sched_u64_add_i64(ggml_backend_sched_now_us() - split_event_record_start_us);
        }

        if (split_agg_stats) {
            const int64_t split_stats_bookkeeping_start_us = ggml_backend_sched_now_us();
            const char * backend_name_c = ggml_backend_name(split_backend);
            const std::string backend_name = backend_name_c != NULL ? backend_name_c : "unknown";
            const ggml_backend_sched_split_agg_counts split_counts =
                ggml_backend_sched_split_agg_count_ops(split, backend_name);
            ggml_backend_sched_split_agg_record_split(
                    split_graph_agg,
                    backend_name,
                    split->graph.n_nodes,
                    split_copy_us,
                    split_compute_us,
                    split_counts);
            ggml_backend_sched_split_agg_record_top_cpu_split(
                    split_graph_agg,
                    split,
                    backend_name,
                    split_copy_us,
                    split_compute_us,
                    split_counts);
            ggml_backend_sched_split_agg_record_cpu_attn_mm_shapes(
                    split_graph_agg,
                    split,
                    backend_name,
                    split_copy_us,
                    split_compute_us);
            split_graph_agg.split_stats_bookkeeping_us +=
                ggml_backend_sched_u64_add_i64(ggml_backend_sched_now_us() - split_stats_bookkeeping_start_us);
        }
    }

    if (split_agg_stats) {
        split_graph_agg.split_loop_wall_us =
            ggml_backend_sched_u64_add_i64(ggml_backend_sched_now_us() - split_loop_start_us);
        if (split_agg_full_stats) {
            const int64_t consumer_analyze_start_us = ggml_backend_sched_now_us();
            split_graph_agg.fpga_output_consumers =
                ggml_backend_sched_analyze_fpga_output_consumers(
                        sched,
                        &split_graph_agg.fpga_output_chains);
            split_graph_agg.consumer_analyze_us =
                ggml_backend_sched_u64_add_i64(ggml_backend_sched_now_us() - consumer_analyze_start_us);
        }
        ggml_backend_sched_split_agg_record_graph(
                split_phase,
                ggml_backend_sched_now_us() - graph_wall_start_us,
                split_graph_agg);
    }

    return GGML_STATUS_SUCCESS;
}

ggml_backend_sched_t ggml_backend_sched_new(
        ggml_backend_t * backends,
        ggml_backend_buffer_type_t * bufts,
        int n_backends,
        size_t graph_size,
        bool parallel,
        bool op_offload) {
    GGML_ASSERT(n_backends > 0);
    GGML_ASSERT(n_backends <= GGML_SCHED_MAX_BACKENDS);
    GGML_ASSERT(ggml_backend_dev_type(ggml_backend_get_device(backends[n_backends - 1])) == GGML_BACKEND_DEVICE_TYPE_CPU);

    struct ggml_backend_sched * sched = (ggml_backend_sched *) calloc(1, sizeof(struct ggml_backend_sched));

    const char * GGML_SCHED_DEBUG = getenv("GGML_SCHED_DEBUG");
    sched->debug = GGML_SCHED_DEBUG ? atoi(GGML_SCHED_DEBUG) : 0;
    sched->n_backends = n_backends;
    sched->n_copies = parallel ? GGML_SCHED_MAX_COPIES : 1;

    // initialize hash table
    // FIXME: needs to be size*2 to account for leafs (do it in graph_split instead)
    sched->hash_set    = ggml_hash_set_new(graph_size);
    sched->hv_tensor_backend_ids = (int *) malloc(sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
    sched->hv_tensor_copies      = (ggml_tensor **) malloc(sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));

    const size_t ggml_sched_max_splits = graph_size; // at most there is one split for each node in the graph
    const size_t nodes_size = graph_size + ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2;
    sched->node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->node_backend_ids[0]));
    sched->leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->leaf_backend_ids[0]));
    sched->prev_node_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_node_backend_ids[0]));
    sched->prev_leaf_backend_ids = (int *) calloc(nodes_size, sizeof(sched->prev_leaf_backend_ids[0]));

    sched->context_buffer_size = ggml_sched_max_splits*GGML_SCHED_MAX_SPLIT_INPUTS*2*sizeof(struct ggml_tensor) + ggml_graph_overhead_custom(graph_size, false);
    sched->context_buffer = (char *) malloc(sched->context_buffer_size);

    const int initial_splits_capacity = 16;
    sched->splits = (ggml_backend_sched_split *) calloc(initial_splits_capacity, sizeof(sched->splits[0]));
    sched->splits_capacity = initial_splits_capacity;

    for (int b = 0; b < n_backends; b++) {
        sched->backends[b] = backends[b];
        sched->bufts[b] = bufts ? bufts[b] : ggml_backend_get_default_buffer_type(backends[b]);
        GGML_ASSERT(ggml_backend_supports_buft(backends[b], sched->bufts[b]));

        if (sched->n_copies > 1) {
            for (int c = 0; c < sched->n_copies; c++) {
                sched->events[b][c] = ggml_backend_event_new(backends[b]->device);
            }
        }
    }

    sched->galloc = ggml_gallocr_new_n(sched->bufts, n_backends);
    sched->op_offload = op_offload;

    ggml_backend_sched_reset(sched);

    return sched;
}

void ggml_backend_sched_free(ggml_backend_sched_t sched) {
    if (sched == NULL) {
        return;
    }
    ggml_backend_sched_split_agg_log();
    for (int b = 0; b < sched->n_backends; b++) {
        for (int c = 0; c < sched->n_copies; c++) {
            ggml_backend_event_free(sched->events[b][c]);
        }
    }
    ggml_gallocr_free(sched->galloc);
    ggml_free(sched->ctx);
    ggml_hash_set_free(&sched->hash_set);
    free(sched->splits);
    free(sched->hv_tensor_backend_ids);
    free(sched->hv_tensor_copies);
    free(sched->node_backend_ids);
    free(sched->leaf_backend_ids);
    free(sched->prev_node_backend_ids);
    free(sched->prev_leaf_backend_ids);
    free(sched->context_buffer);
    free(sched->graph.nodes);
    free(sched->graph.leafs);
    free(sched);
}

void ggml_backend_sched_reset(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    // reset state for the next run
    if (!sched->is_reset) {
        ggml_hash_set_reset(&sched->hash_set);
        memset(sched->hv_tensor_backend_ids, -1, sched->hash_set.size * sizeof(sched->hv_tensor_backend_ids[0]));
        memset(sched->hv_tensor_copies,       0, sched->hash_set.size * sched->n_backends * sched->n_copies * sizeof(struct ggml_tensor *));
        sched->is_reset = true;
    }
    sched->is_alloc = false;
}

bool ggml_backend_sched_reserve(ggml_backend_sched_t sched, struct ggml_cgraph * measure_graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= measure_graph->n_nodes + measure_graph->n_leafs);

    ggml_backend_sched_reset(sched);

    ggml_backend_sched_synchronize(sched);

    ggml_backend_sched_split_graph(sched, measure_graph);

    if (!ggml_gallocr_reserve_n(sched->galloc, &sched->graph, sched->node_backend_ids, sched->leaf_backend_ids)) {
        return false;
    }

    ggml_backend_sched_reset(sched);

    return true;
}

bool ggml_backend_sched_alloc_graph(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    GGML_ASSERT((int)sched->hash_set.size >= graph->n_nodes + graph->n_leafs);
    GGML_ASSERT(!sched->is_alloc);

    sched->cur_copy = sched->next_copy;
    sched->next_copy = (sched->next_copy + 1) % sched->n_copies;

    ggml_backend_sched_split_graph(sched, graph);

    if (!ggml_backend_sched_alloc_splits(sched)) {
        return false;
    }

    sched->is_alloc = true;

    return true;
}

enum ggml_status ggml_backend_sched_graph_compute(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    enum ggml_status err = ggml_backend_sched_graph_compute_async(sched, graph);
    ggml_backend_sched_synchronize(sched);
    return err;
}

enum ggml_status ggml_backend_sched_graph_compute_async(ggml_backend_sched_t sched, struct ggml_cgraph * graph) {
    GGML_ASSERT(sched);
    if (!sched->is_reset && !sched->is_alloc) {
        ggml_backend_sched_reset(sched);
    }

    if (!sched->is_alloc) {
        if (!ggml_backend_sched_alloc_graph(sched, graph)) {
            return GGML_STATUS_ALLOC_FAILED;
        }
    }

    return ggml_backend_sched_compute_splits(sched);
}

void ggml_backend_sched_synchronize(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    for (int i = 0; i < sched->n_backends; i++) {
        ggml_backend_synchronize(sched->backends[i]);
    }
    if (!sched->is_alloc) {
        // if the graph is not already allocated, always use copy 0 after a synchronization
        // this ensures that during generation the same copy is used every time,
        // which avoids changes in the graph that could cause CUDA or other graphs to be disabled
        sched->next_copy = 0;
    }
}

void ggml_backend_sched_set_eval_callback(ggml_backend_sched_t sched, ggml_backend_sched_eval_callback callback, void * user_data) {
    GGML_ASSERT(sched);
    sched->callback_eval = callback;
    sched->callback_eval_user_data = user_data;
}

int ggml_backend_sched_get_n_splits(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_splits;
}

int ggml_backend_sched_get_n_copies(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_copies;
}

int ggml_backend_sched_get_n_backends(ggml_backend_sched_t sched) {
    GGML_ASSERT(sched);
    return sched->n_backends;
}

ggml_backend_t ggml_backend_sched_get_backend(ggml_backend_sched_t sched, int i) {
    GGML_ASSERT(sched);
    GGML_ASSERT(i >= 0 && i < sched->n_backends);
    return sched->backends[i];
}

ggml_backend_buffer_type_t ggml_backend_sched_get_buffer_type(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return sched->bufts[backend_index];
}

size_t ggml_backend_sched_get_buffer_size(ggml_backend_sched_t sched, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);

    return ggml_gallocr_get_buffer_size(sched->galloc, backend_index);
}

void ggml_backend_sched_set_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node, ggml_backend_t backend) {
    GGML_ASSERT(sched);
    int backend_index = ggml_backend_sched_backend_id(sched, backend);
    GGML_ASSERT(backend_index >= 0 && backend_index < sched->n_backends);
    tensor_backend_id(node) = backend_index;
    SET_CAUSE(node, "usr");
    sched->is_reset = false;
}

ggml_backend_t ggml_backend_sched_get_tensor_backend(ggml_backend_sched_t sched, struct ggml_tensor * node) {
    GGML_ASSERT(sched);
    int backend_index = tensor_backend_id(node);
    if (backend_index == -1) {
        return NULL;
    }
    return sched->backends[backend_index];
}

// utils

enum ggml_status ggml_backend_view_init(struct ggml_tensor * tensor) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->view_src != NULL);
    GGML_ASSERT(tensor->view_src->buffer != NULL);
    GGML_ASSERT(tensor->view_src->data != NULL);

    tensor->buffer = tensor->view_src->buffer;
    tensor->data = (char *)tensor->view_src->data + tensor->view_offs;
    return ggml_backend_buffer_init_tensor(tensor->buffer, tensor);
}

enum ggml_status ggml_backend_tensor_alloc(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, void * addr) {
    GGML_ASSERT(tensor);
    GGML_ASSERT(tensor->buffer == NULL);
    GGML_ASSERT(tensor->data == NULL);
    GGML_ASSERT(tensor->view_src == NULL);
    GGML_ASSERT(addr >= ggml_backend_buffer_get_base(buffer));
    GGML_ASSERT((char *)addr + ggml_backend_buffer_get_alloc_size(buffer, tensor) <=
                (char *)ggml_backend_buffer_get_base(buffer) + ggml_backend_buffer_get_size(buffer));

    tensor->buffer = buffer;
    tensor->data = addr;
    return ggml_backend_buffer_init_tensor(buffer, tensor);
}

static struct ggml_tensor * graph_copy_dup_tensor(struct ggml_hash_set hash_set, struct ggml_tensor ** node_copies,
    struct ggml_context * ctx_allocated, struct ggml_context * ctx_unallocated, struct ggml_tensor * src) {

    GGML_ASSERT(src != NULL);
    GGML_ASSERT(src->data && "graph must be allocated");

    size_t id = ggml_hash_insert(&hash_set, src);
    if (id == GGML_HASHSET_ALREADY_EXISTS) {
        return node_copies[ggml_hash_find(&hash_set, src)];
    }

    struct ggml_tensor * dst = ggml_dup_tensor_layout(src->data && !src->view_src ? ctx_allocated : ctx_unallocated, src);
    if (src->view_src != NULL) {
        dst->view_src = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, src->view_src);
        dst->view_offs = src->view_offs;
    }
    dst->op = src->op;
    memcpy(dst->op_params, src->op_params, sizeof(dst->op_params));
    ggml_set_name(dst, src->name);

    // copy src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        dst->src[i] = graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, s);
    }

    node_copies[id] = dst;
    return dst;
}

static void graph_copy_init_tensor(struct ggml_hash_set * hash_set, struct ggml_tensor ** node_copies, bool * node_init, struct ggml_tensor * src) {
    size_t id = ggml_hash_find(hash_set, src);
    if (node_init[id]) {
        return;
    }
    node_init[id] = true;

    struct ggml_tensor * dst = node_copies[id];
    if (dst->view_src != NULL) {
        graph_copy_init_tensor(hash_set, node_copies, node_init, src->view_src);
        enum ggml_status status = ggml_backend_view_init(dst);
        GGML_ASSERT(status == GGML_STATUS_SUCCESS);
    }
    else {
        ggml_backend_tensor_copy(src, dst);
    }

    // init src
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        struct ggml_tensor * s = src->src[i];
        if (s == NULL) {
            continue;
        }
        graph_copy_init_tensor(hash_set, node_copies, node_init, s);
    }
}

struct ggml_backend_graph_copy ggml_backend_graph_copy(ggml_backend_t backend, struct ggml_cgraph * graph) {
    GGML_ASSERT(graph);
    struct ggml_hash_set hash_set = ggml_hash_set_new(graph->visited_hash_set.size);
    struct ggml_tensor ** node_copies = (ggml_tensor **) calloc(hash_set.size, sizeof(node_copies[0])); // NOLINT
    bool * node_init = (bool *) calloc(hash_set.size, sizeof(node_init[0]));

    struct ggml_init_params params = {
        /* .mem_size   = */ ggml_tensor_overhead()*hash_set.size + ggml_graph_overhead_custom(graph->size, false),
        /* .mem_buffer = */ NULL,
        /* .no_alloc   = */ true
    };

    struct ggml_context * ctx_allocated = ggml_init(params);
    struct ggml_context * ctx_unallocated = ggml_init(params);

    if (ctx_allocated == NULL || ctx_unallocated == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate context for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    // dup nodes
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_dup_tensor(hash_set, node_copies, ctx_allocated, ctx_unallocated, node);
    }

    // allocate nodes
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx_allocated, backend);
    if (buffer == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer for graph copy\n", __func__);
        ggml_hash_set_free(&hash_set);
        free(node_copies);
        free(node_init);
        ggml_free(ctx_allocated);
        ggml_free(ctx_unallocated);
        return {
            /* .buffer           = */ NULL,
            /* .ctx_allocated    = */ NULL,
            /* .ctx_unallocated  = */ NULL,
            /* .graph            = */ NULL,
        };
    }

    //printf("copy buffer size: %zu MB\n", ggml_backend_buffer_get_size(buffer) / 1024 / 1024);

    // copy data and init views
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        graph_copy_init_tensor(&hash_set, node_copies, node_init, node);
    }

    // build graph copy
    struct ggml_cgraph * graph_copy = ggml_new_graph_custom(ctx_allocated, graph->size, false);
    for (int i = 0; i < graph->n_nodes; i++) {
        struct ggml_tensor * node = graph->nodes[i];
        struct ggml_tensor * node_copy = node_copies[ggml_hash_find(&hash_set, node)];
        graph_copy->nodes[i] = node_copy;
    }
    graph_copy->n_nodes = graph->n_nodes;

    ggml_hash_set_free(&hash_set);
    free(node_copies);
    free(node_init);

    return {
        /* .buffer           = */ buffer,
        /* .ctx_allocated    = */ ctx_allocated,
        /* .ctx_unallocated  = */ ctx_unallocated,
        /* .graph            = */ graph_copy,
    };
}

void ggml_backend_graph_copy_free(struct ggml_backend_graph_copy copy) {
    ggml_backend_buffer_free(copy.buffer);
    ggml_free(copy.ctx_allocated);
    ggml_free(copy.ctx_unallocated);
}

bool ggml_backend_compare_graph_backend(ggml_backend_t backend1, ggml_backend_t backend2, struct ggml_cgraph * graph, ggml_backend_eval_callback callback, void * user_data, struct ggml_tensor * test_node) {
    struct ggml_backend_graph_copy copy = ggml_backend_graph_copy(backend2, graph);
    if (copy.buffer == NULL) {
        return false;
    }

    struct ggml_cgraph * g1 = graph;
    struct ggml_cgraph * g2 = copy.graph;

    assert(g1->n_nodes == g2->n_nodes);

    if (test_node != nullptr) {
        // Compute the whole graph and only test the output for a specific tensor
        ggml_backend_graph_compute(backend1, g1);
        ggml_backend_graph_compute(backend2, g2);

        int test_node_idx = -1;
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            if (t1 == test_node) {
                test_node_idx = i;
                break;
            }
        }
        GGML_ASSERT(test_node_idx != -1);

        callback(test_node_idx, g1->nodes[test_node_idx], g2->nodes[test_node_idx], user_data);
    } else {
        for (int i = 0; i < g1->n_nodes; i++) {
            struct ggml_tensor * t1 = g1->nodes[i];
            struct ggml_tensor * t2 = g2->nodes[i];

            assert(t1->op == t2->op && ggml_are_same_layout(t1, t2));

            struct ggml_cgraph g1v = ggml_graph_view(g1, i, i + 1);
            struct ggml_cgraph g2v = ggml_graph_view(g2, i, i + 1);

            ggml_backend_graph_compute(backend1, &g1v);
            ggml_backend_graph_compute(backend2, &g2v);

            if (ggml_is_view_op(t1->op)) {
                continue;
            }

            // compare results, calculate rms etc
            if (!callback(i, t1, t2, user_data)) {
                break;
            }
        }
    }
    ggml_backend_graph_copy_free(copy);

    return true;
}

// CPU backend - buffer

static void * ggml_backend_cpu_buffer_get_base(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    uintptr_t data = (uintptr_t)buffer->context;

    // align the buffer
    if (data % TENSOR_ALIGNMENT != 0) {
        data = GGML_PAD(data, TENSOR_ALIGNMENT);
    }

    return (void *)data;
}

static void ggml_backend_cpu_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    GGML_ASSERT(buffer);
    ggml_aligned_free(buffer->context, buffer->size);
}

static void ggml_backend_cpu_buffer_memset_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memset((char *)tensor->data + offset, value, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_set_tensor(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy((char *)tensor->data + offset, data, size);

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_get_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    GGML_ASSERT(tensor);
    memcpy(data, (const char *)tensor->data + offset, size);

    GGML_UNUSED(buffer);
}

static bool ggml_backend_cpu_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    GGML_ASSERT(src);
    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cpu_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    GGML_ASSERT(buffer);
    memset(buffer->context, value, buffer->size);
}

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_i = {
    /* .free_buffer     = */ ggml_backend_cpu_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

static const struct ggml_backend_buffer_i ggml_backend_cpu_buffer_from_ptr_i = {
    /* .free_buffer     = */ NULL, // ptr is not owned by the buffer, so it does not need to be freed
    /* .get_base        = */ ggml_backend_cpu_buffer_get_base,
    /* .init_tensor     = */ NULL, // no initialization required
    /* .memset_tensor   = */ ggml_backend_cpu_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cpu_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cpu_buffer_get_tensor,
    /* .cpy_tensor      = */ ggml_backend_cpu_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cpu_buffer_clear,
    /* .reset           = */ NULL,
};

// CPU backend buffer type

// this buffer type is defined here to make it available to all backends

static const char * ggml_backend_cpu_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cpu_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * data = ggml_aligned_malloc(size);

    if (data == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        return NULL;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_cpu_buffer_i, data, size);
}

static size_t ggml_backend_cpu_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return TENSOR_ALIGNMENT;

    GGML_UNUSED(buft);
}

static bool ggml_backend_cpu_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return true;

    GGML_UNUSED(buft);
}

ggml_backend_buffer_type_t ggml_backend_cpu_buffer_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

static const char * ggml_backend_cpu_buffer_from_ptr_type_get_name(ggml_backend_buffer_type_t buft) {
    return "CPU_Mapped";

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_cpu_buffer_from_ptr_type(void) {
    static struct ggml_backend_buffer_type ggml_backend_cpu_buffer_type = {
        /* .iface   = */ {
            /* .get_name         = */ ggml_backend_cpu_buffer_from_ptr_type_get_name,
            /* .alloc_buffer     = */ ggml_backend_cpu_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type_get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ NULL, // defaults to ggml_nbytes
            /* .is_host          = */ ggml_backend_cpu_buffer_type_is_host,
        },
        /* .device  = */ NULL, // FIXME ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ NULL,
    };

    return &ggml_backend_cpu_buffer_type;
}

ggml_backend_buffer_t ggml_backend_cpu_buffer_from_ptr(void * ptr, size_t size) {
    GGML_ASSERT((uintptr_t)ptr % TENSOR_ALIGNMENT == 0 && "buffer pointer must be aligned");
    return ggml_backend_buffer_init(ggml_backend_cpu_buffer_from_ptr_type(), ggml_backend_cpu_buffer_from_ptr_i, ptr, size);
}
