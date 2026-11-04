#include "tome.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

static float clamp_ratio(float ratio) {
    if (!std::isfinite(ratio)) {
        return 0.0f;
    }
    if (ratio < 0.0f) {
        return 0.0f;
    }
    if (ratio > 1.0f) {
        return 1.0f;
    }
    return ratio;
}

static void copy_tokens(float * dst, const float * src, int n_tokens, int n_embd) {
    const size_t n_bytes = (size_t) n_tokens * (size_t) n_embd * sizeof(float);
    if (n_bytes > 0) {
        std::memmove(dst, src, n_bytes);
    }
}

static void l2_normalize_rows(const float * src, int n_rows, int n_embd, std::vector<float> & dst) {
    dst.resize((size_t) n_rows * (size_t) n_embd);
    for (int i = 0; i < n_rows; ++i) {
        const float * src_row = src + (size_t) i * (size_t) n_embd;
        float * dst_row = dst.data() + (size_t) i * (size_t) n_embd;

        float norm_sq = 0.0f;
        for (int d = 0; d < n_embd; ++d) {
            norm_sq += src_row[d] * src_row[d];
        }

        const float inv_norm = norm_sq > 0.0f ? 1.0f / std::sqrt(norm_sq) : 0.0f;
        for (int d = 0; d < n_embd; ++d) {
            dst_row[d] = src_row[d] * inv_norm;
        }
    }
}

static float dot_product(const float * a, const float * b, int n_embd) {
    float sum = 0.0f;
    for (int d = 0; d < n_embd; ++d) {
        sum += a[d] * b[d];
    }
    return sum;
}

struct merge_pair {
    int ia;
    int ib;
};

} // namespace

int tome_merge(
    const float * tokens_in,
    int           n_tokens,
    int           n_embd,
    float       * tokens_out,
    float         ratio) {
    if (tokens_in == nullptr || tokens_out == nullptr) {
        return 0;
    }
    if (n_embd <= 0) {
        return 0;
    }
    if (n_tokens <= 0) {
        return 0;
    }

    ratio = clamp_ratio(ratio);
    if (ratio <= 0.0f || n_tokens <= 2) {
        copy_tokens(tokens_out, tokens_in, n_tokens, n_embd);
        return n_tokens;
    }

    const int n_a = (n_tokens + 1) / 2;
    const int n_b = n_tokens / 2;
    if (n_a == 0 || n_b == 0) {
        copy_tokens(tokens_out, tokens_in, n_tokens, n_embd);
        return n_tokens;
    }

    int r = (int) (n_tokens * ratio);
    r = std::min(r, std::min(n_a, n_b));
    if (r <= 0) {
        copy_tokens(tokens_out, tokens_in, n_tokens, n_embd);
        return n_tokens;
    }

    std::vector<float> tokens_tmp;
    if (tokens_in == tokens_out) {
        tokens_tmp.assign(tokens_in, tokens_in + (size_t) n_tokens * (size_t) n_embd);
        tokens_in = tokens_tmp.data();
    }

    std::vector<float> tokens_a((size_t) n_a * (size_t) n_embd);
    std::vector<float> tokens_b((size_t) n_b * (size_t) n_embd);
    for (int ia = 0; ia < n_a; ++ia) {
        std::memcpy(
            tokens_a.data() + (size_t) ia * (size_t) n_embd,
            tokens_in + (size_t) (ia * 2) * (size_t) n_embd,
            (size_t) n_embd * sizeof(float));
    }
    for (int ib = 0; ib < n_b; ++ib) {
        std::memcpy(
            tokens_b.data() + (size_t) ib * (size_t) n_embd,
            tokens_in + (size_t) (ib * 2 + 1) * (size_t) n_embd,
            (size_t) n_embd * sizeof(float));
    }

    std::vector<float> a_norm;
    std::vector<float> b_norm;
    l2_normalize_rows(tokens_a.data(), n_a, n_embd, a_norm);
    l2_normalize_rows(tokens_b.data(), n_b, n_embd, b_norm);

    std::vector<int> best_b_for_a(n_a, 0);
    std::vector<float> best_sim(n_a, -INFINITY);
    for (int ia = 0; ia < n_a; ++ia) {
        const float * a_row = a_norm.data() + (size_t) ia * (size_t) n_embd;
        float max_sim = -INFINITY;
        int max_ib = 0;
        for (int ib = 0; ib < n_b; ++ib) {
            const float * b_row = b_norm.data() + (size_t) ib * (size_t) n_embd;
            const float sim = dot_product(a_row, b_row, n_embd);
            if (sim > max_sim) {
                max_sim = sim;
                max_ib = ib;
            }
        }
        best_b_for_a[ia] = max_ib;
        best_sim[ia] = max_sim;
    }

    std::vector<int> sorted_a(n_a);
    for (int ia = 0; ia < n_a; ++ia) {
        sorted_a[ia] = ia;
    }
    std::sort(sorted_a.begin(), sorted_a.end(), [&](int lhs, int rhs) {
        if (best_sim[lhs] == best_sim[rhs]) {
            return lhs < rhs;
        }
        return best_sim[lhs] > best_sim[rhs];
    });

    std::vector<char> merged_a(n_a, 0);
    std::vector<char> merged_b(n_b, 0);
    std::vector<merge_pair> pairs;
    pairs.reserve((size_t) r);

    for (int idx = 0; idx < n_a && (int) pairs.size() < r; ++idx) {
        const int ia = sorted_a[idx];
        const int ib = best_b_for_a[ia];
        if (merged_b[ib]) {
            continue;
        }
        merged_a[ia] = 1;
        merged_b[ib] = 1;
        pairs.push_back({ ia, ib });
    }

    int out_idx = 0;

    for (int ia = 0; ia < n_a; ++ia) {
        if (merged_a[ia]) {
            continue;
        }
        std::memcpy(
            tokens_out + (size_t) out_idx * (size_t) n_embd,
            tokens_in + (size_t) (ia * 2) * (size_t) n_embd,
            (size_t) n_embd * sizeof(float));
        ++out_idx;
    }

    for (int ib = 0; ib < n_b; ++ib) {
        if (merged_b[ib]) {
            continue;
        }
        std::memcpy(
            tokens_out + (size_t) out_idx * (size_t) n_embd,
            tokens_in + (size_t) (ib * 2 + 1) * (size_t) n_embd,
            (size_t) n_embd * sizeof(float));
        ++out_idx;
    }

    for (const merge_pair & pair : pairs) {
        const float * a_ptr = tokens_in + (size_t) (pair.ia * 2) * (size_t) n_embd;
        const float * b_ptr = tokens_in + (size_t) (pair.ib * 2 + 1) * (size_t) n_embd;
        float * out_ptr = tokens_out + (size_t) out_idx * (size_t) n_embd;
        for (int d = 0; d < n_embd; ++d) {
            out_ptr[d] = 0.5f * (a_ptr[d] + b_ptr[d]);
        }
        ++out_idx;
    }

    return out_idx;
}

int tome_self_test(void) {
    const int n_tokens = 8;
    const int n_embd = 4;
    const float tokens[n_tokens * n_embd] = {
        1.00f, 0.00f, 0.00f, 0.00f,
        0.99f, 0.01f, 0.00f, 0.00f,
        0.00f, 1.00f, 0.00f, 0.00f,
        0.00f, 0.98f, 0.02f, 0.00f,
        0.00f, 0.00f, 1.00f, 0.00f,
        0.00f, 0.00f, 0.00f, 1.00f,
        0.50f, 0.50f, 0.00f, 0.00f,
        0.00f, 0.50f, 0.50f, 0.00f,
    };

    std::vector<float> merged((size_t) n_tokens * (size_t) n_embd, 0.0f);
    const int n_out = tome_merge(tokens, n_tokens, n_embd, merged.data(), 0.25f);
    if (n_out != 6) {
        std::fprintf(stderr, "ToMe: self-test failed, expected 6 tokens, got %d\n", n_out);
        return 1;
    }

    const float expected_01[n_embd] = {
        0.995f, 0.005f, 0.000f, 0.000f,
    };
    const float expected_23[n_embd] = {
        0.000f, 0.990f, 0.010f, 0.000f,
    };

    bool found_01 = false;
    bool found_23 = false;
    for (int i = 0; i < n_out; ++i) {
        const float * row = merged.data() + (size_t) i * (size_t) n_embd;
        bool match_01 = true;
        bool match_23 = true;
        for (int d = 0; d < n_embd; ++d) {
            if (std::fabs(row[d] - expected_01[d]) > 1e-6f) {
                match_01 = false;
            }
            if (std::fabs(row[d] - expected_23[d]) > 1e-6f) {
                match_23 = false;
            }
        }
        found_01 = found_01 || match_01;
        found_23 = found_23 || match_23;
    }

    if (!found_01 || !found_23) {
        std::fprintf(stderr, "ToMe: self-test failed, expected merged averages not found\n");
        return 2;
    }

    std::vector<float> merged_one((size_t) n_tokens * (size_t) n_embd, 0.0f);
    const int n_out_one = tome_merge(tokens, n_tokens, n_embd, merged_one.data(), 0.125f);
    if (n_out_one != 7) {
        std::fprintf(stderr, "ToMe: self-test failed, ratio 0.125 expected 7 tokens, got %d\n", n_out_one);
        return 3;
    }

    std::vector<float> copied((size_t) n_tokens * (size_t) n_embd, 0.0f);
    const int n_copy = tome_merge(tokens, n_tokens, n_embd, copied.data(), 0.0f);
    if (n_copy != n_tokens) {
        std::fprintf(stderr, "ToMe: self-test failed, ratio 0 returned %d tokens\n", n_copy);
        return 4;
    }
    if (std::memcmp(tokens, copied.data(), sizeof(tokens)) != 0) {
        std::fprintf(stderr, "ToMe: self-test failed, ratio 0 path changed output\n");
        return 5;
    }

    std::vector<float> merged_max((size_t) n_tokens * (size_t) n_embd, 0.0f);
    const int n_out_max = tome_merge(tokens, n_tokens, n_embd, merged_max.data(), 1.0f);
    if (n_out_max != 4) {
        std::fprintf(stderr, "ToMe: self-test failed, ratio 1 expected 4 tokens, got %d\n", n_out_max);
        return 6;
    }

    return 0;
}
