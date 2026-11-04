#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Apply Token Merging to a set of visual token embeddings.
// tokens layout: row-major float[n_tokens][n_embd]
// tokens_out must be pre-allocated to at least n_tokens * n_embd floats.
// ratio is the target fraction of tokens to remove, clamped to [0, 1].
// Returns the number of output tokens (<= n_tokens).
// If ratio <= 0 or n_tokens <= 2, copies input unchanged.
int tome_merge(
    const float * tokens_in,
    int           n_tokens,
    int           n_embd,
    float       * tokens_out,
    float         ratio);

// Returns 0 on success, non-zero on failure.
int tome_self_test(void);

#ifdef __cplusplus
}
#endif
