// Word layouts of the Gen3 Node (docs/gen2/memory_and_layout.md s3, docs/gen3/node_v3.md s2, g2lib.Layout,
// g3lib.vt_pack / f32_words). A word is 256 bits = 32 bytes, byte 0 = bits 7:0.
//
//   quad   33 words: one scale word, then 4 blocks of 8 words
//   block  8 rows (lanes) x 32 K values; K value k of lane l is byte ((k % 4) / 2) * 16 + 2 l + k % 2 of word k / 4;
//          the 8 row scales are bytes 8 (n % 4) + l of the quad's scale word
//   P8     block(panel, group) = panel * groups + group                   (activations, one panel per band)
//   P32    block(panel, group) = (panel / 4) * groups * 4 + group * 4 + panel % 4   (weights, K rows: 32-row bands)
//   V^T    cell (key group j, dim chunk c) = quad j * G + c, a P32 band of 32 dims x 32 keys (one scale per dim)
//   F32    FP32 rows: value (r, c) at bits 32 (c % 8) of word r * N / 8 + c / 8
#pragma once

#include <cstdint>

namespace g3 {
namespace layout {

inline uint32_t block_word(uint32_t n) { return 33 * (n >> 2) + 1 + 8 * (n & 3); }
inline uint32_t scale_word(uint32_t n) { return 33 * (n >> 2); }
inline uint32_t scale_byte(uint32_t n, uint32_t lane) { return 8 * (n & 3) + lane; }
// word offset inside a block and byte inside the word of K value k (0..31) of lane l (0..7)
inline uint32_t code_word(uint32_t k) { return k >> 2; }
inline uint32_t code_byte(uint32_t k, uint32_t lane) { return ((k & 3) >> 1) * 16 + 2 * lane + (k & 1); }
inline uint32_t p8_block(uint32_t panel, uint32_t group, uint32_t groups) { return panel * groups + group; }
inline uint32_t p32_block(uint32_t panel, uint32_t group, uint32_t groups) {
    return (panel >> 2) * groups * 4 + group * 4 + (panel & 3);
}

}  // namespace layout
}  // namespace g3
