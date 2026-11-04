// Tensor word layouts of the Gen3 Node (g2lib.Layout, g3lib.vt_pack / f32_words / rms_words; docs/gen3/node_v3.md
// s2) over g3::Word (256 bits). MXFP8 tensors travel as codes [rows][cols] and scales [rows][cols / 32].
#pragma once

#include <cstdint>
#include <vector>

#include "g3emu.h"

namespace g3rt {

using g3::Word;

// P8 (activations: a band is one 8-row panel) or P32 (weights, K rows: a band is four panels); `heads` stacks
// `rows` per head, each head padded to whole bands.
struct Layout {
    int rows = 0, cols = 0;
    bool p32 = false;
    int heads = 1;

    Layout() = default;
    Layout(int r, int c, bool p32_ = false, int h = 1) : rows(r), cols(c), p32(p32_), heads(h) {}
    int pb() const { return p32 ? 4 : 1; }
    int groups() const { return (cols + 31) / 32; }
    int head_panels() const { const int p = (rows + 7) / 8; return (p + pb() - 1) / pb() * pb(); }
    int panels() const { return head_panels() * heads; }
    int blocks() const { return panels() * groups(); }
    int words() const { return 33 * ((blocks() + 3) / 4); }
    int block(int panel, int group) const {
        const int band = panel / pb(), q = panel % pb();
        return band * groups() * pb() + group * pb() + q;
    }
    // codes [heads * rows][cols] (head-major), scales [heads * rows][groups] -> words()
    std::vector<Word> pack(const uint8_t* codes, const uint8_t* scales) const;
    void pack_into(const uint8_t* codes, const uint8_t* scales, Word* out) const;
    // words -> codes / scales (padding rows / columns dropped)
    void unpack(const Word* w, uint8_t* codes, uint8_t* scales) const;
    // one row (head-major index, as in pack's codes) of an image: its codes [cols] and scales [groups] placed as
    // pack_into places them, the rest of w kept
    void set_row(Word* w, int row, const uint8_t* codes, const uint8_t* scales) const;
};

inline uint32_t round8(uint32_t w) { return (w + 7) / 8 * 8; }

// FP32 rows: row r, columns 8j..8j+7 in word r * cols / 8 + j (cols % 8 == 0); rows padded to 8 by the Node
inline uint32_t f32_extent(int rows, int cols) { return uint32_t((rows + 7) / 8 * 8) * uint32_t(cols / 8); }
std::vector<Word> f32_pack(const uint32_t* bits, int rows, int cols);
void f32_unpack(const Word* w, int rows, int cols, uint32_t* bits);
// 1/rms words: row 8w + i at bits 32i of word w
std::vector<Word> rms_pack(const uint32_t* r, int rows);
inline uint32_t rms_extent(int rows) { return 4u * uint32_t((rows + 31) / 32); }
// V^T image of one KV head: cell (key group j, dim chunk c) at quad j * G + c, each a P32 band of 32 dims x 32 keys
// with one scale per dim. codes_t [dh][cap], scales_t [dh][cap / 32].
std::vector<Word> vt_pack(const uint8_t* codes_t, const uint8_t* scales_t, int dh, int cap);
void vt_unpack(const Word* w, int dh, int cap, uint8_t* codes_t, uint8_t* scales_t);
inline uint32_t vt_words(int dh, int cap) { return 33u * uint32_t(dh / 32) * uint32_t(cap / 32); }

// FP32 [rows][cols] (cols % 32 == 0) -> MXFP8 codes and scales (g2lib.quantize_matrix)
void quantize(const uint32_t* bits, int rows, int cols, std::vector<uint8_t>& codes, std::vector<uint8_t>& scales);
// codes / scales -> FP32 bits
void dequantize(const uint8_t* codes, const uint8_t* scales, int rows, int cols, uint32_t* bits);

}  // namespace g3rt
