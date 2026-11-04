#include "gen3rt/layout.h"

#include <cstring>

#include "g3emu_kernels.h"
#include "g3emu_layout.h"

namespace g3rt {
namespace L = g3::layout;
namespace {

#if defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
constexpr bool LITTLE_ENDIAN_HOST = true;
#else
constexpr bool LITTLE_ENDIAN_HOST = false;
#endif

// The 32 codes of one lane of a block (base: its 8 code words). Code k is in word k / 4 at byte code_byte(k, lane):
// codes 4w, 4w + 1 at bytes 2 lane, 2 lane + 1 of word w, codes 4w + 2, 4w + 3 at bytes 16 + 2 lane, 17 + 2 lane.
void put_lane(Word* base, int lane, const uint8_t* c) {
    for (int w = 0; w < 8; ++w) {
        std::memcpy(&base[w].b[2 * lane], c + 4 * w, 2);
        std::memcpy(&base[w].b[16 + 2 * lane], c + 4 * w + 2, 2);
    }
}

void get_lane(const Word* base, int lane, uint8_t* c) {
    for (int w = 0; w < 8; ++w) {
        std::memcpy(c + 4 * w, &base[w].b[2 * lane], 2);
        std::memcpy(c + 4 * w + 2, &base[w].b[16 + 2 * lane], 2);
    }
}

}  // namespace

void Layout::pack_into(const uint8_t* codes, const uint8_t* scales, Word* out) const {
    const int W = words(), G = groups();
    std::memset(out, 0, sizeof(Word) * size_t(W));
    for (int h = 0; h < heads; ++h)
        for (int r = 0; r < rows; ++r) {
            const int src = h * rows + r;
            const int srow = h * head_panels() * 8 + r;
            const int panel = srow / 8, lane = srow % 8;
            for (int g = 0; g < G; ++g) {
                const uint32_t n = uint32_t(block(panel, g));
                Word* base = out + L::block_word(n);
                if (32 * g + 32 <= cols) put_lane(base, lane, &codes[size_t(src) * cols + size_t(32 * g)]);
                else
                    for (int k = 0; k < 32; ++k) {
                        const int col = 32 * g + k;
                        const uint8_t v = col < cols ? codes[size_t(src) * cols + col] : 0;
                        base[L::code_word(k)].b[L::code_byte(k, lane)] = v;
                    }
                out[L::scale_word(n)].b[L::scale_byte(n, lane)] = scales[size_t(src) * G + g];
            }
        }
}

std::vector<Word> Layout::pack(const uint8_t* codes, const uint8_t* scales) const {
    std::vector<Word> out(static_cast<size_t>(words()));
    pack_into(codes, scales, out.data());
    return out;
}

void Layout::set_row(Word* w, int row, const uint8_t* codes, const uint8_t* scales) const {
    const int srow = row / rows * head_panels() * 8 + row % rows;
    const int panel = srow / 8, lane = srow % 8;
    for (int g = 0; g < groups(); ++g) {
        const uint32_t n = uint32_t(block(panel, g));
        Word* base = w + L::block_word(n);
        if (32 * g + 32 <= cols) put_lane(base, lane, &codes[32 * g]);
        else
            for (int k = 0; k < 32; ++k) {
                const int col = 32 * g + k;
                base[L::code_word(k)].b[L::code_byte(k, lane)] = col < cols ? codes[col] : 0;
            }
        w[L::scale_word(n)].b[L::scale_byte(n, lane)] = scales[g];
    }
}

void Layout::unpack(const Word* w, uint8_t* codes, uint8_t* scales) const {
    const int G = groups();
    for (int h = 0; h < heads; ++h)
        for (int r = 0; r < rows; ++r) {
            const int dst = h * rows + r;
            const int srow = h * head_panels() * 8 + r;
            const int panel = srow / 8, lane = srow % 8;
            for (int g = 0; g < G; ++g) {
                const uint32_t n = uint32_t(block(panel, g));
                const Word* base = w + L::block_word(n);
                if (32 * g + 32 <= cols) get_lane(base, lane, &codes[size_t(dst) * cols + size_t(32 * g)]);
                else
                    for (int k = 0; k < 32; ++k) {
                        const int col = 32 * g + k;
                        if (col < cols) codes[size_t(dst) * cols + col] = base[L::code_word(k)].b[L::code_byte(k, lane)];
                    }
                scales[size_t(dst) * G + g] = w[L::scale_word(n)].b[L::scale_byte(n, lane)];
            }
        }
}

// Value (r, c) is bytes 4 (r cols + c) .. + 3, least significant first: on a little-endian host the values' own bytes.
std::vector<Word> f32_pack(const uint32_t* bits, int rows, int cols) {
    std::vector<Word> out(static_cast<size_t>(rows) * cols / 8);
    if (LITTLE_ENDIAN_HOST) { std::memcpy(out.data(), bits, size_t(rows) * size_t(cols) * 4); return out; }
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) out[size_t(r) * (cols / 8) + c / 8].set_u32(c % 8, bits[size_t(r) * cols + c]);
    return out;
}

void f32_unpack(const Word* w, int rows, int cols, uint32_t* bits) {
    if (LITTLE_ENDIAN_HOST) { std::memcpy(bits, w, size_t(rows) * size_t(cols) * 4); return; }
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) bits[size_t(r) * cols + c] = w[size_t(r) * (cols / 8) + c / 8].u32(c % 8);
}

std::vector<Word> rms_pack(const uint32_t* r, int rows) {
    std::vector<Word> out(rms_extent(rows));
    for (int i = 0; i < rows; ++i) out[size_t(i / 8)].set_u32(i % 8, r[i]);
    return out;
}

std::vector<Word> vt_pack(const uint8_t* codes_t, const uint8_t* scales_t, int dh, int cap) {
    const int G = dh / 32, J = cap / 32;
    const Layout cell(32, 32, true);
    std::vector<Word> out(vt_words(dh, cap));
    std::vector<uint8_t> c(32 * 32), s(32);
    for (int j = 0; j < J; ++j)
        for (int g = 0; g < G; ++g) {
            for (int d = 0; d < 32; ++d) {
                std::memcpy(&c[size_t(d) * 32], &codes_t[size_t(32 * g + d) * cap + 32 * j], 32);
                s[d] = scales_t[size_t(32 * g + d) * J + j];
            }
            cell.pack_into(c.data(), s.data(), &out[size_t(33) * (j * G + g)]);
        }
    return out;
}

void vt_unpack(const Word* w, int dh, int cap, uint8_t* codes_t, uint8_t* scales_t) {
    const int G = dh / 32, J = cap / 32;
    const Layout cell(32, 32, true);
    std::vector<uint8_t> c(32 * 32), s(32);
    for (int j = 0; j < J; ++j)
        for (int g = 0; g < G; ++g) {
            cell.unpack(w + size_t(33) * (j * G + g), c.data(), s.data());
            for (int d = 0; d < 32; ++d) {
                std::memcpy(&codes_t[size_t(32 * g + d) * cap + 32 * j], &c[size_t(d) * 32], 32);
                scales_t[size_t(32 * g + d) * J + j] = s[d];
            }
        }
}

void quantize(const uint32_t* bits, int rows, int cols, std::vector<uint8_t>& codes, std::vector<uint8_t>& scales) {
    codes.resize(static_cast<size_t>(rows) * cols);
    scales.resize(static_cast<size_t>(rows) * (cols / 32));
    g3::quantize_matrix(bits, rows, cols, cols, codes.data(), scales.data());
}

void dequantize(const uint8_t* codes, const uint8_t* scales, int rows, int cols, uint32_t* bits) {
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            bits[size_t(r) * cols + c] = g3::decode_e4m3(codes[size_t(r) * cols + c], scales[size_t(r) * (cols / 32) + c / 32]);
}

}  // namespace g3rt
