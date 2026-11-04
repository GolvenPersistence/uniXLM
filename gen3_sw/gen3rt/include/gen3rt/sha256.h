#pragma once
#include <algorithm>
#include <array>
#if defined(G3RT_BOARD) && !defined(G3RT_PORTABLE_SHA256)
#include <openssl/evp.h>
#endif
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

namespace g3rt {
// SHA-256 for model/cache identity; independent of host endianness.
#if defined(G3RT_BOARD) && !defined(G3RT_PORTABLE_SHA256)
class Sha256 {
    EVP_MD_CTX *ctx_ = EVP_MD_CTX_new();

  public:
    Sha256() {
        if (!ctx_ || EVP_DigestInit_ex(ctx_, EVP_sha256(), nullptr) != 1) {
            EVP_MD_CTX_free(ctx_);
            throw std::runtime_error("SHA256 initialization failed");
        }
    }
    Sha256(const Sha256 &) = delete;
    Sha256 &operator=(const Sha256 &) = delete;
    ~Sha256() { EVP_MD_CTX_free(ctx_); }
    void update(const void *data, size_t n) {
        if (EVP_DigestUpdate(ctx_, data, n) != 1)
            throw std::runtime_error("SHA256 update failed");
    }
    std::string finish() {
        unsigned char digest[32];
        unsigned len = 0;
        if (EVP_DigestFinal_ex(ctx_, digest, &len) != 1 || len != 32)
            throw std::runtime_error("SHA256 finish failed");
        static const char hex[] = "0123456789abcdef";
        std::string out;
        for (auto b : digest) {
            out += hex[b >> 4];
            out += hex[b & 15];
        }
        return out;
    }
};
#else
class Sha256 {
    uint32_t h_[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    uint8_t block_[64]{};
    uint64_t bytes_ = 0;
    size_t used_ = 0;
    static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }
    void compress() {
        static constexpr uint32_t k[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(block_[4 * i]) << 24) | (uint32_t(block_[4 * i + 1]) << 16) |
                   (uint32_t(block_[4 * i + 2]) << 8) | block_[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            const auto a = w[i - 15], b = w[i - 2];
            w[i] =
                w[i - 16] + (rotr(a, 7) ^ rotr(a, 18) ^ (a >> 3)) + w[i - 7] + (rotr(b, 17) ^ rotr(b, 19) ^ (b >> 10));
        }
        auto a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + k[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
        h_[5] += f;
        h_[6] += g;
        h_[7] += h;
    }

  public:
    void update(const void *data, size_t n) {
        auto p = static_cast<const uint8_t *>(data);
        bytes_ += n;
        while (n) {
            const size_t take = std::min(n, 64 - used_);
            std::memcpy(block_ + used_, p, take);
            used_ += take;
            p += take;
            n -= take;
            if (used_ == 64) {
                compress();
                used_ = 0;
            }
        }
    }
    std::string finish() {
        const uint64_t bits = bytes_ * 8;
        const uint8_t one = 0x80, zero = 0;
        update(&one, 1);
        while (used_ != 56)
            update(&zero, 1);
        uint8_t tail[8];
        for (int i = 0; i < 8; ++i)
            tail[i] = uint8_t(bits >> (56 - 8 * i));
        update(tail, 8);
        static const char hex[] = "0123456789abcdef";
        std::string out;
        for (auto v : h_)
            for (int i = 7; i >= 0; --i)
                out += hex[(v >> (4 * i)) & 15];
        return out;
    }
};
#endif
inline std::string sha256_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("open for SHA256: " + path);
    Sha256 h;
    std::array<char, 65536> b;
    while (f) {
        f.read(b.data(), b.size());
        h.update(b.data(), size_t(f.gcount()));
    }
    if (!f.eof())
        throw std::runtime_error("read for SHA256: " + path);
    return h.finish();
}
} // namespace g3rt
