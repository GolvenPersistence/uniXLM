// Small portable bit helpers (compiler builtins where available, plain loops otherwise).
#pragma once

#include <cstdint>

namespace g3 {

// Number of bits needed for x (0 for x == 0): floor(log2 x) + 1.
inline int bit_length64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return x ? 64 - __builtin_clzll(x) : 0;
#else
    int n = 0;
    while (x) { ++n; x >>= 1; }
    return n;
#endif
}

inline int bit_length32(uint32_t x) { return bit_length64(x); }

}  // namespace g3
