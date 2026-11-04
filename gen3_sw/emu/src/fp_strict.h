// Included first by every source that does FP32 arithmetic: the Node's numerics are separately rounded IEEE binary32
// operations, so a*b+c must never be fused into an FMA (GCC contracts by default outside ISO mode, on aarch64 too).
// The Makefile also passes -ffp-contract=off; this keeps the guarantee when the sources are built elsewhere (ggml).
#pragma once

#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#elif defined(__GNUC__)
#pragma GCC optimize("fp-contract=off")
#endif

#if defined(__FAST_MATH__)
#error "g3emu needs IEEE arithmetic: do not build it with -ffast-math"
#endif
