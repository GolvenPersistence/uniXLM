# PYNQ-ZU Cortex-A53 Q8_0 GEMV

This optional CPU patch accelerates the Q8_0 vocabulary projection in the Qwen
hybrid deployment. It does not modify the FPGA backend, bitstream, quantization
format, model weights, or KV cache. Apply it to the pinned AICAS llama.cpp tree
before rebuilding the CPU library and llama-server:

```sh
python3 gen3_sw/patches/apply_cpu_a53.py /path/to/llama.cpp
cmake --build /path/to/llama.cpp-build-board --target llama-server -j4
```

The helper is enabled by default on AArch64 targets without SDOT, I8MM or SVE.
Set `GGML_GEN3_CPU_A53=0` before launching the process to use the original path.
Use the same Cortex-A53 compiler options as the existing board build; do not use
`-ffast-math`. Application is idempotent and rejects unrecognized source anchors.
This is a patch for the pinned backend, not a compatibility promise for arbitrary
upstream ggml versions. Non-AArch64 targets compile a no-op helper.

Eligible operations: contiguous two-dimensional Q8_0 weights, one contiguous
F32 input vector, contiguous F32 output, K divisible by 64 in [64,8192], and
N >= 256. All other shapes use the original implementation.

Each worker quantizes the input once using ggml's existing quantizer, caches its
block scales on the stack, and owns a disjoint set of output rows. Combining two
signed byte products before widening reduces integer instructions. The input
codes are checked to exclude -128, so the int16 intermediate cannot overflow,
even if external weights contain -128. Nonfinite activation scales fall back.
The original ARM lane grouping, two alternating floating accumulators, FMA and
final reduction order are preserved. No shared scratch allocation or new barrier
is introduced. Scratch size is at most 9728 bytes per worker.

`test_cpu_a53.cpp` builds against ggml, ggml-cpu and their ordinary backend
libraries. Run it on the A53 board after applying the patch. It compares enabled
and disabled execution byte-for-byte over 19 shape/thread combinations, including
K=8192, odd output partitions, -128 weight codes, zero activations, odd-block and
multi-vector fallback, and the 151936 x 896 vocabulary matrix. It prints paired
kernel timings; do not include the cold iteration or zero-input iteration in
normal-input timing summaries. Exact board compile/link commands and full-model
validation are recorded in `docs/reports/20261022_qwen_cpu_opt`.

Only Qwen2.5-0.5B-Instruct Q8_0 has end-to-end validation in this revision. Other
models and quantization formats require their own measurement. The prior CPU
attention implementation remains in place: tested FP16 conversion caching and
Flash Attention alternatives were slower on this board.
