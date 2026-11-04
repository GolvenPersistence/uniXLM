# g3emu: command-level, bit-exact emulator of the Gen3 Node v3

`g3emu` executes Gen3 Node programs (128-bit commands, host pushes, preloads) over a unified buffer (UB, 16384 x
256-bit words) and a byte-addressed external memory. It reproduces what the RTL (`gen3_fpga/node`) writes, bit for
bit, and the completion status of every command. Cycles are not modelled.

- Numerics: v3 only (`docs/gen3/node_v3.md` s1.4-1.5): bf48 block accumulation, flush to +0, FP32 C / D / g (F32
  layout), V^T caches with the decode-tail rewrite, 64-key block-max attention, PWL SiLU / GELU, RoPE, RMSNorm fold.
- Command encoding: ABI v3 (s3 of the spec): BIND base / extent in units of 8 words, 14-bit UB addresses, 16-bit
  MOVE offsets / counts with 40-bit addresses, the binding window `NODE_BASE + gen << 27 + idx << 19 + word << 5`;
  MOVE_IN with `last` (bit 104), BIND APPEND initial rows (bits 118:107), STATIC present (bit 119), STREAM bindings
  for weights, K and V^T. Test programs: `G3_ABI=v3` in the Python generators.
- Admission checks and statuses follow `gen3_fpga/node/rtl/g3_ctrl.sv` line by line (GEMM, ATTN, VECTOR,
  MOVE_OUT, MOVE_IN, BIND, UNBIND, WAIT).

## Build

Windows (llvm-mingw + GnuWin32 make, from Git Bash):

    export PATH=/d/llvm-mingw-20260224-ucrt-x86_64/bin:$PATH
    cd gen3_sw/emu && "/c/Program Files (x86)/GnuWin32/bin/make" -j8

Linux / PYNQ (gcc 11+): `cd gen3_sw/emu && make -j4`. Options: `make OPT="-O3 -march=native"`, `make CXX=clang++`,
`make CXXFLAGS=...` (extra flags; the required flags are always added).

Outputs: `build/libg3emu.a`, `build/g3emu`, `build/g3emu_kcheck` (linked statically on Windows, so they run without
the toolchain's DLLs). Portable C++17 with `std::thread` (`-pthread`); no OS APIs beyond `mkdir`.
The numerics need IEEE binary32 without FMA contraction: the Makefile always passes `-ffp-contract=off`, and
`src/fp_strict.h` repeats it as a pragma (and rejects `-ffast-math`) for builds outside this Makefile.

## Run

    build/g3emu run <case_dir> [--out DIR] [--attn-requantize] [--vector-rmsnorm] [--threads N] [--trace] [--quiet]

Reads `<case_dir>/program.hex` (the format of `tb/gen2/node_program.py`), writes `DIR/hmem_out.hex` (every
external memory word: preloaded or written, `"<word address hex> <64 hex digits>"`, ascending, as the RTL bench)
and `DIR/completions.txt` (`"<sequence> <tag> <status> <data>"` in completion order). `DIR` defaults to
`<case_dir>/emu_out`. `--trace` prints every command with its status; `--attn-requantize` feeds attention the inputs
the Python oracle uses (see below). VECTOR behaves as the v3 Node's `g3_vru_stub.sv` (admitted, result released at watermark 0, status UNSUPPORTED); `--vector-rmsnorm` runs it as gen3_rtl's `g3_vru` (RMSNORM) instead. `--threads N` splits large GEMMs over N threads (every output value is
computed on its own, so the results do not change).

Program-order semantics: each command runs to completion before the next; pushes (record 02) and MOVE_IN write when
they come. The RTL's watermarks and hazard rules make a valid program equivalent. A WAIT completes as soon as its
condition holds (possibly after later commands). Reads of a binding whose writer has not finished in program order
(only possible when a program relies on a later push) are warned about on stderr.

## Checks

    python gen3_sw/emu/tools/kernel_vectors.py && gen3_sw/emu/build/g3emu_kcheck gen3_sw/emu/out/kernels
    python gen3_sw/emu/tools/compare.py            # every RTL case of .codex_tmp/node_v3_reg (--ref DIR for others)
    python gen3_sw/emu/tools/oracle_check.py       # every case of tb/gen3/run_node_tb.py vs the Python oracle
    python gen3_sw/emu/tools/layer_check.py        # SmolVLM-256M text layer (prefill 32 + decode) vs the Python oracle

(Python: `D:/anaconda3/envs/dl/python.exe`, numpy; the tools set `G3_ACC=bf48`.)

1. Kernels vs Python (`g3emu_kcheck`, all bit-exact):

   | group | reference | values |
   |---|---|---:|
   | quant | `g2lib.quantize_row` (3,000 blocks incl. specials, subnormals, ties), `decode_e4m3_fp32` | 7,352 |
   | gemm | `g2lib.gemm_bf48` (10 shapes, scales 0-254, renormalisation pressure, NaN codes, scale 0xFF) | 2,375 |
   | epilogue | `core_vectors.activation_bits` (SiLU, GELU), `rope_bits`, `fp32_add_bits` / `fp32_mul_bits` | 16,410 |
   | fold | `g3lib.inv_rms_rows`, `g2lib.rmsnorm_rows` (W 64-576, eps 0-3) | 4,566 |
   | attn | `Gen3Attention.run(block=64)`, bf48 (10 shapes, Dh 32-224, keys up to 256, causal / not) | 141 rows |
   | vt | V^T re-rounding (all 256 codes x 219 scale pairs), `VtCache` append sequences (prefill + decode tail) | 56,068 |

   The V^T re-rounding reference uses decode + quantise; where `decode_e4m3_fp32` overflows FP32 (old scale >= 247)
   the RTL's code-domain rule is the exact one, and the generator uses the same quantisation at a shifted scale.

2. RTL cases (`compare.py` against `.codex_tmp/node_v3_reg`, RTL run of 2026-10-20 12:18-12:26; also
   `.codex_tmp/node_v3_mi`: 4/4). Every word of the RTL's external memory dump equals the emulator's (no X words
   in these dumps; X placeholders would be skipped), the emulator writes no other word, and every tag's completions
   (status and data) are equal: **40 / 40 bit-exact**.

   | case | result | memory words | completions | | case | result | memory words | completions |
   |---|---|---:|---:|---|---|---|---:|---:|
   | n1_single | PASS | 66 | 5 | | n21_attn_dh160 | PASS | 330 | 6 |
   | n2_tail | PASS | 33 | 5 | | n22_attn_dh96 | PASS | 99 | 6 |
   | n3_add_epilogue | PASS | 66 | 6 | | n23_attn_dh224 | PASS | 231 | 6 |
   | n4_stream_b | PASS | 1188 | 5 | | n24_attn_causal_odd | PASS | 198 | 6 |
   | n5_decode_stream | PASS | 825 | 5 | | n25_append_kv_tail | PASS | 792 | 28 |
   | n6_chain | PASS | 99 | 8 | | n26_fold | PASS | 466 | 24 |
   | n7_heads | PASS | 66 | 5 | | n27_fold_rows | PASS | 533 | 19 |
   | n8_append_kv | PASS | 528 | 20 | | n28_fold_decode | PASS | 364 | 15 |
   | n9_errors | PASS | 33 | 10 | | n29_fold_errors | PASS | 101 | 23 |
   | n10_war_rebind | PASS | 132 | 9 | | n30_f32_add | PASS | 640 | 6 |
   | n11_wait_fence | PASS | 33 | 7 | | n31_f32_fold | PASS | 1328 | 23 |
   | n12_gated_stream | PASS | 594 | 5 | | n32_f32_decode | PASS | 517 | 19 |
   | n13_rope_heads | PASS | 66 | 6 | | n33_f32_none | PASS | 256 | 5 |
   | n14_rmsnorm | PASS | 66 | 5 | | n34_attn_stream | PASS | 726 | 6 |
   | n15_rmsnorm_576_gemm | PASS | 660 | 9 | | n35_attn_stream_vit | PASS | 1122 | 6 |
   | n16_attn_prefill | PASS | 198 | 6 | | n36_attn_stream_decode | PASS | 462 | 6 |
   | n17_attn_decode_gqa | PASS | 165 | 6 | | n37_attn_stream_dh128 | PASS | 924 | 6 |
   | n18_attn_multi_block | PASS | 132 | 6 | | n38_kv_ddr | PASS | 330 | 21 |
   | n19_attn_errors | PASS | 66 | 11 | | n39_kv_ddr_band_start | PASS | 462 | 21 |
   | n20_attn_dh256 | PASS | 1056 | 6 | | n40_move_in | PASS | 3102 | 17 |

   Error statuses covered: n9 (UNSUPPORTED, RANGE, BINDING, ORDER), n19 (UNSUPPORTED x2, RANGE, ORDER), n29
   (UNSUPPORTED, BINDING x2), n40 (RANGE x2, BINDING, ORDER x2).

3. Python oracle (`oracle_check.py`, no RTL needed): 42 / 42 cases (adds n41_gelu, n42_gelu_decode) with the
   regression seed; seeds 101, 202, 404, 505 also 42 / 42; seed 303 see below. `layer_check.py`: the whole
   SmolVLM-256M layer program of `tb/gen3/run_layer_smolvlm.py` (q/k/v with fold + RoPE, K and V^T APPEND arrays,
   GQA attention, o / gate-up / down with folds, decode step on the cache) equals the oracle in every checked tensor,
   with folded norms, VRU norms, and another seed. Emulator time for the layer (prefill + decode, about 117 M
   MACs): 0.045 s with the default build, 0.032 s with `--threads 4`, on the development PC.

### Attention inputs: stored codes (RTL) vs the oracle's re-quantisation

`g3lib.attention` decodes Q, K and V^T and `Gen3Attention.run` quantises them again; for V^T it first zeroes the keys
at or beyond `keys`. With bf48 the integer representation matters (`(pdot * c) >> 2` truncates), so a block whose
re-quantised scale is lower can give different low bits. The RTL uses the stored codes and scales
(`g3f_lane_bf.sv`: PV exponent = P scale + stored V^T scale - 278); the emulator does the same by default, and
`--attn-requantize` (or `--attn-requantize-mask 1|2|4` for Q / K / V^T) reproduces the oracle. All 40 RTL cases and
the regression seed are equal in both modes. Seed 303 of `n24_attn_causal_odd` separates them (V^T cell with random
codes at keys 150..159): 2 O codes differ (0x81 vs 0x80). That program and both results are kept in
`cases/n24_seed303_vt_beyond_keys/` for an RTL run. Caches the Node writes itself keep those keys at code 0, where both
agree.

## Structure

| file | contents |
|---|---|
| `include/g3emu_kernels.h` | numeric kernels, plain functions over plain arrays (for a ggml backend too) |
| `include/g3emu_layout.h` | quad / block / P8 / P32 / V^T word layouts |
| `include/g3emu.h` | `Node` (UB, sparse external memory, binding table, `execute`, `push`, `preload`), program files |
| `src/mxfp8.cpp` | FP32 helpers, E4M3 / E8M0 quantise and decode (Gen1 reference), bf48 rounding |
| `src/gemm.cpp` | K32 exact integer block sums (summed in FP64, exact below 2^53), bf48 accumulation |
| `src/epilogue.cpp` | SiLU / GELU 64-segment PWL, RoPE |
| `src/fold.cpp` | 1/rms (VRU order), rsqrt, VECTOR RMSNORM rows |
| `src/attention.cpp` | 64-key block-max softmax row (P thresholds, c_j, rescale, reciprocal), V^T tail rules |
| `src/node.cpp` | commands: admission checks, GEMM placement (P8 / P32 / APPEND / arrays / V^T / F32 / fold), ATTN, VECTOR, MOVE_OUT / MOVE_IN, pushes |
| `src/program.cpp`, `src/g3emu_main.cpp` | program.hex runner, result files, CLI |
| `src/tables.h` | constants generated from the Python references (`tools/gen_tables.py`) |
| `src/kcheck.cpp` | kernel checker |
| `tools/` | `gen_tables.py`, `kernel_vectors.py`, `compare.py`, `oracle_check.py`, `layer_check.py` |

Kernel API in short: `quantize_block32`, `decode_e4m3`, `gemm_bf48` / `mx_decode_rows` + `gemm_bf48_dot`,
`silu_pwl`, `gelu_pwl`, `rope_pairs`, `inv_rms_row`, `rmsnorm_row`, `attention_row` (one query row against an
`AttnKV` of K rows and V^T), `vt_tail_scale` / `vt_requant_code`.

## Hardware details reproduced (beyond the Python models)

- Rows of a GEMM beyond M (inside the last panel) and columns beyond N (inside the last group) are computed with
  masked operands (zero codes, scale 127), go through the epilogue with whatever C holds there, and are written:
  whole blocks with 8-byte scale strobes; a single-row APPEND writes only its row's bytes and scale byte.
- BIND evicts every live binding owning a granule (8 words) of its range (owner map); UNBIND clears array elements.
- The fold producer writes four 1/rms words per row pass; rows beyond the pass's panels keep the values of the
  output engine's 1/rms memory from earlier passes (as the RTL; it starts at 0 here, X in simulation).
- V^T decode tail: the old cell's "all-zero" test spans all 32 tokens and every token is re-rounded (the Python
  model looks at the tokens before t, equal for caches the Node writes).
- WAIT completion data is the binding's watermark; commit events update watermarks and end writers exactly as
  `g3_ctrl` applies them (array elements, fold outputs, APPEND rows = AROW + M).

Not modelled: cycles and CSR counters, ABORT of a running command (program order: the target has finished; ABORT
completes OK), QUIESCE / RESUME / CONTEXT_DROP / CLEAR / CSR (UNSUPPORTED, as the RTL's default), reader counts
(always 0 between commands in program order), poisoning (never set by this RTL).
