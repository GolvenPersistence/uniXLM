// Gen3 shared encodings (docs/gen3/README.md, gen3_rtl/README.md).
// The command ABI is the Gen2 one (docs/gen2/command_abi.md) with the Gen3 changes listed in
// gen3_rtl/README.md s2: GEMM width ignored, BIND [127] = transposed layout (V^T cache), ATTN
// always uses 64-key blocks.
`ifndef G3_DEFS_VH
`define G3_DEFS_VH

// Opcodes.
`define G3_OP_BIND      8'h01
`define G3_OP_UNBIND    8'h02
`define G3_OP_MOVE_OUT  8'h08
`define G3_OP_MOVE_IN   8'h09            // v3: external memory -> binding (as an RX writer)
`define G3_OP_GEMM      8'h10
`define G3_OP_ATTN      8'h11
`define G3_OP_VECTOR    8'h12
`define G3_OP_FENCE     8'h20
`define G3_OP_QUIESCE   8'h21
`define G3_OP_RESUME    8'h22
`define G3_OP_WAIT      8'h23
`define G3_OP_CTX_DROP  8'h24
`define G3_OP_ABORT     8'h25
`define G3_OP_CLEAR     8'h26
`define G3_OP_CSR       8'h30

// Status.
`define G3_ST_OK          4'd0
`define G3_ST_UNSUPPORTED 4'd1
`define G3_ST_RESERVED    4'd2
`define G3_ST_RANGE       4'd3
`define G3_ST_BINDING     4'd4
`define G3_ST_ORDER       4'd5
`define G3_ST_CAPACITY    4'd6
`define G3_ST_POISONED    4'd7
`define G3_ST_ABORTED     4'd8
`define G3_ST_EXECUTION   4'd9

// Binding kinds and writers.
`define G3_KIND_STATIC 2'd0
`define G3_KIND_APPEND 2'd1
`define G3_KIND_STREAM 2'd2
`define G3_W_NONE   2'd0
`define G3_W_EXT    2'd1
`define G3_W_PARENT 2'd2

// Watermark sentinel handed to executors for a READY binding.
`define G3_WM_READY 13'h1fff

// Epilogues.
`define G3_EPI_NONE     4'd0
`define G3_EPI_ADD      4'd1
`define G3_EPI_MUL      4'd2
`define G3_EPI_SILU     4'd3
`define G3_EPI_GELU     4'd4
`define G3_EPI_SILU_MUL 4'd5
`define G3_EPI_ROPE     4'd6

// GEMM parent descriptor (controller -> g3_gemm_seq), 384 bits.
`define PD_TAG      7:0
`define PD_A_IDX    15:8
`define PD_A_GEN    23:16
`define PD_A_BASE   35:24
`define PD_A_WM     48:36
`define PD_B_IDX    56:49
`define PD_B_GEN    64:57
`define PD_B_BASE   76:65
`define PD_B_WM     89:77
`define PD_B_STREAM 90
`define PD_B_SRC    133:91
`define PD_B_LEN    153:134
`define PD_C_IDX    161:154
`define PD_C_GEN    169:162
`define PD_C_BASE   181:170
`define PD_C_WM     194:182
`define PD_C_VALID  195
`define PD_D_IDX    203:196
`define PD_D_GEN    211:204
`define PD_D_BASE   223:212
`define PD_D_EXT    236:224
`define PD_D_ALEN   240:237
`define PD_D_APPEND 241
`define PD_D_AROW   253:242
`define PD_D_P32    254
`define PD_D_TR     255
`define PD_M        267:256
`define PD_N        279:268
`define PD_G        288:280
`define PD_EPI      292:289
`define PD_AHEADS   296:293
`define PD_DHEADS   300:297
// RMSNorm fold (docs/gen3/README.md s4.4). Producer (epilogue ADD): D = x, D idx+1 = x * g, D idx+2 = 1/rms
// per row (FP32, 8 per word, 4 words per row pass); g = D idx+3 (1 x N, P8). Consumer: 1/rms = A idx+1.
// The derived bindings share the gen of the binding they follow.
`define PD_FP       301          // fold producer
`define PD_FC       302          // fold consumer
`define PD_EPS      304:303
`define PD_XG_BASE  316:305      // producer: x * g base
`define PD_R_BASE   328:317      // producer: 1/rms output base; consumer: 1/rms input base
`define PD_R_EXT    341:329      // producer: 1/rms extent (final commit)
`define PD_G_BASE   353:342      // producer: g base
`define PD_F_WM     366:354      // producer: g watermark; consumer: 1/rms watermark (snapshots)
`define PD_C_F32    367          // v3: C is FP32 rows (F32 layout)
`define PD_G_F32    368          // v3: the producer's g is FP32 rows
`define PD_XG_EXT   381:369      // v3: x * g extent (an FP32 producer commits it on its own)
`define PD_W        384

// UB read segment, 128 bits. Beat i of outer step j reads base + wrap(start + off[i] + j*s1, ext),
// off[0] = 0, off[1..3] = o1..o3 (per-panel offsets: P8 panel strides are not constant).
// The walk is outer-major: (j=0, i=0..n0-1), (j=1, ...).
`define S_CTX   1:0
`define S_BASE  13:2
`define S_EXT   26:14
`define S_START 38:27
`define S_N0    40:39
`define S_O1    52:41
`define S_O2    64:53
`define S_O3    76:65
`define S_N1    88:77
`define S_S1    100:89
`define S_USER  108:101
`define S_CLOSE 109
`define S_W     128
// Beat user byte: [7] scale beat, [6] set_last, [5] full (32 scale bytes), [4:3] destination
// panel, [2:1] source slot (8-byte group of the scale word), [0] half (B, C).

// Array uop record (sequencer -> g3_core), 256 bits: window, tile descriptor, output descriptor.
`define U_OP      1:0          // 0 GEMM, 1 QK, 2 PV, 3 output descriptor only (ATTN O)
`define U_KG      7:2          // K32 groups of the window (1..32)
`define U_ACCUM   8            // not the first K chunk of the tile
`define U_EMIT    9            // last K chunk: the tile completes
`define U_ROWS    41:10        // valid rows
`define U_NL      47:42        // valid columns, left half (0..32)
`define U_NR      53:48        // valid columns, right half (0..32)
`define U_ASRC    54           // A: 0 UB port A, 1 P buffer
`define U_BSRC    55           // B: 0 UB port B, 1 weight buffers
`define U_EPI     59:56
`define U_HASC    61
`define U_SLOT    62           // ATTN context slot
`define U_CSTART  63           // QK: first key block of the context
`define U_LASTKB  64           // PV: last key block
`define U_DP      66:65        // PV / O: Dh pass (64 dims)
`define U_LASTDP  67           // PV: last Dh pass of the key block (P released after it)
`define U_BIG     68           // Dh > 128: the O banks are indexed by the Dh pass
`define U_WIDTH   75:69        // QK / PV: valid keys of the 64-key block
`define U_CAUSAL  76
`define U_LIMB    90:77        // signed: causal_delta - 64*block + 1
`define U_PTB     134:91       // 4 x 11 bits: first token of each row panel
`define U_QKG     138:135      // Dh / 32 (QK scale)
`define U_LAYOUT  140:139      // 0 P8, 1 P32 rows, 2 P32 transposed (V^T)
`define U_GD      149:141      // groups per row block of the output element
`define U_H0      150          // output half 0 valid
`define U_H1      151          // output half 1 valid
`define U_BASE0   163:152
`define U_PAN0    175:164
`define U_G0      184:176
`define U_BASE1   196:185
`define U_PAN1    208:197
`define U_G1      217:209
`define U_NPAN    220:218      // valid row panels (1..4)
`define U_APP1    221          // single-row append (byte strobes)
`define U_AROW8   224:222      // row inside its panel (APP1)
`define U_VTT     229:225      // V^T: token position inside its 32-token cell
`define U_VTTAIL  230          // V^T single-token append into a partial cell (rewrite)
`define U_MARK    231          // the tile's last write is marked (retirement)
// RMSNorm fold, GEMM uops only (bits 91-134 are attention-only fields otherwise):
`define U_FP      232          // fold producer: x * g second output, sum of squares, 1/rms at the last pass
`define U_FC      233          // fold consumer: the accumulator is scaled by the row's 1/rms
`define U_PF      234          // first column pass of the row pass
`define U_PL      235          // last column pass of the row pass
`define U_RPAR    236          // row pass parity (1/rms double buffer)
`define U_EPS     238:237
`define U_XGD     250:239      // x * g base - x base (word addresses, modulo 4096)
`define U_RBASE   102:91       // producer: 1/rms words of the row pass (4)
`define U_INV     134:103      // producer: RNE(1 / (32 * output groups))
`define U_FBASE   88:77        // v3 GEMM with an FP32 D: word of row 0 of the pass, left half (ATTN: U_LIMB)
`define U_CF32    251          // v3: C rows are FP32 (F32 layout)
`define U_GF32    252          // v3: the producer's g row is FP32
`define U_W       256

`define G3_LAYOUT_P8  2'd0
`define G3_LAYOUT_P32 2'd1
`define G3_LAYOUT_VT  2'd2
`define G3_LAYOUT_F32 2'd3          // v3: FP32 rows, 8 values per word, row stride N / 8 words

`endif
