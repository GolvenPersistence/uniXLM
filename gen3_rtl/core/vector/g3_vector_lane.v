`timescale 1ns/1ps
`default_nettype none

// One lane of the Gen3 vector column (gen3_rtl/README.md s3.3): array row r, all 64 columns of a
// pass. Derived from the Gen2 block-maximum lane (rtl/gen2/g2_vector_lane_blockmax.v): the Gen2
// lane's two rows x 32 columns become the two halves (left / right) x 32 columns of one row, so
// the storage map, the step order and the GEMM epilogue are unchanged. What differs:
//   - softmax statistics are per row: block maximum = max of both halves' maxima, one reference,
//     c_j, shift, P scale and l; the row sum adds both halves' (exact) sums;
//   - gated SILU_MUL: the left half holds the gate tile, the right half the up tile of the same
//     columns; gate values go to T0 and the right half's epilogue reads them;
//   - the O row scale spans the whole row (both halves, every Dh pass);
//   - GEMM output modes: row scales (0), magnitudes for a column maximum (1, no pop), external
//     column scales (2) -- the V^T path of the vector column.
//
//   holding heads (left, right) --> step: {half, value}; admission holds
//        v  accumulate stage (one value per cycle)
//   INT42->FP32 (2) --> PV x c_j (4, PV) --> + (3) --> ACC (GEMM, QK) or O (PV)
//        |  last K block of a tile
//        v
//   GEMM: pair hold -> M1 (|| M2 ROPE) -> A1 -> M2 -> 14-bit pre-round -> T1 FIFO (per half)
//   gate (SILU_MUL left half): raw FP32 -> T0      QK: masked score -> T[parity], half maximum
//   Lockstep schedules (started by the vector column):
//   GEMM output: T1 words (base + cp) -> encoder (row or external scale) -> 32-bit chunk
//   softmax:     S; m_j from both halves; k_j, c_j, shift; T -> x S -> - m_j -> P code -> chunk;
//                half sums -> row sum -> l = l 2^-D + sum x c_j 2^shift
//   O output:    max|O| x 1/l -> row scale; O -> x 1/l -> encoder -> chunk
//   RMSNorm fold (docs/gen3/README.md s4.4, GEMM):
//     consumer: the last K group's sum x 1/rms (M0) before the epilogue and the gate store;
//     producer (epilogue ADD): M2 multiplies by g (the D idx+3 row) instead of 1: x = A1 -> T1, x * g -> T0
//     (second row maxima and scales); during the x output the codes give sum x^2 in g3_vru's order
//     (16 partial sums, column k into k mod 16, k increasing: left half codes at once, right half codes
//     replayed after the pass), and after the row pass's last pass the FINAL program of g3_vru
//     (pairwise sum, x 1/H, + eps, magic rsqrt, two Newton steps) on M0 and A2 gives 1/rms.
//
// Chunk layout {right col 2cp+1, right col 2cp, left col 2cp+1, left col 2cp}.
module g3_vector_lane (
    input  wire          clk,
    input  wire          rstn,
    // Holding FIFO heads of the left (0) and right (1) half, and their tiles' lane descriptors.
    input  wire [1:0]    h_valid,
    input  wire [225:0]  h_entry,              // {right, left} x 113
    input  wire [7:0]    h_cp,                 // {right, left} column pair
    input  wire [39:0]   h_desc,               // {right, left} x 20
    input  wire [1:0]    h_owait,              // {right, left}: PV waits for an earlier O job of its slot
    output wire [1:0]    h_pop,
    // c_j word of this row by P bank {bank 1, bank 0}.
    input  wire [63:0]   pv_alpha,
    // Epilogue C tiles: entry e, half h at [(e*2+h)*256], column c byte c.
    input  wire [1023:0] c_rows,
    input  wire [31:0]   c_row_scales,         // (e*2+h)*8
    input  wire [3:0]    c_popped,
    input  wire [1:0]    c_level,
    output reg  [1:0]    c_row_done,
    // RMSNorm fold: producer g rows (entry e, half h at [(e*2+h)*256], column c byte c) and their row 0
    // scales; consumer 1/rms of this row {row pass parity 1, parity 0}.
    input  wire [1023:0] g_rows,
    input  wire [31:0]   g_row_scales,
    input  wire [1:0]    g_level,
    input  wire [63:0]   r_vals,
    input  wire [1:0]    r_valid,
    // GEMM output (lockstep).
    output wire [1:0]    g_ready,              // per half: a tile's scale is available
    output wire [15:0]   g_scales,             // {right, left} row scales of the head tile
    output wire          g_busy,
    input  wire          g_go,
    input  wire [3:0]    g_cp,
    input  wire [1:0]    g_mode,               // 0 row scale, 1 magnitudes, 2 external scales
    input  wire [1:0]    g_halves,             // halves popped at the last column pair
    input  wire          g_last,
    input  wire [31:0]   g_ext_scales,         // {right odd, right even, left odd, left even}
    input  wire          g_out,                // the go reads the x * g FIFO (T0) of a fold producer
    input  wire          g_fold,               // fold producer tile (sum x^2 from the x codes)
    input  wire          g_pfirst,             // first / last column pass of the row pass
    input  wire          g_plast,
    input  wire [31:0]   f_inv,                // 1/H
    input  wire [1:0]    f_eps,
    output wire [1:0]    g_ready2,             // x * g scales (fold producer)
    output wire [15:0]   g_scales2,
    output reg           rms_valid,            // the row pass's 1/rms (fold producer)
    output reg  [31:0]   rms,
    output reg           g_chunk_valid,
    output reg  [31:0]   g_chunk,
    output reg  [51:0]   g_mag,                // 4 x 13-bit pre-rounded magnitudes (mode 1)
    // Softmax (lockstep).
    output wire [1:0]    t_full,               // per parity: both halves hold a complete QK row
    input  wire          s_start,
    input  wire          s_tpar,
    input  wire          s_slot,
    input  wire          s_cstart,
    input  wire          s_vis,
    input  wire [31:0]   s_scale,
    output reg           s_par_valid,
    output reg  [31:0]   s_alpha,
    output reg  [7:0]    s_pscale,
    output reg           s_chunk_valid,
    output reg  [31:0]   s_chunk,
    output reg           s_done,
    // O output (lockstep).
    output reg  [1:0]    pv_end_row,
    input  wire          o_slot,
    output wire [31:0]   o_l,
    input  wire          o_start,
    input  wire [31:0]   o_inv,
    output reg  [7:0]    o_scale,
    input  wire          o_go,
    input  wire [1:0]    o_lg,
    input  wire          o_dh_big,
    input  wire [3:0]    o_cp,
    output reg           o_chunk_valid,
    output reg  [31:0]   o_chunk,
    output wire          busy
);
    `include "g3_numeric_functions.vh"
    localparam [1:0]  OP_GEMM = 0, OP_QK = 1, OP_PV = 2;
    localparam [3:0]  EPI_NONE = 0, EPI_ADD = 1, EPI_MUL = 2, EPI_SILU = 3, EPI_GELU = 4,
                      EPI_SILU_MUL = 5, EPI_ROPE = 6;
    localparam [31:0] ONE = 32'h3f80_0000, NEG_ZERO = 32'h8000_0000, NEG_INF = 32'hff80_0000;
    localparam [2:0]  K_EPI = 1, K_SMAX = 2, K_SCALE = 3, K_SCORE = 4, K_LPROD = 5, K_OSCALE = 6, K_OELEM = 7;
    localparam [31:0] LOG2E = 32'h3fb8_aa3b;
    localparam [7:0]  P_SCALE = 8'd119;
    localparam [2:0]  K2_PAIR = 1, K2_SUM = 2, K2_L = 3, K2_TOT = 4, K2_SQ = 5, K2_FIN = 6;
    localparam [1:0]  M0_PV = 0, M0_SCALE = 1, M0_SQ = 2, M0_FIN = 3;
    localparam [31:0] C_HALF = 32'h3f00_0000, C_NEG1 = 32'hbf80_0000, C_THREEHALF = 32'h3fc0_0000;
    // Accumulate context {op[1:0], end, keep, half, col[4:0], first, descriptor[19:0]}.
    localparam integer CW = 31;
    integer i;

    function automatic [13:0] preround(input [31:0] v);
        preround = {v[31:19], |v[18:0]};
    endfunction
    function automatic [31:0] expand(input [13:0] p);
        expand = {p[13:1], p[0], 18'd0};
    endfunction
    function automatic [31:0] neg(input [31:0] v);
        neg = {~v[31], v[30:0]};
    endfunction
    // P code: E4M3 at scale 119 of 2^d, d = FP32(t - m_j) <= 0 (Gen2 block-maximum lane).
    function automatic [7:0] p_code(input [31:0] d);
        integer e, sh, dfx, u, n, f, m, w, j;
        reg [23:0] mant, q;
        reg sticky;
        begin
            e = d[30:23]; mant = {1'b1, d[22:0]};
            if (e == 0) dfx = (d[31] && d[22:0] != 0) ? -1 : 0;
            else if (e >= 132) dfx = -2000000;
            else begin
                sh = 134 - e;
                if (sh >= 24) begin q = 24'd0; sticky = 1'b1; end
                else begin q = mant >> sh; sticky = |(mant & ((24'd1 << sh) - 24'd1)); end
                dfx = d[31] ? -($signed({8'd0, q}) + sticky) : $signed({8'd0, q});
            end
            if (dfx < -18 * 65536) p_code = 8'd0;
            else begin
                u = dfx + 8 * 65536;
                n = u >>> 16;
                f = u & 32'hffff;
                m = (f >= 5732) + (f >= 16249) + (f >= 25711) + (f >= 34313) + (f >= 42196) +
                    (f >= 49473) + (f >= 56229) + (f >= 62535);
                if (m == 8) begin n = n + 1; m = 0; end
                if (n >= -6) p_code = (n + 7) * 8 + m;
                else begin
                    w = u + 9 * 65536;
                    j = (w > -65536) + (w >= 38337) + (w >= 86634) + (w >= 118447) + (w >= 142209) +
                        (w >= 161182) + (w >= 176977) + (w >= 190506);
                    p_code = j[7:0];
                end
            end
        end
    endfunction
    function automatic [28:0] ceil_frac(input [31:0] mv);
        integer e, ex, sh, mf, fl, k, g;
        reg [23:0] mant, q;
        reg sticky, fracnz;
        begin
            e = mv[30:23]; ex = e - 127; mant = {1'b1, mv[22:0]};
            if (e == 0) begin
                mf = (mv[31] && mv[22:0] != 0) ? -1 : 0; fracnz = mv[22:0] != 0;
            end else if (ex >= 14) begin
                mf = 0; fracnz = 1'b0;
            end else begin
                if (ex >= 11) mf = $signed({8'd0, mant}) << (ex - 11);
                else begin
                    sh = 11 - ex;
                    if (sh >= 24) begin q = 24'd0; sticky = 1'b1; end
                    else begin q = mant >> sh; sticky = |(mant & ((24'd1 << sh) - 24'd1)); end
                    mf = mv[31] ? -($signed({8'd0, q}) + sticky) : $signed({8'd0, q});
                end
                if (ex >= 11 && mv[31]) mf = -mf;
                fracnz = ex >= 23 ? 1'b0 : ex < 0 ? 1'b1 : |(mant & ((24'd1 << (23 - ex)) - 24'd1));
            end
            fl = mf >>> 12;
            k = fl + fracnz;
            g = mf - ((k - 1) << 12);
            if (e != 0 && ex >= 14) begin k = mv[31] ? -16384 : 16384; g = 4096; end
            ceil_frac = {k[15:0], g[12:0]};
        end
    endfunction
    function automatic [31:0] scale_down(input [31:0] x, input [7:0] d);
        if (d == 8'd0 || x[30:23] == 8'd0 || x[30:23] == 8'd255) scale_down = x;
        else if (x[30:23] <= d) scale_down = {x[31], 31'd0};
        else scale_down = {x[31], x[30:23] - d, x[22:0]};
    endfunction
    function automatic [16:0] c_table(input [4:0] i);
        case (i)
        5'd0: c_table = 17'd32768;  5'd1: c_table = 17'd34219;  5'd2: c_table = 17'd35734;
        5'd3: c_table = 17'd37316;  5'd4: c_table = 17'd38968;  5'd5: c_table = 17'd40693;
        5'd6: c_table = 17'd42495;  5'd7: c_table = 17'd44376;  5'd8: c_table = 17'd46341;
        5'd9: c_table = 17'd48393;  5'd10: c_table = 17'd50535; 5'd11: c_table = 17'd52773;
        5'd12: c_table = 17'd55109; 5'd13: c_table = 17'd57549; 5'd14: c_table = 17'd60097;
        5'd15: c_table = 17'd62757; default: c_table = 17'd65536;
        endcase
    endfunction
    function automatic [8:0] c_from_g(input [12:0] g);
        reg [16:0] lo, hi;
        reg [27:0] prod;
        reg [17:0] val;
        begin
            if (g >= 13'd4096) c_from_g = 9'd256;
            else begin
                lo = c_table(g[12:8]); hi = c_table(g[12:8] + 5'd1);
                prod = (hi - lo) * g[7:0] + 28'd128;
                val = lo + prod[27:8];
                c_from_g = 9'((val + 18'd128) >> 8);
            end
        end
    endfunction

    // Lane descriptor fields (20 bits, see g3_vector_column).
    function automatic [3:0] d_epi(input [19:0] d);    d_epi = d[19:16];  endfunction
    function automatic       d_gate(input [19:0] d);   d_gate = d[15];    endfunction
    function automatic       d_has_c(input [19:0] d);  d_has_c = d[14];   endfunction
    function automatic [3:0] d_cidx(input [19:0] d);   d_cidx = d[13:10]; endfunction
    function automatic       d_tpar(input [19:0] d);   d_tpar = d[9];     endfunction
    function automatic       d_slot(input [19:0] d);   d_slot = d[8];     endfunction
    function automatic [1:0] d_group(input [19:0] d);  d_group = d[7:6];  endfunction
    function automatic       d_big(input [19:0] d);    d_big = d[5];      endfunction
    function automatic       d_lastc(input [19:0] d);  d_lastc = d[4];    endfunction
    function automatic       d_lastkb(input [19:0] d); d_lastkb = d[3];   endfunction
    function automatic       d_vis(input [19:0] d);    d_vis = d[2];      endfunction
    function automatic       d_pfirst(input [19:0] d); d_pfirst = d[1];   endfunction
    function automatic       d_pbank(input [19:0] d);  d_pbank = d[0];    endfunction
    // GEMM tiles: RMSNorm fold fields in the attention-only bits.
    function automatic       d_fp(input [19:0] d);     d_fp = d[9];       endfunction
    function automatic       d_fc(input [19:0] d);     d_fc = d[8];       endfunction
    function automatic       d_rpar(input [19:0] d);   d_rpar = d[5];     endfunction
    function automatic [31:0] eps_value(input [1:0] e);
        case (e)
            2'd0: eps_value = 32'h0000_0000; 2'd1: eps_value = 32'h3586_37bd;
            2'd2: eps_value = 32'h3727_c5ac; default: eps_value = 32'h38d1_b717;
        endcase
    endfunction

    // ------------------------------------------------------------------
    // Storage.
    // ------------------------------------------------------------------
    reg         a_wen, a_ren;
    reg  [7:0]  a_wa, a_ra;
    reg  [63:0] a_wd;
    wire [63:0] a_rd;
    reg  [1:0]  t_wen, t_ren;
    reg  [5:0]  t0_wa, t1_wa, t0_ra, t1_ra;
    reg  [31:0] t0_wd, t1_wd;
    wire [31:0] t0_rd, t1_rd;
    wire        o_rd_pair;
    function automatic [7:0] acc_word(input hf, input [3:0] cp);
        acc_word = {3'd0, hf, cp};
    endfunction
    function automatic [7:0] o_word(input bank, input hf, input group0, input [3:0] cp);
        o_word = 8'd32 + {1'b0, bank, hf, group0, cp};
    endfunction
    g3_regfile #(.DEPTH(160), .WIDTH(64)) u_acc_o (
        .clk (clk), .wen (a_wen), .wa (a_wa), .wd (a_wd), .ren (a_ren), .ra (a_ra), .rd (a_rd));
    g3_regfile #(.DEPTH(64), .WIDTH(32)) u_t0 (
        .clk (clk), .wen (t_wen[0]), .wa (t0_wa), .wd (t0_wd), .ren (t_ren[0]), .ra (t0_ra), .rd (t0_rd));
    g3_regfile #(.DEPTH(64), .WIDTH(32)) u_t1 (
        .clk (clk), .wen (t_wen[1]), .wa (t1_wa), .wd (t1_wd), .ren (t_ren[1]), .ra (t1_ra), .rd (t1_rd));

    // Row state.
    reg  [1:0]  tiles_r [0:1];            // GEMM tiles admitted to T1, per half
    reg  [1:0]  t_full_r [0:1];           // [parity][half]
    reg  [31:0] raw_max_r [0:3];          // {parity, half}
    reg  [31:0] qk_max_r [0:1];
    reg  [31:0] l_r [0:1];                // per context slot
    reg  [31:0] o_max_r [0:3];            // {slot, half}
    reg  [15:0] ref_r [0:1];
    reg  [1:0]  have_r;
    reg  [3:0]  rope_gap_r;
    reg  [1:0]  gate_pend_r;              // gate pairs taken whose up pair is not taken yet
    assign t_full = {t_full_r[1][0] && t_full_r[1][1], t_full_r[0][0] && t_full_r[0][1]};
    assign o_l = l_r[o_slot];

    // ------------------------------------------------------------------
    // Step and admission: left pair, right pair; yield on a stalled half.
    // ------------------------------------------------------------------
    reg  [1:0]   step_r;                  // {half, value}
    wire         step_half = step_r[1], step_value = step_r[0];
    wire [112:0] entry = step_half ? h_entry[225:113] : h_entry[112:0];
    wire         entry_valid = h_valid[step_half];
    wire [19:0]  entry_desc = step_half ? h_desc[39:20] : h_desc[19:0];
    wire [1:0]   entry_op = entry[110:109];
    wire         entry_end = entry[107], entry_first = entry[106];
    wire [4:0]   entry_col = {step_half ? h_cp[7:4] : h_cp[3:0], step_value};
    wire         entry_gated = entry_op == OP_GEMM && entry_end && d_gate(entry_desc);
    wire         entry_gate = entry_gated && !step_half;                 // left half of SILU_MUL
    wire         entry_epilogue = entry_op == OP_GEMM && entry_end && !entry_gate;
    // PV: the first K group of the context's first key block starts from +0; a block has up to two
    // K groups (64 keys), the second accumulates onto the first.
    wire         entry_reads = entry_op == OP_PV ? !(d_pfirst(entry_desc) && entry_first) : !entry_first;
    wire [3:0]   c_ahead = d_cidx(entry_desc) - c_popped;
    wire         c_avail = c_ahead < {2'd0, c_level};
    wire         g_avail = c_ahead < {2'd0, g_level};
    wire         hold = !step_value && (
        (entry_epilogue && entry_col == 0 && tiles_r[step_half] == 2'd2) ||
        (entry_epilogue && d_has_c(entry_desc) && !c_avail) ||
        (entry_epilogue && entry_op == OP_GEMM && d_fp(entry_desc) && !g_avail) ||
        (entry_epilogue && entry_op == OP_GEMM && d_fc(entry_desc) && !r_valid[d_rpar(entry_desc)]) ||
        (entry_epilogue && d_epi(entry_desc) == EPI_ROPE && rope_gap_r < 4'd9) ||
        (entry_gated && step_half && gate_pend_r == 0) ||
        (entry_op == OP_QK && entry_end && entry_col == 0 && t_full_r[d_tpar(entry_desc)][step_half]) ||
        (entry_op == OP_PV && h_owait[step_half]) ||
        (o_rd_pair && entry_reads));
    wire         take = entry_valid && !hold;
    assign h_pop = {take && step_half && step_value, take && !step_half && step_value};
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            step_r <= 2'd0;
            rope_gap_r <= 4'd15;
            gate_pend_r <= 0;
        end else begin
            if (take) step_r <= step_r + 1'b1;
            else if (!step_value) step_r <= {!step_half, 1'b0};
            if (take && !step_value && entry_epilogue && d_epi(entry_desc) != EPI_ROPE) rope_gap_r <= 4'd0;
            else if (rope_gap_r != 4'd15) rope_gap_r <= rope_gap_r + 1'b1;
            gate_pend_r <= gate_pend_r + (take && step_value && entry_gate) - (take && step_value && entry_gated && step_half);
        end
    end

    // ------------------------------------------------------------------
    // Accumulate stage: c0 take, c2 conversion and old value, c6 old x alpha, c9 sum.
    // ------------------------------------------------------------------
    wire [CW-1:0] ctx0 = {entry_op, entry_end, entry[104 + step_value], step_half, entry_col, entry_first, entry_desc};
    wire          entry_pv = entry_op == OP_PV;
    wire [1:0]    entry_group = d_group(entry_desc);
    wire          entry_obank = d_big(entry_desc) ? entry_group[1] : d_slot(entry_desc);
    reg           c1_value_r, took_r;
    reg  [31:0]   alpha1_r, alpha2_r, old2_r, old_hi_r;
    always @(posedge clk) begin
        c1_value_r <= step_value;
        took_r <= take;
        alpha1_r <= pv_alpha[{d_pbank(entry_desc), 5'd0} +: 32];
        alpha2_r <= alpha1_r;
        if (!c1_value_r) old_hi_r <= a_rd[63:32];
        old2_r <= c1_value_r ? old_hi_r : a_rd[31:0];
    end
    wire          conv_valid;
    wire [CW-1:0] conv_ctx;
    wire [31:0]   conv_value;
    rc_mxfp8_int42_scaled_to_fp32_staged #(.TAG_WIDTH(CW)) u_convert (
        .clk (clk), .rst_n (rstn), .advance (1'b1), .in_valid (take), .in_tag (ctx0),
        .integer_value (entry[step_value*42 +: 42]), .scale_sum (entry[84 + step_value*9 +: 9]),
        .invalid (entry[102 + step_value]), .out_valid (conv_valid), .out_tag (conv_ctx), .fp32_value (conv_value));
    wire        conv_pv = conv_ctx[30:29] == OP_PV;
    wire        conv_first = conv_ctx[20];
    // M0: PV c_j (result used at c6, beside s1_pipe_r), else (GEMM only) the consumer's 1/rms, the
    // producer's squares and FINAL multiplies (requests below; GEMM and attention never overlap).
    wire [31:0] m0_result;
    wire        m0_valid;
    wire [CW+1:0] m0_tag;                         // {kind, context or destination}
    reg         m0_req;
    reg  [31:0] m0_req_l, m0_req_r;
    reg  [CW+1:0] m0_req_tag;
    rc_mxfp8_fp32_mul_staged #(.TAG_WIDTH(CW+2)) u_m0 (
        .clk (clk), .rst_n (rstn), .advance (1'b1), .in_valid (conv_valid || m0_req),
        .in_tag (m0_req ? m0_req_tag : {M0_PV, conv_ctx}),
        .left (m0_req ? m0_req_l : conv_value), .right (m0_req ? m0_req_r : {alpha2_r[31:16], 16'd0}),
        .out_valid (m0_valid), .out_tag (m0_tag), .result (m0_result));
    reg  [CW+64:0] s1_pipe_r [0:3];
    // PV: the old O is scaled by the block's 2^-D once, with the block's first K group.
    wire [31:0]    old2_in = conv_pv && conv_first ? scale_down(old2_r, alpha2_r[7:0]) : old2_r;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) for (i = 0; i < 4; i = i + 1) s1_pipe_r[i] <= 0;
        else begin
            s1_pipe_r[0] <= {conv_valid, conv_ctx, conv_value, old2_in};
            for (i = 1; i < 4; i = i + 1) s1_pipe_r[i] <= s1_pipe_r[i-1];
        end
    end
    wire          c6_valid = s1_pipe_r[3][CW+64];
    wire [CW-1:0] c6_ctx = s1_pipe_r[3][CW+63:64];
    wire          c6_pv = c6_ctx[30:29] == OP_PV;
    wire          sum_valid;
    wire [CW-1:0] sum_ctx;
    wire [31:0]   sum_value;
    rc_mxfp8_fp32_add_staged #(.TAG_WIDTH(CW)) u_a0 (
        .clk (clk), .rst_n (rstn), .advance (1'b1), .in_valid (c6_valid), .in_tag (c6_ctx),
        .left  (c6_pv ? (d_pfirst(c6_ctx[19:0]) && c6_ctx[20] ? 32'd0 : s1_pipe_r[3][31:0])
                      : (c6_ctx[20] ? NEG_ZERO : s1_pipe_r[3][31:0])),
        .right (c6_pv ? (d_vis(c6_ctx[19:0]) ? m0_result : NEG_ZERO) : s1_pipe_r[3][63:32]),
        .out_valid (sum_valid), .out_tag (sum_ctx), .result (sum_value));
    wire [1:0]  sum_op = sum_ctx[30:29];
    wire        sum_end = sum_ctx[28], sum_half = sum_ctx[26];
    wire [4:0]  sum_col = sum_ctx[25:21];
    wire [19:0] sum_desc = sum_ctx[19:0];
    wire [1:0]  sum_group = d_group(sum_desc);
    wire        sum_obank = d_big(sum_desc) ? sum_group[1] : d_slot(sum_desc);
    wire [31:0] sum_out = (sum_op == OP_QK && sum_end && !sum_ctx[27]) ? NEG_INF : sum_value;
    wire        sum_gemm_end = sum_valid && sum_op == OP_GEMM && sum_end;
    wire        sum_fc = sum_gemm_end && d_fc(sum_desc);                 // to M0 (x 1/rms)
    wire        q_scaled = m0_valid && m0_tag[CW+1:CW] == M0_SCALE;
    wire        q_end = (sum_gemm_end && !sum_fc) || q_scaled;
    wire [CW-1:0] q_ctx = q_scaled ? m0_tag[CW-1:0] : sum_ctx;
    wire [31:0] q_value = q_scaled ? m0_result : sum_value;
    wire [19:0] q_desc = q_ctx[19:0];
    wire        q_half = q_ctx[26];
    wire [4:0]  q_col = q_ctx[25:21];
    wire        sum_gate = q_end && d_gate(q_desc) && !q_half;
    wire        sum_qk_end = sum_valid && sum_op == OP_QK && sum_end;
    wire        sum_pv = sum_valid && sum_op == OP_PV;
    wire [31:0] r_row = d_rpar(sum_desc) ? r_vals[63:32] : r_vals[31:0];
    wire [31:0] qk_max_n = sum_col == 0 ? max_value(sum_out, sum_out, 1'b1)
                                        : max_value(qk_max_r[sum_half], sum_out, 1'b1);

    // ------------------------------------------------------------------
    // GEMM epilogue (Gen2): pair hold (e0), C decode and gate read (e1), M1 || M2, A1, M2.
    // ------------------------------------------------------------------
    wire          epi_in = q_end && !sum_gate;
    wire          odd_in = epi_in && q_col[0];
    reg           even_valid_r, odd_valid_r;
    reg  [31:0]   even_x_r, odd_x_r, odd_partner_r;
    reg  [CW-1:0] even_ctx_r, odd_ctx_r;
    wire          e0_valid = odd_in || odd_valid_r;
    wire [31:0]   e0_x = odd_in ? even_x_r : odd_x_r;
    wire [31:0]   e0_partner = odd_in ? q_value : odd_partner_r;
    wire [CW-1:0] e0_ctx = odd_in ? even_ctx_r : odd_ctx_r;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            even_valid_r <= 1'b0;
            odd_valid_r <= 1'b0;
        end else begin
            if (epi_in && !q_col[0]) even_valid_r <= 1'b1;
            else if (odd_in) even_valid_r <= 1'b0;
            odd_valid_r <= odd_in;
        end
    end
    always @(posedge clk) begin
        if (epi_in && !q_col[0]) begin even_x_r <= q_value; even_ctx_r <= q_ctx; end
        if (odd_in) begin odd_x_r <= q_value; odd_partner_r <= even_x_r; odd_ctx_r <= q_ctx; end
    end
    wire [3:0]   e0_epi = e0_ctx[19:16];
    wire         e0_half = e0_ctx[26];
    wire [4:0]   e0_col = e0_ctx[25:21];
    wire         e0_entry = e0_ctx[10];
    wire [255:0] e0_c_row = c_rows[{e0_entry, e0_half, 8'd0} +: 256];
    wire [7:0]   e0_c_scale = c_row_scales[{e0_entry, e0_half, 3'd0} +: 8];
    wire [255:0] e0_g_row = g_rows[{e0_entry, e0_half, 8'd0} +: 256];
    wire [7:0]   e0_g_scale = g_row_scales[{e0_entry, e0_half, 3'd0} +: 8];
    reg          e1_valid_r;
    reg  [31:0]  e1_x_r, e1_partner_r, e1_c_r, e1_cos_r, e1_sin_r, e1_g_r;
    reg  [CW-1:0] e1_ctx_r;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            e1_valid_r <= 1'b0;
            c_row_done <= 2'b00;
        end else begin
            e1_valid_r <= e0_valid;
            c_row_done <= 2'b00;
            if (e0_valid && d_has_c(e0_ctx[19:0]) && e0_col == 5'd31) c_row_done[e0_half] <= 1'b1;
        end
    end
    always @(posedge clk) begin
        e1_x_r <= e0_x;
        e1_partner_r <= e0_partner;
        e1_ctx_r <= e0_ctx;
        e1_c_r <= decode_element(e0_c_row[{e0_col, 3'd0} +: 8], e0_c_scale);
        e1_cos_r <= decode_element(e0_c_row[{e0_col[4:1], 1'b0, 3'd0} +: 8], e0_c_scale);
        e1_sin_r <= decode_element(e0_c_row[{e0_col[4:1], 1'b1, 3'd0} +: 8], e0_c_scale);
        e1_g_r <= decode_element(e0_g_row[{e0_col, 3'd0} +: 8], e0_g_scale);
    end
    wire [3:0]  e1_epi = e1_ctx_r[19:16];
    wire [31:0] e1_act = e1_epi == EPI_SILU_MUL ? t0_rd : e1_x_r;      // gate read at e0
    wire [63:0] e1_coef = e1_epi == EPI_GELU ? gelu64_coeff(gelu64_code(e1_act)) : silu64_coeff(silu64_code(e1_act));
    reg  [31:0] epi_m1_left, epi_m1_right, epi_addend;
    always @* begin
        epi_m1_left = e1_x_r; epi_m1_right = ONE; epi_addend = NEG_ZERO;
        case (e1_epi)
        EPI_ADD:  epi_addend = e1_c_r;
        EPI_MUL:  epi_m1_right = e1_c_r;
        EPI_SILU, EPI_GELU, EPI_SILU_MUL:
                  begin epi_m1_left = e1_act; epi_m1_right = e1_coef[31:0]; epi_addend = e1_coef[63:32]; end
        EPI_ROPE: epi_m1_right = e1_cos_r;
        default: ;
        endcase
    end
    reg  [CW:0] epi_ctx_r [0:10];
    reg  [31:0] epi_add_r [0:3];
    reg  [31:0] epi_up_r [0:6];
    always @(posedge clk or negedge rstn) begin
        if (!rstn) for (i = 0; i < 11; i = i + 1) epi_ctx_r[i] <= 0;
        else begin
            epi_ctx_r[0] <= {e1_valid_r, e1_ctx_r};
            for (i = 1; i < 11; i = i + 1) epi_ctx_r[i] <= epi_ctx_r[i-1];
        end
    end
    always @(posedge clk) begin
        epi_add_r[0] <= epi_addend;
        epi_up_r[0] <= d_fp(e1_ctx_r[19:0]) ? e1_g_r : e1_x_r;   // M2: up (SILU_MUL) or g (fold producer)
        for (i = 1; i < 4; i = i + 1) epi_add_r[i] <= epi_add_r[i-1];
        for (i = 1; i < 7; i = i + 1) epi_up_r[i] <= epi_up_r[i-1];
    end
    wire          e5_valid = epi_ctx_r[3][CW];
    wire          e5_rope = epi_ctx_r[3][19:16] == EPI_ROPE;
    wire          e8_valid = epi_ctx_r[6][CW];
    wire [3:0]    e8_epi = epi_ctx_r[6][19:16];
    wire          e8_fp = d_fp(epi_ctx_r[6][19:0]);
    wire          e12_valid = epi_ctx_r[10][CW];
    wire [CW-1:0] e12_ctx = epi_ctx_r[10][CW-1:0];

    // Shared units M1, A1, M2.
    reg         seq_m1_valid, seq_a1_valid, seq_m2_valid;
    reg  [31:0] seq_m1_left, seq_m1_right, seq_a1_left, seq_a1_right, seq_m2_left, seq_m2_right;
    reg  [7:0]  seq_m1_tag, seq_a1_tag, seq_m2_tag;
    wire        m1_valid, a1_valid, m2_valid;
    wire [31:0] m1_result, a1_result, m2_result;
    wire [7:0]  m1_tag, a1_tag, m2_tag;
    wire        m2_pre = e1_valid_r && e1_epi == EPI_ROPE;
    wire        m2_post = e8_valid && e8_epi != EPI_ROPE;
    rc_mxfp8_fp32_mul_staged #(.TAG_WIDTH(8)) u_m1 (
        .clk (clk), .rst_n (rstn), .advance (1'b1),
        .in_valid (e1_valid_r || seq_m1_valid), .in_tag (e1_valid_r ? {K_EPI, 5'd0} : seq_m1_tag),
        .left (e1_valid_r ? epi_m1_left : seq_m1_left), .right (e1_valid_r ? epi_m1_right : seq_m1_right),
        .out_valid (m1_valid), .out_tag (m1_tag), .result (m1_result));
    rc_mxfp8_fp32_mul_staged #(.TAG_WIDTH(8)) u_m2 (
        .clk (clk), .rst_n (rstn), .advance (1'b1),
        .in_valid (m2_pre || m2_post || seq_m2_valid), .in_tag ((m2_pre || m2_post) ? {K_EPI, 5'd0} : seq_m2_tag),
        .left  (m2_pre ? e1_partner_r : m2_post ? a1_result : seq_m2_left),
        .right (m2_pre ? (e1_ctx_r[21] ? e1_sin_r : neg(e1_sin_r)) :
                m2_post ? ((e8_epi == EPI_SILU_MUL || e8_fp) ? epi_up_r[6] : ONE) : seq_m2_right),
        .out_valid (m2_valid), .out_tag (m2_tag), .result (m2_result));
    rc_mxfp8_fp32_add_staged #(.TAG_WIDTH(8)) u_a1 (
        .clk (clk), .rst_n (rstn), .advance (1'b1),
        .in_valid (e5_valid || seq_a1_valid), .in_tag (e5_valid ? {K_EPI, 5'd0} : seq_a1_tag),
        .left (e5_valid ? m1_result : seq_a1_left),
        .right (e5_valid ? (e5_rope ? m2_result : epi_add_r[3]) : seq_a1_right),
        .out_valid (a1_valid), .out_tag (a1_tag), .result (a1_result));
    reg  [31:0] rope_r [0:3];
    always @(posedge clk) begin
        rope_r[0] <= a1_result;
        for (i = 1; i < 4; i = i + 1) rope_r[i] <= rope_r[i-1];
    end
    // Fold producer: x = A1 (as ROPE's, delayed to e12), x * g = M2.
    wire        e12_fp = d_fp(e12_ctx[19:0]);
    wire [31:0] y = (e12_ctx[19:16] == EPI_ROPE || e12_fp) ? rope_r[3] : m2_result;
    wire        y_half = e12_ctx[26];
    wire [4:0]  y_col = e12_ctx[25:21];
    wire [13:0] y_pre = preround(y);
    wire [13:0] y2_pre = preround(m2_result);
    wire        e12_xg = e12_valid && e12_fp;

    // RMSNorm fold producer: sum of x^2 and FINAL (declarations; the logic follows the encoder).
    reg  [31:0] sacc_r [0:15];                 // 16 partial sums, then FINAL registers 0..6
    reg         sq_v_r, sq_first_r;
    reg  [3:0]  sq_j_r;
    reg  [31:0] sq_x_r;
    reg  [7:0]  rbuf_r [0:31];                 // right half codes of the pass, replayed after it
    reg  [7:0]  rsc_r;
    reg         rp_act_r, fin_pend_r;
    reg  [4:0]  rp_i_r;
    reg  [6:0]  sq_out_r;
    reg         fs_act_r;
    reg  [4:0]  fs_step_r;
    reg  [3:0]  fs_sub_r;
    reg  [2:0]  fs_out_r;
    reg         fs_issue, fs_mul, fs_int;
    reg  [3:0]  fs_dst;
    reg  [31:0] fs_a, fs_b;
    reg  [3:0]  m0_req_d;
    wire        sq_add = m0_valid && m0_tag[CW+1:CW] == M0_SQ;
    wire        fold_busy = sq_v_r || rp_act_r || fin_pend_r || sq_out_r != 0 || fs_act_r || (|m0_req_d);

    // ------------------------------------------------------------------
    // Requant FIFO in T1: half h holds words {h, 5-bit index}, two pre-rounded values per word;
    // tile k of a half starts at rd_base; per-half |max| -> two latched row scales per half.
    // ------------------------------------------------------------------
    reg  [12:0] row_max_r [0:1];
    reg  [13:0] y_even_r [0:1];
    reg  [4:0]  fifo_wr_ptr_r [0:1], rd_base_r [0:1];
    reg  [7:0]  scale_r [0:3];            // {half, entry}
    reg  [1:0]  scale_level_r [0:1];
    reg         scale_rd_ptr_r [0:1];
    wire [12:0] row_max_n = (y_col == 0 || y_pre[12:0] > row_max_r[y_half]) ? y_pre[12:0] : row_max_r[y_half];
    wire        scale_push = e12_valid && y_col == 5'd31;
    assign g_ready = {scale_level_r[1] != 0, scale_level_r[0] != 0};
    assign g_scales = {scale_r[{1'b1, scale_rd_ptr_r[1]}], scale_r[{1'b0, scale_rd_ptr_r[0]}]};
    // Fold producer: x * g in T0, same layout as T1, its own maxima and scales.
    reg  [12:0] row_max2_r [0:1];
    reg  [13:0] y2_even_r [0:1];
    reg  [4:0]  fifo_wr_ptr2_r [0:1], rd_base2_r [0:1];
    reg  [7:0]  scale2_r [0:3];
    reg  [1:0]  scale_level2_r [0:1];
    reg         scale_rd_ptr2_r [0:1];
    wire [12:0] row_max2_n = (y_col == 0 || y2_pre[12:0] > row_max2_r[y_half]) ? y2_pre[12:0] : row_max2_r[y_half];
    wire        scale2_push = e12_xg && y_col == 5'd31;
    assign g_ready2 = {scale_level2_r[1] != 0, scale_level2_r[0] != 0};
    assign g_scales2 = {scale2_r[{1'b1, scale_rd_ptr2_r[1]}], scale2_r[{1'b0, scale_rd_ptr2_r[0]}]};

    // GEMM output schedule (one command per 4 cycles, overlapping): read left (0), right (1);
    // encode 2..5; pop 6 (last column pair, modes 0 and 2); chunk 8.
    reg  [7:1]  g_go_r, g_last_r;
    reg  [3:0]  g_cp_r;
    reg  [1:0]  g_mode_r [1:7];
    reg  [1:0]  g_halves_r [1:7];
    reg  [31:0] g_ext0_r, g_ext_r;         // external scales: latched at go, used at encode 2..5
    reg  [31:0] g_word0_r, g_word1_r;
    reg  [12:0] enc_mag_r, mag_r2;              // magnitude beside the encode pipeline (mode 1)
    reg  [38:0] mag_sr_r;
    wire        g_rd0 = g_go, g_rd1 = g_go_r[1];
    wire        g_pop = g_go_r[6] && g_last_r[6] && g_mode_r[6] != 2'd1;
    reg  [7:1]  g_out_d;                       // the go's output: 0 x (T1), 1 x * g (T0)
    reg  [3:0]  g_cp_d [1:5];
    reg         g_fold_r, g_pf_r, g_pl_r;      // fold producer pass, first / last of its row pass
    reg  [31:0] f_inv_r;
    reg  [1:0]  f_eps_r;
    wire        g_pop1 = g_pop && !g_out_d[6], g_pop0 = g_pop && g_out_d[6];
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            for (i = 0; i < 2; i = i + 1) begin
                row_max_r[i] <= 0; fifo_wr_ptr_r[i] <= 0; rd_base_r[i] <= 0;
                scale_level_r[i] <= 0; scale_rd_ptr_r[i] <= 1'b0; tiles_r[i] <= 0;
                row_max2_r[i] <= 0; fifo_wr_ptr2_r[i] <= 0; rd_base2_r[i] <= 0;
                scale_level2_r[i] <= 0; scale_rd_ptr2_r[i] <= 1'b0;
            end
            g_out_d <= 0;
            g_go_r <= 0;
            g_last_r <= 0;
        end else begin
            g_go_r <= {g_go_r[6:1], g_go};
            g_last_r <= {g_last_r[6:1], g_last};
            g_out_d <= {g_out_d[6:1], g_out};
            if (e12_valid) row_max_r[y_half] <= row_max_n;
            if (e12_valid && y_col[0]) fifo_wr_ptr_r[y_half] <= fifo_wr_ptr_r[y_half] + 1'b1;
            if (e12_xg) row_max2_r[y_half] <= row_max2_n;
            if (e12_xg && y_col[0]) fifo_wr_ptr2_r[y_half] <= fifo_wr_ptr2_r[y_half] + 1'b1;
            for (i = 0; i < 2; i = i + 1) begin
                scale_level_r[i] <= scale_level_r[i] + (scale_push && y_half == i) - (g_pop1 && g_halves_r[6][i]);
                if (g_pop1 && g_halves_r[6][i]) begin
                    scale_rd_ptr_r[i] <= !scale_rd_ptr_r[i];
                    rd_base_r[i] <= rd_base_r[i] + 5'd16;
                end
                scale_level2_r[i] <= scale_level2_r[i] + (scale2_push && y_half == i) - (g_pop0 && g_halves_r[6][i]);
                if (g_pop0 && g_halves_r[6][i]) begin
                    scale_rd_ptr2_r[i] <= !scale_rd_ptr2_r[i];
                    rd_base2_r[i] <= rd_base2_r[i] + 5'd16;
                end
                // a fold producer's tile leaves both FIFOs at its x * g pop
                tiles_r[i] <= tiles_r[i] + (take && !step_value && entry_epilogue && entry_col == 0 && step_half == i)
                                         - ((g_pop1 && !g_fold_r || g_pop0) && g_halves_r[6][i]);
            end
        end
    end
    always @(posedge clk) begin
        if (g_go) begin g_cp_r <= g_cp; g_mode_r[1] <= g_mode; g_halves_r[1] <= g_halves; g_ext0_r <= g_ext_scales; end
        if (g_go) begin g_fold_r <= g_fold; g_pf_r <= g_pfirst; g_pl_r <= g_plast; f_inv_r <= f_inv; f_eps_r <= f_eps; end
        g_cp_d[1] <= g_cp;
        for (i = 2; i < 6; i = i + 1) g_cp_d[i] <= g_cp_d[i-1];
        if (e12_xg && !y_col[0]) y2_even_r[y_half] <= y2_pre;
        if (scale2_push)
            scale2_r[{y_half, scale_rd_ptr2_r[y_half] ^ scale_level2_r[y_half][0]}] <= choose_scale(expand({1'b0, row_max2_n}));
        if (g_go_r[1]) g_ext_r <= g_ext0_r;
        for (i = 2; i < 8; i = i + 1) begin g_mode_r[i] <= g_mode_r[i-1]; g_halves_r[i] <= g_halves_r[i-1]; end
        if (e12_valid && !y_col[0]) y_even_r[y_half] <= y_pre;
        if (scale_push)
            scale_r[{y_half, scale_rd_ptr_r[y_half] ^ scale_level_r[y_half][0]}] <= choose_scale(expand({1'b0, row_max_n}));
        if (g_go_r[1]) g_word0_r <= g_out_d[1] ? t0_rd : t1_rd;
        if (g_go_r[2]) g_word1_r <= g_out_d[2] ? t0_rd : t1_rd;
    end
    reg epi_busy;
    always @* begin
        epi_busy = even_valid_r || odd_valid_r || e1_valid_r;
        for (i = 0; i < 11; i = i + 1) epi_busy = epi_busy | epi_ctx_r[i][CW];
    end
    assign g_busy = tiles_r[0] != 0 || tiles_r[1] != 0 || epi_busy || g_go || (|g_go_r) ||
                    scale_level_r[0] != 0 || scale_level_r[1] != 0 || scale_level2_r[0] != 0 ||
                    scale_level2_r[1] != 0 || fold_busy;

    // ------------------------------------------------------------------
    // Softmax schedule (s_count_r after s_start):
    //   S = scale x log2 e at 0; m_j = max(left max, right max) x S at 5 (result 9);
    //   k_j, g at 10; c_j, shift_j at 11; FP32 words, P scale at 12; exported at 16
    //   element n = 0..63 (half n[1], column {n[5:2], n[0]}): T read 18+n, M1 19+n, A1 23+n, P code,
    //   chunk 37+4cp; pair-first sums of the decoded P per half on A2
    //   A2 row sum = left + right at 101; M2 sum x c_j 2^shift at 105; A2 l at 110; done at 113
    // ------------------------------------------------------------------
    reg         s_act_r;
    reg  [6:0]  s_count_r;
    reg         s_par_r, s_slot_r, s_cst_r, s_vis_r;
    reg  [31:0] s_scale_r;
    reg  [31:0] big_s_r, scaled_max_r, new_m_r, c_r, c_shift_r, l_prod_r, tot_r;
    reg  [31:0] row_sum_r [0:1], p_even_r [0:1];
    reg  [7:0]  p_scale_r;
    reg         rs_valid_r;
    reg  [28:0] rs_kg_r;
    reg  [7:0]  delta_r, cs_delta_r;
    reg         cs_valid_r;
    reg  [8:0]  cs_c_r;
    reg  [16:0] cs_shift_r;
    wire [15:0] rs_k = rs_kg_r[28:13];
    wire        rs_have = !s_cst_r && have_r[s_slot_r];
    wire [15:0] rs_ref = ref_r[s_slot_r];
    wire [15:0] rs_new_ref = (rs_have && $signed(rs_ref) > $signed(rs_k)) ? rs_ref : rs_k;
    wire [16:0] rs_grow = rs_have ? $signed({rs_new_ref[15], rs_new_ref}) - $signed({rs_ref[15], rs_ref}) : 17'd0;
    wire [7:0]  rs_delta = rs_grow > 17'd255 ? 8'd255 : rs_grow[7:0];
    wire [6:0]  s_elem = s_count_r - 7'd18;
    wire        s_t_rd = s_act_r && s_count_r >= 7'd18 && s_count_r < 7'd82;
    reg         t_rd_valid_r;
    reg  [1:0]  t_rd_elem_r;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            t_rd_valid_r <= 1'b0;
            t_rd_elem_r <= 0;
        end else begin
            t_rd_valid_r <= s_t_rd;
            t_rd_elem_r <= s_elem[1:0];
        end
    end
    wire [31:0] t_data = s_par_r ? t1_rd : t0_rd;
    wire        a1_score = a1_valid && a1_tag[7:5] == K_SCORE;
    reg  [4:0]  pu_valid_r;
    reg  [7:0]  pu_tag_r [0:4];
    reg  [31:0] pu_d_r;
    reg  [7:0]  pu_code_r [1:4];
    always @(posedge clk or negedge rstn) begin
        if (!rstn) pu_valid_r <= 5'd0;
        else pu_valid_r <= {pu_valid_r[3:0], a1_score};
    end
    always @(posedge clk) begin
        pu_d_r <= a1_result;
        pu_tag_r[0] <= a1_tag;
        pu_code_r[1] <= p_code(pu_d_r);
        for (i = 1; i < 5; i = i + 1) pu_tag_r[i] <= pu_tag_r[i-1];
        for (i = 2; i < 5; i = i + 1) pu_code_r[i] <= pu_code_r[i-1];
    end
    wire        exp_score = pu_valid_r[4];
    wire [7:0]  exp_tag = pu_tag_r[4];
    wire        exp_half = exp_tag[4];
    wire [7:0]  p_code_vis = (s_vis_r && !exp_tag[1]) ? pu_code_r[4] : 8'd0;
    wire [31:0] p_value = decode_element(p_code_vis, P_SCALE);
    reg         a2_in_valid;
    reg  [31:0] a2_in_left, a2_in_right;
    reg  [7:0]  a2_in_tag;
    wire        a2_valid;
    wire [31:0] a2_result;
    wire [7:0]  a2_tag;
    rc_mxfp8_fp32_add_staged #(.TAG_WIDTH(8)) u_a2 (
        .clk (clk), .rst_n (rstn), .advance (1'b1), .in_valid (a2_in_valid), .in_tag (a2_in_tag),
        .left (a2_in_left), .right (a2_in_right), .out_valid (a2_valid), .out_tag (a2_tag), .result (a2_result));
    always @* begin
        a2_in_valid = 1'b0; a2_in_left = 32'd0; a2_in_right = 32'd0; a2_in_tag = 8'd0;
        if (exp_score && exp_tag[0]) begin                     // odd column: pair add
            a2_in_valid = 1'b1; a2_in_left = p_even_r[exp_half]; a2_in_right = p_value;
            a2_in_tag = {K2_PAIR, exp_half, 4'd0};
        end else if (a2_valid && a2_tag[7:5] == K2_PAIR) begin  // add the pair to the half sum
            a2_in_valid = 1'b1; a2_in_left = row_sum_r[a2_tag[4]]; a2_in_right = a2_result;
            a2_in_tag = {K2_SUM, a2_tag[4], 4'd0};
        end else if (s_act_r && s_count_r == 7'd101) begin      // row sum (exact)
            a2_in_valid = 1'b1; a2_in_left = row_sum_r[0]; a2_in_right = row_sum_r[1];
            a2_in_tag = {K2_TOT, 5'd0};
        end else if (s_act_r && s_count_r == 7'd110) begin
            a2_in_valid = 1'b1; a2_in_left = s_cst_r ? 32'd0 : scale_down(l_r[s_slot_r], delta_r);
            a2_in_right = l_prod_r; a2_in_tag = {K2_L, 5'd0};
        end else if (sq_add) begin                                  // fold: partial sum + x^2
            a2_in_valid = 1'b1; a2_in_left = m0_tag[4] ? 32'd0 : sacc_r[m0_tag[3:0]]; a2_in_right = m0_result;
            a2_in_tag = {K2_SQ, 1'b0, m0_tag[3:0]};
        end else if (fs_issue && !fs_mul && !fs_int) begin           // fold: FINAL addition
            a2_in_valid = 1'b1; a2_in_left = fs_a; a2_in_right = fs_b; a2_in_tag = {K2_FIN, 1'b0, fs_dst};
        end
    end

    // ------------------------------------------------------------------
    // O output schedules: row scale (after o_start: M2 at 0, result 4) and elements (after o_go:
    // O read 0-3, M2 1-4, encode 5-8, chunk 11; one command per 4 cycles).
    // ------------------------------------------------------------------
    reg         o_scale_act_r, o_slot_r, o_go_act_r, o_big_r;
    reg  [2:0]  o_scale_count_r;
    reg  [3:0]  o_go_count_r, o_cp_r;
    reg  [1:0]  o_group_r;
    reg  [31:0] o_inv_r;
    wire        o_rd = o_go_act_r && o_go_count_r < 4'd4;
    wire        o_rd_half = o_go_count_r[1];
    wire        o_rd_bank = o_big_r ? o_group_r[1] : o_slot_r;
    assign      o_rd_pair = o_rd && !o_go_count_r[0];
    reg         o_rd_valid_r, o_rd_half_r, o_rd_odd_r;
    reg  [31:0] o_hi_r;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            o_rd_valid_r <= 1'b0;
            o_rd_half_r <= 1'b0;
            o_rd_odd_r <= 1'b0;
        end else begin
            o_rd_valid_r <= o_rd;
            o_rd_half_r <= o_rd_half;
            o_rd_odd_r <= o_go_count_r[0];
        end
    end
    always @(posedge clk) if (o_rd_valid_r && !o_rd_odd_r) o_hi_r <= a_rd[63:32];
    wire [31:0] o_rd_value = o_rd_odd_r ? o_hi_r : a_rd[31:0];
    wire [31:0] o_max_row = max_value(o_max_r[{o_slot_r, 1'b0}], o_max_r[{o_slot_r, 1'b1}], 1'b0);

    // Shared-unit requests of the lockstep schedules (mutually exclusive by construction).
    always @* begin
        seq_m1_valid = 1'b0; seq_m1_left = 32'd0; seq_m1_right = 32'd0; seq_m1_tag = 8'd0;
        seq_a1_valid = 1'b0; seq_a1_left = 32'd0; seq_a1_right = 32'd0; seq_a1_tag = 8'd0;
        seq_m2_valid = 1'b0; seq_m2_left = 32'd0; seq_m2_right = 32'd0; seq_m2_tag = 8'd0;
        if (s_act_r && s_count_r == 7'd0) begin
            seq_m1_valid = 1'b1; seq_m1_left = s_scale_r; seq_m1_right = LOG2E; seq_m1_tag = {K_SCALE, 5'd0};
        end
        if (s_act_r && s_count_r == 7'd5) begin
            seq_m1_valid = 1'b1;
            seq_m1_left = max_value(raw_max_r[{s_par_r, 1'b0}], raw_max_r[{s_par_r, 1'b1}], 1'b1);
            seq_m1_right = big_s_r; seq_m1_tag = {K_SMAX, 5'd0};
        end
        if (t_rd_valid_r) begin
            seq_m1_valid = 1'b1; seq_m1_left = t_data; seq_m1_right = big_s_r;
            seq_m1_tag = {K_SCORE, t_rd_elem_r[1], 2'd0, t_data == NEG_INF, t_rd_elem_r[0]};
        end
        if (m1_valid && m1_tag[7:5] == K_SCORE) begin
            seq_a1_valid = 1'b1; seq_a1_left = m1_result; seq_a1_right = neg(new_m_r); seq_a1_tag = m1_tag;
        end
        if (s_act_r && s_count_r == 7'd105) begin
            seq_m2_valid = 1'b1; seq_m2_left = tot_r; seq_m2_right = c_shift_r; seq_m2_tag = {K_LPROD, 5'd0};
        end
        if (o_scale_act_r && o_scale_count_r == 3'd0) begin
            seq_m2_valid = 1'b1; seq_m2_left = o_max_row; seq_m2_right = o_inv_r; seq_m2_tag = {K_OSCALE, 5'd0};
        end
        if (o_rd_valid_r) begin
            seq_m2_valid = 1'b1; seq_m2_left = o_rd_value; seq_m2_right = o_inv_r;
            seq_m2_tag = {K_OELEM, o_rd_half_r, 4'd0};
        end
    end

    // Encoder: registered request, registered code (two cycles).
    reg         enc_req_valid, enc_req_direct;
    reg  [31:0] enc_req_value;
    reg  [7:0]  enc_req_scale;
    reg         enc_valid_r, code_valid_r, enc_direct_r;
    reg  [7:0]  enc_dcode_r;
    reg  [31:0] enc_value_r;
    reg  [7:0]  enc_scale_r, code_r;
    reg  [23:0] chunk_r;
    reg  [1:0]  chunk_count_r;
    reg  [6:0]  enc_info_r, code_info_r;       // {x code of a fold producer, k, column pair}
    reg  [7:0]  code_scale_r;
    wire [15:0] enc_code;
    rc_mxfp8_requant2 u_encode (
        .accumulator_data ({32'd0, enc_value_r}), .row_scale (enc_scale_r), .row_valid (1'b1),
        .column_valid (2'b01), .quantized_data (enc_code));
    wire        m2_oelem = m2_valid && m2_tag[7:5] == K_OELEM;
    wire [31:0] o_norm = m2_result[30:0] == 31'd0 ? 32'd0 : m2_result;
    // GEMM output value k (0 left even, 1 left odd, 2 right even, 3 right odd).
    wire [1:0]  g_k = g_go_r[2] ? 2'd0 : g_go_r[3] ? 2'd1 : g_go_r[4] ? 2'd2 : 2'd3;
    wire [13:0] g_val = g_k == 0 ? g_word0_r[13:0] : g_k == 1 ? g_word0_r[27:14] :
                        g_k == 2 ? g_word1_r[13:0] : g_word1_r[27:14];
    wire [1:0]  g_kmode = g_go_r[2] ? g_mode_r[2] : g_go_r[3] ? g_mode_r[3] : g_go_r[4] ? g_mode_r[4] : g_mode_r[5];
    wire        g_kout = g_go_r[2] ? g_out_d[2] : g_go_r[3] ? g_out_d[3] : g_go_r[4] ? g_out_d[4] : g_out_d[5];
    wire [3:0]  g_kcp = g_go_r[2] ? g_cp_d[2] : g_go_r[3] ? g_cp_d[3] : g_go_r[4] ? g_cp_d[4] : g_cp_d[5];
    always @* begin
        enc_req_valid = 1'b0; enc_req_value = 32'd0; enc_req_scale = 8'd0; enc_req_direct = 1'b0;
        if (|g_go_r[5:2]) begin
            enc_req_valid = 1'b1;
            enc_req_value = expand(g_val);
            enc_req_scale = g_kmode == 2'd2 ? g_ext_r[g_k*8 +: 8] :
                            g_kout ? scale2_r[{g_k[1], scale_rd_ptr2_r[g_k[1]]}] : scale_r[{g_k[1], scale_rd_ptr_r[g_k[1]]}];
        end
        if (exp_score) begin
            enc_req_valid = 1'b1; enc_req_direct = 1'b1;
        end
        if (m2_oelem) begin
            enc_req_valid = 1'b1; enc_req_value = expand(preround(o_norm)); enc_req_scale = o_scale;
        end
    end
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            enc_valid_r <= 1'b0;
            code_valid_r <= 1'b0;
            chunk_count_r <= 0;
        end else begin
            enc_valid_r <= enc_req_valid;
            code_valid_r <= enc_valid_r;
            if (code_valid_r) chunk_count_r <= chunk_count_r + 1'b1;
        end
    end
    always @(posedge clk) begin
        enc_value_r <= enc_req_value;
        enc_scale_r <= enc_req_scale;
        enc_direct_r <= enc_req_direct;
        enc_dcode_r <= p_code_vis;
        enc_info_r <= {(|g_go_r[5:2]) && g_fold_r && !g_kout && g_kmode == 2'd0, g_k, g_kcp};
        code_info_r <= enc_info_r;
        code_scale_r <= enc_scale_r;
        code_r <= enc_direct_r ? enc_dcode_r : enc_code[7:0];
        if (code_valid_r) chunk_r <= {code_r, chunk_r[23:8]};
        enc_mag_r <= g_val[12:0];
        mag_r2 <= enc_mag_r;
        if (code_valid_r) mag_sr_r <= {mag_r2, mag_sr_r[38:13]};
    end
    wire        chunk_done = code_valid_r && chunk_count_r == 2'd3;
    wire [31:0] chunk_word = {code_r, chunk_r};

    // ------------------------------------------------------------------
    // RMSNorm fold producer: sum of x^2 over the x codes in g3_vru's order (column k of the row into
    // partial sum k mod 16, k increasing; a pass is 64 columns: the left half's 32 codes are added as they
    // are encoded, the right half's are kept and replayed after the pass), x^2 on M0, + on A2; the first
    // 16 columns of the row pass start from +0. After the row pass's last pass: g3_vru's FINAL program
    // (pairwise sums, x 1/H, + eps, 5f375a86 - (v >> 1), two Newton steps), in place on the 16 registers.
    // M0 also scales a consumer's GEMM sums by its 1/rms.
    // ------------------------------------------------------------------
    wire        cap_v = code_valid_r && code_info_r[6];
    wire        cap_right = code_info_r[5];
    wire [4:0]  cap_col = {code_info_r[3:0], code_info_r[4]};
    wire        cap_left = cap_v && !cap_right;
    wire [3:0]  fs_ops = fs_step_r == 5'd0 ? 4'd8 : fs_step_r == 5'd1 ? 4'd4 : fs_step_r == 5'd2 ? 4'd2 : 4'd1;
    wire        fs_start = fin_pend_r && sq_out_r == 0 && !sq_v_r && !rp_act_r;
    wire        fs_step_end = fs_act_r && fs_sub_r == fs_ops && fs_out_r == 0;
    always @* begin : fs_decode
        fs_mul = 1'b0; fs_int = 1'b0; fs_a = 32'd0; fs_b = 32'd0; fs_dst = 4'd0;
        case (fs_step_r)
        5'd0, 5'd1, 5'd2, 5'd3: begin
            fs_a = sacc_r[{fs_sub_r[2:0], 1'b0}]; fs_b = sacc_r[{fs_sub_r[2:0], 1'b1}]; fs_dst = fs_sub_r;
        end
        5'd4: begin fs_mul = 1'b1; fs_a = sacc_r[0]; fs_b = f_inv_r; fs_dst = 4'd1; end
        5'd5: begin fs_a = sacc_r[1]; fs_b = eps_value(f_eps_r); fs_dst = 4'd0; end
        5'd6: begin fs_int = 1'b1; fs_dst = 4'd1; end
        5'd7, 5'd13:  begin fs_mul = 1'b1; fs_a = sacc_r[1]; fs_b = sacc_r[1]; fs_dst = 4'd2; end
        5'd8, 5'd14:  begin fs_mul = 1'b1; fs_a = sacc_r[0]; fs_b = sacc_r[2]; fs_dst = 4'd3; end
        5'd9, 5'd15:  begin fs_mul = 1'b1; fs_a = sacc_r[3]; fs_b = C_HALF; fs_dst = 4'd4; end
        5'd10, 5'd16: begin fs_mul = 1'b1; fs_a = sacc_r[4]; fs_b = C_NEG1; fs_dst = 4'd5; end
        5'd11, 5'd17: begin fs_a = sacc_r[5]; fs_b = C_THREEHALF; fs_dst = 4'd6; end
        default:      begin fs_mul = 1'b1; fs_a = sacc_r[1]; fs_b = sacc_r[6]; fs_dst = 4'd1; end   // 12, 18
        endcase
        fs_issue = fs_act_r && fs_sub_r < fs_ops;
    end
    // M0 requests (GEMM): consumer sums x 1/rms, producer squares, FINAL multiplies.
    always @* begin
        m0_req = 1'b0; m0_req_l = 32'd0; m0_req_r = 32'd0; m0_req_tag = 0;
        if (sum_fc) begin
            m0_req = 1'b1; m0_req_l = sum_value; m0_req_r = r_row; m0_req_tag = {M0_SCALE, sum_ctx};
        end else if (sq_v_r) begin
            m0_req = 1'b1; m0_req_l = sq_x_r; m0_req_r = sq_x_r;
            m0_req_tag = {M0_SQ, {(CW-5){1'b0}}, sq_first_r, sq_j_r};
        end else if (fs_issue && fs_mul) begin
            m0_req = 1'b1; m0_req_l = fs_a; m0_req_r = fs_b; m0_req_tag = {M0_FIN, {(CW-4){1'b0}}, fs_dst};
        end
    end
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            sq_v_r <= 1'b0; rp_act_r <= 1'b0; fin_pend_r <= 1'b0; sq_out_r <= 0; fs_act_r <= 1'b0;
            fs_out_r <= 0; rms_valid <= 1'b0; m0_req_d <= 0;
        end else begin
            rms_valid <= 1'b0;
            m0_req_d <= {m0_req_d[2:0], m0_req};
            sq_v_r <= cap_left || rp_act_r;
            if (cap_v && cap_right && cap_col == 5'd31) begin rp_act_r <= 1'b1; rp_i_r <= 5'd0; end
            else if (rp_act_r) begin
                rp_i_r <= rp_i_r + 1'b1;
                if (rp_i_r == 5'd31) begin rp_act_r <= 1'b0; if (g_pl_r) fin_pend_r <= 1'b1; end
            end
            sq_out_r <= sq_out_r + (sq_v_r ? 7'd1 : 7'd0) - ((a2_valid && a2_tag[7:5] == K2_SQ) ? 7'd1 : 7'd0);
            if (fs_start) begin fs_act_r <= 1'b1; fin_pend_r <= 1'b0; fs_step_r <= 5'd0; fs_sub_r <= 4'd0; end
            else if (fs_issue) fs_sub_r <= fs_sub_r + 1'b1;
            else if (fs_step_end) begin
                fs_sub_r <= 4'd0;
                if (fs_step_r == 5'd18) begin fs_act_r <= 1'b0; rms_valid <= 1'b1; end
                else fs_step_r <= fs_step_r + 1'b1;
            end
            fs_out_r <= fs_out_r + ((fs_issue && !fs_int) ? 3'd1 : 3'd0) -
                        (((m0_valid && m0_tag[CW+1:CW] == M0_FIN) || (a2_valid && a2_tag[7:5] == K2_FIN)) ? 3'd1 : 3'd0);
        end
    end
    always @(posedge clk) begin
        if (a2_valid && (a2_tag[7:5] == K2_SQ || a2_tag[7:5] == K2_FIN)) sacc_r[a2_tag[3:0]] <= a2_result;
        if (m0_valid && m0_tag[CW+1:CW] == M0_FIN) sacc_r[m0_tag[3:0]] <= m0_result;
        if (fs_issue && fs_int) sacc_r[1] <= 32'h5f37_5a86 - {1'b0, sacc_r[0][31:1]};
        if (cap_left) begin
            sq_x_r <= decode_element(code_r, code_scale_r); sq_j_r <= cap_col[3:0]; sq_first_r <= g_pf_r && !cap_col[4];
        end else if (rp_act_r) begin
            sq_x_r <= decode_element(rbuf_r[rp_i_r], rsc_r); sq_j_r <= rp_i_r[3:0]; sq_first_r <= 1'b0;
        end
        if (cap_v && cap_right) begin rbuf_r[cap_col] <= code_r; rsc_r <= code_scale_r; end
        if (fs_step_end && fs_step_r == 5'd18) rms <= sacc_r[1];
    end

    // ------------------------------------------------------------------
    // Port control: ACC, T, O.
    // ------------------------------------------------------------------
    wire [5:0] s_t_addr = {s_elem[1], s_elem[5:2], s_elem[0]};
    reg  [31:0] sum_lo_r;
    always @(posedge clk) if (sum_valid && !sum_col[0]) sum_lo_r <= sum_value;
    always @* begin
        a_ren = (take && !step_value && entry_reads) || o_rd_pair;
        a_ra = o_rd_pair ? o_word(o_rd_bank, o_rd_half, o_group_r[0], o_cp_r) :
               entry_pv  ? o_word(entry_obank, step_half, entry_group[0], entry_col[4:1]) :
                           acc_word(step_half, entry_col[4:1]);
        a_wen = sum_valid && sum_col[0] && (sum_op == OP_PV || !sum_end);
        a_wa = sum_op == OP_PV ? o_word(sum_obank, sum_half, sum_group[0], sum_col[4:1])
                               : acc_word(sum_half, sum_col[4:1]);
        a_wd = {sum_value, sum_lo_r};
        t_wen = 2'b00; t_ren = 2'b00;
        t0_wa = {sum_half, sum_col}; t0_wd = sum_out;
        t1_wa = {sum_half, sum_col}; t1_wd = sum_out;
        t0_ra = {1'b0, e0_col};
        t1_ra = g_rd1 ? {1'b1, rd_base_r[1] + {1'b0, g_cp_r}} : {1'b0, rd_base_r[0] + {1'b0, g_cp}};
        if (sum_qk_end) t_wen[d_tpar(sum_desc)] = 1'b1;
        if (sum_gate) begin t_wen[0] = 1'b1; t0_wa = {1'b0, q_col}; t0_wd = q_value; end
        if (e12_valid && y_col[0]) begin
            t_wen[1] = 1'b1; t1_wa = {y_half, fifo_wr_ptr_r[y_half]}; t1_wd = {4'd0, y_pre, y_even_r[y_half]};
        end
        if (e0_valid && e0_epi == EPI_SILU_MUL) t_ren[0] = 1'b1;
        if ((g_rd0 && !g_out) || (g_rd1 && !g_out_d[1])) t_ren[1] = 1'b1;
        if (e12_xg && y_col[0]) begin
            t_wen[0] = 1'b1; t0_wa = {y_half, fifo_wr_ptr2_r[y_half]}; t0_wd = {4'd0, y2_pre, y2_even_r[y_half]};
        end
        if (g_rd0 && g_out) begin t_ren[0] = 1'b1; t0_ra = {1'b0, rd_base2_r[0] + {1'b0, g_cp}}; end
        if (g_rd1 && g_out_d[1]) begin t_ren[0] = 1'b1; t0_ra = {1'b1, rd_base2_r[1] + {1'b0, g_cp_r}}; end
        if (s_t_rd) begin
            t_ren[s_par_r] = 1'b1;
            if (s_par_r) t1_ra = s_t_addr; else t0_ra = s_t_addr;
        end
    end

    // ------------------------------------------------------------------
    // Sequencers and row state.
    // ------------------------------------------------------------------
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            s_act_r <= 1'b0; s_count_r <= 0; s_par_valid <= 1'b0; s_chunk_valid <= 1'b0; s_done <= 1'b0;
            g_chunk_valid <= 1'b0; o_chunk_valid <= 1'b0; pv_end_row <= 2'b00;
            o_scale_act_r <= 1'b0; o_scale_count_r <= 0; o_go_act_r <= 1'b0; o_go_count_r <= 0;
            t_full_r[0] <= 2'b00; t_full_r[1] <= 2'b00;
            l_r[0] <= 32'd0; l_r[1] <= 32'd0;
            have_r <= 2'd0;
        end else begin
            s_par_valid <= 1'b0; s_chunk_valid <= 1'b0; s_done <= 1'b0;
            g_chunk_valid <= 1'b0; o_chunk_valid <= 1'b0; pv_end_row <= 2'b00;
            if (sum_qk_end && sum_col == 5'd31) t_full_r[d_tpar(sum_desc)][sum_half] <= 1'b1;
            if (sum_pv && sum_end && d_lastkb(sum_desc) && d_lastc(sum_desc) && sum_col == 5'd31) pv_end_row[sum_half] <= 1'b1;
            if (s_start) begin s_act_r <= 1'b1; s_count_r <= 0; end
            else if (s_act_r) begin
                s_count_r <= s_count_r + 1'b1;
                if (s_count_r == 7'd113) begin s_act_r <= 1'b0; s_done <= 1'b1; t_full_r[s_par_r] <= 2'b00; end
            end
            if (s_act_r && s_count_r == 7'd16) s_par_valid <= 1'b1;
            if (rs_valid_r && s_vis_r) begin
                ref_r[s_slot_r] <= rs_new_ref;
                have_r[s_slot_r] <= 1'b1;
            end else if (rs_valid_r && s_cst_r)
                have_r[s_slot_r] <= 1'b0;
            if (a2_valid && a2_tag[7:5] == K2_L) l_r[s_slot_r] <= a2_result;
            if (chunk_done && s_act_r) s_chunk_valid <= 1'b1;
            if (chunk_done && g_go_r[7]) g_chunk_valid <= 1'b1;
            if (chunk_done && !s_act_r && !g_go_r[7]) o_chunk_valid <= 1'b1;
            if (o_start) begin o_scale_act_r <= 1'b1; o_scale_count_r <= 0; end
            else if (o_scale_act_r) begin
                o_scale_count_r <= o_scale_count_r + 1'b1;
                if (o_scale_count_r == 3'd6) o_scale_act_r <= 1'b0;
            end
            if (o_go) begin o_go_act_r <= 1'b1; o_go_count_r <= 0; end
            else if (o_go_act_r) begin
                o_go_count_r <= o_go_count_r + 1'b1;
                if (o_go_count_r == 4'd11) o_go_act_r <= 1'b0;
            end
            // PV: max|O| per half over the Dh passes, on the last key block.
            if (sum_pv && sum_end && d_lastkb(sum_desc))
                o_max_r[{d_slot(sum_desc), sum_half}] <= (sum_group == 0 && sum_col == 0) ? max_value(sum_value, sum_value, 1'b0)
                                                     : max_value(o_max_r[{d_slot(sum_desc), sum_half}], sum_value, 1'b0);
            if (sum_qk_end) qk_max_r[sum_half] <= qk_max_n;
            if (sum_qk_end && sum_col == 5'd31) raw_max_r[{d_tpar(sum_desc), sum_half}] <= qk_max_n;
        end
    end
    always @(posedge clk) begin
        if (s_start) begin
            s_par_r <= s_tpar; s_slot_r <= s_slot; s_cst_r <= s_cstart; s_vis_r <= s_vis; s_scale_r <= s_scale;
            row_sum_r[0] <= 32'd0; row_sum_r[1] <= 32'd0;
        end
        if (m1_valid && m1_tag[7:5] == K_SCALE) big_s_r <= m1_result;
        if (m1_valid && m1_tag[7:5] == K_SMAX) scaled_max_r <= m1_result;
        rs_valid_r <= s_act_r && s_count_r == 7'd10;
        if (s_act_r && s_count_r == 7'd10) begin
            rs_kg_r <= ceil_frac(scaled_max_r);
            new_m_r <= scaled_max_r;
        end
        cs_valid_r <= rs_valid_r;
        if (rs_valid_r) begin
            cs_c_r <= c_from_g(rs_kg_r[12:0]);
            cs_shift_r <= $signed({rs_k[15], rs_k}) - $signed({rs_new_ref[15], rs_new_ref});
            cs_delta_r <= rs_delta;
        end
        if (cs_valid_r) begin : b_correction
            reg [8:0]  c;
            reg signed [16:0] shift;
            integer ps, ce;
            c = cs_c_r;
            shift = cs_shift_r;
            ps = 119 + shift;
            ce = (c == 9'd256 ? 127 : 126) + shift;
            if (!s_vis_r) begin
                c_r <= 32'd0; c_shift_r <= 32'd0; p_scale_r <= 8'd127; delta_r <= 8'd0;
            end else begin
                c_r <= {c == 9'd256 ? 16'h3f80 : {1'b0, 8'd126, c[6:0]}, 8'd0, cs_delta_r};
                delta_r <= cs_delta_r;
                c_shift_r <= ce >= 255 ? 32'h7f80_0000 : ce <= 0 ? 32'd0 :
                             {1'b0, ce[7:0], c == 9'd256 ? 7'd0 : c[6:0], 16'd0};
                p_scale_r <= ps < 0 ? 8'd0 : ps > 254 ? 8'd254 : ps[7:0];
            end
        end
        if (s_act_r && s_count_r == 7'd16) begin
            s_alpha <= c_r;
            s_pscale <= p_scale_r;
        end
        if (a2_valid && a2_tag[7:5] == K2_TOT) tot_r <= a2_result;
        if (m2_valid && m2_tag[7:5] == K_LPROD) l_prod_r <= m2_result;
        if (exp_score && !exp_tag[0]) p_even_r[exp_half] <= p_value;
        if (a2_valid && a2_tag[7:5] == K2_SUM) row_sum_r[a2_tag[4]] <= a2_result;
        if (chunk_done && s_act_r) s_chunk <= chunk_word;
        if (chunk_done && g_go_r[7]) begin
            g_chunk <= chunk_word;
            g_mag <= {mag_r2, mag_sr_r};
        end
        if (chunk_done && !s_act_r && !g_go_r[7]) o_chunk <= chunk_word;
        if (o_start) begin o_slot_r <= o_slot; o_inv_r <= o_inv; end
        if (m2_valid && m2_tag[7:5] == K_OSCALE) o_scale <= choose_scale(m2_result);
        if (o_go) begin o_group_r <= o_lg; o_big_r <= o_dh_big; o_cp_r <= o_cp; end
    end
    assign busy = g_busy || s_act_r || o_scale_act_r || o_go_act_r || took_r || conv_valid || c6_valid ||
                  sum_valid || s1_pipe_r[0][CW+64] || s1_pipe_r[1][CW+64] || s1_pipe_r[2][CW+64] ||
                  enc_valid_r || code_valid_r;

    // synthesis translate_off
    reg sum_valid_r, sum_col_r0;
    always @(posedge clk) begin sum_valid_r <= sum_valid; sum_col_r0 <= sum_col[0]; end
    always @(posedge clk) if (rstn) begin
        if (e1_valid_r && seq_m1_valid) $fatal(1, "G3_LANE_M1_CONFLICT %m");
        if (e5_valid && seq_a1_valid) $fatal(1, "G3_LANE_A1_CONFLICT %m");
        if ((m2_pre || m2_post) && seq_m2_valid) $fatal(1, "G3_LANE_M2_CONFLICT %m");
        if (m2_pre && m2_post) $fatal(1, "G3_LANE_ROPE_M2_CONFLICT %m");
        if (exp_score && exp_tag[0] && a2_valid && a2_tag[7:5] == K2_PAIR) $fatal(1, "G3_LANE_A2_CONFLICT %m");
        if (o_rd_pair && take && !step_value && entry_reads) $fatal(1, "G3_LANE_A_PORT %m");
        if (took_r && !c1_value_r && !(take && step_value)) $fatal(1, "G3_LANE_PAIR_TAKE %m");
        if (sum_valid && sum_col[0] && !(sum_valid_r && !sum_col_r0)) $fatal(1, "G3_LANE_PAIR_SUM %m");
        if (s_t_rd && ((!s_par_r && e0_valid && e0_epi == EPI_SILU_MUL) || (s_par_r && (g_rd0 || g_rd1))))
            $fatal(1, "G3_LANE_T_PORT %m");
        if (sum_gate && sum_qk_end) $fatal(1, "G3_LANE_T0_PORT %m");
        if (sum_fc && (sq_v_r || (fs_issue && fs_mul))) $fatal(1, "G3_LANE_M0_CONFLICT %m");
        if (sum_gemm_end && !sum_fc && q_scaled) $fatal(1, "G3_LANE_SUM_PATH_CONFLICT %m");
        if (sum_fc && conv_valid && conv_pv) $fatal(1, "G3_LANE_M0_PV %m");
        if (e12_xg && y_col[0] && sum_gate) $fatal(1, "G3_LANE_T0_FOLD %m");
        if ((sq_add || (fs_issue && !fs_mul && !fs_int)) &&
            ((exp_score && exp_tag[0]) || (a2_valid && a2_tag[7:5] == K2_PAIR) || s_act_r))
            $fatal(1, "G3_LANE_A2_FOLD_CONFLICT %m");
    end
    // synthesis translate_on
endmodule

`default_nettype wire
