`timescale 1ns/1ps
`default_nettype none

// Gen3 vector column (docs/gen3/README.md s3-s4, gen3_rtl/README.md s3.3): the 32 lanes between
// the array halves, the attention softmax / P buffer / O control, and every reformatting on the
// way to the UB (the write path): row-major P8 / P32 tiles, V^T cells (column scales from a
// 32-way maximum), the single-token V^T rewrite of a partial cell, O output.
//
//   left / right row exports --> holding FIFOs (32 rows x 2 halves x 2) --> 32 x g3_vector_lane
//   tile descriptors (8, by tile): epilogue, C index, T parity, slot, Dh pass, per-row key limit
//   C beats (UB port C, stream 2) --> C tiles (2 entries: 2 halves x 4 panels x 8 words)
//   fold beats (UB port C, stream 1) --> g rows (producer: 2 entries beside the C tiles, 2 halves x 32 codes)
//                                    --> 1/rms (consumer: 2 row passes x 32 rows, FP32)
//   old V^T cells (UB port C, stream 0) --> rewrite: scan (old scales, zero dims), modify
//                                   lockstep chunks: 32 x 32 bit (column pair cp, both halves)
//   +-------------------------------------------+----------------------------------+
//   v                                           v                                  v
//   output engine: GEMM tiles and O passes   softmax -> P buffer (2 banks:     O control: 1/l
//   (gos, chunk buffer of 2 groups, drain:   2 K groups x 16 ticks x 512 b)    (shared recip),
//    words, scale words, marks) --> wb_*     --> PV A source (p_*)             row scales
//
// Control:
//   output engine  E_IDLE --GEMM tile (lanes hold its halves) | O pass--> [E_MAG (V^T: 16
//                  magnitude gos) --> E_MAGW (scan done: new column scales) -->] E_ENC (16 gos;
//                  chunk groups of 2 (rows) or 4 (V^T) gos, double buffered) --> E_DRAIN (words)
//                  --> E_SCALE (scale words) | E_RMW (V^T rewrite: old words requantised, the new
//                  token inserted, new scale word) --> E_FIN --> E_IDLE
//                  RMSNorm fold producer: E_SCALE (x) --> E_XG --> E_ENC, E_DRAIN, E_SCALE (x * g) -->
//                  [last pass of the row pass: E_RMS (the 32 rows' 1/rms, 4 words)] --> E_FIN
//   O control      D_IDLE --O job, last PV in every row--> D_RECIP (32 reciprocals) --> D_SCALE
//                  (row scales) --> D_PASS (output engine pass per O descriptor) --> D_DONE
//   softmax        idle --QK info, T full in every lane, P bank free, O quiet--> busy (114 cycles)
// GEMM and attention never overlap here (g3_core interlocks the two classes).
module g3_vector_column (
    input  wire              clk,
    input  wire              rstn,
    // Array row exports.
    input  wire [31:0]       left_valid,
    input  wire [32*113-1:0] left_data,
    input  wire [31:0]       right_valid,
    input  wire [32*113-1:0] right_data,
    output wire              array_ready,
    // Tile descriptors (uop records of emitting GEMM, QK and PV windows), pushed at plan.
    input  wire              td_valid,
    output wire              td_ready,
    input  wire [255:0]      td_uop,
    // GEMM output descriptors (emitting GEMM uops, in tile order).
    input  wire              desc_valid,
    output reg               desc_ready,
    input  wire [255:0]      desc_uop,
    // O descriptors (op 3 uops, one per Dh pass).
    input  wire              od_valid,
    output reg               od_ready,
    input  wire [255:0]      od_uop,
    // PV windows (pushed at plan): K groups, P release after the window.
    input  wire              pw_valid,
    input  wire [1:0]        pw_kg,
    input  wire              pw_release,
    output wire              p_plan_ok,
    output wire              p_valid,
    output wire [511:0]      p_data,
    output wire [255:0]      p_scale,
    input  wire              p_pop,
    // UB port C lane: stream 2 = C tiles, stream 0 = old V^T cells.
    input  wire              c_beat_valid,
    input  wire [1:0]        c_beat_stream,
    input  wire [1:0]        c_beat_inner_idx,
    input  wire              c_beat_inner_last,
    input  wire              c_beat_seg_last,
    input  wire [7:0]        c_beat_user,
    input  wire [255:0]      c_beat_data,
    output wire [2:0]        c_credit,
    // UB D writes.
    output wire              wb_valid,
    input  wire              wb_ready,
    output wire [11:0]       wb_addr,
    output wire [255:0]      wb_data,
    output wire [31:0]       wb_strb,
    output wire              wb_mark,
    output wire              gemm_active,
    output wire              fa_active,
    output wire              busy
);
    `include "g3_defs.vh"
    `include "g3_numeric_functions.vh"
    localparam [1:0]  OP_GEMM = 0, OP_QK = 1, OP_PV = 2;
    localparam [31:0] ONE = 32'h3f80_0000;
    integer i, j, k;

    function automatic [31:0] qk_scale(input [3:0] groups);
        case (groups)
        4'd1: qk_scale = 32'h3e3504f3; 4'd2: qk_scale = 32'h3e000000;
        4'd3: qk_scale = 32'h3dd105ec; 4'd4: qk_scale = 32'h3db504f3;
        4'd5: qk_scale = 32'h3da1e89b; 4'd6: qk_scale = 32'h3d93cd3a;
        4'd7: qk_scale = 32'h3d88d677; default: qk_scale = 32'h3d800000;
        endcase
    endfunction
    function automatic [15:0] blk(input [1:0] layout, input [11:0] panel, input [8:0] gd, input [8:0] g);
        blk = layout == `G3_LAYOUT_P8 ? 16'(panel) * 16'(gd) + 16'(g)
                                       : 16'(panel >> 2) * 16'(gd) * 16'd4 + 16'(g) * 16'd4 + 16'(panel[1:0]);
    endfunction
    function automatic [11:0] block_word(input [15:0] n);
        block_word = 33 * (n >> 2) + 1 + 8 * n[1:0];
    endfunction
    function automatic [11:0] scale_word(input [15:0] n);
        scale_word = 33 * (n >> 2);
    endfunction
    function automatic [31:0] expand(input [13:0] p);
        expand = {p[13:1], p[0], 18'd0};
    endfunction

    // ------------------------------------------------------------------
    // Holding FIFOs: index rh = 2 * row + half.
    // ------------------------------------------------------------------
    wire [63:0]  fifo_space, head_valid, pop;
    wire [112:0] head [0:63];
    wire [3:0]   hcp [0:63];
    wire [2:0]   htile [0:63];
    for (genvar rh = 0; rh < 64; rh = rh + 1) begin : g_hold
        localparam integer R = rh / 2, H = rh % 2;
        wire in_v = H != 0 ? right_valid[R] : left_valid[R];
        wire [112:0] in_d = H != 0 ? right_data[R*113 +: 113] : left_data[R*113 +: 113];
        reg [112:0] h0_r, h1_r;
        reg [1:0] level_r;
        reg rd_r, wr_r;
        reg [3:0] cp_r;
        reg [2:0] tile_r;
        wire [112:0] hd = rd_r ? h1_r : h0_r;
        assign fifo_space[rh] = level_r != 2'd2;
        assign head_valid[rh] = level_r != 0;
        assign head[rh] = hd;
        assign hcp[rh] = cp_r;
        assign htile[rh] = tile_r;
        always @(posedge clk or negedge rstn) begin
            if (!rstn) begin
                level_r <= 0; rd_r <= 0; wr_r <= 0; cp_r <= 0; tile_r <= 0;
            end else begin
                if (in_v) wr_r <= !wr_r;
                if (pop[rh]) begin
                    rd_r <= !rd_r;
                    cp_r <= cp_r + 1'b1;
                    if (cp_r == 4'd15 && hd[107]) tile_r <= tile_r + 1'b1;
                end
                level_r <= level_r + in_v - pop[rh];
            end
        end
        always @(posedge clk) if (in_v) begin
            if (wr_r) h1_r <= in_d; else h0_r <= in_d;
        end
    end
    assign array_ready = &fifo_space;

    // ------------------------------------------------------------------
    // Tile descriptors: {attn, epi[3:0], gated, has_c, cidx[3:0], tpar, slot, dp[1:0], big, lastdp,
    // lastkb} (18 bits), {pfirst, pbank}, per-row key limits.
    // ------------------------------------------------------------------
    reg  [7:0]   td_live_r;
    reg  [63:0]  td_done_r [0:7];
    reg  [2:0]   td_wr_r;
    reg  [3:0]   c_count_r;
    reg          qk_par_r, p_rd_bank_r;
    reg  [1:0]   first_block_r;
    reg  [17:0]  td_mem [0:7];
    reg  [1:0]   td_pmem [0:7];
    reg  [223:0] td_lim [0:7];
    // O jobs per context slot (the next item may start on a slot before the slot's O job has run):
    // issued (last PV accepted), l read (reciprocals taken), O read (output passes done). A tile
    // records the slot's issued count; its softmax waits until that many O jobs of the slot have
    // read l, its PV accumulation until they have read O.
    reg  [1:0]   oj_issued_r [0:1], oj_lread_r [0:1], oj_oread_r [0:1];
    reg  [1:0]   td_ojobs [0:7];
    localparam integer QW = 2 + 1 + 1 + 32 + 32;  // QK info {O jobs before, slot, cstart, vis[32], scale32}
    reg  [QW-1:0] qk_mem [0:3];
    reg  [1:0]   qk_rd_r, qk_wr_r; reg [2:0] qk_level_r;
    reg          oj_mem [0:1];                    // O jobs {slot}
    reg          oj_rd_r, oj_wr_r; reg [1:0] oj_level_r;
    wire [1:0]   td_op = td_uop[`U_OP];
    wire         td_ojob = td_op == OP_PV && td_uop[`U_LASTKB] && td_uop[`U_LASTDP];
    assign td_ready = !td_live_r[td_wr_r] && (td_op != OP_QK || qk_level_r != 3'd4) && (!td_ojob || oj_level_r != 2'd2);
    wire         td_fire = td_valid && td_ready;
    // Per-row key limit: rows[r] ? clamp(min(width, causal ? token(r) + limb : 64), 0, 64) : 0.
    reg  [223:0] td_lim_new;
    reg  [31:0]  td_vis_new;
    always @* begin : lim_calc
        reg signed [15:0] tok, lim;
        reg [6:0] w;
        w = td_uop[`U_WIDTH];
        for (i = 0; i < 32; i = i + 1) begin
            tok = $signed({5'd0, td_uop[91 + (i/8)*11 +: 11]}) + (i % 8);
            lim = td_uop[`U_CAUSAL] ? tok + $signed(td_uop[`U_LIMB]) : 16'sd64;
            if (lim > $signed({9'd0, w})) lim = $signed({9'd0, w});
            if (lim < 0) lim = 0;
            if (!td_uop[10 + i]) lim = 0;
            td_lim_new[i*7 +: 7] = lim[6:0];
            td_vis_new[i] = lim != 0;
        end
    end
    wire         qk_pop, oj_pop;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            td_live_r <= 0; td_wr_r <= 0; c_count_r <= 0; qk_par_r <= 0; p_rd_bank_r <= 0; first_block_r <= 0;
            qk_rd_r <= 0; qk_wr_r <= 0; qk_level_r <= 0; oj_rd_r <= 0; oj_wr_r <= 0; oj_level_r <= 0;
            for (i = 0; i < 8; i = i + 1) td_done_r[i] <= 0;
            oj_issued_r[0] <= 0; oj_issued_r[1] <= 0;
        end else begin
            if (td_fire) begin
                td_live_r[td_wr_r] <= 1'b1;
                td_wr_r <= td_wr_r + 1'b1;
                if (td_op == OP_GEMM && td_uop[`U_HASC]) c_count_r <= c_count_r + 1'b1;
                if (td_op == OP_QK) begin
                    qk_par_r <= !qk_par_r; qk_wr_r <= qk_wr_r + 1'b1;
                    if (td_uop[`U_CSTART]) first_block_r[td_uop[`U_SLOT]] <= 1'b1;
                end
                if (td_op == OP_PV && td_uop[`U_LASTDP]) begin
                    first_block_r[td_uop[`U_SLOT]] <= 1'b0;
                    p_rd_bank_r <= !p_rd_bank_r;
                end
                if (td_ojob) begin
                    oj_wr_r <= !oj_wr_r;
                    oj_issued_r[td_uop[`U_SLOT]] <= oj_issued_r[td_uop[`U_SLOT]] + 1'b1;
                end
            end
            qk_level_r <= qk_level_r + (td_fire && td_op == OP_QK) - qk_pop;
            if (qk_pop) qk_rd_r <= qk_rd_r + 1'b1;
            oj_level_r <= oj_level_r + (td_fire && td_ojob) - oj_pop;
            if (oj_pop) oj_rd_r <= !oj_rd_r;
            for (i = 0; i < 8; i = i + 1) begin
                if (td_live_r[i] && &td_done_r[i]) begin
                    td_live_r[i] <= 1'b0;
                    td_done_r[i] <= 0;
                end else
                    for (j = 0; j < 64; j = j + 1)
                        if (pop[j] && hcp[j] == 4'd15 && head[j][107] && htile[j] == i[2:0]) td_done_r[i][j] <= 1'b1;
            end
        end
    end
    always @(posedge clk) begin
        if (td_fire) begin
            // GEMM tiles carry the fold fields in the attention-only bits: {fp, fc, pf, pl, rpar, 00}.
            td_mem[td_wr_r] <= {td_op != OP_GEMM, td_uop[`U_EPI], td_uop[`U_EPI] == `G3_EPI_SILU_MUL, td_uop[`U_HASC],
                                c_count_r,
                                td_op == OP_GEMM ? {td_uop[`U_FP], td_uop[`U_FC], td_uop[`U_PF], td_uop[`U_PL],
                                                    td_uop[`U_RPAR], 2'b00}
                                                 : {qk_par_r, td_uop[`U_SLOT], td_uop[`U_DP], td_uop[`U_BIG],
                                                    td_uop[`U_LASTDP], td_uop[`U_LASTKB]}};
            td_pmem[td_wr_r] <= {first_block_r[td_uop[`U_SLOT]], p_rd_bank_r};
            td_lim[td_wr_r] <= td_lim_new;
            td_ojobs[td_wr_r] <= oj_issued_r[td_uop[`U_SLOT]];
        end
        if (td_fire && td_op == OP_QK)
            qk_mem[qk_wr_r] <= {oj_issued_r[td_uop[`U_SLOT]], td_uop[`U_SLOT], td_uop[`U_CSTART], td_vis_new,
                                qk_scale(td_uop[`U_QKG])};
        if (td_fire && td_ojob) oj_mem[oj_wr_r] <= td_uop[`U_SLOT];
    end
    // Lane view of (row, half): descriptor {epi, gate, has_c, cidx, tpar, slot, dp, big, lastdp, lastkb,
    // vis, pfirst, pbank} and the head with its attention keep bits (key < row limit).
    wire [19:0]  hdesc [0:63];
    wire [112:0] hentry [0:63];
    wire [63:0]  howait;                          // the head's PV must wait for an earlier O job of its slot
    for (genvar rh = 0; rh < 64; rh = rh + 1) begin : g_view
        localparam integer R = rh / 2, H = rh % 2;
        wire [17:0] t = td_mem[htile[rh]];
        wire [223:0] lims = td_lim[htile[rh]];
        wire [6:0]  lim = lims[R*7 +: 7];
        wire [1:0]  pm = td_pmem[htile[rh]];
        wire [6:0]  key0 = 7'(32 * H) + {2'd0, hcp[rh], 1'b0};
        wire        attn = t[17];
        assign hdesc[rh] = {t[16:13], t[12], t[11], t[10:7], t[6], t[5], t[4:3], t[2], t[1], t[0], lim != 7'd0, pm};
        assign howait[rh] = oj_oread_r[t[5]] != td_ojobs[htile[rh]];
        assign hentry[rh] = {head[rh][112:106], attn ? (key0 + 7'd1 < lim) : 1'b1, attn ? (key0 < lim) : 1'b1,
                             head[rh][103:0]};
    end

    // ------------------------------------------------------------------
    // C tiles (stream 2) and old V^T cells (stream 0) from UB port C.
    // ------------------------------------------------------------------
    localparam integer BW = 1 + 1 + 2 + 8 + 256;  // {seg_last, inner_last, idx, user, data}
    reg [BW-1:0] craw_mem [0:7];
    reg [2:0] craw_rd_r, craw_wr_r; reg [3:0] craw_level_r;
    reg [BW-1:0] vraw_mem [0:7];
    reg [2:0] vraw_rd_r, vraw_wr_r; reg [3:0] vraw_level_r;
    wire [BW-1:0] craw_head = craw_mem[craw_rd_r];
    wire [BW-1:0] vh = vraw_mem[vraw_rd_r];
    reg  [255:0] c_tile [0:127];                   // {entry, half, panel, word}
    reg  [63:0]  c_scale [0:15];                   // {entry, half, panel}
    reg  [2:0]   c_w_r [0:1];
    reg          c_wr_sel_r;
    reg  [1:0]   c_level_r;
    reg  [3:0]   c_popped_r;
    reg          c_pop_r;
    wire         c_take = craw_level_r != 0 && c_level_r != 2'd2;
    wire         v_take;
    // Stream 1 (RMSNorm fold): g rows of a producer pass (scale beat: row 0 scale at byte 0 of the slot;
    // payload word w: row 0 codes of columns 4w..4w+3 at bytes 0, 1, 16, 17), or a consumer's 1/rms words
    // (user[5] on a payload beat, user[0] = row pass parity; 8 FP32 rows per word).
    reg  [BW-1:0] fraw_mem [0:7];
    reg  [2:0]   fraw_rd_r, fraw_wr_r; reg [3:0] fraw_level_r;
    wire [BW-1:0] fh = fraw_mem[fraw_rd_r];
    wire         fh_r = !fh[263] && fh[261];
    reg  [255:0] g_row_r [0:3];                    // {entry, half}: 32 codes
    reg  [7:0]   g_rsc_r [0:3];                    // {entry, half}: row 0 scale
    reg  [2:0]   g_w_r [0:1];
    reg  [1:0]   g_level_r;
    // The g entry being filled belongs to C entry c_popped + g_level (a producer fetches g with every C
    // tile; the C entries of earlier GEMMs are all popped when it starts), so it takes that entry's slot.
    wire         g_wr_sel = c_popped_r[0] ^ g_level_r[0];
    reg  [31:0]  r_buf_r [0:63];                   // {row pass parity, row}
    reg  [1:0]   r_valid_r, r_w_r;
    wire         f_take = fraw_level_r != 0 && (fh_r ? !r_valid_r[fh[256]] : g_level_r != 2'd2);
    reg          r_clear_r, r_clear_par_r;
    assign c_credit = {c_take, f_take, v_take};
    reg  [3:0]   c_done_r [0:63];
    wire [63:0]  c_row_done;
    reg          c_all_done;
    always @* begin
        c_all_done = 1'b1;
        for (i = 0; i < 64; i = i + 1) if (c_done_r[i] == c_popped_r) c_all_done = 1'b0;
    end
    wire         ch_half = craw_head[256];
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            craw_rd_r <= 0; craw_wr_r <= 0; craw_level_r <= 0; vraw_rd_r <= 0; vraw_wr_r <= 0; vraw_level_r <= 0;
            c_w_r[0] <= 0; c_w_r[1] <= 0; c_wr_sel_r <= 0; c_level_r <= 0; c_popped_r <= 0; c_pop_r <= 0;
            fraw_rd_r <= 0; fraw_wr_r <= 0; fraw_level_r <= 0; g_w_r[0] <= 0; g_w_r[1] <= 0;
            g_level_r <= 0; r_valid_r <= 0; r_w_r <= 0;
            for (i = 0; i < 64; i = i + 1) c_done_r[i] <= 0;
        end else begin
            if (c_beat_valid && c_beat_stream == 2'd2) begin
                craw_mem[craw_wr_r] <= {c_beat_seg_last, c_beat_inner_last, c_beat_inner_idx, c_beat_user, c_beat_data};
                craw_wr_r <= craw_wr_r + 1'b1;
            end
            if (c_beat_valid && c_beat_stream == 2'd0) begin
                vraw_mem[vraw_wr_r] <= {c_beat_seg_last, c_beat_inner_last, c_beat_inner_idx, c_beat_user, c_beat_data};
                vraw_wr_r <= vraw_wr_r + 1'b1;
            end
            if (c_beat_valid && c_beat_stream == 2'd1) begin
                fraw_mem[fraw_wr_r] <= {c_beat_seg_last, c_beat_inner_last, c_beat_inner_idx, c_beat_user, c_beat_data};
                fraw_wr_r <= fraw_wr_r + 1'b1;
            end
            if (f_take) fraw_rd_r <= fraw_rd_r + 1'b1;
            fraw_level_r <= fraw_level_r + (c_beat_valid && c_beat_stream == 2'd1) - f_take;
            if (f_take && !fh_r && !fh[263]) begin
                g_w_r[fh[256]] <= g_w_r[fh[256]] + 1'b1;
                if (fh[267] && fh[262]) begin g_w_r[0] <= 0; g_w_r[1] <= 0; end
            end
            // a g entry leaves with its C entry, in the cycle c_popped_r counts it (g_wr_sel stays put)
            g_level_r <= g_level_r + (f_take && !fh_r && !fh[263] && fh[267] && fh[262]) -
                         (c_all_done && !c_pop_r && g_level_r != 2'd0);
            if (f_take && fh_r) r_w_r <= fh[267] ? 2'd0 : r_w_r + 1'b1;
            if (f_take && fh_r && fh[267]) r_valid_r[fh[256]] <= 1'b1;
            else if (r_clear_r) r_valid_r[r_clear_par_r] <= 1'b0;
            if (c_take) craw_rd_r <= craw_rd_r + 1'b1;
            if (v_take) vraw_rd_r <= vraw_rd_r + 1'b1;
            craw_level_r <= craw_level_r + (c_beat_valid && c_beat_stream == 2'd2) - c_take;
            vraw_level_r <= vraw_level_r + (c_beat_valid && c_beat_stream == 2'd0) - v_take;
            if (c_take && !craw_head[263]) begin
                if (craw_head[266]) c_w_r[ch_half] <= c_w_r[ch_half] + 1'b1;
                if (craw_head[267] && craw_head[262]) begin
                    c_wr_sel_r <= !c_wr_sel_r; c_w_r[0] <= 0; c_w_r[1] <= 0;
                end
            end
            c_level_r <= c_level_r + (c_take && !craw_head[263] && craw_head[267] && craw_head[262]) - c_pop_r;
            c_pop_r <= c_all_done && !c_pop_r;
            if (c_all_done && !c_pop_r) c_popped_r <= c_popped_r + 1'b1;
            for (i = 0; i < 64; i = i + 1) if (c_row_done[i]) c_done_r[i] <= c_done_r[i] + 1'b1;
        end
    end
    always @(posedge clk) if (c_take) begin
        if (craw_head[263]) c_scale[{c_wr_sel_r, ch_half, craw_head[260:259]}] <= craw_head[craw_head[258:257]*64 +: 64];
        else c_tile[{c_wr_sel_r, ch_half, craw_head[265:264], c_w_r[ch_half]}] <= craw_head[255:0];
    end
    always @(posedge clk) if (f_take) begin : fold_beat
        integer b;
        if (fh_r) for (b = 0; b < 8; b = b + 1) r_buf_r[{fh[256], r_w_r, 3'(b)}] <= fh[b*32 +: 32];
        else if (fh[263]) g_rsc_r[{g_wr_sel, fh[256]}] <= fh[fh[258:257]*64 +: 8];
        else begin
            g_row_r[{g_wr_sel, fh[256]}][g_w_r[fh[256]]*32 +: 16] <= fh[15:0];
            g_row_r[{g_wr_sel, fh[256]}][g_w_r[fh[256]]*32 + 16 +: 16] <= fh[143:128];
        end
    end
    // Row r, column c of a C tile (native block words: panel r/8, word c/4): direct wiring.
    wire [255:0] c_row [0:3][0:31];               // [{entry, half}][row]
    for (genvar eh = 0; eh < 4; eh = eh + 1) begin : g_crow_e
        for (genvar r = 0; r < 32; r = r + 1) begin : g_crow_r
            for (genvar c = 0; c < 32; c = c + 1) begin : g_crow_c
                assign c_row[eh][r][c*8 +: 8] = c_tile[{2'(eh), 2'(r/8), 3'(c/4)}][((c/2)%2)*128 + (r%8)*16 + (c%2)*8 +: 8];
            end
        end
    end

    // ------------------------------------------------------------------
    // Lanes.
    // ------------------------------------------------------------------
    reg  [31:0]  alpha_r [0:63];                  // {bank, row}
    reg  [7:0]   pscale_r [0:63];                 // {bank, row}
    wire [31:0]  lane_g_ready0, lane_g_ready1, lane_g_busy, lane_g_cv, lane_s_pv, lane_s_cv, lane_s_done;
    wire [31:0]  lane_o_cv, lane_busy, lane_t0, lane_t1;
    wire [15:0]  lane_g_scales [0:31];
    wire [7:0]   lane_s_pscale [0:31], lane_o_scale [0:31];
    wire [31:0]  lane_g_chunk [0:31], lane_s_chunk [0:31], lane_o_chunk [0:31], lane_s_alpha [0:31], lane_o_l [0:31];
    wire [51:0]  lane_g_mag [0:31];
    wire [1:0]   lane_pv_end [0:31];
    wire [31:0]  lane_g_ready20, lane_g_ready21, lane_rms_v;
    wire [15:0]  lane_g_scales2 [0:31];
    wire [31:0]  lane_rms [0:31];
    reg          g_out_r, e_fp_r, e_pf_r, e_pl_r;
    reg  [31:0]  e_inv_r;
    reg  [1:0]   e_eps_r;
    reg          g_go_r, g_last_r, s_start_r, o_start_r, o_go_r, o_big_r;
    reg  [3:0]   g_cp_r, o_cp_r;
    reg  [1:0]   g_mode_r, g_halves_r, o_group_r;
    reg  [31:0]  g_ext_r;
    reg  [31:0]  inv_r [0:31];
    wire         s_slot, s_cstart, o_slot;
    wire [1:0]   s_ojobs;
    wire [31:0]  s_vis, s_scale;
    reg          s_par_r;
    for (genvar r = 0; r < 32; r = r + 1) begin : g_lane
        wire [1:0] lpop, cdone, gr, tf;
        assign pop[2*r] = lpop[0];
        assign pop[2*r+1] = lpop[1];
        assign c_row_done[2*r] = cdone[0];
        assign c_row_done[2*r+1] = cdone[1];
        assign lane_g_ready0[r] = gr[0];
        assign lane_g_ready1[r] = gr[1];
        assign lane_t0[r] = tf[0];
        assign lane_t1[r] = tf[1];
        wire [1:0] gr2;
        assign lane_g_ready20[r] = gr2[0];
        assign lane_g_ready21[r] = gr2[1];
        wire [63:0] cs0 = c_scale[{1'b0, 1'b0, 2'(r/8)}];
        wire [63:0] cs1 = c_scale[{1'b0, 1'b1, 2'(r/8)}];
        wire [63:0] cs2 = c_scale[{1'b1, 1'b0, 2'(r/8)}];
        wire [63:0] cs3 = c_scale[{1'b1, 1'b1, 2'(r/8)}];
        g3_vector_lane u_lane (
            .clk           (clk),
            .rstn          (rstn),
            .h_valid       ({head_valid[2*r+1], head_valid[2*r]}),
            .h_entry       ({hentry[2*r+1], hentry[2*r]}),
            .h_cp          ({hcp[2*r+1], hcp[2*r]}),
            .h_desc        ({hdesc[2*r+1], hdesc[2*r]}),
            .h_owait       ({howait[2*r+1], howait[2*r]}),
            .h_pop         (lpop),
            .pv_alpha      ({alpha_r[32 + r], alpha_r[r]}),
            .c_rows        ({c_row[3][r], c_row[2][r], c_row[1][r], c_row[0][r]}),
            .c_row_scales  ({cs3[(r%8)*8 +: 8], cs2[(r%8)*8 +: 8], cs1[(r%8)*8 +: 8], cs0[(r%8)*8 +: 8]}),
            .c_popped      (c_popped_r),
            .c_level       (c_level_r),
            .c_row_done    (cdone),
            .g_rows        ({g_row_r[3], g_row_r[2], g_row_r[1], g_row_r[0]}),
            .g_row_scales  ({g_rsc_r[3], g_rsc_r[2], g_rsc_r[1], g_rsc_r[0]}),
            .g_level       (g_level_r),
            .r_vals        ({r_buf_r[32 + r], r_buf_r[r]}),
            .r_valid       (r_valid_r),
            .g_ready       (gr),
            .g_scales      (lane_g_scales[r]),
            .g_busy        (lane_g_busy[r]),
            .g_go          (g_go_r),
            .g_cp          (g_cp_r),
            .g_mode        (g_mode_r),
            .g_halves      (g_halves_r),
            .g_last        (g_last_r),
            .g_ext_scales  (g_ext_r),
            .g_out         (g_out_r),
            .g_fold        (e_fp_r),
            .g_pfirst      (e_pf_r),
            .g_plast       (e_pl_r),
            .f_inv         (e_inv_r),
            .f_eps         (e_eps_r),
            .g_ready2      (gr2),
            .g_scales2     (lane_g_scales2[r]),
            .rms_valid     (lane_rms_v[r]),
            .rms           (lane_rms[r]),
            .g_chunk_valid (lane_g_cv[r]),
            .g_chunk       (lane_g_chunk[r]),
            .g_mag         (lane_g_mag[r]),
            .t_full        (tf),
            .s_start       (s_start_r),
            .s_tpar        (s_par_r),
            .s_slot        (s_slot),
            .s_cstart      (s_cstart),
            .s_vis         (s_vis[r]),
            .s_scale       (s_scale),
            .s_par_valid   (lane_s_pv[r]),
            .s_alpha       (lane_s_alpha[r]),
            .s_pscale      (lane_s_pscale[r]),
            .s_chunk_valid (lane_s_cv[r]),
            .s_chunk       (lane_s_chunk[r]),
            .s_done        (lane_s_done[r]),
            .pv_end_row    (lane_pv_end[r]),
            .o_slot        (o_slot),
            .o_l           (lane_o_l[r]),
            .o_start       (o_start_r),
            .o_inv         (inv_r[r]),
            .o_scale       (lane_o_scale[r]),
            .o_go          (o_go_r),
            .o_lg          (o_group_r),
            .o_dh_big      (o_big_r),
            .o_cp          (o_cp_r),
            .o_chunk_valid (lane_o_cv[r]),
            .o_chunk       (lane_o_chunk[r]),
            .busy          (lane_busy[r])
        );
    end

    // ------------------------------------------------------------------
    // Softmax and the P buffer: 2 banks x 2 K groups x 16 ticks of 512 bits (32 rows x K pair).
    // ------------------------------------------------------------------
    reg         s_busy_r, p_wr_bank_r;
    reg  [1:0]  p_full_r;
    reg  [4:0]  s_chunks_r, o_quiet_r;
    reg         p2_pend_r;
    reg  [3:0]  p2_tick_r;
    reg  [511:0] p2_data_r;
    assign {s_ojobs, s_slot, s_cstart, s_vis, s_scale} = qk_mem[qk_rd_r];
    wire        s_ready = !s_busy_r && qk_level_r != 0 && (s_par_r ? &lane_t1 : &lane_t0) && !p_full_r[p_wr_bank_r] &&
                          oj_lread_r[s_slot] == s_ojobs;
    wire        s_go = s_ready && o_quiet_r == 5'd16;
    assign qk_pop = lane_s_done[0];
    reg  [511:0] s_word0, s_word1;
    always @* for (i = 0; i < 32; i = i + 1) begin
        s_word0[i*16 +: 16] = lane_s_chunk[i][15:0];
        s_word1[i*16 +: 16] = lane_s_chunk[i][31:16];
    end
    reg         pm_wen;
    reg  [5:0]  pm_wa;
    reg  [511:0] pm_wd;
    wire [511:0] pm_rd;
    reg         pr_act_r, pr_rel_r, pr_bank_r, pr_g_r, p_plan_bank_r, ph_valid_r, pr_pend_r;
    reg  [1:0]  pr_kg_r;
    reg  [3:0]  pr_t_r;
    reg  [511:0] ph_data_r;
    reg  [255:0] ph_scale_r;                       // P scales of the word in the holding register
    reg          pr_pend_bank_r;
    reg  [2:0]  pwq_mem [0:3];
    reg  [1:0]  pwq_rd_r, pwq_wr_r; reg [2:0] pwq_level_r;
    wire        pr_end = pr_t_r == 4'd15 && (pr_kg_r == 2'd1 || pr_g_r);
    wire        pr_issue = pr_act_r && !pr_pend_r && (!ph_valid_r || p_pop);
    wire [5:0]  pm_ra = {pr_bank_r, pr_g_r, pr_t_r};
    rc_sram_1r1w #(.WIDTH(512), .ADDR_WIDTH(6)) u_pmem (
        .clk (clk), .write_enable (pm_wen), .write_address (pm_wa), .write_data (pm_wd),
        .read_enable (pr_issue), .read_address (pm_ra), .read_data (pm_rd));
    always @* begin
        pm_wen = 1'b0; pm_wa = 0; pm_wd = 0;
        if (lane_s_cv[0]) begin pm_wen = 1'b1; pm_wa = {p_wr_bank_r, 1'b0, s_chunks_r[3:0]}; pm_wd = s_word0; end
        else if (p2_pend_r) begin pm_wen = 1'b1; pm_wa = {p_wr_bank_r, 1'b1, p2_tick_r}; pm_wd = p2_data_r; end
    end
    assign p_valid = ph_valid_r;
    assign p_data = ph_data_r;
    reg  [255:0] p_scale_w;
    always @* for (i = 0; i < 32; i = i + 1) p_scale_w[i*8 +: 8] = pscale_r[{pr_pend_bank_r, 5'(i)}];
    assign p_scale = ph_scale_r;
    assign p_plan_ok = p_full_r[p_plan_bank_r];
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            s_busy_r <= 1'b0; s_par_r <= 1'b0; p_wr_bank_r <= 1'b0; s_start_r <= 1'b0; s_chunks_r <= 0;
            p_full_r <= 0; p2_pend_r <= 1'b0; pwq_rd_r <= 0; pwq_wr_r <= 0; pwq_level_r <= 0;
            pr_act_r <= 1'b0; pr_bank_r <= 1'b0; p_plan_bank_r <= 1'b0; pr_g_r <= 0; pr_t_r <= 0;
            ph_valid_r <= 1'b0; pr_pend_r <= 1'b0; pr_kg_r <= 0; pr_rel_r <= 1'b0;
        end else begin
            s_start_r <= s_go;
            if (s_go) s_busy_r <= 1'b1;
            else if (lane_s_done[0]) s_busy_r <= 1'b0;
            if (lane_s_done[0]) begin s_par_r <= !s_par_r; p_wr_bank_r <= !p_wr_bank_r; end
            if (s_go) s_chunks_r <= 0;
            else if (lane_s_cv[0]) s_chunks_r <= s_chunks_r + 1'b1;
            p2_pend_r <= lane_s_cv[0];
            if (lane_s_cv[0]) begin p2_tick_r <= s_chunks_r[3:0]; p2_data_r <= s_word1; end
            if (pw_valid) begin
                pwq_mem[pwq_wr_r] <= {pw_kg, pw_release}; pwq_wr_r <= pwq_wr_r + 1'b1;
                if (pw_release) p_plan_bank_r <= !p_plan_bank_r;
            end
            if (!pr_act_r && pwq_level_r != 0) begin
                pr_act_r <= 1'b1; {pr_kg_r, pr_rel_r} <= pwq_mem[pwq_rd_r]; pwq_rd_r <= pwq_rd_r + 1'b1;
                pr_g_r <= 1'b0; pr_t_r <= 0;
            end
            pwq_level_r <= pwq_level_r + pw_valid - (!pr_act_r && pwq_level_r != 0);
            pr_pend_r <= pr_issue;
            if (pr_issue) pr_pend_bank_r <= pr_bank_r;
            if (pr_issue) begin
                if (pr_end) begin
                    pr_act_r <= 1'b0;
                    if (pr_rel_r) pr_bank_r <= !pr_bank_r;
                end else if (pr_t_r == 4'd15) begin pr_t_r <= 0; pr_g_r <= 1'b1; end
                else pr_t_r <= pr_t_r + 1'b1;
            end
            if (pr_pend_r) begin ph_valid_r <= 1'b1; ph_data_r <= pm_rd; ph_scale_r <= p_scale_w; end
            else if (p_pop) ph_valid_r <= 1'b0;
            // Bank occupancy: full when its softmax ends, free after the releasing window's last read.
            for (i = 0; i < 2; i = i + 1)
                if (lane_s_done[0] && p_wr_bank_r == i) p_full_r[i] <= 1'b1;
                else if (pr_issue && pr_end && pr_rel_r && pr_bank_r == i) p_full_r[i] <= 1'b0;
        end
    end
    always @(posedge clk) if (lane_s_pv[0]) for (i = 0; i < 32; i = i + 1) begin
        alpha_r[{p_wr_bank_r, 5'(i)}] <= lane_s_alpha[i];
        pscale_r[{p_wr_bank_r, 5'(i)}] <= lane_s_pscale[i];
    end

    // ------------------------------------------------------------------
    // O control: reciprocals of l and row scales; O descriptors are output engine passes.
    // ------------------------------------------------------------------
    localparam [2:0] D_IDLE = 0, D_RECIP = 1, D_RECIP_WAIT = 2, D_SCALE = 3, D_PASS = 4, D_DONE = 5;
    reg  [2:0]  d_state_r;
    reg  [5:0]  d_issued_r, d_received_r;
    reg  [3:0]  d_wait_r;
    reg  [1:0]  pv_end_count_r [0:63];
    reg  [1:0]  o_done_count_r;
    reg         pv_all_done;
    assign o_slot = oj_mem[oj_rd_r];
    always @* begin
        pv_all_done = 1'b1;
        for (i = 0; i < 64; i = i + 1) if (pv_end_count_r[i] == o_done_count_r) pv_all_done = 1'b0;
    end
    wire [31:0] l_row = lane_o_l[d_issued_r[4:0]];
    wire        recip_valid;
    wire [4:0]  recip_tag;
    wire [31:0] recip_result;
    sfu_fp32_recip_staged #(.TAG_WIDTH(5)) u_recip (
        .clk (clk), .rst_n (rstn), .advance (1'b1), .in_valid (d_state_r == D_RECIP), .in_tag (d_issued_r[4:0]),
        .operand (l_row == 32'd0 ? ONE : l_row), .out_valid (recip_valid), .out_tag (recip_tag), .result (recip_result));
    wire        d_scale_go = d_state_r == D_RECIP_WAIT && d_received_r == 6'd32 && !s_ready && !s_busy_r;
    wire        d_scale_done = d_state_r == D_SCALE && d_wait_r == 4'd8;
    reg         o_pass_done;                      // output engine finished the last O pass of the job
    assign oj_pop = d_state_r == D_DONE;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            d_state_r <= D_IDLE; d_issued_r <= 0; d_received_r <= 0; d_wait_r <= 0; o_done_count_r <= 0;
            o_start_r <= 1'b0;
            oj_lread_r[0] <= 0; oj_lread_r[1] <= 0; oj_oread_r[0] <= 0; oj_oread_r[1] <= 0;
            for (i = 0; i < 64; i = i + 1) pv_end_count_r[i] <= 0;
        end else begin
            for (i = 0; i < 32; i = i + 1) begin
                if (lane_pv_end[i][0]) pv_end_count_r[2*i] <= pv_end_count_r[2*i] + 1'b1;
                if (lane_pv_end[i][1]) pv_end_count_r[2*i+1] <= pv_end_count_r[2*i+1] + 1'b1;
            end
            case (d_state_r)
            D_IDLE:       if (oj_level_r != 0 && pv_all_done) d_state_r <= D_RECIP;
            D_RECIP:      if (d_issued_r == 6'd31) d_state_r <= D_RECIP_WAIT;
            D_RECIP_WAIT: if (d_scale_go) d_state_r <= D_SCALE;
            D_SCALE:      if (d_scale_done) d_state_r <= D_PASS;
            D_PASS:       if (o_pass_done) d_state_r <= D_DONE;
            default:      d_state_r <= D_IDLE;
            endcase
            if (d_state_r == D_IDLE) d_issued_r <= 0;
            else if (d_state_r == D_RECIP) d_issued_r <= d_issued_r + 1'b1;
            if (d_state_r == D_IDLE) d_received_r <= 0;
            else if (recip_valid) d_received_r <= d_received_r + 1'b1;
            o_start_r <= d_scale_go;
            if (d_scale_go) d_wait_r <= 0;
            else if (d_state_r == D_SCALE) d_wait_r <= d_wait_r + 1'b1;
            if (d_state_r == D_DONE) o_done_count_r <= o_done_count_r + 1'b1;
            if (d_scale_go) oj_lread_r[o_slot] <= oj_lread_r[o_slot] + 1'b1;
            if (oj_pop) oj_oread_r[o_slot] <= oj_oread_r[o_slot] + 1'b1;
        end
    end
    always @(posedge clk) if (recip_valid) inv_r[recip_tag] <= recip_result;

    // ------------------------------------------------------------------
    // Output engine.
    // ------------------------------------------------------------------
    wire [7:0]   max_scale [0:63];                          // scale of each column maximum (V^T)
    localparam [3:0] E_IDLE = 0, E_MAG = 1, E_MAGW = 2, E_ENC = 3, E_DRAIN = 4, E_SCALE = 5, E_RMW = 6, E_FIN = 7,
                     E_XG = 8, E_RMS = 9;
    reg  [3:0]   e_state_r;
    reg          e_o_r;                            // O pass (else a GEMM tile)
    reg  [1:0]   e_layout_r;
    reg          e_vt_r, e_tail_r, e_app1_r, e_mark_r;
    reg  [1:0]   e_hv_r;
    reg  [2:0]   e_npan_r, e_arow8_r;
    reg  [4:0]   e_vtt_r;
    reg  [11:0]  e_base_r [0:1];
    reg  [11:0]  e_pan_r [0:1];
    reg  [8:0]   e_g_r [0:1];
    reg  [8:0]   e_gd_r;
    reg  [1:0]   e_dp_r;
    reg          e_big_r;
    reg  [4:0]   e_issued_r, e_arrived_r;
    reg  [1:0]   e_gap_r;
    reg  [31:0]  e_buf [0:255];                    // {group parity, go in group (2b), lane (5b)}
    reg  [1:0]   e_full_r;
    reg  [3:0]   e_dgrp_r;                         // next group to drain
    reg  [4:0]   e_dw_r;                           // word inside the group
    reg  [3:0]   e_si_r;                           // scale write index
    reg  [15:0]  e_rsc_r [0:31];                   // lane row scales {half 1, half 0} (GEMM)
    reg  [12:0]  colmax_r [0:63];                  // {half, column}
    for (genvar c = 0; c < 64; c = c + 1) begin : g_max_scale
        assign max_scale[c] = choose_scale(expand({1'b0, colmax_r[c]}));
    end
    reg  [7:0]   colscale_r [0:63];
    reg  [7:0]   sold_r [0:63];
    reg  [63:0]  zero_r;
    reg  [7:0]   newcode_r [0:63];
    reg          scan_done_r, scan_h_r;
    reg  [5:0]   scan_i_r;
    reg  [5:0]   rmw_i_r;                          // 0 old scale beat, 1..32 words, 33 new scale word
    reg          rmw_h_r;
    reg          od_busy_r;
    wire [1:0]   g_need = desc_uop[`U_EPI] == `G3_EPI_SILU_MUL ? 2'b10 : 2'b11;
    wire         lanes_ready = (!g_need[0] || &lane_g_ready0) && (!g_need[1] || &lane_g_ready1) &&
                               (!desc_uop[`U_FP] || ((!g_need[0] || &lane_g_ready20) && (!g_need[1] || &lane_g_ready21)));
    // RMSNorm fold: the x * g pass of a producer tile; the 1/rms of the row pass (producer, last pass).
    reg          e_xg_r, e_fc_r, e_rpar_r;
    reg  [11:0]  e_xgd_r, e_rbase_r;
    reg  [31:0]  rms_r [0:31];
    reg  [31:0]  rms_have_r;
    reg  [2:0]   rms_i_r;
    wire         e_scale_last = e_fp_r && (!e_xg_r || e_pl_r);  // a later write of the pass carries the mark
    wire [3:0]   e_after_scale = e_fp_r && !e_xg_r ? E_XG : e_fp_r && e_pl_r ? E_RMS : E_FIN;
    reg  [255:0] rms_word;
    always @* begin : rms_pack
        integer b;
        for (b = 0; b < 8; b = b + 1) rms_word[b*32 +: 32] = rms_r[{rms_i_r[1:0], 3'(b)}];
    end
    wire         rms_now = e_state_r == E_RMS && &rms_have_r && rms_i_r != 3'd4;
    // A descriptor FIFO pops one cycle after E_FIN (registered ready): no start while it pops.
    wire         start_gemm = e_state_r == E_IDLE && desc_valid && lanes_ready && !desc_ready;
    wire         start_o = e_state_r == E_IDLE && !start_gemm && d_state_r == D_PASS && od_valid && !od_busy_r &&
                           !od_ready;
    wire [255:0] load_uop = start_gemm ? desc_uop : od_uop;   // descriptor taken by E_IDLE
    wire [4:0]   iss = e_issued_r;
    wire         grp_first = e_vt_r ? iss[1:0] == 2'd0 : !iss[0];
    wire         grp_par = e_vt_r ? iss[2] : iss[1];
    wire         enc_buf_ok = e_tail_r || !grp_first || !e_full_r[grp_par];
    wire         o_type_ok = !e_o_r || (!s_ready && !s_busy_r);
    wire         issue_mag = e_state_r == E_MAG && iss != 5'd16 && e_gap_r == 0;
    wire         issue_enc = e_state_r == E_ENC && iss != 5'd16 && e_gap_r == 0 && enc_buf_ok && o_type_ok;
    wire         go_now = issue_mag || issue_enc;
    wire         arr = e_o_r ? lane_o_cv[0] : lane_g_cv[0];
    wire [3:0]   arr_cp = e_arrived_r[3:0];
    wire         arr_mag = arr && (e_state_r == E_MAG || e_state_r == E_MAGW);
    wire         arr_enc = arr && !arr_mag;
    wire         arr_par = e_vt_r ? arr_cp[2] : arr_cp[1];
    wire [1:0]   arr_slot = e_vt_r ? arr_cp[1:0] : {1'b0, arr_cp[0]};
    wire         arr_end = e_vt_r ? arr_cp[1:0] == 2'd3 : arr_cp[0];
    // Drain word map: rows: (half, panel) over valid halves x panels; V^T: (half, token quad).
    wire [1:0]   hv_cnt = e_hv_r[0] + e_hv_r[1];
    wire [2:0]   pan_n = e_app1_r ? 3'd1 : e_npan_r;
    wire [4:0]   grp_words = e_vt_r ? {hv_cnt, 3'd0} : hv_cnt * pan_n;
    wire         d_par = e_dgrp_r[0];
    wire         drain_go = e_full_r[d_par] && !e_tail_r;
    reg          dw_h;
    reg  [2:0]   dw_x;
    always @* begin : drain_map
        reg [4:0] w;
        w = e_dw_r;
        if (e_vt_r) begin
            dw_h = !e_hv_r[0] || w >= 8; dw_x = w[2:0];
        end else begin
            dw_h = !e_hv_r[0] || w >= pan_n; dw_x = e_hv_r[0] && w >= pan_n ? w - pan_n : w[2:0];
        end
    end
    reg  [255:0] dw_data;
    reg  [31:0]  dw_strb;
    reg  [11:0]  dw_addr;
    always @* begin : drain_word
        integer r, kk, g;
        reg [15:0] nb;
        dw_data = 0; dw_strb = 32'hffff_ffff;
        nb = blk(e_layout_r, e_pan_r[dw_h] + (e_vt_r ? {8'd0, e_dgrp_r} : {9'd0, dw_x}), e_gd_r, e_g_r[dw_h]);
        if (e_vt_r) begin
            // V^T cell word (panel p = group, token quad w): byte (k/2)*16 + r*2 + k%2 = token 4w+k,
            // dim 8p+r = column pair 4p + r/2 (go r/2 of the group), value r%2 of half h.
            for (r = 0; r < 8; r = r + 1)
                for (kk = 0; kk < 4; kk = kk + 1)
                    dw_data[(kk/2)*128 + r*16 + (kk%2)*8 +: 8] =
                        e_buf[{d_par, 2'(r/2), 5'(dw_x*4 + kk)}][(2*dw_h + r%2)*8 +: 8];
            dw_addr = e_base_r[dw_h] + block_word(nb) + {9'd0, dw_x};
        end else begin
            // Row word (panel p, word w = group): low half from the even go, high from the odd go.
            for (g = 0; g < 2; g = g + 1)
                for (r = 0; r < 8; r = r + 1)
                    dw_data[g*128 + r*16 +: 16] = e_buf[{d_par, 2'(g), 5'(dw_x*8 + r)}][dw_h*16 +: 16];
            if (e_app1_r) begin
                dw_data = 0;
                dw_data[e_arow8_r*16 +: 16] = e_buf[{d_par, 2'd0, 5'd0}][dw_h*16 +: 16];
                dw_data[128 + e_arow8_r*16 +: 16] = e_buf[{d_par, 2'd1, 5'd0}][dw_h*16 +: 16];
                dw_strb = (32'h3 << (2*e_arow8_r)) | (32'h3 << (16 + 2*e_arow8_r));
            end
            dw_addr = e_base_r[dw_h] + block_word(nb) + {8'd0, e_dgrp_r};
        end
    end
    // Scale writes: rows: one per valid (half, panel); V^T: one full word per valid half.
    wire [3:0]   sw_n = e_vt_r ? {2'd0, hv_cnt} : hv_cnt * pan_n;
    reg          sw_h;
    reg  [1:0]   sw_p;
    reg  [255:0] sw_data;
    reg  [31:0]  sw_strb;
    reg  [11:0]  sw_addr;
    always @* begin : scale_word_data
        integer r;
        reg [15:0] nb;
        if (e_vt_r) begin sw_h = !e_hv_r[0] || e_si_r != 0; sw_p = 0; end
        else begin sw_h = !e_hv_r[0] || e_si_r >= pan_n; sw_p = e_hv_r[0] && e_si_r >= pan_n ? e_si_r - pan_n : e_si_r[1:0]; end
        nb = blk(e_layout_r, e_pan_r[sw_h] + {10'd0, sw_p}, e_gd_r, e_g_r[sw_h]);
        sw_addr = e_base_r[sw_h] + scale_word(nb);
        sw_data = 0;
        if (e_vt_r) begin
            for (r = 0; r < 32; r = r + 1) sw_data[r*8 +: 8] = colscale_r[{sw_h, 5'(r)}];
            sw_strb = 32'hffff_ffff;
        end else if (e_app1_r) begin
            sw_data[nb[1:0]*64 + e_arow8_r*8 +: 8] = e_o_r ? lane_o_scale[0] : e_rsc_r[0][sw_h*8 +: 8];
            sw_strb = 32'h1 << (nb[1:0]*8 + e_arow8_r);
        end else begin
            for (r = 0; r < 8; r = r + 1)
                sw_data[nb[1:0]*64 + r*8 +: 8] = e_o_r ? lane_o_scale[sw_p*8 + r] : e_rsc_r[sw_p*8 + r][sw_h*8 +: 8];
            sw_strb = 32'hff << (nb[1:0]*8);
        end
    end
    // V^T rewrite word: old codes re-rounded to the new scale, the new token inserted. Two steps: the
    // word's codes, scale increases and flags are registered (rq_*), then re-rounded into the engine's
    // write register. The tail rule never lowers a scale, so re-rounding a code to a scale larger by
    // d >= 0 is an exponent decrease with round-to-nearest-even on the 4-bit significand; it equals
    // quant_e4m3(decode_element(code, s), s + d).
    function automatic [7:0] requant_down(input [7:0] code, input [7:0] d);
        reg [3:0] s, q, rem, half;
        reg [8:0] en;
        reg [3:0] k;
        begin
            s = code[6:3] == 4'd0 ? {1'b0, code[2:0]} : {1'b1, code[2:0]};
            en = (code[6:3] == 4'd0 ? 9'd1 : {5'd0, code[6:3]}) - {1'b0, d};       // new exponent field
            if (!en[8] && en != 9'd0) begin
                requant_down = s[3] ? {code[7], en[3:0], s[2:0]} : code;           // still normal / unchanged
            end else begin
                k = (9'd1 - en) > 9'd5 ? 4'd5 : 4'(9'd1 - en);                   // right shift into subnormals
                q = s >> k; rem = s & ((4'd1 << k) - 4'd1); half = 4'd1 << (k - 4'd1);
                if (k == 4'd5) q = 4'd0;
                else if (rem > half || (rem == half && q[0])) q = q + 4'd1;
                requant_down = {code[7], 3'd0, q};
            end
        end
    endfunction
    reg  [255:0] rq_code_r;                           // per byte: the code (or the new token's code)
    reg  [255:0] rq_d_r;                              // per byte: scale increase
    reg  [31:0]  rq_req_r, rq_nan_r;                  // per byte: re-round, invalid scale
    reg          rq_v_r;
    reg  [11:0]  rq_addr_r;
    reg  [255:0] rq_code_n, rq_d_n;
    reg  [31:0]  rq_req_n, rq_nan_n, rq_dec_n;        // rq_dec_n: a scale decrease (checked in simulation)
    always @* begin : rmw_inputs
        integer r, kk, d, tk, p, w, b;
        reg [7:0] code, so, sn;
        p = (rmw_i_r - 1) / 8; w = (rmw_i_r - 1) % 8;
        rq_code_n = 0; rq_d_n = 0; rq_req_n = 0; rq_nan_n = 0; rq_dec_n = 0;
        for (r = 0; r < 8; r = r + 1)
            for (kk = 0; kk < 4; kk = kk + 1) begin
                d = 8 * p + r; tk = 4 * w + kk; b = (kk/2)*16 + r*2 + kk%2;
                code = vh[b*8 +: 8];
                so = sold_r[{rmw_h_r, 5'(d)}]; sn = colscale_r[{rmw_h_r, 5'(d)}];
                if (tk == e_vtt_r) rq_code_n[b*8 +: 8] = newcode_r[{rmw_h_r, 5'(d)}];
                else begin
                    rq_code_n[b*8 +: 8] = code;
                    rq_req_n[b] = sn != so && code[6:0] != 7'd0;
                    rq_nan_n[b] = so == 8'hff || sn == 8'hff || code[6:0] == 7'h7f;
                    rq_d_n[b*8 +: 8] = sn - so;
                    rq_dec_n[b] = sn < so;
                end
            end
    end
    reg  [255:0] rmw_data;
    always @* begin : rmw_word
        integer b;
        for (b = 0; b < 32; b = b + 1)
            rmw_data[b*8 +: 8] = !rq_req_r[b] ? rq_code_r[b*8 +: 8] :
                                 rq_nan_r[b] ? 8'h7f : requant_down(rq_code_r[b*8 +: 8], rq_d_r[b*8 +: 8]);
    end
    wire [15:0]  rmw_nb = blk(e_layout_r, e_pan_r[rmw_h_r], e_gd_r, e_g_r[rmw_h_r]);
    wire [11:0]  rmw_cell = e_base_r[rmw_h_r] + scale_word(rmw_nb);
    reg  [255:0] rmw_sw;
    always @* for (i = 0; i < 32; i = i + 1) rmw_sw[i*8 +: 8] = colscale_r[{rmw_h_r, 5'(i)}];
    wire         rmw_last_half = rmw_h_r || !e_hv_r[1];
    // Engine write (one per cycle) into the write register (wb_*).
    reg          ow_valid, ow_mark;
    reg  [255:0] ow_data;
    reg  [31:0]  ow_strb;
    reg  [11:0]  ow_addr;
    wire         drain_now = e_state_r != E_IDLE && drain_go;
    wire         rmw_word_now = e_state_r == E_RMW && rmw_i_r >= 1 && rmw_i_r <= 32 && vraw_level_r != 0;
    wire         rmw_sw_now = e_state_r == E_RMW && rmw_i_r == 6'd33 && !rq_v_r;
    always @* begin
        ow_valid = 1'b0; ow_mark = 1'b0; ow_data = 0; ow_strb = 0; ow_addr = 0;
        if (drain_now) begin ow_valid = 1'b1; ow_data = dw_data; ow_strb = dw_strb; ow_addr = dw_addr; end
        else if (e_state_r == E_SCALE && !e_tail_r && e_si_r < sw_n) begin
            ow_valid = 1'b1; ow_data = sw_data; ow_strb = sw_strb; ow_addr = sw_addr;
            ow_mark = e_mark_r && e_si_r + 1 == sw_n && !e_scale_last;
        end else if (rms_now) begin
            ow_valid = 1'b1; ow_data = rms_word; ow_strb = 32'hffff_ffff; ow_addr = e_rbase_r + {9'd0, rms_i_r};
            ow_mark = e_mark_r && rms_i_r == 3'd3;
        end else if (rq_v_r) begin
            ow_valid = 1'b1; ow_data = rmw_data; ow_strb = 32'hffff_ffff; ow_addr = rq_addr_r;
        end else if (rmw_sw_now) begin
            ow_valid = 1'b1; ow_data = rmw_sw; ow_strb = 32'hffff_ffff; ow_addr = rmw_cell;
            ow_mark = e_mark_r && rmw_last_half;
        end
    end
    reg          wr_v_r, wr_mark_r;
    reg  [255:0] wr_data_r;
    reg  [31:0]  wr_strb_r;
    reg  [11:0]  wr_addr_r;
    wire         wr_free = !wr_v_r || wb_ready;
    wire         ow_take = ow_valid && wr_free;
    wire         rq_load = rmw_word_now && (!rq_v_r || ow_take);
    wire         scan_take = !scan_done_r && vraw_level_r != 0;
    wire         rmw_take = e_state_r == E_RMW && vraw_level_r != 0 && (rmw_i_r == 0 || rq_load);
    assign v_take = scan_take || rmw_take;
    always @(posedge clk or negedge rstn) begin : write_state
        if (!rstn) begin wr_v_r <= 1'b0; rq_v_r <= 1'b0; end
        else begin
            if (wr_free) wr_v_r <= ow_valid;
            if (rq_load) rq_v_r <= 1'b1;
            else if (ow_take) rq_v_r <= 1'b0;
        end
    end
    always @(posedge clk) begin : write_data
        if (ow_take) begin wr_data_r <= ow_data; wr_strb_r <= ow_strb; wr_addr_r <= ow_addr; wr_mark_r <= ow_mark; end
        if (rq_load) begin
            rq_code_r <= rq_code_n; rq_d_r <= rq_d_n; rq_req_r <= rq_req_n; rq_nan_r <= rq_nan_n;
            rq_addr_r <= rmw_cell + {6'd0, rmw_i_r};
        end
    end
    assign wb_valid = wr_v_r;
    assign wb_addr = wr_addr_r;
    assign wb_data = wr_data_r;
    assign wb_strb = wr_strb_r;
    assign wb_mark = wr_v_r && wr_mark_r;
    // Column maximum of the 4 values of a magnitude chunk over the 32 lanes.
    // Column maximum over the 32 lanes (V^T magnitude pass), a pipelined tree: 32 -> 8 (registered)
    // -> 2 (registered) -> 1, written to colmax_r two cycles after the arrival.
    function automatic [12:0] max13(input [12:0] a, input [12:0] b);
        max13 = a > b ? a : b;
    endfunction
    reg  [12:0] mx1_r [0:3][0:7];
    reg  [12:0] mx2_r [0:3][0:1];
    reg         mx1_v_r, mx2_v_r;
    reg  [3:0]  mx1_cp_r, mx2_cp_r;
    wire [12:0] mag_max [0:3];
    always @(posedge clk) begin : col_max_tree
        integer kk, gg;
        for (kk = 0; kk < 4; kk = kk + 1) begin
            for (gg = 0; gg < 8; gg = gg + 1)
                mx1_r[kk][gg] <= max13(max13(lane_g_mag[4*gg][kk*13 +: 13], lane_g_mag[4*gg + 1][kk*13 +: 13]),
                                       max13(lane_g_mag[4*gg + 2][kk*13 +: 13], lane_g_mag[4*gg + 3][kk*13 +: 13]));
            for (gg = 0; gg < 2; gg = gg + 1)
                mx2_r[kk][gg] <= max13(max13(mx1_r[kk][4*gg], mx1_r[kk][4*gg + 1]),
                                       max13(mx1_r[kk][4*gg + 2], mx1_r[kk][4*gg + 3]));
        end
        mx1_cp_r <= arr_cp;
        mx2_cp_r <= mx1_cp_r;
    end
    always @(posedge clk or negedge rstn)
        if (!rstn) begin mx1_v_r <= 1'b0; mx2_v_r <= 1'b0; end
        else begin mx1_v_r <= arr_mag; mx2_v_r <= mx1_v_r; end
    for (genvar m = 0; m < 4; m = m + 1) begin : g_mag_max
        assign mag_max[m] = max13(mx2_r[m][0], mx2_r[m][1]);
    end
    wire all_drained = e_full_r == 2'b00 && e_dgrp_r == (e_vt_r ? 4'd4 : 4'd8);
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            e_state_r <= E_IDLE; e_o_r <= 1'b0; e_issued_r <= 0; e_arrived_r <= 0; e_gap_r <= 0;
            e_full_r <= 0; e_dgrp_r <= 0; e_dw_r <= 0; e_si_r <= 0;
            g_go_r <= 1'b0; g_last_r <= 1'b0; o_go_r <= 1'b0; desc_ready <= 1'b0; od_ready <= 1'b0; od_busy_r <= 1'b0;
            scan_done_r <= 1'b1; scan_i_r <= 0; scan_h_r <= 1'b0; rmw_i_r <= 0; rmw_h_r <= 1'b0;
            o_quiet_r <= 5'd16; o_pass_done <= 1'b0; e_vt_r <= 1'b0; e_tail_r <= 1'b0;
            e_xg_r <= 1'b0; e_fp_r <= 1'b0; rms_have_r <= 0; rms_i_r <= 0; r_clear_r <= 1'b0; g_out_r <= 1'b0;
        end else begin
            desc_ready <= 1'b0; od_ready <= 1'b0; o_pass_done <= 1'b0;
            g_go_r <= go_now && !e_o_r;
            if (go_now) g_out_r <= e_xg_r;
            r_clear_r <= 1'b0;
            for (i = 0; i < 32; i = i + 1)
                if (lane_rms_v[i]) begin rms_have_r[i] <= 1'b1; rms_r[i] <= lane_rms[i]; end
            o_go_r <= go_now && e_o_r;
            g_last_r <= iss == 5'd15;
            if (go_now) begin
                g_cp_r <= iss[3:0]; o_cp_r <= iss[3:0];
                g_ext_r <= {colscale_r[{1'b1, iss[3:0], 1'b1}], colscale_r[{1'b1, iss[3:0], 1'b0}],
                            colscale_r[{1'b0, iss[3:0], 1'b1}], colscale_r[{1'b0, iss[3:0], 1'b0}]};
            end
            if (go_now) e_gap_r <= 2'd3;
            else if (e_gap_r != 0) e_gap_r <= e_gap_r - 1'b1;
            if (o_start_r || (go_now && e_o_r)) o_quiet_r <= 0;
            else if (o_quiet_r != 5'd16) o_quiet_r <= o_quiet_r + 1'b1;
            if (go_now) e_issued_r <= e_issued_r + 1'b1;
            if (arr) e_arrived_r <= e_arrived_r + 1'b1;
            // Old V^T scan (first read of the cells): old scales and dims with only zero codes.
            if (scan_take) begin : scan
                integer r, kk, p;
                p = (scan_i_r - 1) / 8;
                if (scan_i_r == 0) for (r = 0; r < 32; r = r + 1) sold_r[{scan_h_r, 5'(r)}] <= vh[r*8 +: 8];
                else for (r = 0; r < 8; r = r + 1)
                    for (kk = 0; kk < 4; kk = kk + 1)
                        if (vh[(kk/2)*128 + r*16 + (kk%2)*8 +: 7] != 7'd0) zero_r[{scan_h_r, 5'(8*p + r)}] <= 1'b0;
                if (scan_i_r == 6'd32) begin
                    scan_i_r <= 0;
                    if (scan_h_r || !e_hv_r[1]) scan_done_r <= 1'b1; else scan_h_r <= 1'b1;
                end else scan_i_r <= scan_i_r + 1'b1;
            end
            case (e_state_r)
            E_IDLE: begin
                e_issued_r <= 0; e_arrived_r <= 0; e_dgrp_r <= 0; e_dw_r <= 0; e_si_r <= 0; e_full_r <= 0;
                if (start_gemm || start_o) begin : load
                    e_o_r <= !start_gemm;
                    e_layout_r <= load_uop[`U_LAYOUT]; e_vt_r <= load_uop[`U_LAYOUT] == `G3_LAYOUT_VT; e_tail_r <= load_uop[`U_VTTAIL];
                    e_app1_r <= load_uop[`U_APP1]; e_mark_r <= load_uop[`U_MARK]; e_hv_r <= {load_uop[`U_H1], load_uop[`U_H0]};
                    e_npan_r <= load_uop[`U_NPAN]; e_arow8_r <= load_uop[`U_AROW8]; e_vtt_r <= load_uop[`U_VTT];
                    e_base_r[0] <= load_uop[`U_BASE0]; e_base_r[1] <= load_uop[`U_BASE1];
                    e_pan_r[0] <= load_uop[`U_PAN0]; e_pan_r[1] <= load_uop[`U_PAN1];
                    e_g_r[0] <= load_uop[`U_G0]; e_g_r[1] <= load_uop[`U_G1]; e_gd_r <= load_uop[`U_GD];
                    o_group_r <= load_uop[`U_DP]; o_big_r <= load_uop[`U_BIG];
                    e_xg_r <= 1'b0; rms_i_r <= 0;
                    e_fp_r <= start_gemm && load_uop[`U_FP]; e_fc_r <= start_gemm && load_uop[`U_FC];
                    e_pf_r <= load_uop[`U_PF]; e_pl_r <= load_uop[`U_PL]; e_rpar_r <= load_uop[`U_RPAR];
                    e_xgd_r <= load_uop[`U_XGD]; e_rbase_r <= load_uop[`U_RBASE];
                    e_inv_r <= load_uop[`U_INV]; e_eps_r <= load_uop[`U_EPS];
                    if (start_gemm) begin
                        for (i = 0; i < 32; i = i + 1) e_rsc_r[i] <= lane_g_scales[i];
                        e_state_r <= load_uop[`U_LAYOUT] == `G3_LAYOUT_VT ? E_MAG : E_ENC;
                        g_mode_r <= load_uop[`U_LAYOUT] == `G3_LAYOUT_VT ? 2'd1 : 2'd0;
                        g_halves_r <= load_uop[`U_LAYOUT] == `G3_LAYOUT_VT ? 2'b00 : g_need;
                        if (load_uop[`U_VTTAIL]) begin
                            scan_done_r <= 1'b0; scan_i_r <= 0; scan_h_r <= !load_uop[`U_H0]; zero_r <= {64{1'b1}};
                        end
                    end else begin
                        e_state_r <= E_ENC; od_busy_r <= 1'b1;
                    end
                end
            end
            E_MAG: if (arr_mag && e_arrived_r == 5'd15) e_state_r <= E_MAGW;
            E_MAGW: if (scan_done_r && !mx1_v_r && !mx2_v_r) begin   // the maximum tree has drained
                // New scales: the column maximum, or (rewrite) the decode-tail rule.
                for (i = 0; i < 64; i = i + 1) begin : newscale
                    if (!e_tail_r || zero_r[i]) colscale_r[i] <= max_scale[i];
                    else if (colmax_r[i] == 13'd0) colscale_r[i] <= sold_r[i];
                    else colscale_r[i] <= max_scale[i] > sold_r[i] ? max_scale[i] : sold_r[i];
                end
                e_state_r <= E_ENC; e_issued_r <= 0; e_arrived_r <= 0;
                g_mode_r <= 2'd2; g_halves_r <= 2'b11;
            end
            E_ENC: if (arr_enc && e_arrived_r == 5'd15) e_state_r <= E_DRAIN;
            E_DRAIN: if (e_tail_r || all_drained) begin e_state_r <= E_SCALE; e_si_r <= 0; end
            E_SCALE: begin
                if (e_tail_r) begin e_state_r <= E_RMW; rmw_i_r <= 0; rmw_h_r <= !e_hv_r[0]; end
                else if (e_si_r >= sw_n) e_state_r <= e_after_scale;
                else if (ow_take) begin
                    e_si_r <= e_si_r + 1'b1;
                    if (e_si_r + 1 == sw_n) e_state_r <= e_after_scale;
                end
            end
            E_XG: begin                                    // the producer tile's x * g: T0, second scales
                e_xg_r <= 1'b1; e_state_r <= E_ENC;
                e_issued_r <= 0; e_arrived_r <= 0; e_dgrp_r <= 0; e_dw_r <= 0; e_si_r <= 0; e_full_r <= 0;
                e_base_r[0] <= e_base_r[0] + e_xgd_r; e_base_r[1] <= e_base_r[1] + e_xgd_r;
                for (i = 0; i < 32; i = i + 1) e_rsc_r[i] <= lane_g_scales2[i];
            end
            E_RMS: begin                                   // after the row pass's last x * g: 1/rms words
                if (rms_now && ow_take) rms_i_r <= rms_i_r + 1'b1;
                if (rms_i_r == 3'd4) begin e_state_r <= E_FIN; rms_have_r <= 0; end
            end
            E_RMW: begin
                if (rmw_take) rmw_i_r <= rmw_i_r + 1'b1;
                if (rmw_sw_now && ow_take) begin
                    if (rmw_last_half) e_state_r <= E_FIN;
                    else begin rmw_h_r <= 1'b1; rmw_i_r <= 0; end
                end
            end
            E_FIN: begin
                e_state_r <= E_IDLE;
                if (!e_o_r && e_fc_r && e_pl_r) begin r_clear_r <= 1'b1; r_clear_par_r <= e_rpar_r; end
                if (e_o_r) begin od_ready <= 1'b1; od_busy_r <= 1'b0; o_pass_done <= e_mark_r; end
                else desc_ready <= 1'b1;
            end
            default: e_state_r <= E_IDLE;
            endcase
            // Chunk capture.
            if (arr_enc && !e_tail_r) begin
                for (i = 0; i < 32; i = i + 1)
                    e_buf[{arr_par, arr_slot, 5'(i)}] <= e_o_r ? lane_o_chunk[i] : lane_g_chunk[i];
                if (arr_end) e_full_r[arr_par] <= 1'b1;
            end
            if (arr_enc && e_tail_r)
                for (k = 0; k < 4; k = k + 1) newcode_r[{1'(k/2), arr_cp, 1'(k%2)}] <= lane_g_chunk[0][k*8 +: 8];
            if (mx2_v_r)
                for (k = 0; k < 4; k = k + 1) colmax_r[{1'(k/2), mx2_cp_r, 1'(k%2)}] <= mag_max[k];
            // Drain progress.
            if (drain_now && ow_take) begin
                if (e_dw_r + 1 == grp_words) begin
                    e_dw_r <= 0; e_full_r[d_par] <= 1'b0; e_dgrp_r <= e_dgrp_r + 1'b1;
                end else e_dw_r <= e_dw_r + 1'b1;
            end
        end
    end

    // ------------------------------------------------------------------
    // Activity.
    // ------------------------------------------------------------------
    reg td_gemm, td_attn;
    always @* begin
        td_gemm = 1'b0;
        td_attn = 1'b0;
        for (i = 0; i < 8; i = i + 1) begin
            if (td_live_r[i] && !td_mem[i][17]) td_gemm = 1'b1;
            if (td_live_r[i] && td_mem[i][17]) td_attn = 1'b1;
        end
    end
    assign gemm_active = td_gemm || (|lane_g_busy) || (e_state_r != E_IDLE && !e_o_r);
    assign fa_active = td_attn || qk_level_r != 0 || s_busy_r || oj_level_r != 0 || d_state_r != D_IDLE ||
                       (e_state_r != E_IDLE && e_o_r) || p_full_r != 0 || pwq_level_r != 0 || pr_act_r ||
                       (|lane_t0) || (|lane_t1);
    assign busy = gemm_active || fa_active || (|head_valid) || (|lane_busy) || c_pop_r || craw_level_r != 0 ||
                  vraw_level_r != 0 || c_level_r != 0 || wr_v_r || fraw_level_r != 0 || g_level_r != 0;
    // synthesis translate_off
    always @(posedge clk) if (rstn && rq_load) begin : rewrite_check
        integer b;
        for (b = 0; b < 32; b = b + 1)
            if (rq_req_n[b] && !rq_nan_n[b] && rq_dec_n[b])
                $fatal(1, "G3_VT_REWRITE_SCALE_DECREASE byte=%0d", b);
    end
    // synthesis translate_on

    // synthesis translate_off
    always @(posedge clk) if (rstn) begin
        if (lane_g_cv != 32'd0 && lane_g_cv != 32'hffff_ffff) $fatal(1, "G3_VC_LANES_OUT_OF_STEP g");
        if (lane_s_cv != 32'd0 && lane_s_cv != 32'hffff_ffff) $fatal(1, "G3_VC_LANES_OUT_OF_STEP s");
        if (lane_o_cv != 32'd0 && lane_o_cv != 32'hffff_ffff) $fatal(1, "G3_VC_LANES_OUT_OF_STEP o");
        if (arr_enc && !e_tail_r && e_full_r[arr_par] && arr_slot == 0) $fatal(1, "G3_VC_CHUNK_BUFFER_OVERRUN");
    end
    // synthesis translate_on
endmodule

`default_nettype wire
