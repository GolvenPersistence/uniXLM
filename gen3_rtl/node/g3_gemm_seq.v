`timescale 1ns/1ps
`default_nettype none

// Gen3 GEMM sequencer (gen3_rtl/README.md s6.1): one GEMM parent on the single array.
//
//   pd --> shape: row passes (32 rows), column passes (64 columns = B bands 2c, 2c+1; gated: one
//          output tile per pass, gate band left, up band right), K chunks (<= 32 groups)
//          |
//          +--> STREAM: two pull programs (even / odd bands) + weight-buffer cell counts
//          +--> per pass: uop per K chunk (the last carries the output descriptor);
//               per K group: A scales (one beat per panel) + A payload (4 panels, per-panel offsets),
//               B cells from the UB (left / right contexts) unless STREAM;
//               C tile (epilogue operand) after the last chunk; V^T rewrite: two reads of the old cells;
//               RMSNorm fold producer: g bands of the pass (port C, context 1) after the C tile;
//               fold consumer: the row pass's 1/rms words (port C, context 1) before its first pass
//   ev_* --> watch registers A, B, C --> segment gates
//   mark_retired (one mark per pass) --> P8 single tensor: quad-prefix commit per row pass;
//                                        otherwise final commits (APPEND: rows, arrays: elements)
//
// Control:
//   G_IDLE -> G_SETUP -> G_PASS -> G_UOP -> G_GRP (A scale x np, A payload, B left, B right)
//   -> ... -> G_CSEG (C) -> G_TAIL (V^T old reads) -> G_NEXT -> G_PASS ... -> G_DRAIN (every pass
//   retired) -> G_FINAL (element commits) -> G_DONE
module g3_gemm_seq (
    input  wire          clk,
    input  wire          rstn,
    input  wire          start,
    input  wire [383:0]  pd,
    input  wire          abort,
    input  wire          ev_valid,
    input  wire [7:0]    ev_idx,
    input  wire [7:0]    ev_gen,
    input  wire [12:0]   ev_wm,
    output reg           cm_valid,
    input  wire          cm_ready,
    output reg  [7:0]    cm_idx,
    output reg  [7:0]    cm_gen,
    output reg  [12:0]   cm_wm,
    output reg           cm_final,
    // UB segments: ports A, B, C.
    output reg           sa_valid,
    input  wire          sa_ready,
    output reg  [127:0]  sa,
    output reg           sb_valid,
    input  wire          sb_ready,
    output reg  [127:0]  sb,
    output reg           sc_valid,
    input  wire          sc_ready,
    output reg  [127:0]  sc,
    // Uops to the core.
    output reg           uop_valid,
    input  wire          uop_ready,
    output reg  [255:0]  uop,
    input  wire          mark_retired,
    // STREAM.
    output reg  [1:0]    p_start,
    output reg  [319:0]  p_prog,
    output reg           wb_start,
    output reg  [31:0]   wb_cells,
    output reg           sources_closed,
    output reg           done,
    output reg  [3:0]    done_status,
    output reg           busy
);
    `include "g3_defs.vh"
    function automatic [11:0] block_word(input [15:0] n);
        block_word = 33 * (n >> 2) + 1 + 8 * n[1:0];
    endfunction
    function automatic [11:0] scale_word(input [15:0] n);
        scale_word = 33 * (n >> 2);
    endfunction
    function automatic [127:0] mkseg(input [1:0] ctx, input [11:0] start_, input [1:0] n0m1, input [11:0] o1,
                                     input [11:0] o2, input [11:0] o3, input [11:0] n1, input [11:0] s1,
                                     input [7:0] user);
        mkseg = {18'd0, 1'b0, user, s1, n1 - 12'd1, o3, o2, o1, n0m1, start_, 13'd4096, 12'd0, ctx};
    endfunction
    // user: [7] scale [6] set_last [5] full [4:3] dst panel [2:1] slot [0] half
    function automatic [7:0] su(input set_last, input full, input [1:0] dst, input [1:0] slot, input hf);
        su = {1'b1, set_last, full, dst, slot, hf};
    endfunction

    // RNE(1/n) for n = 32 G (gen3_rtl/core/vru/g3_vru.v inv_count): the fold producer's 1/H.
    function automatic [31:0] inv_count(input [13:0] nval);
        integer e, i2;
        reg [63:0] num, q, rem;
        begin
            e = 0;
            for (i2 = 0; i2 < 14; i2 = i2 + 1) if (nval[i2]) e = i2;
            if (nval == (14'd1 << e)) inv_count = {1'b0, 8'(127 - e), 23'd0};
            else begin
                num = 64'd1 << (e + 1 + 23);
                q = num / nval; rem = num % nval;
                if (rem * 2 > nval || (rem * 2 == nval && q[0])) q = q + 1;
                if (q[24]) inv_count = {1'b0, 8'(127 - e), 23'd0};
                else inv_count = {1'b0, 8'(127 - e - 1), q[22:0]};
            end
        end
    endfunction

    reg [383:0] pd_r;
    reg [12:0] a_wm_r, b_wm_r, c_wm_r, f_wm_r;
    reg [31:0] inv_r;
    // Shape.
    wire [11:0] m = pd_r[`PD_M], n = pd_r[`PD_N];
    wire [8:0]  g = pd_r[`PD_G];
    wire [3:0]  epi = pd_r[`PD_EPI];
    wire        gated = epi == `G3_EPI_SILU_MUL;
    wire        rope = epi == `G3_EPI_ROPE;
    wire        has_c = pd_r[`PD_C_VALID];
    wire        stream = pd_r[`PD_B_STREAM];
    wire [7:0]  nt = 8'((n + 31) >> 5);                       // output groups
    wire [8:0]  nbands = gated ? {nt, 1'b0} : {1'b0, nt}; // B bands
    wire [7:0]  cpn = gated ? nt : (nt + 1) >> 1;         // column passes
    wire [7:0]  rpn = 8'((m + 31) >> 5);                       // row passes
    wire [8:0]  panels = (m + 7) >> 3;
    wire [3:0]  a_heads = pd_r[`PD_AHEADS] + 1'b1;
    wire [3:0]  d_heads = pd_r[`PD_DHEADS] + 1'b1;
    wire [8:0]  a_gh = g / a_heads;
    wire [3:0]  alen = pd_r[`PD_D_ALEN] + 1'b1;
    wire        d_array = alen > 1;
    wire        d_append = pd_r[`PD_D_APPEND];
    wire        d_p32 = pd_r[`PD_D_P32];
    wire        d_tr = pd_r[`PD_D_TR];
    wire [11:0] arow = pd_r[`PD_D_AROW];
    wire [8:0]  d_g = nt / d_heads;
    wire [12:0] elem_ext = pd_r[`PD_D_EXT];
    wire        simple_d = !d_array && !d_append && d_heads == 1 && !d_p32 && !d_tr;
    wire        fp = pd_r[`PD_FP], fc = pd_r[`PD_FC];
    wire [7:0]  f_idx = fp ? pd_r[`PD_C_IDX] + 8'd1 : pd_r[`PD_A_IDX] + 8'd1;
    wire [7:0]  f_gen = fp ? pd_r[`PD_C_GEN] : pd_r[`PD_A_GEN];
    wire [8:0]  c_g = rope ? d_g : {1'b0, nt};
    wire [8:0]  nb_left = (nbands + 1) >> 1, nb_right = nbands >> 1;

    // ------------------------------------------------------------------
    // Walk state.
    // ------------------------------------------------------------------
    localparam [3:0] G_IDLE = 0, G_SETUP = 1, G_PASS = 2, G_UOP = 3, G_GRP = 4, G_CSEG = 5, G_TAIL = 6,
                     G_NEXT = 7, G_DRAIN = 8, G_FINAL = 9, G_DONE = 10, G_GSEG = 11, G_RSEG = 12;
    reg [3:0] state_r;
    reg [7:0] rp_r, cp_r;
    reg [8:0] group_r, chunk_end_r;
    reg [3:0] seg_i_r;
    reg [15:0] passes_retired_r, passes_issued_r;
    reg [7:0] rp_committed_r;
    reg [3:0] fin_elem_r;
    reg       aborted_r;
    // Current pass.
    wire [11:0] rows_here = m - 32 * rp_r;
    wire [5:0]  rows_n = rows_here > 32 ? 6'd32 : rows_here[5:0];
    wire [31:0] rows_mask = rows_n == 6'd32 ? 32'hffff_ffff : (32'h1 << rows_n) - 1;
    wire [8:0]  pan_here = panels - 4 * rp_r;
    wire [2:0]  np = pan_here > 4 ? 3'd4 : pan_here[2:0];
    wire [8:0]  bl = gated ? {cp_r, 1'b0} : {cp_r, 1'b0};             // B bands of the pass
    wire [8:0]  br = bl + 1'b1;
    wire        right_b = br < nbands;
    wire [11:0] cols_l = gated ? n - 32 * cp_r : n - 64 * cp_r;
    wire [5:0]  nl = cols_l > 32 ? 6'd32 : cols_l[5:0];
    wire [11:0] cols_r = gated ? cols_l : (cols_l > 32 ? cols_l - 12'd32 : 12'd0);
    wire [5:0]  nr = !right_b ? 6'd0 : cols_r > 32 ? 6'd32 : cols_r[5:0];
    wire [8:0]  g_rem = g - group_r;
    wire [5:0]  kg = g_rem > 9'd32 ? 6'd32 : g_rem[5:0];
    wire        last_chunk = g_rem <= 9'd32;
    // A blocks of group group_r: head ah = g / a_gh, local gl; panel P = ah * a_hp + 4 rp + p.
    wire [8:0]  a_h = group_r / a_gh;
    wire [8:0]  a_gl = group_r % a_gh;
    // A block of (head ah, local group agl, panel rp4 + p): (ah * panels + rp4 + p) * a_gh + agl.
    function automatic [15:0] a_blk(input [8:0] ah, input [8:0] agl, input [11:0] rp4, input [1:0] p,
                                    input [8:0] pan_n, input [8:0] gh);
        a_blk = (16'(ah) * 16'(pan_n) + 16'(rp4) + 16'(p)) * 16'(gh) + 16'(agl);
    endfunction
    wire [15:0] an [0:3];
    for (genvar p = 0; p < 4; p = p + 1) begin : g_an
        assign an[p] = a_blk(a_h, a_gl, 12'(4 * rp_r), 2'(p), panels, a_gh);
    end
    // Output groups of the pass per half (gated: one tile, carried by the right half).
    wire        h0_valid = !gated && {cp_r, 1'b0} < nt;
    wire        h1_valid = gated ? 1'b1 : {cp_r, 1'b0} + 1 < nt;
    wire [8:0]  go0 = {cp_r, 1'b0}, go1 = gated ? {1'b0, cp_r} : {cp_r, 1'b0} + 1'b1;
    wire [8:0]  cg0 = rope ? go0 % d_g : go0, cg1 = rope ? go1 % d_g : go1;
    // Output placement per half: {element base, first panel, local group}.
    function automatic [32:0] place(input [8:0] go, input [3:0] dhs, input [8:0] dg, input arr, input [11:0] dbase,
                                    input [11:0] eext, input tr, input app, input [11:0] ar, input m1,
                                    input [7:0] rp, input [8:0] pan_n);
        reg [3:0] dh;
        reg [8:0] gl;
        reg [11:0] base, pan;
        begin
            dh = dhs > 1 ? go / dg : 4'd0;
            gl = dhs > 1 ? go % dg : go;
            base = arr ? dbase + dh * eext : dbase;
            if (tr) pan = 4 * ((ar >> 5) + rp);
            else if (app) pan = (ar >> 3) + (m1 ? 12'd0 : 4 * rp);
            else if (arr) pan = 4 * rp;
            else pan = dh * pan_n + 4 * rp;
            place = {base, pan, gl};
        end
    endfunction
    wire [32:0] pl0 = place(go0, d_heads, d_g, d_array, pd_r[`PD_D_BASE], elem_ext[11:0], d_tr, d_append, arow,
                            m == 12'd1, rp_r, panels);
    wire [32:0] pl1 = place(go1, d_heads, d_g, d_array, pd_r[`PD_D_BASE], elem_ext[11:0], d_tr, d_append, arow,
                            m == 12'd1, rp_r, panels);
    // Watermark needs (words, or rows for APPEND): A rows of the pass, B cell, C tile.
    wire [15:0] a_last = a_blk(a_heads - 1, a_gh - 1, 12'(4 * rp_r + np - 1), 2'd0, panels, a_gh);
    wire [12:0] a_need = 33 * ((a_last >> 2) + 1);
    wire [8:0]  b_band = seg_i_r >= 4'd7 ? br : bl;
    wire [12:0] b_need = 33 * (b_band * g + group_r + 1);
    wire [15:0] c_last = (4 * rp_r + np - 1) * c_g + (h1_valid ? cg1 : cg0);
    wire [12:0] c_need = 33 * ((c_last >> 2) + 1);
    wire a_ok = a_wm_r == `G3_WM_READY || a_wm_r >= (a_need > 13'd4096 ? 13'd4096 : a_need);
    wire b_ok = stream || b_wm_r == `G3_WM_READY || b_wm_r >= b_need;
    wire c_ok = !has_c || c_wm_r == `G3_WM_READY || c_wm_r >= c_need;
    // Fold inputs: g block of the step's band (one row: block = band); the row pass's 1/rms words.
    wire [8:0]  g_band = seg_i_r[1] ? br : bl;
    wire [12:0] g_need = 33 * ((g_band >> 2) + 1);
    wire [12:0] r_need = 13'(4 * (rp_r + 1));
    wire f_ok = f_wm_r == `G3_WM_READY || f_wm_r >= (state_r == G_GSEG ? g_need : r_need);

    // Segment of the current step. G_GRP: 0..3 A scale p, 4 A payload, 5 B left scale, 6 B left payload,
    // 7 B right scale, 8 B right payload. G_CSEG: 0..3 C scale (half 0), 4 payload, 5..8 scale (half 1),
    // 9 payload. G_TAIL: 0, 1 first read (half 0, 1), 2, 3 second read.
    reg [1:0]   seg_port;          // 0 A, 1 B, 2 C
    reg         seg_here;          // the step issues a segment (else it is skipped)
    reg [127:0] seg_d;
    always @* begin : seg_mux
        reg [15:0] nb, nb0;
        reg [8:0] band;
        reg hf;
        reg [2:0] pi;
        reg [8:0] cg;
        seg_port = 0; seg_here = 1'b0; seg_d = 0; nb = 0; nb0 = 0; band = 0; hf = 0; pi = 0; cg = 0;
        case (state_r)
        G_GRP: begin
            if (seg_i_r < 4) begin
                seg_port = 0; seg_here = seg_i_r < np;
                seg_d = mkseg(2'd0, pd_r[`PD_A_BASE] + scale_word(an[seg_i_r[1:0]]), 2'd0, 0, 0, 0, 12'd1, 12'd0,
                              su(seg_i_r + 1 == np, 1'b0, seg_i_r[1:0], an[seg_i_r[1:0]][1:0], 1'b0));
            end else if (seg_i_r == 4) begin
                seg_port = 0; seg_here = 1'b1;
                seg_d = mkseg(2'd0, pd_r[`PD_A_BASE] + block_word(an[0]), 2'(np - 1),
                              block_word(an[1]) - block_word(an[0]), block_word(an[2]) - block_word(an[0]),
                              block_word(an[3]) - block_word(an[0]), 12'd8, 12'd1, 8'd0);
            end else begin
                hf = seg_i_r >= 7;
                band = hf ? br : bl;
                seg_port = 1; seg_here = !stream && (!hf || right_b);
                nb = band * g + group_r;            // cell (quad) index
                if (seg_i_r == 5 || seg_i_r == 7)
                    seg_d = mkseg({1'b0, hf}, pd_r[`PD_B_BASE] + 33 * nb, 2'd0, 0, 0, 0, 12'd1, 12'd0,
                                  su(1'b1, 1'b1, 2'd0, 2'd0, hf));
                else
                    seg_d = mkseg({1'b0, hf}, pd_r[`PD_B_BASE] + 33 * nb + 1, 2'd3, 12'd8, 12'd16, 12'd24,
                                  12'd8, 12'd1, 8'd0);
            end
        end
        G_CSEG: begin
            hf = seg_i_r >= 5;
            pi = hf ? seg_i_r - 5 : seg_i_r;
            cg = hf ? cg1 : cg0;
            seg_port = 2;
            seg_here = has_c && (hf ? h1_valid : h0_valid) && (pi == 4 || pi < np);
            nb0 = (4 * rp_r) * c_g + cg;
            nb = (4 * rp_r + pi) * c_g + cg;
            if (pi < 4)
                seg_d = mkseg(2'd2, pd_r[`PD_C_BASE] + scale_word(nb), 2'd0, 0, 0, 0, 12'd1, 12'd0,
                              su(1'b0, 1'b0, pi[1:0], nb[1:0], hf));
            else
                seg_d = mkseg(2'd2, pd_r[`PD_C_BASE] + block_word(nb0), 2'(np - 1),
                              block_word(nb0 + c_g) - block_word(nb0), block_word(nb0 + 2 * c_g) - block_word(nb0),
                              block_word(nb0 + 3 * c_g) - block_word(nb0), 12'd8, 12'd1,
                              {1'b0, hf || !h1_valid, 5'd0, hf});
        end
        G_GSEG: begin
            hf = seg_i_r[1];
            seg_port = 2;
            seg_here = hf ? h1_valid : h0_valid;
            nb = {7'd0, g_band};
            if (!seg_i_r[0])
                seg_d = mkseg(2'd1, pd_r[`PD_G_BASE] + scale_word(nb), 2'd0, 0, 0, 0, 12'd1, 12'd0,
                              su(1'b0, 1'b0, 2'd0, nb[1:0], hf));
            else
                seg_d = mkseg(2'd1, pd_r[`PD_G_BASE] + block_word(nb), 2'd0, 0, 0, 0, 12'd8, 12'd1,
                              {1'b0, hf || !h1_valid, 5'd0, hf});
        end
        G_RSEG: begin
            seg_port = 2;
            seg_here = 1'b1;
            seg_d = mkseg(2'd1, pd_r[`PD_R_BASE] + 12'(4 * rp_r), 2'd0, 0, 0, 0, 12'd4, 12'd1,
                          {1'b0, 1'b1, 1'b1, 4'd0, rp_r[0]});
        end
        G_TAIL: begin
            hf = seg_i_r[0];
            seg_port = 2;
            seg_here = hf ? h1_valid : h0_valid;
            nb = hf ? ((pl1[20:9] >> 2) * d_g * 4 + pl1[8:0] * 4) : ((pl0[20:9] >> 2) * d_g * 4 + pl0[8:0] * 4);
            seg_d = mkseg(2'd0, (hf ? pl1[32:21] : pl0[32:21]) + scale_word(nb), 2'd0, 0, 0, 0, 12'd33, 12'd1, 8'd0);
        end
        default: ;
        endcase
    end
    wire seg_busy = (sa_valid && seg_port == 0) || (sb_valid && seg_port == 1) || (sc_valid && seg_port == 2);
    wire seg_slot = !seg_busy;
    wire vt_tail = d_tr && m == 12'd1 && arow[4:0] != 0;
    // Last step indices.
    wire grp_last_step = seg_i_r == 4'd8;
    wire group_last = group_r + 1 == chunk_end_r;
    wire groups_done = chunk_end_r == g;
    wire drained = !sa_valid && !sb_valid && !sc_valid && !uop_valid && passes_retired_r == passes_issued_r;
    wire step_ok = state_r == G_GRP ? (seg_i_r < 5 || seg_i_r > 8 || b_ok || !seg_here) :
                   state_r == G_CSEG ? c_ok : (state_r == G_GSEG || state_r == G_RSEG) ? f_ok : 1'b1;
    wire step_go = (state_r == G_GRP || state_r == G_CSEG || state_r == G_TAIL || state_r == G_GSEG ||
                    state_r == G_RSEG) && !abort && (!seg_here || (seg_slot && step_ok));
    wire uop_go = state_r == G_UOP && !abort && !uop_valid && a_ok;

    // Output descriptor of the pass (emitting uop).
    wire [1:0]  layout = d_tr ? `G3_LAYOUT_VT : d_p32 ? `G3_LAYOUT_P32 : `G3_LAYOUT_P8;
    wire        app1 = d_append && !d_tr && m == 12'd1;
    reg  [255:0] u_w;
    always @* begin
        u_w = 0;
        u_w[`U_OP] = 2'd0;
        u_w[`U_KG] = kg;
        u_w[`U_ACCUM] = group_r != 0;
        u_w[`U_EMIT] = last_chunk;
        u_w[`U_ROWS] = rows_mask;
        u_w[`U_NL] = nl;
        u_w[`U_NR] = nr;
        u_w[`U_ASRC] = 1'b0;
        u_w[`U_BSRC] = stream;
        u_w[`U_EPI] = epi;
        u_w[`U_HASC] = has_c;
        u_w[`U_LAYOUT] = layout;
        u_w[`U_GD] = d_g;
        u_w[`U_H0] = h0_valid;
        u_w[`U_H1] = h1_valid;
        u_w[`U_BASE0] = pl0[32:21]; u_w[`U_PAN0] = pl0[20:9]; u_w[`U_G0] = pl0[8:0];
        u_w[`U_BASE1] = pl1[32:21]; u_w[`U_PAN1] = pl1[20:9]; u_w[`U_G1] = pl1[8:0];
        u_w[`U_NPAN] = app1 ? 3'd1 : np;
        u_w[`U_APP1] = app1;
        u_w[`U_AROW8] = arow[2:0];
        u_w[`U_VTT] = arow[4:0];
        u_w[`U_VTTAIL] = vt_tail;
        u_w[`U_MARK] = 1'b1;
        u_w[`U_FP] = fp;
        u_w[`U_FC] = fc;
        u_w[`U_PF] = cp_r == 0;
        u_w[`U_PL] = cp_r + 1 == cpn;
        u_w[`U_RPAR] = rp_r[0];
        u_w[`U_EPS] = pd_r[`PD_EPS];
        u_w[`U_XGD] = pd_r[`PD_XG_BASE] - pd_r[`PD_D_BASE];
        if (fp) begin
            u_w[`U_RBASE] = pd_r[`PD_R_BASE] + 12'(4 * rp_r);
            u_w[`U_INV] = inv_r;
        end
    end

    // ------------------------------------------------------------------
    // FSM.
    // ------------------------------------------------------------------
    wire        rp_done_now = simple_d && rp_committed_r < rpn &&
                              passes_retired_r >= (rp_committed_r + 1) * cpn;
    // Final element commits: arrays (each element), APPEND (rows); a fold producer's x * g and 1/rms.
    wire        fin_all = simple_d ? (!fp || fin_elem_r == 4'd2) : fin_elem_r == (d_array ? alen : 4'd1);
    function automatic [12:0] band_wm(input [7:0] rp, input [8:0] pan_n, input [7:0] ntg);
        reg [19:0] blocks;
        begin
            blocks = ((4 * rp + 4) > pan_n ? pan_n : (4 * rp + 4)) * ntg;
            band_wm = 33 * (blocks >> 2);
        end
    endfunction
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state_r <= G_IDLE; cm_valid <= 0; sa_valid <= 0; sb_valid <= 0; sc_valid <= 0; uop_valid <= 0;
            p_start <= 0; wb_start <= 0; sources_closed <= 0; done <= 0; busy <= 0; done_status <= 0;
            rp_r <= 0; cp_r <= 0; group_r <= 0; seg_i_r <= 0; passes_retired_r <= 0; passes_issued_r <= 0;
            rp_committed_r <= 0; fin_elem_r <= 0; aborted_r <= 0; chunk_end_r <= 0;
        end else begin
            p_start <= 0; wb_start <= 0; sources_closed <= 0; done <= 0;
            if (sa_valid && sa_ready) sa_valid <= 1'b0;
            if (sb_valid && sb_ready) sb_valid <= 1'b0;
            if (sc_valid && sc_ready) sc_valid <= 1'b0;
            if (uop_valid && uop_ready) uop_valid <= 1'b0;
            if (cm_valid && cm_ready) cm_valid <= 1'b0;
            if (mark_retired && state_r != G_IDLE) passes_retired_r <= passes_retired_r + 1'b1;
            // Per row pass commit (P8 single tensor).
            if (rp_done_now && !cm_valid) begin
                cm_valid <= 1'b1; cm_idx <= pd_r[`PD_D_IDX]; cm_gen <= pd_r[`PD_D_GEN];
                cm_wm <= rp_committed_r + 1 == rpn ? pd_r[`PD_D_EXT] : band_wm(rp_committed_r, panels, nt);
                cm_final <= rp_committed_r + 1 == rpn;
                rp_committed_r <= rp_committed_r + 1'b1;
            end
            if (step_go && seg_here) begin
                case (seg_port)
                2'd0: begin sa_valid <= 1'b1; sa <= seg_d; end
                2'd1: begin sb_valid <= 1'b1; sb <= seg_d; end
                default: begin sc_valid <= 1'b1; sc <= seg_d; end
                endcase
            end
            case (state_r)
            G_IDLE: if (start) begin
                state_r <= G_SETUP; busy <= 1'b1; pd_r <= pd; aborted_r <= 1'b0;
                a_wm_r <= pd[`PD_A_WM]; b_wm_r <= pd[`PD_B_WM]; c_wm_r <= pd[`PD_C_WM]; f_wm_r <= pd[`PD_F_WM];
                passes_retired_r <= 0; passes_issued_r <= 0; rp_committed_r <= 0; fin_elem_r <= 0;
            end
            G_SETUP: begin
                state_r <= G_PASS; rp_r <= 0; cp_r <= 0;
                inv_r <= inv_count(14'({6'd0, nt} * 14'd32));
                if (stream) begin
                    p_start <= {nb_right != 0, 1'b1};
                    wb_start <= 1'b1;
                    wb_cells <= {16'(rpn * nb_right * g), 16'(rpn * nb_left * g)};
                    // {src, tile_words, stride, ntiles, rounds, ring_base, ring_ext}
                    p_prog[159:0] <= {41'd0, {pd_r[`PD_B_SRC], 5'd0}, 14'(33 * g), 16'(66 * g), nb_left[7:0], rpn,
                                      12'd0, 13'd132};
                    p_prog[319:160] <= {41'd0, {pd_r[`PD_B_SRC], 5'd0} + 48'(33 * 32 * g), 14'(33 * g), 16'(66 * g),
                                        nb_right[7:0], rpn, 12'd0, 13'd132};
                end
            end
            G_PASS: begin
                if (abort) state_r <= G_DRAIN;
                else if (rp_r >= rpn) state_r <= G_DRAIN;
                else begin state_r <= (fc && cp_r == 0) ? G_RSEG : G_UOP; group_r <= 0; seg_i_r <= 0; end
            end
            G_RSEG: if (abort) state_r <= G_DRAIN;
                    else if (step_go) state_r <= G_UOP;
            G_UOP: if (abort) state_r <= G_DRAIN;
                   else if (uop_go) begin
                       uop_valid <= 1'b1; uop <= u_w; chunk_end_r <= group_r + kg; seg_i_r <= 0;
                       state_r <= G_GRP;
                       if (last_chunk) passes_issued_r <= passes_issued_r + 1'b1;
                   end
            G_GRP: if (abort) state_r <= G_DRAIN;
                   else if (step_go) begin
                       if (!grp_last_step) seg_i_r <= seg_i_r + 1'b1;
                       else begin
                           seg_i_r <= 0;
                           group_r <= group_r + 1'b1;
                           if (group_last) begin
                               if (!groups_done) state_r <= G_UOP;
                               else if (has_c) state_r <= G_CSEG;
                               else if (vt_tail) state_r <= G_TAIL;
                               else state_r <= G_NEXT;
                           end
                       end
                   end
            G_CSEG: if (abort) state_r <= G_DRAIN;
                    else if (step_go) begin
                        if (seg_i_r != 4'd9) seg_i_r <= seg_i_r + 1'b1;
                        else begin seg_i_r <= 0; state_r <= fp ? G_GSEG : vt_tail ? G_TAIL : G_NEXT; end
                    end
            G_GSEG: if (abort) state_r <= G_DRAIN;
                    else if (step_go) begin
                        if (seg_i_r != 4'd3) seg_i_r <= seg_i_r + 1'b1;
                        else begin seg_i_r <= 0; state_r <= G_NEXT; end
                    end
            G_TAIL: if (abort) state_r <= G_DRAIN;
                    else if (step_go) begin
                        if (seg_i_r != 4'd3) seg_i_r <= seg_i_r + 1'b1;
                        else begin seg_i_r <= 0; state_r <= G_NEXT; end
                    end
            G_NEXT: begin
                state_r <= G_PASS;
                if (cp_r + 1 == cpn) begin cp_r <= 0; rp_r <= rp_r + 1'b1; end
                else cp_r <= cp_r + 1'b1;
            end
            G_DRAIN: if (drained || abort || aborted_r) begin
                if (abort || aborted_r) begin state_r <= G_DONE; done_status <= `G3_ST_ABORTED; end
                else state_r <= G_FINAL;
            end
            G_FINAL: if (!cm_valid && !rp_done_now) begin
                if (fin_all) begin
                    state_r <= G_DONE; done_status <= `G3_ST_OK; sources_closed <= 1'b1;
                end else begin
                    cm_valid <= 1'b1; cm_gen <= pd_r[`PD_D_GEN]; cm_final <= 1'b1;
                    cm_idx <= pd_r[`PD_D_IDX] + (simple_d ? 8'd1 : 8'd0) + {4'd0, fin_elem_r};
                    cm_wm <= simple_d ? (fin_elem_r == 0 ? elem_ext : pd_r[`PD_R_EXT]) : d_append ? arow + m : elem_ext;
                    fin_elem_r <= fin_elem_r + 1'b1;
                end
            end
            G_DONE: begin state_r <= G_IDLE; done <= 1'b1; busy <= 1'b0; end
            default: state_r <= G_IDLE;
            endcase
            if (abort) aborted_r <= 1'b1;
            // Watch registers.
            if (ev_valid && state_r != G_IDLE) begin
                if (ev_idx == pd_r[`PD_A_IDX] && ev_gen == pd_r[`PD_A_GEN] && a_wm_r != `G3_WM_READY && ev_wm > a_wm_r)
                    a_wm_r <= ev_wm;
                if (ev_idx == pd_r[`PD_B_IDX] && ev_gen == pd_r[`PD_B_GEN] && b_wm_r != `G3_WM_READY && ev_wm > b_wm_r)
                    b_wm_r <= ev_wm;
                if (ev_idx == pd_r[`PD_C_IDX] && ev_gen == pd_r[`PD_C_GEN] && c_wm_r != `G3_WM_READY && ev_wm > c_wm_r)
                    c_wm_r <= ev_wm;
                if ((fp || fc) && ev_idx == f_idx && ev_gen == f_gen && f_wm_r != `G3_WM_READY && ev_wm > f_wm_r)
                    f_wm_r <= ev_wm;
            end
        end
    end
endmodule

`default_nettype wire
