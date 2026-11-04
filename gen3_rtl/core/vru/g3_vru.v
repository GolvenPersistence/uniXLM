`timescale 1ns/1ps
`default_nettype none

// Gen3 VRU: the Gen2 streaming VRU (rtl/gen2/g2_vru.v, docs/gen2/vector_and_numerics.md s3) unchanged:
// row RMSNorm. It issues Gen2 96-bit UB segments; g3_node widens them to the Gen3 format.
//
// Block diagram (one 8-row panel of MAIN at a time, W = 32*G columns):
//
//   seg_* --> UB port 4 --> eg_* --> MAIN buffer / AUX buffer --> block reader (scale + 8 words)
//                                                                    |
//       PASS1: decode --> 16 lanes: x*x --> acc[r][k%16] += ----------+
//       FINAL: per row: pairwise reduce 16 lanes, * RNE(1/W), + eps, rsqrt
//              (magic + 2 Newton steps); one mul and one add unit, rows interleaved (fin_rf_mem)
//       PASS2: MAIN and AUX (gamma row) re-read --> ((x*s)+0)*gamma+0 --> output block
//              (double buffer, 8 x 32) --> per-row MXFP8 quantise --> 8 words + scale beat
//                                                                    |
//                                                     wb_* (UB port 4) --> RESULT
//   ev_* --> watch registers MAIN, AUX           mark_retired --> panel commit (cm_*)
//
// Control:
//
//   V_IDLE --start--> V_DIV (clear accumulators) --> V_P1 (read + accumulate)
//     --> V_FIN (micro-program, 19 steps) --> V_P2 (re-read, scale, quantise, write)
//     --> V_COMMIT (panel watermark after its writes retired)
//     --> next panel: V_DIV | last panel: V_DONE (final commit, sources_closed, done)
//
// No replay storage and no FP16 rounding point (Gen1 differs exactly there).
module g3_vru (
    input  wire          clk,
    input  wire          rstn,
    input  wire          start,
    input  wire [255:0]  desc,         // see field map below
    output reg           done,
    output reg  [3:0]    done_status,
    output reg           sources_closed,
    // Event broadcast (watch MAIN / AUX).
    input  wire          ev_valid,
    input  wire [7:0]    ev_idx,
    input  wire [7:0]    ev_gen,
    input  wire [12:0]   ev_wm,
    // Commit events.
    output reg           cm_valid,
    input  wire          cm_ready,
    output reg  [7:0]    cm_idx,
    output reg  [7:0]    cm_gen,
    output reg  [12:0]   cm_wm,
    output reg           cm_final,
    // UB port (V): segments, egress, credits, ingress, retirement.
    output reg           seg_valid,
    input  wire          seg_ready,
    output reg  [95:0]   seg,
    input  wire          eg_valid,
    input  wire [1:0]    eg_stream,
    input  wire [255:0]  eg_data,
    output wire [2:0]    credit_return,
    output wire          wb_valid,
    input  wire          wb_ready,
    output wire [11:0]   wb_addr,
    output wire [255:0]  wb_data,
    output wire [31:0]   wb_strb,
    output wire          wb_mark,
    input  wire          mark_retired,
    output wire          busy
);
    `include "g3_numeric_functions.vh"
    `include "g3_defs.vh"

    // Descriptor (LSB first):
    //   MAIN   [7:0] idx  [15:8] gen  [27:16] base  [40:28] wm
    //   AUX    [48:41] idx  [56:49] gen  [68:57] base  [81:69] wm
    //   RESULT [89:82] idx  [97:90] gen  [109:98] base  [122:110] ext
    //   [134:123] rows  [143:135] groups  [145:144] eps
    wire [12:0] main_wm0 = desc[40:28];
    wire [12:0] aux_wm0 = desc[81:69];
    wire [8:0]  groups = desc[143:135];

    reg [255:0] desc_r;
    reg [12:0] main_wm_r, aux_wm_r;
    wire [11:0] r_rows = desc_r[134:123];
    wire [8:0]  r_g = desc_r[143:135];
    wire [11:0] r_main_base = desc_r[27:16], r_aux_base = desc_r[68:57], r_res_base = desc_r[109:98];
    wire [12:0] r_res_ext = desc_r[122:110];
    wire [8:0]  panels = (r_rows + 7) >> 3;

    function automatic [11:0] block_word(input [15:0] nb);
        block_word = 33 * (nb >> 2) + 1 + 8 * nb[1:0];
    endfunction
    function automatic [11:0] scale_word(input [15:0] nb);
        scale_word = 33 * (nb >> 2);
    endfunction
    function automatic [95:0] mkseg(input [1:0] ctx, input [11:0] start, input [11:0] n1, input [7:0] user);
        mkseg = {6'd0, 1'b0, 4'd0, user, 12'd1, n1 - 12'd1, 12'd0, 2'd0, start, 13'd4096, 12'd0, ctx};
    endfunction

    // ------------------------------------------------------------------
    // Input buffers (8 beats per stream: MAIN 0, AUX 2).
    // ------------------------------------------------------------------
    reg [255:0] main_buf_mem [0:7], aux_buf_mem [0:7];
    reg [2:0] main_rd_ptr_r, main_wr_ptr_r, aux_rd_ptr_r, aux_wr_ptr_r;
    reg [3:0] main_level_r, aux_level_r;
    reg main_pop_r, aux_pop_r;                    // credits return the cycle after a pop
    assign credit_return = {aux_pop_r, 1'b0, main_pop_r};
    wire [255:0] m_head = main_buf_mem[main_rd_ptr_r];
    wire [255:0] a_head = aux_buf_mem[aux_rd_ptr_r];

    // ------------------------------------------------------------------
    // Pass 1 lanes: decode, square, accumulate (mul 4 + add 3 stages; the
    // same (row, lane) recurs every 8 half-words, after the pipeline).
    // ------------------------------------------------------------------
    reg [31:0] acc_mem [0:7][0:15];
    reg [63:0] row_scale_r; // current block's 8 row scales
    reg [63:0] g_scale_r;   // gamma row scale (row 0 of the AUX block)
    reg [2:0] word_idx_r;   // word within the block
    reg half_r;             // half-word within the word
    reg [1:0] phase_r;      // 0 PASS1, 1 PASS2
    wire [15:0] lane_valid;
    wire [16*32-1:0] lane_sum;
    wire [16*5-1:0] lane_col;
    reg run_block_r, have_scale_r;
    reg ob_wr_sel_r, ob_rd_sel_r;                  // output block buffer being written / quantised
    wire ob_free;
    wire issue1 = phase_r == 2'd0 && run_block_r && main_level_r != 0 && have_scale_r;
    wire issue2 = phase_r == 2'd1 && run_block_r && main_level_r != 0 && aux_level_r != 0 && have_scale_r && ob_free;
    reg [31:0] s_row_r [0:7];          // per-row inverse std
    localparam [2:0] V_IDLE = 0, V_DIV = 1, V_P1 = 2, V_FIN = 3, V_P2 = 4, V_COMMIT = 5, V_DONE = 6;
    reg [2:0] state_r, state_n;
    for (genvar l = 0; l < 16; l = l + 1) begin : g_lane
        localparam integer R = l / 2;
        localparam integer J = l % 2;
        wire [7:0] code = m_head[half_r*128 + R*16 + J*8 +: 8];
        wire [31:0] x = decode_element(code, row_scale_r[R*8 +: 8]);
        wire [7:0] gcode = a_head[half_r*128 + J*8 +: 8];        // gamma: row 0 lane
        wire [31:0] gm = decode_element(gcode, g_scale_r[7:0]);
        wire [3:0] lane16 = {word_idx_r[1:0], half_r, 1'b0} + J;     // k % 16 for k = 4w+2h+J
        // PASS1: sq = x*x -> acc += sq.
        wire sq_valid; wire [31:0] sq_result; wire [3:0] sq_tag;
        rc_mxfp8_fp32_mul_staged #(
            .TAG_WIDTH (4)
        ) u_sq (
            .clk       (clk),
            .rst_n     (rstn),
            .advance   (1'b1),
            .in_valid  (issue1),
            .in_tag    (lane16),
            .left      (x),
            .right     (x),
            .out_valid (sq_valid),
            .out_tag   (sq_tag),
            .result    (sq_result)
        );
        wire acc_valid; wire [31:0] acc_result; wire [3:0] acc_tag;
        rc_mxfp8_fp32_add_staged #(
            .TAG_WIDTH (4)
        ) u_acc (
            .clk       (clk),
            .rst_n     (rstn),
            .advance   (1'b1),
            .in_valid  (sq_valid),
            .in_tag    (sq_tag),
            .left      (acc_mem[R][sq_tag]),
            .right     (sq_result),
            .out_valid (acc_valid),
            .out_tag   (acc_tag),
            .result    (acc_result)
        );
        // One writer per accumulator word (word 2a + J of row R; acc_tag[0] == J): cleared in V_DIV.
        for (genvar a = 0; a < 8; a = a + 1) begin : g_acc_word
            always @(posedge clk) begin
                if (state_r == V_DIV) acc_mem[R][2*a + J] <= 32'd0;
                else if (acc_valid && acc_tag[3:1] == a) acc_mem[R][2*a + J] <= acc_result;
            end
        end
        // PASS2: ((x*s)+0)*gamma+0, with the output column carried as tag.
        wire [4:0] col = {word_idx_r, half_r, 1'b0} + J;             // k within the group
        wire p1_valid, p2_valid, p3_valid, p4_valid; wire [31:0] p1_result, p2_result, p3_result, p4_result;
        wire [36:0] p1_tag, p2_tag, p3_tag; wire [4:0] p4_tag;
        rc_mxfp8_fp32_mul_staged #(
            .TAG_WIDTH (37)
        ) u_p1 (
            .clk       (clk),
            .rst_n     (rstn),
            .advance   (1'b1),
            .in_valid  (issue2),
            .in_tag    ({col, gm}),
            .left      (x),
            .right     (s_row_r[R]),
            .out_valid (p1_valid),
            .out_tag   (p1_tag),
            .result    (p1_result)
        );
        rc_mxfp8_fp32_add_staged #(
            .TAG_WIDTH (37)
        ) u_p2 (
            .clk       (clk),
            .rst_n     (rstn),
            .advance   (1'b1),
            .in_valid  (p1_valid),
            .in_tag    (p1_tag),
            .left      (p1_result),
            .right     (32'd0),
            .out_valid (p2_valid),
            .out_tag   (p2_tag),
            .result    (p2_result)
        );
        rc_mxfp8_fp32_mul_staged #(
            .TAG_WIDTH (37)
        ) u_p3 (
            .clk       (clk),
            .rst_n     (rstn),
            .advance   (1'b1),
            .in_valid  (p2_valid),
            .in_tag    (p2_tag),
            .left      (p2_result),
            .right     (p2_tag[31:0]),
            .out_valid (p3_valid),
            .out_tag   (p3_tag),
            .result    (p3_result)
        );
        rc_mxfp8_fp32_add_staged #(
            .TAG_WIDTH (5)
        ) u_p4 (
            .clk       (clk),
            .rst_n     (rstn),
            .advance   (1'b1),
            .in_valid  (p3_valid),
            .in_tag    (p3_tag[36:32]),
            .left      (p3_result),
            .right     (32'd0),
            .out_valid (p4_valid),
            .out_tag   (p4_tag),
            .result    (p4_result)
        );
        assign lane_valid[l] = p4_valid;
        assign lane_sum[l*32 +: 32] = p4_result;
        assign lane_col[l*5 +: 5] = p4_tag;       // output column
    end
    wire out_valid = lane_valid[0];
    // In-flight tracking of the pass-1 and pass-2 chains.
    reg [7:0] p1_pipe_r;
    wire pass1_busy = |p1_pipe_r;

    // Results belong to the buffer being written when they were issued; with
    // ob_wr_sel_r flipping only after a block's last issue, results of that block
    // may still land after the flip: track the issue buffer through the tag.
    // (Simplification: the issue buffer is carried in a shift register of the
    // chain latency.)
    localparam integer CHAIN = 4 + 3 + 4 + 3;
    reg [CHAIN-1:0] obw_pipe_r, iss_pipe_r;
    always @(posedge clk or negedge rstn)
        if (!rstn) begin obw_pipe_r <= 0; iss_pipe_r <= 0; end
        else begin
            obw_pipe_r <= {obw_pipe_r[CHAIN-2:0], ob_wr_sel_r};
            iss_pipe_r <= {iss_pipe_r[CHAIN-2:0], issue2};
        end
    wire ob_wr_sel_res = obw_pipe_r[CHAIN-1];
    wire p2_inflight = |iss_pipe_r;

    // ------------------------------------------------------------------
    // Output block buffers (double) and per-row quantisation.
    // ------------------------------------------------------------------
    reg [31:0] ob_mem [0:1][0:7][0:31];
    reg [4:0] ob_cnt_r [0:1];          // results still to land in each buffer
    reg [1:0] ob_full_r;
    reg [15:0] ob_blk_r [0:1];         // block index for the address
    assign ob_free = !ob_full_r[ob_wr_sel_r];

    // Quantiser: one row per cycle.
    reg q_run_r; reg [3:0] q_row_r; reg [7:0] q_bytes_mem [0:7][0:31]; reg [7:0] q_scale_r [0:7];
    reg [3:0] emit_idx_r; reg emit_run_r;
    reg [255:0] w_data_r; reg [31:0] w_strb_r; reg [11:0] w_addr_r; reg w_mark_r, w_valid_r;
    assign wb_valid = w_valid_r; assign wb_data = w_data_r; assign wb_strb = w_strb_r;
    assign wb_addr = w_addr_r; assign wb_mark = w_mark_r;

    // ------------------------------------------------------------------
    // FINAL: scalar micro-program on one mul and one add unit, rows interleaved.
    // ------------------------------------------------------------------
    reg [31:0] fin_rf_mem [0:7][0:31];
    reg [31:0] inv_count_r;
    reg fin_run_r; reg [4:0] fin_step_r; reg [3:0] fin_row_r; reg [3:0] fin_sub_r;
    reg fin_mul_valid_r, fin_add_valid_r; reg [31:0] fin_a_r, fin_b_r; reg [8:0] fin_tag_r;
    wire fm_valid, fa_valid; wire [31:0] fm_result, fa_result; wire [8:0] fm_tag, fa_tag;
    rc_mxfp8_fp32_mul_staged #(
        .TAG_WIDTH (9)
    ) u_fmul (
        .clk       (clk),
        .rst_n     (rstn),
        .advance   (1'b1),
        .in_valid  (fin_mul_valid_r),
        .in_tag    (fin_tag_r),
        .left      (fin_a_r),
        .right     (fin_b_r),
        .out_valid (fm_valid),
        .out_tag   (fm_tag),
        .result    (fm_result)
    );
    rc_mxfp8_fp32_add_staged #(
        .TAG_WIDTH (9)
    ) u_fadd (
        .clk       (clk),
        .rst_n     (rstn),
        .advance   (1'b1),
        .in_valid  (fin_add_valid_r),
        .in_tag    (fin_tag_r),
        .left      (fin_a_r),
        .right     (fin_b_r),
        .out_valid (fa_valid),
        .out_tag   (fa_tag),
        .result    (fa_result)
    );
    // Micro-program: step -> {op(0 add, 1 mul, 2 int), dst, a, b(reg or const)}.
    //  0: 16..23 = 0..15 pairwise   (8 ops per row)
    //  1: 24..27 = 16..23 pairwise  (4)
    //  2: 28..29                    (2)
    //  3: 30 = 28 + 29              (1)
    //  4: 31 = 30 * inv_count_r
    //  5: 0  = 31 + eps
    //  6: 1  = magic - (0 >> 1)     (integer)
    //  7..12, 13..18: Newton: 2=1*1, 3=0*2, 4=3*0.5, 5=4*-1, 6=5+1.5, 1=1*6
    localparam [31:0] C_HALF = 32'h3f00_0000, C_NEG1 = 32'hbf80_0000, C_THREEHALF = 32'h3fc0_0000;
    reg [3:0] fin_ops_in_step;
    always @* begin
        case (fin_step_r)
            0: fin_ops_in_step = 8; 1: fin_ops_in_step = 4; 2: fin_ops_in_step = 2;
            default: fin_ops_in_step = 1;
        endcase
    end
    reg [7:0] fin_outstanding_r;


    // ------------------------------------------------------------------
    // Top FSM.
    // ------------------------------------------------------------------
    reg [8:0] panel_r, seg_group_r, use_group_r;       // segment issue group, consumption group
    reg [2:0] seg_idx_r;
    reg [15:0] blk_rd_idx_r;
    reg [11:0] marks_pending_r;

    wire [15:0] main_blk_seg = panel_r * r_g + seg_group_r;
    wire [15:0] main_last_blk = (panel_r + 1) * r_g - 1;
    wire main_ok = main_wm_r == `G3_WM_READY || main_wm_r >= 33 * ((main_last_blk >> 2) + 1);
    wire aux_ok = aux_wm_r == `G3_WM_READY || aux_wm_r >= 33 * (((r_g - 1) >> 2) + 1);

    // ------------------------------------------------------------------
    // FSM: state register, next state, control events.
    // ------------------------------------------------------------------
    wire p1_done    = !run_block_r && !pass1_busy && !issue1;              // pass 1 drained
    wire fin_quiet  = fin_outstanding_r == 0 && !fm_valid && !fa_valid && !fin_mul_valid_r && !fin_add_valid_r;
    wire p2_done    = !run_block_r && ob_full_r == 0 && !q_run_r && !emit_run_r && !w_valid_r && !p2_inflight;
    wire commit_ok  = !cm_valid && marks_pending_r == 0;                   // panel writes retired
    wire last_panel = panel_r + 1 == panels;

    always @(posedge clk or negedge rstn)
        if (!rstn) state_r <= V_IDLE;
        else state_r <= state_n;

    always @* begin
        state_n = state_r;
        case (state_r)
        V_IDLE:   if (start) state_n = V_DIV;
        V_DIV:    state_n = V_P1;
        V_P1:     if (p1_done) state_n = V_FIN;
        V_FIN:    if (!fin_run_r) state_n = V_P2;
        V_P2:     if (p2_done) state_n = V_COMMIT;
        V_COMMIT: if (commit_ok) state_n = last_panel ? V_DONE : V_DIV;
        V_DONE:   if (!cm_valid) state_n = V_IDLE;
        default:  state_n = V_IDLE;
        endcase
    end

    wire start_accept  = state_r == V_IDLE && start;
    wire p2_setup      = state_r == V_FIN && !fin_run_r;
    wire pass_setup    = state_r == V_DIV || p2_setup;                     // start pass 1 / pass 2
    wire seg_issue     = !seg_valid && seg_group_r < r_g &&
                         ((state_r == V_P1 && main_ok) || (state_r == V_P2 && aux_ok));
    wire fin_load      = state_r == V_P1 && p1_done;                       // accumulators into the RF
    wire fin_issue_en  = state_r == V_FIN && fin_run_r && fin_row_r < 8;   // one op per cycle
    wire fin_step_done = state_r == V_FIN && fin_run_r && !(fin_row_r < 8) && fin_quiet;
    wire commit_issue  = state_r == V_COMMIT && commit_ok;
    wire done_en       = state_r == V_DONE && !cm_valid;

    // Segment issued this cycle. PASS1 per group: MAIN scale, MAIN payload; PASS2 adds the
    // AUX (gamma) scale and payload.
    reg [95:0] issue_seg;
    always @* begin
        if (state_r == V_P2)
            case (seg_idx_r)
            0: issue_seg = mkseg(2'd0, r_main_base + scale_word(main_blk_seg), 12'd1, 8'h80);
            1: issue_seg = mkseg(2'd0, r_main_base + block_word(main_blk_seg), 12'd8, 8'h00);
            2: issue_seg = mkseg(2'd2, r_aux_base + scale_word(seg_group_r), 12'd1, 8'h80);
            default: issue_seg = mkseg(2'd2, r_aux_base + block_word(seg_group_r), 12'd8, 8'h00);
            endcase
        else
            issue_seg = seg_idx_r == 0 ? mkseg(2'd0, r_main_base + scale_word(main_blk_seg), 12'd1, 8'h80)
                                       : mkseg(2'd0, r_main_base + block_word(main_blk_seg), 12'd8, 8'h00);
    end

    // Block consumption: the scale beat (with the gamma row scale in PASS2), then 8 words in
    // half-word steps.
    wire scale_take = run_block_r && !have_scale_r && main_level_r != 0 && (phase_r == 0 || aux_level_r != 0);
    wire half_take  = !scale_take && (issue1 || issue2);
    wire word_take  = half_take && half_r;
    wire block_end  = word_take && word_idx_r == 3'd7;
    wire main_pop   = scale_take || word_take;
    wire aux_pop    = main_pop && phase_r == 2'd1;

    // FINAL micro-program decode of the current op (fin_step_r, fin_row_r, fin_sub_r).
    reg [31:0] fin_op_a, fin_op_b; reg fin_op_mul, fin_op_int; reg [4:0] fin_op_dst;
    always @* begin
        fin_op_mul = 0; fin_op_int = 0; fin_op_a = 0; fin_op_b = 0; fin_op_dst = 0;
        case (fin_step_r)
        0: begin fin_op_a = fin_rf_mem[fin_row_r][2*fin_sub_r]; fin_op_b = fin_rf_mem[fin_row_r][2*fin_sub_r+1];
                 fin_op_dst = 16 + fin_sub_r; end
        1: begin fin_op_a = fin_rf_mem[fin_row_r][16+2*fin_sub_r]; fin_op_b = fin_rf_mem[fin_row_r][17+2*fin_sub_r];
                 fin_op_dst = 24 + fin_sub_r; end
        2: begin fin_op_a = fin_rf_mem[fin_row_r][24+2*fin_sub_r]; fin_op_b = fin_rf_mem[fin_row_r][25+2*fin_sub_r];
                 fin_op_dst = 28 + fin_sub_r; end
        3: begin fin_op_a = fin_rf_mem[fin_row_r][28]; fin_op_b = fin_rf_mem[fin_row_r][29]; fin_op_dst = 30; end
        4: begin fin_op_mul = 1; fin_op_a = fin_rf_mem[fin_row_r][30]; fin_op_b = inv_count_r; fin_op_dst = 31; end
        5: begin fin_op_a = fin_rf_mem[fin_row_r][31]; fin_op_b = eps_value(desc_r[145:144]); fin_op_dst = 0; end
        6: begin fin_op_int = 1; fin_op_dst = 1; end
        7, 13:  begin fin_op_mul = 1; fin_op_a = fin_rf_mem[fin_row_r][1]; fin_op_b = fin_rf_mem[fin_row_r][1];
                      fin_op_dst = 2; end
        8, 14:  begin fin_op_mul = 1; fin_op_a = fin_rf_mem[fin_row_r][0]; fin_op_b = fin_rf_mem[fin_row_r][2];
                      fin_op_dst = 3; end
        9, 15:  begin fin_op_mul = 1; fin_op_a = fin_rf_mem[fin_row_r][3]; fin_op_b = C_HALF; fin_op_dst = 4; end
        10, 16: begin fin_op_mul = 1; fin_op_a = fin_rf_mem[fin_row_r][4]; fin_op_b = C_NEG1; fin_op_dst = 5; end
        11, 17: begin fin_op_a = fin_rf_mem[fin_row_r][5]; fin_op_b = C_THREEHALF; fin_op_dst = 6; end
        default: begin                                                                     // 12, 18
            fin_op_mul = 1; fin_op_a = fin_rf_mem[fin_row_r][1]; fin_op_b = fin_rf_mem[fin_row_r][6]; fin_op_dst = 1;
        end
        endcase
    end

    // ------------------------------------------------------------------
    // Registered outputs and data path.
    // ------------------------------------------------------------------
    integer r, c;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            done <= 0; sources_closed <= 0; cm_valid <= 0; seg_valid <= 0; seg <= 0;
            main_rd_ptr_r <= 0;
            main_wr_ptr_r <= 0;
            main_level_r <= 0;
            aux_rd_ptr_r <= 0;
            aux_wr_ptr_r <= 0;
            aux_level_r <= 0;
            main_pop_r <= 0; aux_pop_r <= 0;
            run_block_r <= 0; have_scale_r <= 0; phase_r <= 0; word_idx_r <= 0; half_r <= 0;
            ob_wr_sel_r <= 0; ob_rd_sel_r <= 0; ob_full_r <= 0; ob_cnt_r[0] <= 0; ob_cnt_r[1] <= 0;
            q_run_r <= 0; emit_run_r <= 0; w_valid_r <= 0; fin_run_r <= 0; marks_pending_r <= 0;
            fin_mul_valid_r <= 0; fin_add_valid_r <= 0; fin_outstanding_r <= 0; p1_pipe_r <= 0;
        end else begin
            done <= done_en;
            sources_closed <= done_en;
            if (seg_issue) seg_valid <= 1'b1;
            else if (seg_valid && seg_ready) seg_valid <= 1'b0;
            if (seg_issue) seg <= issue_seg;
            if (commit_issue) cm_valid <= 1'b1;
            else if (cm_valid && cm_ready) cm_valid <= 1'b0;
            // Stream buffers: egress beats in, consumed beats out (credits one cycle later).
            if (eg_valid && eg_stream == 2'd0) main_wr_ptr_r <= main_wr_ptr_r + 1'b1;
            if (eg_valid && eg_stream == 2'd2) aux_wr_ptr_r <= aux_wr_ptr_r + 1'b1;
            if (main_pop) main_rd_ptr_r <= main_rd_ptr_r + 1'b1;
            if (aux_pop) aux_rd_ptr_r <= aux_rd_ptr_r + 1'b1;
            main_level_r <= main_level_r + (eg_valid && eg_stream == 2'd0) - main_pop;
            aux_level_r <= aux_level_r + (eg_valid && eg_stream == 2'd2) - aux_pop;
            main_pop_r <= main_pop;
            aux_pop_r <= aux_pop;
            if (mark_retired) marks_pending_r <= marks_pending_r - 1'b1;

            // ---- consumption of MAIN (and AUX in PASS2): scale beat, then 8 words.
            if (pass_setup) phase_r <= state_r == V_DIV ? 2'd0 : 2'd1;
            if (pass_setup) run_block_r <= 1'b1;
            else if (block_end && use_group_r + 1 == r_g) run_block_r <= 1'b0;
            if (pass_setup) have_scale_r <= 1'b0;
            else if (scale_take) have_scale_r <= 1'b1;
            else if (block_end) have_scale_r <= 1'b0;
            if (scale_take) begin
                word_idx_r <= 0; half_r <= 0;
            end else if (half_take) begin
                half_r <= !half_r;
                if (half_r) word_idx_r <= word_idx_r + 1'b1;
            end
            // The block index is latched with the full flag: the next scale beat may be consumed
            // while this buffer is still emitting.
            if (block_end && issue2) begin
                ob_full_r[ob_wr_sel_r] <= 1'b1;
                ob_blk_r[ob_wr_sel_r] <= blk_rd_idx_r;
                ob_wr_sel_r <= !ob_wr_sel_r;
            end

            // ---- PASS2 results into the output block buffer.
            if (out_valid)
                for (c = 0; c < 16; c = c + 1)
                    ob_mem[ob_wr_sel_res][c/2][lane_col[c*5 +: 5]] <= lane_sum[c*32 +: 32];
            // Per-buffer count of results still in the chain.
            for (c = 0; c < 2; c = c + 1)
                ob_cnt_r[c] <= ob_cnt_r[c] + (issue2 && ob_wr_sel_r == c) - (out_valid && ob_wr_sel_res == c);
            p1_pipe_r <= {p1_pipe_r[6:0], issue1};

            // ---- quantise a full buffer, then emit its 8 words and scale beat.
            if (!q_run_r && !emit_run_r && ob_full_r[ob_rd_sel_r] && ob_cnt_r[ob_rd_sel_r] == 0) begin
                q_run_r <= 1'b1;
                q_row_r <= 0;
            end
            if (q_run_r) begin : quant
                reg [31:0] mx; reg [7:0] sc;
                mx = 0;
                for (c = 0; c < 32; c = c + 1) mx = max_value(c == 0 ? ob_mem[ob_rd_sel_r][q_row_r][0] : mx,
                                                              ob_mem[ob_rd_sel_r][q_row_r][c], 1'b0);
                sc = choose_scale(mx);
                q_scale_r[q_row_r] <= sc;
                for (c = 0; c < 32; c = c + 1) q_bytes_mem[q_row_r][c] <= quant_e4m3(ob_mem[ob_rd_sel_r][q_row_r][c], sc);
                q_row_r <= q_row_r + 1'b1;
                if (q_row_r == 4'd7) begin q_run_r <= 1'b0; emit_run_r <= 1'b1; emit_idx_r <= 0; end
            end
            if (w_valid_r && wb_ready) w_valid_r <= 1'b0;
            if (emit_run_r && (!w_valid_r || wb_ready)) begin : emit
                reg [255:0] wd;
                integer k;
                if (emit_idx_r < 8) begin
                    for (r = 0; r < 8; r = r + 1)
                        for (k = 0; k < 4; k = k + 1)
                            wd[(k/2)*128 + r*16 + (k%2)*8 +: 8] = q_bytes_mem[r][emit_idx_r*4 + k];
                    w_data_r <= wd; w_strb_r <= 32'hffff_ffff; w_mark_r <= 1'b0;
                    w_addr_r <= r_res_base + block_word(ob_blk_r[ob_rd_sel_r]) + emit_idx_r;
                end else begin
                    wd = 0;
                    for (r = 0; r < 8; r = r + 1) wd[ob_blk_r[ob_rd_sel_r][1:0]*64 + r*8 +: 8] = q_scale_r[r];
                    w_data_r <= wd; w_strb_r <= 32'hff << (ob_blk_r[ob_rd_sel_r][1:0] * 8); w_mark_r <= 1'b1;
                    w_addr_r <= r_res_base + scale_word(ob_blk_r[ob_rd_sel_r]);
                end
                w_valid_r <= 1'b1;
                emit_idx_r <= emit_idx_r + 1'b1;
                if (emit_idx_r == 4'd8) begin
                    emit_run_r <= 1'b0; ob_full_r[ob_rd_sel_r] <= 1'b0; ob_rd_sel_r <= !ob_rd_sel_r;
                    marks_pending_r <= marks_pending_r + 1'b1 - mark_retired;
                end
            end

            // ---- FINAL micro-program: issue one op per cycle, wait for a step's results.
            fin_outstanding_r <= fin_outstanding_r + (fin_mul_valid_r || fin_add_valid_r) - (fm_valid || fa_valid);
            fin_mul_valid_r <= fin_issue_en && !fin_op_int && fin_op_mul;
            fin_add_valid_r <= fin_issue_en && !fin_op_int && !fin_op_mul;
            if (fin_load) fin_run_r <= 1'b1;
            else if (fin_step_done && fin_step_r == 5'd18) fin_run_r <= 1'b0;
        end
    end

    always @(posedge clk) begin
        if (start_accept) begin
            desc_r <= desc; main_wm_r <= main_wm0; aux_wm_r <= aux_wm0;
            inv_count_r <= inv_count(groups * 32);
        end else if (ev_valid && state_r != V_IDLE) begin
            // Watch registers.
            if (ev_idx == desc_r[7:0] && ev_gen == desc_r[15:8] && main_wm_r != `G3_WM_READY &&
                ev_wm > main_wm_r) main_wm_r <= ev_wm;
            if (ev_idx == desc_r[48:41] && ev_gen == desc_r[56:49] && aux_wm_r != `G3_WM_READY &&
                ev_wm > aux_wm_r) aux_wm_r <= ev_wm;
        end
        if (eg_valid && eg_stream == 2'd0) main_buf_mem[main_wr_ptr_r] <= eg_data;
        if (eg_valid && eg_stream == 2'd2) aux_buf_mem[aux_wr_ptr_r] <= eg_data;
        if (start_accept) panel_r <= 0;
        else if (commit_issue && !last_panel) panel_r <= panel_r + 1'b1;
        if (scale_take) begin
            row_scale_r <= m_head[blk_rd_idx_r[1:0]*64 +: 64];
            // gamma row 0 scale of AUX block g (slot g % 4).
            if (phase_r == 2'd1) g_scale_r <= a_head[use_group_r[1:0]*64 +: 64];
        end
        if (pass_setup) begin
            use_group_r <= 0; blk_rd_idx_r <= panel_r * r_g;
        end else if (block_end) begin
            use_group_r <= use_group_r + 1'b1; blk_rd_idx_r <= blk_rd_idx_r + 1'b1;
        end
        if (pass_setup) begin
            seg_group_r <= 0; seg_idx_r <= 0;
        end else if (seg_issue) begin
            if (seg_idx_r == (state_r == V_P2 ? 3'd3 : 3'd1)) begin
                seg_idx_r <= 0; seg_group_r <= seg_group_r + 1'b1;
            end else seg_idx_r <= state_r == V_P2 ? seg_idx_r + 1'b1 : 3'd1;
        end

        // FINAL: results, the accumulator load, and the integer op write the register file.
        if (fm_valid) fin_rf_mem[fm_tag[8:6]][fm_tag[4:0]] <= fm_result;
        if (fa_valid) fin_rf_mem[fa_tag[8:6]][fa_tag[4:0]] <= fa_result;
        if (fin_load)
            for (integer i = 0; i < 8; i = i + 1)
                for (integer j = 0; j < 16; j = j + 1) fin_rf_mem[i][j] <= acc_mem[i][j];
        if (fin_issue_en && fin_op_int) fin_rf_mem[fin_row_r][1] <= 32'h5f375a86 - (fin_rf_mem[fin_row_r][0] >> 1);
        if (fin_issue_en && !fin_op_int) begin
            fin_a_r <= fin_op_a; fin_b_r <= fin_op_b; fin_tag_r <= {fin_row_r[2:0], 1'b0, fin_op_dst};
        end
        if (fin_load) begin
            fin_step_r <= 0; fin_row_r <= 0; fin_sub_r <= 0;
        end else if (fin_issue_en) begin
            if (fin_sub_r + 1 == fin_ops_in_step) begin fin_sub_r <= 0; fin_row_r <= fin_row_r + 1'b1; end
            else fin_sub_r <= fin_sub_r + 1'b1;
        end else if (fin_step_done) begin
            fin_row_r <= 0;
            if (fin_step_r == 5'd18) for (integer i = 0; i < 8; i = i + 1) s_row_r[i] <= fin_rf_mem[i][1];
            else fin_step_r <= fin_step_r + 1'b1;
        end

        if (commit_issue) begin
            cm_idx <= desc_r[89:82]; cm_gen <= desc_r[97:90];
            cm_final <= panel_r + 1 == panels;
            cm_wm <= panel_r + 1 == panels ? r_res_ext : 33 * (((panel_r + 1) * r_g) >> 2);
        end
        if (done_en) done_status <= 4'd0;
    end

    function automatic [31:0] eps_value(input [1:0] p);
        case (p)
            2'd0: eps_value = 32'h0000_0000; 2'd1: eps_value = 32'h3586_37bd;
            2'd2: eps_value = 32'h3727_c5ac; default: eps_value = 32'h38d1_b717;
        endcase
    endfunction

    // RNE(1/n), n = 32*G >= 32 (exact long division in a function; the
    // hardware form is a 26-step restoring divider started with the job).
    function automatic [31:0] inv_count(input [13:0] nval);
        integer e, i2;
        reg [63:0] num, q, rem;
        begin
            e = 0;
            for (i2 = 0; i2 < 14; i2 = i2 + 1) if (nval[i2]) e = i2;      // 2^e <= n
            if (nval == (14'd1 << e)) inv_count = {1'b0, 8'(127 - e), 23'd0};
            else begin
                // 1/n = 2^-(e+1) * (2^(e+1)/n), q = floor(2^(e+1+23)/n) in [2^23, 2^24)
                num = 64'd1 << (e + 1 + 23);
                q = num / nval; rem = num % nval;
                if (rem * 2 > nval || (rem * 2 == nval && q[0])) q = q + 1;
                if (q[24]) inv_count = {1'b0, 8'(127 - e), 23'd0};
                else inv_count = {1'b0, 8'(127 - e - 1), q[22:0]};
            end
        end
    endfunction

    assign busy = state_r != V_IDLE;
endmodule

`default_nettype wire
