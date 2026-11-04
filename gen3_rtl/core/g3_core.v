`timescale 1ns/1ps
`default_nettype none

// Gen3 compute core (gen3_rtl/README.md s3): feed, 32 x 64 array, vector column. Pure compute: the
// sequencers program the UB streams and send matching uops.
//
//   uop --> uop queue (4) --+--> plan: feed window + tile descriptor (+ GEMM output descriptor FIFO (4),
//                           |    + P window for PV)
//                           +--> O descriptor (op 3) --> O descriptor FIFO (8)
//   UB A / B / C lanes, weight buffers --> g3_feed --> g3_array (rc_clk_en: 1 of R) --> exports
//                                            ^                                          |
//                                            +----- P (A of PV windows) -- g3_vector_column <--+
//                                                                              |
//                                                                              v  d_* (UB D)
// Control (uop at the queue head):
//   GEMM          needs a tile descriptor slot when it emits, an output descriptor slot, and no
//                 attention work in the vector column                       -> plan
//   QK / PV       need a tile descriptor slot and no GEMM work; PV also its full P bank
//   O descriptor  needs O descriptor FIFO space
module g3_core #(
    parameter integer R = 4
) (
    input  wire          clk,
    input  wire          rstn,
    input  wire          uop_valid,
    output wire          uop_ready,
    input  wire [255:0]  uop,
    // UB port A lane.
    input  wire          a_beat_valid,
    input  wire [1:0]    a_beat_inner_idx,
    input  wire          a_beat_inner_last,
    input  wire [7:0]    a_beat_user,
    input  wire [255:0]  a_beat_data,
    output wire          a_credit,
    // UB port B lane.
    input  wire          b_beat_valid,
    input  wire [1:0]    b_beat_stream,
    input  wire [1:0]    b_beat_inner_idx,
    input  wire          b_beat_inner_last,
    input  wire [7:0]    b_beat_user,
    input  wire [255:0]  b_beat_data,
    output wire [1:0]    b_credit,
    // UB port C lane.
    input  wire          c_beat_valid,
    input  wire [1:0]    c_beat_stream,
    input  wire [1:0]    c_beat_inner_idx,
    input  wire          c_beat_inner_last,
    input  wire          c_beat_seg_last,
    input  wire [7:0]    c_beat_user,
    input  wire [255:0]  c_beat_data,
    output wire [2:0]    c_credit,
    // Weight buffers.
    input  wire          wb_start,
    input  wire [31:0]   wb_cells,
    output wire [1:0]    wb_rd_en,
    output wire [15:0]   wb_rd_addr,
    input  wire [511:0]  wb_rd_data,
    input  wire [39:0]   wb_pulled,
    output wire [39:0]   wb_released,
    // UB D writes.
    output wire          d_valid,
    input  wire          d_ready,
    output wire [11:0]   d_addr,
    output wire [255:0]  d_data,
    output wire [31:0]   d_strb,
    output wire          d_mark,
    output wire          feed_tick_fire,
    output wire          busy
);
    `include "g3_defs.vh"
    // ------------------------------------------------------------------
    // Uop queue.
    // ------------------------------------------------------------------
    reg [255:0] uq_mem [0:3];
    reg [1:0] uq_rd_r, uq_wr_r; reg [2:0] uq_level_r;
    assign uop_ready = uq_level_r != 4;
    wire [255:0] u = uq_mem[uq_rd_r];
    wire uq_avail = uq_level_r != 0;
    wire [1:0] u_op = u[`U_OP];
    wire u_odesc = u_op == 2'd3;
    wire u_attn = u_op == 2'd1 || u_op == 2'd2;
    wire u_pv = u_op == 2'd2;
    wire u_emit = u[`U_EMIT];
    wire u_td = u_attn || u_emit;
    // Output descriptors (GEMM emit uops), O descriptors.
    reg [255:0] dq_mem [0:3];
    reg [1:0] dq_rd_r, dq_wr_r; reg [2:0] dq_level_r;
    reg [255:0] oq_mem [0:7];
    reg [2:0] oq_rd_r, oq_wr_r; reg [3:0] oq_level_r;
    wire td_ready, vc_gemm, vc_fa, p_plan_ok, plan_ready, desc_ready, od_ready;
    wire plan_valid = uq_avail && !u_odesc && (!u_td || td_ready) &&
                      (u_attn ? !vc_gemm && (!u_pv || p_plan_ok) : !vc_fa && (!u_emit || dq_level_r != 4));
    wire od_push = uq_avail && u_odesc && oq_level_r != 8;
    wire plan_fire = plan_valid && plan_ready;
    wire uq_pop = plan_fire || od_push;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            uq_rd_r <= 0; uq_wr_r <= 0; uq_level_r <= 0;
            dq_rd_r <= 0; dq_wr_r <= 0; dq_level_r <= 0;
            oq_rd_r <= 0; oq_wr_r <= 0; oq_level_r <= 0;
        end else begin
            if (uop_valid && uop_ready) begin uq_mem[uq_wr_r] <= uop; uq_wr_r <= uq_wr_r + 1'b1; end
            if (uq_pop) uq_rd_r <= uq_rd_r + 1'b1;
            uq_level_r <= uq_level_r + (uop_valid && uop_ready) - uq_pop;
            if (plan_fire && !u_attn && u_emit) begin dq_mem[dq_wr_r] <= u; dq_wr_r <= dq_wr_r + 1'b1; end
            if (desc_ready) dq_rd_r <= dq_rd_r + 1'b1;
            dq_level_r <= dq_level_r + (plan_fire && !u_attn && u_emit) - desc_ready;
            if (od_push) begin oq_mem[oq_wr_r] <= u; oq_wr_r <= oq_wr_r + 1'b1; end
            if (od_ready) oq_rd_r <= oq_rd_r + 1'b1;
            oq_level_r <= oq_level_r + od_push - od_ready;
        end
    end

    // ------------------------------------------------------------------
    // Synchronous 1:R array clock enable.
    // ------------------------------------------------------------------
    reg [2:0] rc_phase_r;
    always @(posedge clk or negedge rstn)
        if (!rstn) rc_phase_r <= 0;
        else rc_phase_r <= rc_phase_r == R - 1 ? 3'd0 : rc_phase_r + 1'b1;
    wire rc_clk_en = rc_phase_r == R - 1;

    // ------------------------------------------------------------------
    // Feed, array, vector column.
    // ------------------------------------------------------------------
    wire cfg_valid, cfg_ready, cfg_accum, cfg_emit;
    wire [5:0] cfg_kg, cfg_nl, cfg_nr; wire [31:0] cfg_rows; wire [1:0] cfg_op;
    wire feed_window_open, tick_valid, tick_ready, tick_last;
    wire [511:0] fa_payload, fbl_payload, fbr_payload; wire [255:0] fa_scale, fbl_scale, fbr_scale;
    wire p_valid, p_pop; wire [511:0] p_data; wire [255:0] p_scale;
    wire feed_busy;
    g3_feed u_feed (
        .clk               (clk),
        .rstn              (rstn),
        .a_beat_valid      (a_beat_valid),
        .a_beat_inner_idx  (a_beat_inner_idx),
        .a_beat_inner_last (a_beat_inner_last),
        .a_beat_user       (a_beat_user),
        .a_beat_data       (a_beat_data),
        .a_credit          (a_credit),
        .b_beat_valid      (b_beat_valid),
        .b_beat_stream     (b_beat_stream),
        .b_beat_inner_idx  (b_beat_inner_idx),
        .b_beat_inner_last (b_beat_inner_last),
        .b_beat_user       (b_beat_user),
        .b_beat_data       (b_beat_data),
        .b_credit          (b_credit),
        .wb_start          (wb_start),
        .wb_cells          (wb_cells),
        .wb_rd_en          (wb_rd_en),
        .wb_rd_addr        (wb_rd_addr),
        .wb_rd_data        (wb_rd_data),
        .wb_pulled         (wb_pulled),
        .wb_released       (wb_released),
        .plan_valid        (plan_valid),
        .plan_ready        (plan_ready),
        .plan_kg           (u[`U_KG]),
        .plan_rows         (u[`U_ROWS]),
        .plan_nl           (u[`U_NL]),
        .plan_nr           (u[`U_NR]),
        .plan_accum        (u[`U_ACCUM]),
        .plan_emit         (u[`U_EMIT] || u_attn),
        .plan_op           (u_op),
        .plan_asrc         (u[`U_ASRC]),
        .cfg_valid         (cfg_valid),
        .cfg_ready         (cfg_ready),
        .cfg_kg            (cfg_kg),
        .cfg_rows          (cfg_rows),
        .cfg_nl            (cfg_nl),
        .cfg_nr            (cfg_nr),
        .cfg_accum         (cfg_accum),
        .cfg_emit          (cfg_emit),
        .cfg_op            (cfg_op),
        .feed_window_open  (feed_window_open),
        .feed_tick_valid   (tick_valid),
        .feed_tick_ready   (tick_ready),
        .feed_tick_last    (tick_last),
        .feed_a_payload    (fa_payload),
        .feed_a_scale      (fa_scale),
        .feed_bl_payload   (fbl_payload),
        .feed_bl_scale     (fbl_scale),
        .feed_br_payload   (fbr_payload),
        .feed_br_scale     (fbr_scale),
        .p_valid           (p_valid),
        .p_data            (p_data),
        .p_scale           (p_scale),
        .p_pop             (p_pop),
        .busy              (feed_busy)
    );
    wire array_ready, array_quiescent;
    wire [31:0] lv, rv; wire [32*113-1:0] ld, rd;
    g3_array u_array (
        .clk               (clk),
        .rstn              (rstn),
        .rc_clk_en         (rc_clk_en),
        .array_ready       (array_ready),
        .left_valid        (lv),
        .left_data         (ld),
        .right_valid       (rv),
        .right_data        (rd),
        .cfg_valid         (cfg_valid),
        .cfg_ready         (cfg_ready),
        .cfg_k_group_count (cfg_kg),
        .cfg_rows          (cfg_rows),
        .cfg_nl            (cfg_nl),
        .cfg_nr            (cfg_nr),
        .cfg_accumulate    (cfg_accum),
        .cfg_emit_result   (cfg_emit),
        .cfg_operation     (cfg_op),
        .feed_window_open  (feed_window_open),
        .feed_tick_valid   (tick_valid),
        .feed_tick_ready   (tick_ready),
        .feed_tick_last    (tick_last),
        .feed_a_payload    (fa_payload),
        .feed_a_scale      (fa_scale),
        .feed_bl_payload   (fbl_payload),
        .feed_bl_scale     (fbl_scale),
        .feed_br_payload   (fbr_payload),
        .feed_br_scale     (fbr_scale),
        .quiescent         (array_quiescent)
    );
    assign feed_tick_fire = tick_valid && tick_ready;
    wire vc_busy;
    g3_vector_column u_vc (
        .clk               (clk),
        .rstn              (rstn),
        .left_valid        (lv),
        .left_data         (ld),
        .right_valid       (rv),
        .right_data        (rd),
        .array_ready       (array_ready),
        .td_valid          (plan_fire && u_td),
        .td_ready          (td_ready),
        .td_uop            (u),
        .desc_valid        (dq_level_r != 0),
        .desc_ready        (desc_ready),
        .desc_uop          (dq_mem[dq_rd_r]),
        .od_valid          (oq_level_r != 0),
        .od_ready          (od_ready),
        .od_uop            (oq_mem[oq_rd_r]),
        .pw_valid          (plan_fire && u_pv),
        .pw_kg             (u[`U_KG] == 6'd1 ? 2'd1 : 2'd2),
        .pw_release        (u[`U_LASTDP]),
        .p_plan_ok         (p_plan_ok),
        .p_valid           (p_valid),
        .p_data            (p_data),
        .p_scale           (p_scale),
        .p_pop             (p_pop),
        .c_beat_valid      (c_beat_valid),
        .c_beat_stream     (c_beat_stream),
        .c_beat_inner_idx  (c_beat_inner_idx),
        .c_beat_inner_last (c_beat_inner_last),
        .c_beat_seg_last   (c_beat_seg_last),
        .c_beat_user       (c_beat_user),
        .c_beat_data       (c_beat_data),
        .c_credit          (c_credit),
        .wb_valid          (d_valid),
        .wb_ready          (d_ready),
        .wb_addr           (d_addr),
        .wb_data           (d_data),
        .wb_strb           (d_strb),
        .wb_mark           (d_mark),
        .gemm_active       (vc_gemm),
        .fa_active         (vc_fa),
        .busy              (vc_busy)
    );
    assign busy = uq_level_r != 0 || dq_level_r != 0 || oq_level_r != 0 || feed_busy || !array_quiescent || vc_busy;
endmodule

`default_nettype wire
