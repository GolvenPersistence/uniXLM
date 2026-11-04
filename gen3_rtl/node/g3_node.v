`timescale 1ns/1ps
`default_nettype none

// Gen3 Node top (docs/gen3/README.md s2, gen3_rtl/README.md s1): controller, GEMM and ATTN
// sequencers, one compute core (32 x 64 array + vector column), UB, VRU, NI, weight buffers.
// External interface identical to the Gen2 Node (rtl/gen2/g2_node.v).
//
//   cmd --> +-------------------------------------------------------------+ --> cpl
//           | g3_ctrl: command FIFO, admission, URF + owner map, one array |
//           | parent, event arbiter -> URF -> broadcast (ev_*)             |
//           +---+--------------+------------------+--------------+--------+
//               | gemm pd      | attn desc        | vru desc     | tx desc  ^ commit events
//               v              v                  v              v          |
//        g3_gemm_seq     g3_attn_seq           g3_vru        g3_ni: RX windows, MOVE_OUT x2,
//          |  segs A/B/C, uops, pull programs    |            |    pull x2 -> weight buffers
//          +-------------+-------+               |            |              |
//                        v       v               v            v              v
//   +------------------------------------------------------------------+  g3_weight_buffer
//   | g3_ub: 4 banks; ports 0 A, 1 B (left / right), 2 C, 3 VRU, 4 NI   |  (2 x 132 words)
//   +----+-----------+-----------+--------------------------------------+          |
//        | A         | B         | C            ^ D                                 |
//        v           v           v              |                                   v
//   +------------------------------------------------------------------------------------+
//   | g3_core: feed, 32 x 64 array (two halves around the vector column), vector column   |
//   +------------------------------------------------------------------------------------+
module g3_node #(
    parameter [47:0] NODE_BASE = 48'h0000_1000_0000
) (
    input  wire          clk,
    input  wire          rstn,
    input  wire          cmd_valid,
    output wire          cmd_ready,
    input  wire [127:0]  cmd_data,
    output wire          cpl_valid,
    input  wire          cpl_ready,
    output wire [7:0]    cpl_tag,
    output wire [3:0]    cpl_status,
    output wire [31:0]   cpl_data,
    input  wire          s_aw_valid,
    output wire          s_aw_ready,
    input  wire [47:0]   s_aw_addr,
    input  wire [7:0]    s_aw_len,
    input  wire          s_w_valid,
    output wire          s_w_ready,
    input  wire [255:0]  s_w_data,
    input  wire          s_w_last,
    output wire          s_b_valid,
    input  wire          s_b_ready,
    output wire [1:0]    s_b_resp,
    output wire          m_aw_valid,
    input  wire          m_aw_ready,
    output wire [47:0]   m_aw_addr,
    output wire [7:0]    m_aw_len,
    output wire          m_w_valid,
    input  wire          m_w_ready,
    output wire [255:0]  m_w_data,
    output wire          m_w_last,
    input  wire          m_b_valid,
    output wire          m_b_ready,
    output wire          m_ar_valid,
    input  wire          m_ar_ready,
    output wire [47:0]   m_ar_addr,
    output wire [7:0]    m_ar_len,
    output wire          m_ar_id,
    input  wire          m_r_valid,
    output wire          m_r_ready,
    input  wire [255:0]  m_r_data,
    input  wire          m_r_id,
    output wire          feed_fire,
    output wire          idle
);
    // ------------------------------------------------------------------
    // Controller.
    // ------------------------------------------------------------------
    wire gemm_cm_valid, gemm_cm_ready, gemm_cm_final; wire [7:0] gemm_cm_idx, gemm_cm_gen; wire [12:0] gemm_cm_wm;
    wire ni_ev_valid, ni_ev_ready, ni_ev_final; wire [7:0] ni_ev_idx, ni_ev_gen; wire [12:0] ni_ev_wm;
    wire ev_valid; wire [7:0] ev_idx, ev_gen; wire [12:0] ev_wm;
    wire [7:0] lk_idx, lk_gen; wire lk_valid; wire [11:0] lk_base; wire [12:0] lk_ext; wire [1:0] lk_writer;
    wire gemm_start, gemm_abort, gemm_sources_closed, gemm_done, gemm_busy; wire [383:0] gemm_pd; wire [3:0] gemm_status;
    wire [1:0] tx_start, tx_done; wire [255:0] tx_desc;
    wire ctrl_idle;
    wire vru_start, vru_done, vru_sources_closed, vru_cm_valid, vru_cm_ready, vru_cm_final, vru_busy;
    wire [255:0] vru_desc; wire [3:0] vru_status; wire [7:0] vru_cm_idx, vru_cm_gen; wire [12:0] vru_cm_wm;
    wire attn_start, attn_done, attn_sources_closed, attn_cm_valid, attn_cm_ready, attn_cm_final, attn_busy;
    wire [511:0] attn_desc; wire [3:0] attn_status;
    wire [7:0] attn_cm_idx, attn_cm_gen; wire [12:0] attn_cm_wm;
    g3_ctrl u_ctrl (
        .clk                 (clk),
        .rstn                (rstn),
        .cmd_valid           (cmd_valid),
        .cmd_ready           (cmd_ready),
        .cmd_data            (cmd_data),
        .cpl_valid           (cpl_valid),
        .cpl_ready           (cpl_ready),
        .cpl_tag             (cpl_tag),
        .cpl_status          (cpl_status),
        .cpl_data            (cpl_data),
        .gemm_cm_valid       (gemm_cm_valid),
        .gemm_cm_ready       (gemm_cm_ready),
        .gemm_cm_idx         (gemm_cm_idx),
        .gemm_cm_gen         (gemm_cm_gen),
        .gemm_cm_wm          (gemm_cm_wm),
        .gemm_cm_final       (gemm_cm_final),
        .ni_ev_valid         (ni_ev_valid),
        .ni_ev_ready         (ni_ev_ready),
        .ni_ev_idx           (ni_ev_idx),
        .ni_ev_gen           (ni_ev_gen),
        .ni_ev_wm            (ni_ev_wm),
        .ni_ev_final         (ni_ev_final),
        .ev_valid            (ev_valid),
        .ev_idx              (ev_idx),
        .ev_gen              (ev_gen),
        .ev_wm               (ev_wm),
        .lk_idx              (lk_idx),
        .lk_valid            (lk_valid),
        .lk_gen              (lk_gen),
        .lk_base             (lk_base),
        .lk_ext              (lk_ext),
        .lk_writer           (lk_writer),
        .gemm_start          (gemm_start),
        .gemm_pd             (gemm_pd),
        .gemm_abort          (gemm_abort),
        .gemm_sources_closed (gemm_sources_closed),
        .gemm_done           (gemm_done),
        .gemm_status         (gemm_status),
        .vru_start           (vru_start),
        .vru_desc            (vru_desc),
        .vru_done            (vru_done),
        .vru_status          (vru_status),
        .vru_sources_closed  (vru_sources_closed),
        .vru_cm_valid        (vru_cm_valid),
        .vru_cm_ready        (vru_cm_ready),
        .vru_cm_idx          (vru_cm_idx),
        .vru_cm_gen          (vru_cm_gen),
        .vru_cm_wm           (vru_cm_wm),
        .vru_cm_final        (vru_cm_final),
        .attn_start          (attn_start),
        .attn_desc           (attn_desc),
        .attn_done           (attn_done),
        .attn_status         (attn_status),
        .attn_sources_closed (attn_sources_closed),
        .attn_cm_valid       (attn_cm_valid),
        .attn_cm_ready       (attn_cm_ready),
        .attn_cm_idx         (attn_cm_idx),
        .attn_cm_gen         (attn_cm_gen),
        .attn_cm_wm          (attn_cm_wm),
        .attn_cm_final       (attn_cm_final),
        .tx_start            (tx_start),
        .tx_desc             (tx_desc),
        .tx_done             (tx_done),
        .idle                (ctrl_idle),
        .stat_admitted       (),
        .stat_bind_wait      (),
        .stat_lease_wait     ()
    );

    // ------------------------------------------------------------------
    // UB.
    // ------------------------------------------------------------------
    wire [4:0] seg_valid, seg_ready; wire [5*128-1:0] seg_data; wire [14:0] seg_closed;
    wire [4:0] eg_valid, eg_inner_last, eg_seg_last; wire [9:0] eg_stream, eg_inner_idx;
    wire [39:0] eg_user; wire [5*256-1:0] eg_data; wire [14:0] credit_return;
    wire [4:0] in_valid, in_ready, in_mark, mark_retired; wire [59:0] in_addr;
    wire [5*256-1:0] in_data; wire [5*32-1:0] in_strb;
    wire ub_busy;
    g3_ub u_ub (
        .clk                 (clk),
        .rstn                (rstn),
        .seg_valid           (seg_valid),
        .seg_ready           (seg_ready),
        .seg_data            (seg_data),
        .seg_closed          (seg_closed),
        .eg_valid            (eg_valid),
        .eg_stream           (eg_stream),
        .eg_inner_idx        (eg_inner_idx),
        .eg_inner_last       (eg_inner_last),
        .eg_seg_last         (eg_seg_last),
        .eg_user             (eg_user),
        .eg_data             (eg_data),
        .credit_return       (credit_return),
        .in_valid            (in_valid),
        .in_ready            (in_ready),
        .in_addr             (in_addr),
        .in_data             (in_data),
        .in_strb             (in_strb),
        .in_mark             (in_mark),
        .mark_retired        (mark_retired),
        .stat_read_beats     (),
        .stat_write_beats    (),
        .stat_bank_conflicts (),
        .busy                (ub_busy)
    );

    // ------------------------------------------------------------------
    // Sequencers (one array parent at a time) and the segment / uop mux.
    // ------------------------------------------------------------------
    wire g_sa_valid, g_sb_valid, g_sc_valid, g_uop_valid; wire [127:0] g_sa, g_sb, g_sc; wire [255:0] g_uop;
    wire t_sa_valid, t_sb_valid, t_uop_valid; wire [127:0] t_sa, t_sb; wire [255:0] t_uop;
    wire uop_ready;
    wire [1:0] p_start; wire [319:0] p_prog; wire wb_start; wire [31:0] wb_cells;
    wire attn_active = attn_busy;
    g3_gemm_seq u_gemm (
        .clk            (clk),
        .rstn           (rstn),
        .start          (gemm_start),
        .pd             (gemm_pd),
        .abort          (gemm_abort),
        .ev_valid       (ev_valid),
        .ev_idx         (ev_idx),
        .ev_gen         (ev_gen),
        .ev_wm          (ev_wm),
        .cm_valid       (gemm_cm_valid),
        .cm_ready       (gemm_cm_ready),
        .cm_idx         (gemm_cm_idx),
        .cm_gen         (gemm_cm_gen),
        .cm_wm          (gemm_cm_wm),
        .cm_final       (gemm_cm_final),
        .sa_valid       (g_sa_valid),
        .sa_ready       (seg_ready[0] && !attn_active),
        .sa             (g_sa),
        .sb_valid       (g_sb_valid),
        .sb_ready       (seg_ready[1] && !attn_active),
        .sb             (g_sb),
        .sc_valid       (g_sc_valid),
        .sc_ready       (seg_ready[2]),
        .sc             (g_sc),
        .uop_valid      (g_uop_valid),
        .uop_ready      (uop_ready && !attn_active),
        .uop            (g_uop),
        .mark_retired   (mark_retired[0]),
        .p_start        (p_start),
        .p_prog         (p_prog),
        .wb_start       (wb_start),
        .wb_cells       (wb_cells),
        .sources_closed (gemm_sources_closed),
        .done           (gemm_done),
        .done_status    (gemm_status),
        .busy           (gemm_busy)
    );
    g3_attn_seq u_attn (
        .clk            (clk),
        .rstn           (rstn),
        .start          (attn_start),
        .desc           (attn_desc),
        .ev_valid       (ev_valid),
        .ev_idx         (ev_idx),
        .ev_gen         (ev_gen),
        .ev_wm          (ev_wm),
        .cm_valid       (attn_cm_valid),
        .cm_ready       (attn_cm_ready),
        .cm_idx         (attn_cm_idx),
        .cm_gen         (attn_cm_gen),
        .cm_wm          (attn_cm_wm),
        .cm_final       (attn_cm_final),
        .sa_valid       (t_sa_valid),
        .sa_ready       (seg_ready[0] && attn_active),
        .sa             (t_sa),
        .sb_valid       (t_sb_valid),
        .sb_ready       (seg_ready[1] && attn_active),
        .sb             (t_sb),
        .uop_valid      (t_uop_valid),
        .uop_ready      (uop_ready && attn_active),
        .uop            (t_uop),
        .mark_retired   (mark_retired[0]),
        .sources_closed (attn_sources_closed),
        .done           (attn_done),
        .done_status    (attn_status),
        .busy           (attn_busy)
    );
    assign seg_valid[0] = attn_active ? t_sa_valid : g_sa_valid;
    assign seg_data[0*128 +: 128] = attn_active ? t_sa : g_sa;
    assign seg_valid[1] = attn_active ? t_sb_valid : g_sb_valid;
    assign seg_data[1*128 +: 128] = attn_active ? t_sb : g_sb;
    assign seg_valid[2] = g_sc_valid;
    assign seg_data[2*128 +: 128] = g_sc;
    wire uop_valid = attn_active ? t_uop_valid : g_uop_valid;
    wire [255:0] uop = attn_active ? t_uop : g_uop;

    // ------------------------------------------------------------------
    // Core and weight buffers.
    // ------------------------------------------------------------------
    wire [1:0] wb_rd_en; wire [15:0] wb_rd_addr; wire [511:0] wb_rd_data; wire [39:0] wb_pulled, wb_released;
    wire [1:0] wbuf_valid, wbuf_retired; wire [23:0] wbuf_addr; wire [511:0] wbuf_data;
    wire core_busy;
    wire a_credit; wire [1:0] b_credit; wire [2:0] c_credit;
    g3_core #(
        .R (4)
    ) u_core (
        .clk               (clk),
        .rstn              (rstn),
        .uop_valid         (uop_valid),
        .uop_ready         (uop_ready),
        .uop               (uop),
        .a_beat_valid      (eg_valid[0]),
        .a_beat_inner_idx  (eg_inner_idx[1:0]),
        .a_beat_inner_last (eg_inner_last[0]),
        .a_beat_user       (eg_user[7:0]),
        .a_beat_data       (eg_data[255:0]),
        .a_credit          (a_credit),
        .b_beat_valid      (eg_valid[1]),
        .b_beat_stream     (eg_stream[3:2]),
        .b_beat_inner_idx  (eg_inner_idx[3:2]),
        .b_beat_inner_last (eg_inner_last[1]),
        .b_beat_user       (eg_user[15:8]),
        .b_beat_data       (eg_data[511:256]),
        .b_credit          (b_credit),
        .c_beat_valid      (eg_valid[2]),
        .c_beat_stream     (eg_stream[5:4]),
        .c_beat_inner_idx  (eg_inner_idx[5:4]),
        .c_beat_inner_last (eg_inner_last[2]),
        .c_beat_seg_last   (eg_seg_last[2]),
        .c_beat_user       (eg_user[23:16]),
        .c_beat_data       (eg_data[767:512]),
        .c_credit          (c_credit),
        .wb_start          (wb_start),
        .wb_cells          (wb_cells),
        .wb_rd_en          (wb_rd_en),
        .wb_rd_addr        (wb_rd_addr),
        .wb_rd_data        (wb_rd_data),
        .wb_pulled         (wb_pulled),
        .wb_released       (wb_released),
        .d_valid           (in_valid[0]),
        .d_ready           (in_ready[0]),
        .d_addr            (in_addr[11:0]),
        .d_data            (in_data[255:0]),
        .d_strb            (in_strb[31:0]),
        .d_mark            (in_mark[0]),
        .feed_tick_fire    (feed_fire),
        .busy              (core_busy)
    );
    assign credit_return[2:0] = {2'b00, a_credit};
    assign credit_return[5:3] = {1'b0, b_credit};
    assign credit_return[8:6] = c_credit;
    assign in_valid[2:1] = 2'b00;
    assign in_addr[35:12] = 24'd0;
    assign in_data[767:256] = 512'd0;
    assign in_strb[95:32] = 64'd0;
    assign in_mark[2:1] = 2'b00;
    g3_weight_buffer u_wbuf (
        .clk        (clk),
        .rstn       (rstn),
        .wr_valid   (wbuf_valid),
        .wr_addr    (wbuf_addr),
        .wr_data    (wbuf_data),
        .wr_retired (wbuf_retired),
        .rd_en      (wb_rd_en),
        .rd_addr    (wb_rd_addr),
        .rd_data    (wb_rd_data)
    );

    // ------------------------------------------------------------------
    // VRU on UB port 3 (Gen2 segments widened to the Gen3 format).
    // ------------------------------------------------------------------
    wire [95:0] vru_seg96;
    g3_vru u_vru (
        .clk            (clk),
        .rstn           (rstn),
        .start          (vru_start),
        .desc           (vru_desc),
        .done           (vru_done),
        .done_status    (vru_status),
        .sources_closed (vru_sources_closed),
        .ev_valid       (ev_valid),
        .ev_idx         (ev_idx),
        .ev_gen         (ev_gen),
        .ev_wm          (ev_wm),
        .cm_valid       (vru_cm_valid),
        .cm_ready       (vru_cm_ready),
        .cm_idx         (vru_cm_idx),
        .cm_gen         (vru_cm_gen),
        .cm_wm          (vru_cm_wm),
        .cm_final       (vru_cm_final),
        .seg_valid      (seg_valid[3]),
        .seg_ready      (seg_ready[3]),
        .seg            (vru_seg96),
        .eg_valid       (eg_valid[3]),
        .eg_stream      (eg_stream[7:6]),
        .eg_data        (eg_data[1023:768]),
        .credit_return  (credit_return[11:9]),
        .wb_valid       (in_valid[3]),
        .wb_ready       (in_ready[3]),
        .wb_addr        (in_addr[47:36]),
        .wb_data        (in_data[1023:768]),
        .wb_strb        (in_strb[127:96]),
        .wb_mark        (in_mark[3]),
        .mark_retired   (mark_retired[3]),
        .busy           (vru_busy)
    );
    // Gen2 segment {.., close[89], .., user[84:77], s1[76:65], n1-1[64:53], s0[52:41], n0-1[40:39], start,
    // ext, base, ctx} -> Gen3 {close, user, s1, n1-1, o3 = 3 s0, o2 = 2 s0, o1 = s0, n0-1, start, ext, base, ctx}.
    wire [11:0] vru_s0 = vru_seg96[52:41];
    assign seg_data[3*128 +: 128] = {18'd0, vru_seg96[89], vru_seg96[84:77], vru_seg96[76:65], vru_seg96[64:53],
                                     12'(3 * vru_s0), 12'(2 * vru_s0), vru_s0, vru_seg96[40:0]};

    // ------------------------------------------------------------------
    // NI on UB port 4.
    // ------------------------------------------------------------------
    g3_ni #(
        .NODE_BASE (NODE_BASE)
    ) u_ni (
        .clk           (clk),
        .rstn          (rstn),
        .s_aw_valid    (s_aw_valid),
        .s_aw_ready    (s_aw_ready),
        .s_aw_addr     (s_aw_addr),
        .s_aw_len      (s_aw_len),
        .s_w_valid     (s_w_valid),
        .s_w_ready     (s_w_ready),
        .s_w_data      (s_w_data),
        .s_w_last      (s_w_last),
        .s_b_valid     (s_b_valid),
        .s_b_ready     (s_b_ready),
        .s_b_resp      (s_b_resp),
        .m_aw_valid    (m_aw_valid),
        .m_aw_ready    (m_aw_ready),
        .m_aw_addr     (m_aw_addr),
        .m_aw_len      (m_aw_len),
        .m_w_valid     (m_w_valid),
        .m_w_ready     (m_w_ready),
        .m_w_data      (m_w_data),
        .m_w_last      (m_w_last),
        .m_b_valid     (m_b_valid),
        .m_b_ready     (m_b_ready),
        .m_ar_valid    (m_ar_valid),
        .m_ar_ready    (m_ar_ready),
        .m_ar_addr     (m_ar_addr),
        .m_ar_len      (m_ar_len),
        .m_ar_id       (m_ar_id),
        .m_r_valid     (m_r_valid),
        .m_r_ready     (m_r_ready),
        .m_r_data      (m_r_data),
        .m_r_id        (m_r_id),
        .lk_idx        (lk_idx),
        .lk_valid      (lk_valid),
        .lk_gen        (lk_gen),
        .lk_base       (lk_base),
        .lk_ext        (lk_ext),
        .lk_writer     (lk_writer),
        .ev_out_valid  (ni_ev_valid),
        .ev_out_ready  (ni_ev_ready),
        .ev_out_idx    (ni_ev_idx),
        .ev_out_gen    (ni_ev_gen),
        .ev_out_wm     (ni_ev_wm),
        .ev_out_final  (ni_ev_final),
        .ev_valid      (ev_valid),
        .ev_idx        (ev_idx),
        .ev_gen        (ev_gen),
        .ev_wm         (ev_wm),
        .tx_start      (tx_start),
        .tx_desc       (tx_desc),
        .tx_done       (tx_done),
        .p_start       (p_start),
        .p_prog        (p_prog),
        .p_released    (wb_released),
        .p_pulled      (wb_pulled),
        .wbuf_valid    (wbuf_valid),
        .wbuf_addr     (wbuf_addr),
        .wbuf_data     (wbuf_data),
        .wbuf_retired  (wbuf_retired),
        .seg_valid     (seg_valid[4]),
        .seg_ready     (seg_ready[4]),
        .seg           (seg_data[4*128 +: 128]),
        .eg_valid      (eg_valid[4]),
        .eg_stream     (eg_stream[9:8]),
        .eg_data       (eg_data[1279:1024]),
        .credit_return (credit_return[14:12]),
        .in_valid      (in_valid[4]),
        .in_ready      (in_ready[4]),
        .in_addr       (in_addr[59:48]),
        .in_data       (in_data[1279:1024]),
        .in_mark       (in_mark[4]),
        .mark_retired  (mark_retired[4])
    );
    assign in_strb[159:128] = 32'hffff_ffff;

    assign idle = ctrl_idle && !ub_busy && !core_busy && !gemm_busy && !vru_busy && !attn_busy;
endmodule

`default_nettype wire
