`timescale 1ns/1ps
`default_nettype none

// Gen3 network interface (docs/gen2/command_abi.md s6; gen3_rtl/README.md s5): the Gen2 NI
// (rtl/gen2/g2_ni.v) with the pull engines writing the weight buffers of the array halves
// instead of a UB ring; the UB N port carries RX pushes and MOVE_OUT only.
//
// Block diagram:
//
//   AXI sub AW/W --> RX: URF lookup of the binding window {gen, idx, word} --> in_* (UB N port,
//   AXI sub B   <--  after the burst retired and its commit event              every beat marked)
//   mark_retired --> RX retire count
//
//   pull 0 (g3_pull) <--> AR/R (id 0) --> weight buffer left  (wbuf_*: ring position, retired)
//   pull 1 (g3_pull) <--> AR/R (id 1) --> weight buffer right
//
//   MOVE_OUT x2: desc + watch wm (ev_*) --> chunk (<= 16 words): UB read segment + AXI AW
//                eg_* (N port) --> channel buffer (8) --> AXI W (AW order) --> B --> tx_done
//
// Control:
//
//   RX: RX_IDLE --AW (window valid, writer EXT)--> RX_DATA --last W--> RX_WAIT
//       --all beats retired--> RX_EVENT (commit event) --> RX_RESP (B) --> RX_IDLE
//   TX: per channel, issue a chunk while the source watermark covers it and the burst
//       queue has room; the channel finishes after its last B response.
module g3_ni #(
    parameter [47:0] NODE_BASE = 48'h0000_1000_0000
) (
    input  wire          clk,
    input  wire          rstn,
    // AXI subordinate (push into binding windows).
    input  wire          s_aw_valid,
    output wire          s_aw_ready,
    input  wire [47:0]   s_aw_addr,
    input  wire [7:0]    s_aw_len,
    input  wire          s_w_valid,
    output wire          s_w_ready,
    input  wire [255:0]  s_w_data,
    input  wire          s_w_last,
    output reg           s_b_valid,
    input  wire          s_b_ready,
    output reg  [1:0]    s_b_resp,
    // AXI manager write (MOVE_OUT).
    output reg           m_aw_valid,
    input  wire          m_aw_ready,
    output reg  [47:0]   m_aw_addr,
    output reg  [7:0]    m_aw_len,
    output wire          m_w_valid,
    input  wire          m_w_ready,
    output wire [255:0]  m_w_data,
    output wire          m_w_last,
    input  wire          m_b_valid,
    output wire          m_b_ready,
    // AXI manager read (pull).
    output wire          m_ar_valid,
    input  wire          m_ar_ready,
    output wire [47:0]   m_ar_addr,
    output wire [7:0]    m_ar_len,
    output wire          m_ar_id,
    input  wire          m_r_valid,
    output wire          m_r_ready,
    input  wire [255:0]  m_r_data,
    input  wire          m_r_id,
    // URF lookup for RX.
    output wire [7:0]    lk_idx,
    input  wire          lk_valid,
    input  wire [7:0]    lk_gen,
    input  wire [11:0]   lk_base,
    input  wire [12:0]   lk_ext,
    input  wire [1:0]    lk_writer,
    // Commit events (RX).
    output reg           ev_out_valid,
    input  wire          ev_out_ready,
    output reg  [7:0]    ev_out_idx,
    output reg  [7:0]    ev_out_gen,
    output reg  [12:0]   ev_out_wm,
    output reg           ev_out_final,
    // Event broadcast (TX watch registers).
    input  wire          ev_valid,
    input  wire [7:0]    ev_idx,
    input  wire [7:0]    ev_gen,
    input  wire [12:0]   ev_wm,
    // MOVE_OUT channels: desc {idx, gen, base, wm, offset, words, dst}.
    input  wire [1:0]    tx_start,
    input  wire [2*128-1:0] tx_desc,
    output reg  [1:0]    tx_done,
    // Pull engines.
    input  wire [1:0]    p_start,
    input  wire [2*160-1:0] p_prog,
    input  wire [2*20-1:0] p_released,
    output wire [2*20-1:0] p_pulled,
    // Weight buffer writes (pull engines).
    output wire [1:0]    wbuf_valid,
    output wire [23:0]   wbuf_addr,
    output wire [511:0]  wbuf_data,
    input  wire [1:0]    wbuf_retired,
    // UB N port.
    output reg           seg_valid,
    input  wire          seg_ready,
    output reg  [127:0]  seg,
    input  wire          eg_valid,
    input  wire [1:0]    eg_stream,
    input  wire [255:0]  eg_data,
    output wire [2:0]    credit_return,
    output wire          in_valid,
    input  wire          in_ready,
    output wire [11:0]   in_addr,
    output wire [255:0]  in_data,
    output wire          in_mark,
    input  wire          mark_retired
);
    // ------------------------------------------------------------------
    // RX: one burst at a time.
    // ------------------------------------------------------------------
    localparam [2:0] RX_IDLE = 0, RX_DATA = 1, RX_WAIT = 2, RX_EVENT = 3, RX_RESP = 4;
    reg [2:0] rx_state_r, rx_state_n;
    reg [7:0] rx_idx_r, rx_gen_r;
    reg [11:0] rx_word_r;
    reg [8:0] rx_beats_r, rx_sent_r, rx_retired_r;
    reg [11:0] rx_base_r;
    reg [12:0] rx_ext_r;
    wire [47:0] aw_off = s_aw_addr - NODE_BASE;
    assign lk_idx = rx_state_r == RX_IDLE ? aw_off[24:17] : rx_idx_r;
    wire aw_ok = lk_valid && lk_gen == aw_off[32:25] && lk_writer == 2'd1;
    assign s_aw_ready = rx_state_r == RX_IDLE && aw_ok;
    wire rx_w_valid = rx_state_r == RX_DATA && s_w_valid;

    // ------------------------------------------------------------------
    // Pull engines.
    // ------------------------------------------------------------------
    wire [1:0] pl_ar_valid; wire [1:0] pl_ar_ready; wire [2*48-1:0] pl_ar_addr; wire [2*8-1:0] pl_ar_len;
    wire [1:0] pl_r_ready; wire [1:0] pl_wb_valid; wire [1:0] pl_wb_ready;
    wire [2*12-1:0] pl_wb_addr; wire [2*256-1:0] pl_wb_data; wire [1:0] pl_retired;
    for (genvar k = 0; k < 2; k = k + 1) begin : g_pull
        g3_pull u_pull (
            .clk        (clk),
            .rstn       (rstn),
            .start      (p_start[k]),
            .prog       (p_prog[k*160 +: 160]),
            .released   (p_released[k*20 +: 20]),
            .pulled     (p_pulled[k*20 +: 20]),
            .ar_valid   (pl_ar_valid[k]),
            .ar_ready   (pl_ar_ready[k]),
            .ar_addr    (pl_ar_addr[k*48 +: 48]),
            .ar_len     (pl_ar_len[k*8 +: 8]),
            .r_valid    (m_r_valid && m_r_id == k),
            .r_ready    (pl_r_ready[k]),
            .r_data     (m_r_data),
            .wb_valid   (pl_wb_valid[k]),
            .wb_ready   (pl_wb_ready[k]),
            .wb_addr    (pl_wb_addr[k*12 +: 12]),
            .wb_data    (pl_wb_data[k*256 +: 256]),
            .wb_retired (pl_retired[k])
        );
    end
    reg ar_sel_r;
    wire ar_pick = pl_ar_valid[0] && (!pl_ar_valid[1] || !ar_sel_r) ? 1'b0 : 1'b1;
    assign m_ar_valid = |pl_ar_valid;
    assign m_ar_id = ar_pick;
    assign m_ar_addr = pl_ar_addr[ar_pick*48 +: 48];
    assign m_ar_len = pl_ar_len[ar_pick*8 +: 8];
    assign pl_ar_ready = {m_ar_ready && ar_pick, m_ar_ready && !ar_pick};
    assign m_r_ready = pl_r_ready[m_r_id];

    // ------------------------------------------------------------------
    // N-port ingress: RX only (every beat marked). Pull engines -> weight buffers.
    // ------------------------------------------------------------------
    wire rx_sel = rx_w_valid;
    assign in_valid = rx_sel;
    assign in_addr = rx_base_r + rx_word_r + rx_sent_r;
    assign in_data = s_w_data;
    assign in_mark = 1'b1;
    assign s_w_ready = rx_sel && in_ready;
    assign wbuf_valid = pl_wb_valid;
    assign wbuf_addr = pl_wb_addr;
    assign wbuf_data = pl_wb_data;
    assign pl_wb_ready = 2'b11;
    assign pl_retired = wbuf_retired;
    wire rx_beat_retired = mark_retired;

    // ------------------------------------------------------------------
    // RX FSM: state register, next state, control events.
    // ------------------------------------------------------------------
    wire rx_aw_fire     = s_aw_valid && s_aw_ready;        // RX_IDLE only
    wire rx_w_fire      = s_w_valid && s_w_ready;          // RX_DATA only
    wire rx_all_retired = rx_retired_r == rx_beats_r;

    always @(posedge clk or negedge rstn)
        if (!rstn) rx_state_r <= RX_IDLE;
        else rx_state_r <= rx_state_n;

    always @* begin
        rx_state_n = rx_state_r;
        case (rx_state_r)
        RX_IDLE:  if (rx_aw_fire) rx_state_n = RX_DATA;
        RX_DATA:  if (rx_w_fire && s_w_last) rx_state_n = RX_WAIT;
        RX_WAIT:  if (rx_all_retired) rx_state_n = RX_EVENT;
        RX_EVENT: if (!ev_out_valid) rx_state_n = RX_RESP;
        RX_RESP:  if (!s_b_valid) rx_state_n = RX_IDLE;
        default:  rx_state_n = RX_IDLE;
        endcase
    end

    wire rx_event_issue = rx_state_r == RX_WAIT && rx_all_retired;   // commit event of the burst
    wire rx_resp_issue  = rx_state_r == RX_EVENT && !ev_out_valid;    // B once the event was taken

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            s_b_valid <= 0; ev_out_valid <= 0;
            ar_sel_r <= 0; rx_sent_r <= 0; rx_retired_r <= 0;
        end else begin
            if (m_ar_valid && m_ar_ready) ar_sel_r <= !ar_pick;
            if (rx_event_issue) ev_out_valid <= 1'b1;
            else if (ev_out_valid && ev_out_ready) ev_out_valid <= 1'b0;
            if (rx_resp_issue) s_b_valid <= 1'b1;
            else if (s_b_valid && s_b_ready) s_b_valid <= 1'b0;
            if (rx_aw_fire) rx_retired_r <= 0;
            else if (rx_beat_retired) rx_retired_r <= rx_retired_r + 1'b1;
            if (rx_aw_fire) rx_sent_r <= 0;
            else if (rx_w_fire) rx_sent_r <= rx_sent_r + 1'b1;
        end
    end

    always @(posedge clk) begin
        if (rx_aw_fire) begin
            rx_idx_r <= aw_off[24:17]; rx_gen_r <= aw_off[32:25]; rx_word_r <= aw_off[16:5];
            rx_beats_r <= s_aw_len + 1'b1;
            rx_base_r <= lk_base; rx_ext_r <= lk_ext;
        end
        if (rx_event_issue) begin
            ev_out_idx <= rx_idx_r; ev_out_gen <= rx_gen_r;
            ev_out_wm <= rx_word_r + rx_beats_r; ev_out_final <= (rx_word_r + rx_beats_r) == rx_ext_r;
        end
        if (rx_resp_issue) s_b_resp <= 2'b00;
    end

    // ------------------------------------------------------------------
    // TX channels (MOVE_OUT): chunks of <= 16 words, gated by the source
    // watermark, one AXI burst per chunk; the write port serves one burst at
    // a time in request order.
    // ------------------------------------------------------------------
    reg [1:0] tx_run_r;
    reg [7:0] tx_idx_r [0:1], tx_gen_r [0:1];
    reg [11:0] tx_base_r [0:1], tx_off_r [0:1];
    reg [12:0] tx_wm_r [0:1];
    reg [12:0] tx_words_r [0:1], tx_issued_r [0:1];
    reg [47:0] tx_dst_r [0:1];
    reg [8:0] tx_bpend_r [0:1];
    // Egress buffers (8 beats) per channel.
    reg [255:0] eb_mem [0:1][0:7];
    reg [2:0] eb_rd_ptr_r [0:1], eb_wr_ptr_r [0:1];
    reg [3:0] eb_level_r [0:1];
    // Burst queue: {channel, beats} in AW order.
    reg [5:0] bq_mem [0:7];
    reg [2:0] bq_rd_ptr_r, bq_wr_ptr_r; reg [3:0] bq_level_r;
    reg [4:0] w_beat_r;
    wire [5:0] bq_head = bq_mem[bq_rd_ptr_r];
    wire w_ch = bq_head[5];
    assign m_w_valid = bq_level_r != 0 && eb_level_r[w_ch] != 0;
    assign m_w_data = eb_mem[w_ch][eb_rd_ptr_r[w_ch]];
    assign m_w_last = w_beat_r + 1 == bq_head[4:0];
    assign m_b_ready = 1'b1;
    reg [0:0] b_owner_mem [0:15];
    reg [3:0] bo_rd_ptr_r, bo_wr_ptr_r;
    assign credit_return = {1'b0, m_w_valid && m_w_ready && w_ch, m_w_valid && m_w_ready && !w_ch};

    integer c;
    reg [1:0] bp_inc;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            tx_run_r <= 0;
            tx_done <= 0;
            seg_valid <= 0;
            seg <= 0;
            m_aw_valid <= 0;
            bq_rd_ptr_r <= 0;
            bq_wr_ptr_r <= 0;
            bq_level_r <= 0;
            w_beat_r <= 0; bo_rd_ptr_r <= 0; bo_wr_ptr_r <= 0;
            for (c = 0; c < 2; c = c + 1) begin
                eb_rd_ptr_r[c] <= 0;
                eb_wr_ptr_r[c] <= 0;
                eb_level_r[c] <= 0;
                tx_bpend_r[c] <= 0;
            end
        end else begin
            tx_done <= 0;
            if (seg_valid && seg_ready) seg_valid <= 1'b0;
            if (m_aw_valid && m_aw_ready) m_aw_valid <= 1'b0;
            for (c = 0; c < 2; c = c + 1) begin
                if (tx_start[c]) begin
                    {tx_idx_r[c], tx_gen_r[c], tx_base_r[c], tx_wm_r[c], tx_off_r[c], tx_words_r[c], tx_dst_r[c]} <=
                        tx_desc[c*128 +: 114];
                    tx_issued_r[c] <= 0; tx_bpend_r[c] <= 0; tx_run_r[c] <= 1'b1;
                end
                if (ev_valid && tx_run_r[c] && ev_idx == tx_idx_r[c] && ev_gen == tx_gen_r[c] &&
                    tx_wm_r[c] != 13'h1fff && ev_wm > tx_wm_r[c]) tx_wm_r[c] <= ev_wm;
            end
            // Issue one chunk (segment + AW) per cycle, channel 0 first.
            bp_inc = 2'b00;
            begin : issue
                reg done_one;
                reg [12:0] left, need;
                reg [4:0] cl;
                reg [47:0] a;
                reg [7:0] to4k;
                done_one = 0;
                for (c = 0; c < 2; c = c + 1) begin
                    left = tx_words_r[c] - tx_issued_r[c];
                    a = tx_dst_r[c] + tx_issued_r[c] * 32;
                    to4k = (13'h1000 - a[11:0]) >> 5;
                    cl = left > 16 ? 5'd16 : left[4:0];
                    if (to4k != 0 && to4k < cl) cl = to4k[4:0];
                    need = tx_off_r[c] + tx_issued_r[c] + cl;
                    if (!done_one && tx_run_r[c] && left != 0 && !seg_valid && !m_aw_valid && bq_level_r < 7 &&
                        (tx_wm_r[c] == 13'h1fff || tx_wm_r[c] >= need)) begin
                        seg_valid <= 1'b1;
                        // Gen3 segment: one inner step, cl outer steps of one word.
                        seg <= {18'd0, 1'b0, 8'd0, 12'd1, 12'(cl - 1), 12'd0, 12'd0, 12'd0, 2'd0,
                                12'(tx_off_r[c] + tx_issued_r[c]), 13'd4096, tx_base_r[c], c[1:0]};
                        m_aw_valid <= 1'b1; m_aw_addr <= a; m_aw_len <= cl - 1'b1;
                        bq_mem[bq_wr_ptr_r] <= {c[0], cl}; bq_wr_ptr_r <= bq_wr_ptr_r + 1'b1;
                        b_owner_mem[bo_wr_ptr_r] <= c[0]; bo_wr_ptr_r <= bo_wr_ptr_r + 1'b1;
                        tx_issued_r[c] <= tx_issued_r[c] + cl;
                        bp_inc[c] = 1'b1;
                        done_one = 1;
                    end
                end
            end
            // Egress beats into the channel buffers.
            if (eg_valid && eg_stream < 2) begin
                eb_mem[eg_stream[0]][eb_wr_ptr_r[eg_stream[0]]] <= eg_data;
                eb_wr_ptr_r[eg_stream[0]] <= eb_wr_ptr_r[eg_stream[0]] + 1'b1;
            end
            for (c = 0; c < 2; c = c + 1)
                eb_level_r[c] <= eb_level_r[c] + (eg_valid && eg_stream == c) - (m_w_valid && m_w_ready && w_ch == c);
            if (m_w_valid && m_w_ready) begin
                eb_rd_ptr_r[w_ch] <= eb_rd_ptr_r[w_ch] + 1'b1;
                if (m_w_last) begin w_beat_r <= 0; bq_rd_ptr_r <= bq_rd_ptr_r + 1'b1; end
                else w_beat_r <= w_beat_r + 1'b1;
            end
            bq_level_r <= bq_level_r + (m_aw_valid && m_aw_ready) - (m_w_valid && m_w_ready && m_w_last);
            // B responses complete chunks; a channel finishes after its last B.
            if (m_b_valid) bo_rd_ptr_r <= bo_rd_ptr_r + 1'b1;
            for (c = 0; c < 2; c = c + 1)
                tx_bpend_r[c] <= tx_bpend_r[c] + bp_inc[c] - (m_b_valid && b_owner_mem[bo_rd_ptr_r] == c);
            for (c = 0; c < 2; c = c + 1)
                if (tx_run_r[c] && tx_issued_r[c] == tx_words_r[c] && tx_bpend_r[c] == 0 && !tx_start[c]) begin
                    tx_run_r[c] <= 1'b0; tx_done[c] <= 1'b1;
                end
        end
    end
endmodule

`default_nettype wire
