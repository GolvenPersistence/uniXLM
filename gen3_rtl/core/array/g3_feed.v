`timescale 1ns/1ps
`default_nettype none

// Gen3 operand feed (gen3_rtl/README.md s3.2): turns UB beats, weight-buffer cells and P words
// into array ticks.
//
//   UB port A beats --> raw A FIFO (8) --> A parser --> A quads (4) + A scale sets (2) --+
//   P buffer words (PV windows, a_src = 1) ----------------------------------------------+
//   UB port B beats --> raw B FIFO left (8) / right (8) (contexts 0 / 1) --+              |
//   weight buffer left / right (STREAM: cell reader per half) -------------+-> B parser  |
//                                       per half: B quads (4) + B scale sets (2) --------+--> tick
//   credits: one per raw FIFO pop, per context.
//
// A quad: word w of the (up to) four 8-row panels of a K32 group (beats in panel order, the
// segment walk is outer-major); a tick uses the low or high 128 bits (K pair). B quad: word w of
// the four panels of a 32 x 32 cell. A scale set: 8 bytes per panel (one beat per panel, user
// destination panel and source slot). B scale set: the cell's scale word (full).
//
// Control: plan (from the uop at the queue head) -> cfg to the array; window queue (4): per window
// kg groups x 16 ticks; a tick needs A (or P) and, for each half with valid columns, its quad and
// scales. STREAM: the cell reader of a half prefetches its whole cell stream (started by
// wb_start) and the parser takes only reader beats while the reader has cells left.
module g3_feed (
    input  wire          clk,
    input  wire          rstn,
    // UB port A lane.
    input  wire          a_beat_valid,
    input  wire [1:0]    a_beat_inner_idx,
    input  wire          a_beat_inner_last,
    input  wire [7:0]    a_beat_user,
    input  wire [255:0]  a_beat_data,
    output wire          a_credit,
    // UB port B lane (stream 0 left, 1 right).
    input  wire          b_beat_valid,
    input  wire [1:0]    b_beat_stream,
    input  wire [1:0]    b_beat_inner_idx,
    input  wire          b_beat_inner_last,
    input  wire [7:0]    b_beat_user,
    input  wire [255:0]  b_beat_data,
    output wire [1:0]    b_credit,
    // Weight buffers (cell rings of 132 words), per half.
    input  wire          wb_start,                // a STREAM GEMM starts: counters reset
    input  wire [31:0]   wb_cells,                // cells per half {right, left}
    output wire [1:0]    wb_rd_en,
    output wire [15:0]   wb_rd_addr,              // {right, left} x 8
    input  wire [511:0]  wb_rd_data,              // {right, left} x 256, one cycle after rd_en
    input  wire [39:0]   wb_pulled,               // words landed {right, left} x 20
    output wire [39:0]   wb_released,             // words consumed
    // Plan from the uop queue head.
    input  wire          plan_valid,
    output wire          plan_ready,
    input  wire [5:0]    plan_kg,
    input  wire [31:0]   plan_rows,
    input  wire [5:0]    plan_nl,
    input  wire [5:0]    plan_nr,
    input  wire          plan_accum,
    input  wire          plan_emit,
    input  wire [1:0]    plan_op,
    input  wire          plan_asrc,
    // Array configuration and feed.
    output wire          cfg_valid,
    input  wire          cfg_ready,
    output wire [5:0]    cfg_kg,
    output wire [31:0]   cfg_rows,
    output wire [5:0]    cfg_nl,
    output wire [5:0]    cfg_nr,
    output wire          cfg_accum,
    output wire          cfg_emit,
    output wire [1:0]    cfg_op,
    output wire          feed_window_open,
    output wire          feed_tick_valid,
    input  wire          feed_tick_ready,
    output wire          feed_tick_last,
    output wire [511:0]  feed_a_payload,
    output wire [255:0]  feed_a_scale,
    output wire [511:0]  feed_bl_payload,
    output wire [255:0]  feed_bl_scale,
    output wire [511:0]  feed_br_payload,
    output wire [255:0]  feed_br_scale,
    // P (PV windows).
    input  wire          p_valid,
    input  wire [511:0]  p_data,
    input  wire [255:0]  p_scale,
    output wire          p_pop,
    output wire          busy
);
    localparam integer BW = 1 + 2 + 8 + 256;     // {inner_last, idx, user, data}
    localparam integer RING = 132;

    function automatic [63:0] slot_bytes(input [255:0] w, input [1:0] s);
        slot_bytes = w[s*64 +: 64];
    endfunction

    // ------------------------------------------------------------------
    // A: raw FIFO, parser, quads and scale sets.
    // ------------------------------------------------------------------
    reg [BW-1:0] raw_a_mem [0:7];
    reg [2:0] raw_a_rd_r, raw_a_wr_r; reg [3:0] raw_a_level_r;
    wire [BW-1:0] raw_a_head = raw_a_mem[raw_a_rd_r];
    reg [1023:0] a_quad_mem [0:3];
    reg [255:0]  a_scale_mem [0:1];
    reg [1:0] a_quad_rd_r, a_quad_wr_r; reg [2:0] a_quad_level_r;
    reg a_scale_rd_r, a_scale_wr_r; reg [1:0] a_scale_level_r;
    reg [1023:0] a_quad_asm_r;
    reg [255:0]  a_scale_asm_r;
    wire a_head_scale = raw_a_head[263];
    wire a_pop = raw_a_level_r != 0 && (a_head_scale ? (!raw_a_head[262] || a_scale_level_r != 2)
                                                     : (!raw_a_head[266] || a_quad_level_r != 4));
    assign a_credit = a_pop;

    // ------------------------------------------------------------------
    // B halves: raw FIFOs, cell readers, parsers.
    // ------------------------------------------------------------------
    reg [BW-1:0] raw_b_mem [0:1][0:7];
    reg [2:0] raw_b_rd_r [0:1], raw_b_wr_r [0:1]; reg [3:0] raw_b_level_r [0:1];
    reg [1023:0] b_quad_mem [0:1][0:3];
    reg [255:0]  b_scale_mem [0:1][0:1];
    reg [1:0] b_quad_rd_r [0:1], b_quad_wr_r [0:1]; reg [2:0] b_quad_level_r [0:1];
    reg b_scale_rd_r [0:1], b_scale_wr_r [0:1]; reg [1:0] b_scale_level_r [0:1];
    reg [1023:0] b_quad_asm_r [0:1];
    // Cell readers.
    reg [19:0] rd_cells_left_r [0:1];       // cells still to read
    reg [19:0] rd_consumed_r [0:1];         // words read (logical stream position of the next cell)
    reg [7:0]  rd_cell_phys_r [0:1];        // ring position of the current cell
    reg [5:0]  rd_word_r [0:1];             // next word of the cell (0 = scale, 1..32 = payload)
    reg        rd_pend_r [0:1];             // read issued last cycle
    reg [5:0]  rd_pend_word_r [0:1];
    reg [19:0] released_r [0:1];
    reg [1:0]  rd_quads_pend_r [0:1];       // quads being read (not yet complete in the parser)
    reg        rd_scale_pend_r [0:1];

    wire [1:0] h_beat_valid;                // a beat for the parser of each half this cycle
    wire [BW-1:0] h_beat [0:1];
    wire [1:0] raw_b_pop;
    assign b_credit = raw_b_pop;
    wire [1:0] rd_issue;
    for (genvar h = 0; h < 2; h = h + 1) begin : g_bh
        wire [BW-1:0] raw_head = raw_b_mem[h][raw_b_rd_r[h]];
        wire reader_on = rd_cells_left_r[h] != 0 || rd_pend_r[h];
        // Reader beat: registered read data of the previous cycle.
        wire [5:0] pw = rd_pend_word_r[h];
        wire [BW-1:0] rd_beat = pw == 0 ? {1'b1, 2'd0, 8'b1110_0000, wb_rd_data[h*256 +: 256]}
                                        : {(pw - 6'd1) % 4 == 3, 2'((pw - 6'd1) % 4), 8'd0, wb_rd_data[h*256 +: 256]};
        wire head_scale = raw_head[263];
        wire raw_ok = raw_b_level_r[h] != 0 && (head_scale ? (!raw_head[262] || b_scale_level_r[h] != 2)
                                                             : (!raw_head[266] || b_quad_level_r[h] != 4));
        assign raw_b_pop[h] = !reader_on && raw_ok;
        assign h_beat_valid[h] = rd_pend_r[h] || raw_b_pop[h];
        assign h_beat[h] = rd_pend_r[h] ? rd_beat : raw_head;
        // Issue the next read of the current cell: the cell has landed, and the parser has room
        // (a quad starts only if a quad slot is free counting the quads being read).
        wire [19:0] avail = wb_pulled[h*20 +: 20] - rd_consumed_r[h];
        wire cell_in = avail >= 20'd33;
        wire [5:0] w = rd_word_r[h];
        wire quad_start = w != 0 && (w - 6'd1) % 4 == 0;
        assign rd_issue[h] = rd_cells_left_r[h] != 0 && cell_in &&
                             (w == 0 ? (b_scale_level_r[h] + rd_scale_pend_r[h] < 2)
                                     : (!quad_start || b_quad_level_r[h] + rd_quads_pend_r[h] < 4));
        // Payload word w (1..32) of panel p = (w-1)%4, quad word (w-1)/4: cell word 1 + 8p + q.
        wire [7:0] cw = w == 0 ? 8'd0 : 8'(1 + 8 * ((w - 6'd1) % 4) + (w - 6'd1) / 4);
        wire [8:0] pa = rd_cell_phys_r[h] + cw;
        assign wb_rd_addr[h*8 +: 8] = pa >= RING ? pa - RING : pa[7:0];
        assign wb_rd_en[h] = rd_issue[h];
        assign wb_released[h*20 +: 20] = released_r[h];
    end

    // ------------------------------------------------------------------
    // Window plan queue and tick assembly.
    // ------------------------------------------------------------------
    reg win_r;
    reg [5:0] win_kg_r, win_group_r;
    reg [3:0] win_tick_r;
    reg win_asrc_r, win_l_r, win_r_r;
    reg [5:0] pq_kg_mem [0:3];
    reg [2:0] pq_flags_mem [0:3];          // {right, left, a_src}
    reg [1:0] pq_rd_r, pq_wr_r; reg [2:0] pq_level_r;
    wire a_from_ub = !win_asrc_r;
    wire tick_avail = win_r && (a_from_ub ? (a_quad_level_r != 0 && a_scale_level_r != 0) : p_valid) &&
                      (!win_l_r || (b_quad_level_r[0] != 0 && b_scale_level_r[0] != 0)) &&
                      (!win_r_r || (b_quad_level_r[1] != 0 && b_scale_level_r[1] != 0));
    assign feed_tick_valid = tick_avail;
    assign feed_window_open = win_r;
    wire tick_fire = tick_avail && feed_tick_ready;
    wire last_tick = win_group_r == win_kg_r - 1'b1 && win_tick_r == 4'd15;
    assign feed_tick_last = last_tick;
    wire half = win_tick_r[0];
    wire [1023:0] aq = a_quad_mem[a_quad_rd_r];
    wire [1023:0] blq = b_quad_mem[0][b_quad_rd_r[0]];
    wire [1023:0] brq = b_quad_mem[1][b_quad_rd_r[1]];
    function automatic [511:0] quad_tick(input [1023:0] q, input hf);
        quad_tick = {q[768 + hf*128 +: 128], q[512 + hf*128 +: 128], q[256 + hf*128 +: 128], q[hf*128 +: 128]};
    endfunction
    assign feed_a_payload = a_from_ub ? quad_tick(aq, half) : p_data;
    assign feed_a_scale = a_from_ub ? a_scale_mem[a_scale_rd_r] : p_scale;
    assign p_pop = tick_fire && !a_from_ub;
    assign feed_bl_payload = win_l_r ? quad_tick(blq, half) : 512'd0;
    assign feed_bl_scale = win_l_r ? b_scale_mem[0][b_scale_rd_r[0]] : {32{8'd127}};
    assign feed_br_payload = win_r_r ? quad_tick(brq, half) : 512'd0;
    assign feed_br_scale = win_r_r ? b_scale_mem[1][b_scale_rd_r[1]] : {32{8'd127}};

    assign cfg_valid = plan_valid && pq_level_r != 4;
    assign plan_ready = cfg_valid && cfg_ready;
    assign cfg_kg = plan_kg;
    assign cfg_rows = plan_rows;
    assign cfg_nl = plan_nl;
    assign cfg_nr = plan_nr;
    assign cfg_accum = plan_accum;
    assign cfg_emit = plan_emit;
    assign cfg_op = plan_op;

    integer i;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            raw_a_rd_r <= 0; raw_a_wr_r <= 0; raw_a_level_r <= 0;
            a_quad_rd_r <= 0; a_quad_wr_r <= 0; a_quad_level_r <= 0;
            a_scale_rd_r <= 0; a_scale_wr_r <= 0; a_scale_level_r <= 0;
            a_quad_asm_r <= 0; a_scale_asm_r <= 0;
            for (i = 0; i < 2; i = i + 1) begin
                raw_b_rd_r[i] <= 0; raw_b_wr_r[i] <= 0; raw_b_level_r[i] <= 0;
                b_quad_rd_r[i] <= 0; b_quad_wr_r[i] <= 0; b_quad_level_r[i] <= 0;
                b_scale_rd_r[i] <= 0; b_scale_wr_r[i] <= 0; b_scale_level_r[i] <= 0;
                b_quad_asm_r[i] <= 0;
                rd_cells_left_r[i] <= 0; rd_consumed_r[i] <= 0; rd_cell_phys_r[i] <= 0; rd_word_r[i] <= 0;
                rd_pend_r[i] <= 1'b0; rd_pend_word_r[i] <= 0; released_r[i] <= 0;
                rd_quads_pend_r[i] <= 0; rd_scale_pend_r[i] <= 1'b0;
            end
            win_r <= 1'b0; win_kg_r <= 0; win_group_r <= 0; win_tick_r <= 0; win_asrc_r <= 1'b0;
            win_l_r <= 1'b0; win_r_r <= 1'b0;
            pq_rd_r <= 0; pq_wr_r <= 0; pq_level_r <= 0;
        end else begin
            // --- A raw FIFO and parser.
            if (a_beat_valid) begin
                raw_a_mem[raw_a_wr_r] <= {a_beat_inner_last, a_beat_inner_idx, a_beat_user, a_beat_data};
                raw_a_wr_r <= raw_a_wr_r + 1'b1;
            end
            if (a_pop) raw_a_rd_r <= raw_a_rd_r + 1'b1;
            raw_a_level_r <= raw_a_level_r + a_beat_valid - a_pop;
            if (a_pop) begin
                if (a_head_scale) begin
                    if (raw_a_head[262]) begin
                        for (i = 0; i < 4; i = i + 1)
                            a_scale_mem[a_scale_wr_r][i*64 +: 64] <= (i == raw_a_head[260:259]) ?
                                slot_bytes(raw_a_head[255:0], raw_a_head[258:257]) : a_scale_asm_r[i*64 +: 64];
                        a_scale_wr_r <= !a_scale_wr_r; a_scale_asm_r <= 0;
                    end else a_scale_asm_r[raw_a_head[260:259]*64 +: 64] <= slot_bytes(raw_a_head[255:0], raw_a_head[258:257]);
                end else begin
                    if (raw_a_head[266]) begin
                        for (i = 0; i < 4; i = i + 1)
                            a_quad_mem[a_quad_wr_r][i*256 +: 256] <= (i == raw_a_head[265:264]) ? raw_a_head[255:0]
                                                                                                 : a_quad_asm_r[i*256 +: 256];
                        a_quad_wr_r <= a_quad_wr_r + 1'b1; a_quad_asm_r <= 0;
                    end else a_quad_asm_r[raw_a_head[265:264]*256 +: 256] <= raw_a_head[255:0];
                end
            end
            // --- B: raw FIFOs, readers, parsers.
            for (i = 0; i < 2; i = i + 1) begin : b_half
                if (b_beat_valid && b_beat_stream == i) begin
                    raw_b_mem[i][raw_b_wr_r[i]] <= {b_beat_inner_last, b_beat_inner_idx, b_beat_user, b_beat_data};
                    raw_b_wr_r[i] <= raw_b_wr_r[i] + 1'b1;
                end
                if (raw_b_pop[i]) raw_b_rd_r[i] <= raw_b_rd_r[i] + 1'b1;
                raw_b_level_r[i] <= raw_b_level_r[i] + (b_beat_valid && b_beat_stream == i) - raw_b_pop[i];
                // Reader.
                rd_pend_r[i] <= rd_issue[i];
                rd_pend_word_r[i] <= rd_word_r[i];
                if (wb_start) begin
                    rd_cells_left_r[i] <= wb_cells[i*16 +: 16]; rd_consumed_r[i] <= 0; rd_cell_phys_r[i] <= 0;
                    rd_word_r[i] <= 0; released_r[i] <= 0;
                end else if (rd_issue[i]) begin
                    if (rd_word_r[i] == 6'd32) begin
                        rd_word_r[i] <= 0;
                        rd_cells_left_r[i] <= rd_cells_left_r[i] - 1'b1;
                        rd_consumed_r[i] <= rd_consumed_r[i] + 20'd33;
                        rd_cell_phys_r[i] <= rd_cell_phys_r[i] + 8'd33 >= RING ? rd_cell_phys_r[i] + 8'd33 - RING
                                                                               : rd_cell_phys_r[i] + 8'd33;
                    end else rd_word_r[i] <= rd_word_r[i] + 1'b1;
                end
                // The ring words of a cell are free once its last read returned.
                if (!wb_start && rd_pend_r[i] && rd_pend_word_r[i] == 6'd32) released_r[i] <= released_r[i] + 20'd33;
                rd_scale_pend_r[i] <= rd_issue[i] && rd_word_r[i] == 0;
                rd_quads_pend_r[i] <= rd_quads_pend_r[i] + (rd_issue[i] && rd_word_r[i] != 0 && (rd_word_r[i] - 6'd1) % 4 == 0)
                                                         - (h_beat_valid[i] && !h_beat[i][263] && h_beat[i][266] && rd_pend_r[i]);
                // Parser.
                if (h_beat_valid[i]) begin
                    if (h_beat[i][263]) begin
                        // Scale beats of a B cell are full (32 bytes, one per column).
                        b_scale_mem[i][b_scale_wr_r[i]] <= h_beat[i][255:0];
                        b_scale_wr_r[i] <= !b_scale_wr_r[i];
                    end else begin
                        if (h_beat[i][266]) begin
                            for (integer q = 0; q < 4; q = q + 1)
                                b_quad_mem[i][b_quad_wr_r[i]][q*256 +: 256] <= (q == h_beat[i][265:264]) ? h_beat[i][255:0]
                                                                                                  : b_quad_asm_r[i][q*256 +: 256];
                            b_quad_wr_r[i] <= b_quad_wr_r[i] + 1'b1; b_quad_asm_r[i] <= 0;
                        end else b_quad_asm_r[i][h_beat[i][265:264]*256 +: 256] <= h_beat[i][255:0];
                    end
                end
            end
            // --- Window plan queue.
            if (plan_ready) begin
                pq_kg_mem[pq_wr_r] <= plan_kg; pq_flags_mem[pq_wr_r] <= {plan_nr != 0, plan_nl != 0, plan_asrc};
                pq_wr_r <= pq_wr_r + 1'b1;
            end
            if (!win_r && pq_level_r != 0) begin
                win_r <= 1'b1; win_kg_r <= pq_kg_mem[pq_rd_r];
                {win_r_r, win_l_r, win_asrc_r} <= pq_flags_mem[pq_rd_r];
                win_group_r <= 0; win_tick_r <= 0; pq_rd_r <= pq_rd_r + 1'b1;
            end
            pq_level_r <= pq_level_r + plan_ready - (!win_r && pq_level_r != 0);
            if (tick_fire) begin
                win_tick_r <= win_tick_r + 1'b1;
                if (win_tick_r == 4'd15) begin
                    win_group_r <= win_group_r + 1'b1;
                    if (a_from_ub) a_scale_rd_r <= !a_scale_rd_r;
                    if (win_l_r) b_scale_rd_r[0] <= !b_scale_rd_r[0];
                    if (win_r_r) b_scale_rd_r[1] <= !b_scale_rd_r[1];
                end
                if (half) begin
                    if (a_from_ub) a_quad_rd_r <= a_quad_rd_r + 1'b1;
                    if (win_l_r) b_quad_rd_r[0] <= b_quad_rd_r[0] + 1'b1;
                    if (win_r_r) b_quad_rd_r[1] <= b_quad_rd_r[1] + 1'b1;
                end
                if (last_tick) win_r <= 1'b0;
            end
            a_quad_level_r <= a_quad_level_r + (a_pop && !a_head_scale && raw_a_head[266])
                                             - (tick_fire && half && a_from_ub);
            a_scale_level_r <= a_scale_level_r + (a_pop && a_head_scale && raw_a_head[262])
                                               - (tick_fire && win_tick_r == 4'd15 && a_from_ub);
            for (i = 0; i < 2; i = i + 1) begin
                b_quad_level_r[i] <= b_quad_level_r[i] + (h_beat_valid[i] && !h_beat[i][263] && h_beat[i][266])
                                                       - (tick_fire && half && (i == 0 ? win_l_r : win_r_r));
                b_scale_level_r[i] <= b_scale_level_r[i] + (h_beat_valid[i] && h_beat[i][263])
                                                         - (tick_fire && win_tick_r == 4'd15 && (i == 0 ? win_l_r : win_r_r));
            end
        end
    end
    assign busy = win_r || pq_level_r != 0 || raw_a_level_r != 0 || raw_b_level_r[0] != 0 || raw_b_level_r[1] != 0 ||
                  a_quad_level_r != 0 || a_scale_level_r != 0 || b_quad_level_r[0] != 0 || b_quad_level_r[1] != 0 ||
                  b_scale_level_r[0] != 0 || b_scale_level_r[1] != 0 || rd_cells_left_r[0] != 0 ||
                  rd_cells_left_r[1] != 0 || rd_pend_r[0] || rd_pend_r[1];
endmodule

`default_nettype wire
