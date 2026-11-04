`timescale 1ns/1ps
`default_nettype none

// Gen3 array (docs/gen3/README.md s3, gen3_rtl/README.md s3.1): one 32 x 64 output array, two
// halves of 32 rows x 16 GEMUs on either side of the vector column. A is skewed once (row r
// delayed r) and injected into both halves at the middle; each half has its own B (32 columns).
//
//   cfg (K groups, rows, valid columns per half, op) --> execution window (kg x 16 ticks)
//   feed A (32 rows x K pair) --> 32 A skew lines --+--> left half  (32 x 16 GEMU) --> left exports
//                                                   +--> right half (32 x 16 GEMU) --> right exports
//   feed B left / right (32 columns x K pair each) -----^ (skewed inside the half)
//
//   Row r exports one column pair of each half per tick (column pair c at tick 16b + r + c), so a
//   lane receives four values per tick: the Gen2 rate of one value per Node cycle.
//
// Control: array_advance = rc_clk_en && array_ready moves every array register (R = 4 Node
// cycles per tick); cfg is accepted when the previous window takes its last tick.
module g3_array (
    input  wire              clk,
    input  wire              rstn,
    input  wire              rc_clk_en,
    input  wire              array_ready,
    // Row exports of both halves.
    output wire [31:0]       left_valid,
    output wire [32*113-1:0] left_data,
    output wire [31:0]       right_valid,
    output wire [32*113-1:0] right_data,
    // Window configuration.
    input  wire              cfg_valid,
    output reg               cfg_ready,
    input  wire [5:0]        cfg_k_group_count,
    input  wire [31:0]       cfg_rows,
    input  wire [5:0]        cfg_nl,
    input  wire [5:0]        cfg_nr,
    input  wire              cfg_accumulate,
    input  wire              cfg_emit_result,
    input  wire [1:0]        cfg_operation,
    // Operand feed (one tick = one K pair).
    input  wire              feed_window_open,
    input  wire              feed_tick_valid,
    output reg               feed_tick_ready,
    input  wire              feed_tick_last,
    input  wire [511:0]      feed_a_payload,     // row r at [r*16 +: 16]
    input  wire [255:0]      feed_a_scale,       // row r at [r*8 +: 8]
    input  wire [511:0]      feed_bl_payload,    // column c at [c*16 +: 16]
    input  wire [255:0]      feed_bl_scale,
    input  wire [511:0]      feed_br_payload,
    input  wire [255:0]      feed_br_scale,
    output wire              quiescent
);
    localparam integer A_ITEM_WIDTH = 31;
    reg        exec_r;
    reg [3:0]  pair_r;
    reg [5:0]  group_r;
    reg [31:0] rows_r;
    reg [5:0]  nl_r, nr_r;
    reg        accum_r, emit_r;
    reg [1:0]  op_r;
    wire array_advance = rc_clk_en && array_ready;
    wire tick_fire = feed_tick_valid && feed_tick_ready;
    wire cfg_fire = cfg_valid && cfg_ready;
    wire block_last = pair_r == 4'd15;
    wire release_exec = tick_fire && feed_tick_last;

    always @* begin
        cfg_ready = |cfg_rows && (cfg_nl != 0 || cfg_nr != 0) && cfg_k_group_count != 0 && array_advance &&
                    (!exec_r || release_exec);
        feed_tick_ready = exec_r && array_advance;
    end

    // A skew: {op, emit, numeric valid, first block, block last, final, scale, K pair}.
    wire [31:0] a_dvalid, a_busy;
    wire [32*A_ITEM_WIDTH-1:0] a_ditem;
    for (genvar r = 0; r < 32; r = r + 1) begin : g_a_input
        wire row_ok = rows_r[r];
        wire [A_ITEM_WIDTH-1:0] item = {op_r, emit_r, row_ok, group_r == 0 && !accum_r, block_last, feed_tick_last,
                                        row_ok ? feed_a_scale[r*8 +: 8] : 8'd127,
                                        row_ok ? feed_a_payload[r*16 +: 16] : 16'd0};
        rc_mxfp8_delay_line #(
            .WIDTH (A_ITEM_WIDTH),
            .DEPTH (r)
        ) u_a_skew (
            .clk       (clk),
            .rstn      (rstn),
            .advance   (array_advance),
            .in_valid  (tick_fire),
            .in_data   (item),
            .out_valid (a_dvalid[r]),
            .out_data  (a_ditem[r*A_ITEM_WIDTH +: A_ITEM_WIDTH]),
            .busy      (a_busy[r])
        );
    end

    wire left_busy, right_busy;
    g3_array_half u_left (
        .clk            (clk),
        .rstn           (rstn),
        .array_advance  (array_advance),
        .feed_tick_fire (tick_fire),
        .a_valid        (a_dvalid),
        .a_item         (a_ditem),
        .b_payload      (feed_bl_payload),
        .b_scale        (feed_bl_scale),
        .b_valid_cols   (nl_r),
        .row_valid      (left_valid),
        .row_data       (left_data),
        .busy           (left_busy)
    );
    g3_array_half u_right (
        .clk            (clk),
        .rstn           (rstn),
        .array_advance  (array_advance),
        .feed_tick_fire (tick_fire),
        .a_valid        (a_dvalid),
        .a_item         (a_ditem),
        .b_payload      (feed_br_payload),
        .b_scale        (feed_br_scale),
        .b_valid_cols   (nr_r),
        .row_valid      (right_valid),
        .row_data       (right_data),
        .busy           (right_busy)
    );
    assign quiescent = !feed_window_open && !exec_r && !(|a_busy) && !left_busy && !right_busy;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            exec_r <= 1'b0; pair_r <= 0; group_r <= 0; rows_r <= 0; nl_r <= 0; nr_r <= 0;
            accum_r <= 1'b0; emit_r <= 1'b1; op_r <= 0;
        end else begin
            if (tick_fire) begin
                if (feed_tick_last) begin exec_r <= 1'b0; pair_r <= 0; group_r <= 0; end
                else if (block_last) begin pair_r <= 0; group_r <= group_r + 1'b1; end
                else pair_r <= pair_r + 1'b1;
            end
            if (cfg_fire) begin
                exec_r <= 1'b1; rows_r <= cfg_rows; nl_r <= cfg_nl; nr_r <= cfg_nr;
                accum_r <= cfg_accumulate; emit_r <= cfg_emit_result; op_r <= cfg_operation;
            end
        end
    end
endmodule

`default_nettype wire
