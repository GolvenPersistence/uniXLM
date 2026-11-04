`timescale 1ns/1ps
`default_nettype none

// One half of the Gen3 array (gen3_rtl/README.md s3.1): 32 rows x 16 GEMUs (two output columns
// each), the Gen1/Gen2 rc_mxfp8_gemu grid. Physical column 0 is next to the vector column: A enters
// there (already skewed by row) and moves away from the middle; B enters at the top, column c
// delayed c/2 ticks, and moves down. PE (r, c) sees both operands at t + r + c/2.
//
//   B (32 columns, skewed) --> +----------------------------------------+
//                              | row r: PE c=0 ... PE c=15 (A moves out) |
//   A (32 rows, skewed) -----> |  dot2 x 2 columns, INT42 K32 block      |
//                              +-------------------+--------------------+
//                                                  | per row: OR of the 16 PE update requests
//                                                  v (one column pair per tick, column c at r + c)
//                                       row_valid[r] / row_data[r] (113 bits, Gen2 export format)
//
// Control: all registers move on array_advance only (synchronous clock enable of the parent).
module g3_array_half #(
    parameter integer ROWS = 32,
    parameter integer COLS = 16                   // GEMUs per row (2 output columns each)
) (
    input  wire                  clk,
    input  wire                  rstn,
    input  wire                  array_advance,
    input  wire                  feed_tick_fire,
    // Skewed A items (shared by both halves).
    input  wire [ROWS-1:0]       a_valid,
    input  wire [ROWS*31-1:0]    a_item,
    // B of this half: 2*COLS columns x K pair, scales, valid columns.
    input  wire [2*COLS*16-1:0]  b_payload,
    input  wire [2*COLS*8-1:0]   b_scale,
    input  wire [5:0]            b_valid_cols,
    // Row export.
    output wire [ROWS-1:0]       row_valid,
    output wire [ROWS*113-1:0]   row_data,
    output wire                  busy
);
    localparam integer NB = 2 * COLS;             // logical B columns
    localparam integer NPE = ROWS * COLS;
    localparam integer A_ITEM_WIDTH = 31;
    localparam integer B_ITEM_WIDTH = 25;
    localparam integer A_SCALE_LSB = 16;
    localparam integer A_SCALE_MSB = 23;
    localparam integer A_FINAL = 24;
    localparam integer A_BLOCK_LAST = 25;
    localparam integer A_FIRST_BLOCK = 26;
    localparam integer A_NUMERIC_VALID = 27;
    localparam integer A_EMIT_RESULT = 28;
    localparam integer A_OP_LSB = 29;
    localparam integer A_OP_MSB = 30;
    localparam integer B_NUMERIC_VALID = 24;

    // ------------------------------------------------------------------
    // B skew lines (logical column c delayed by its physical column c/2).
    // ------------------------------------------------------------------
    wire [B_ITEM_WIDTH-1:0] b_delayed_item [0:NB-1];
    wire [NB-1:0] b_delayed_valid, b_delay_busy;
    for (genvar c = 0; c < NB; c = c + 1) begin : g_b_input
        localparam integer DELAY = c / 2;
        wire col_valid = c < b_valid_cols;
        wire [B_ITEM_WIDTH-1:0] item = {col_valid, col_valid ? b_scale[c*8 +: 8] : 8'd127,
                                        col_valid ? b_payload[c*16 +: 16] : 16'd0};
        rc_mxfp8_delay_line #(
            .WIDTH (B_ITEM_WIDTH),
            .DEPTH (DELAY)
        ) u_b_skew (
            .clk       (clk),
            .rstn      (rstn),
            .advance   (array_advance),
            .in_valid  (feed_tick_fire),
            .in_data   (item),
            .out_valid (b_delayed_valid[c]),
            .out_data  (b_delayed_item[c]),
            .busy      (b_delay_busy[c])
        );
    end

    // ------------------------------------------------------------------
    // Grid: A moves along the row (away from the middle), B moves down.
    // ------------------------------------------------------------------
    reg [A_ITEM_WIDTH-1:0] a_grid_item_r [0:NPE-1];
    reg                    a_grid_valid_r [0:NPE-1];
    reg [B_ITEM_WIDTH-1:0] b_grid_item_r [0:2*NPE-1];
    reg                    b_grid_valid_r [0:2*NPE-1];

    always @(posedge clk or negedge rstn) begin : update_grid_valid
        integer r, c, l, pe;
        if (!rstn) begin
            for (pe = 0; pe < NPE; pe = pe + 1) a_grid_valid_r[pe] <= 1'b0;
            for (pe = 0; pe < 2*NPE; pe = pe + 1) b_grid_valid_r[pe] <= 1'b0;
        end else if (array_advance) begin
            for (r = 0; r < ROWS; r = r + 1)
                for (c = 0; c < COLS; c = c + 1) begin
                    pe = r*COLS + c;
                    a_grid_valid_r[pe] <= c == 0 ? a_valid[r] : a_grid_valid_r[pe-1];
                    for (l = 0; l < 2; l = l + 1)
                        b_grid_valid_r[pe*2+l] <= r == 0 ? b_delayed_valid[c*2+l] : b_grid_valid_r[(pe-COLS)*2+l];
                end
        end
    end
    always @(posedge clk) begin : update_grid_data
        integer r, c, l, pe;
        if (array_advance)
            for (r = 0; r < ROWS; r = r + 1)
                for (c = 0; c < COLS; c = c + 1) begin
                    pe = r*COLS + c;
                    a_grid_item_r[pe] <= c == 0 ? a_item[r*A_ITEM_WIDTH +: A_ITEM_WIDTH] : a_grid_item_r[pe-1];
                    for (l = 0; l < 2; l = l + 1)
                        b_grid_item_r[pe*2+l] <= r == 0 ? b_delayed_item[c*2+l] : b_grid_item_r[(pe-COLS)*2+l];
                end
    end

    // ------------------------------------------------------------------
    // GEMUs. The PE decodes the operand one tick late: A from the next PE of the row (or an edge
    // register), B from the PE below (or an edge register), as in the Gen1/Gen2 grid.
    // ------------------------------------------------------------------
    wire [2*NPE-1:0]  upd_valid, upd_invalid, pipe_busy;
    wire [NPE*84-1:0] upd_integer;
    wire [NPE*18-1:0] upd_scales;
    wire [NPE-1:0]    upd_first;
    wire [3:0]        upd_meta [0:NPE-1];
    for (genvar pe = 0; pe < NPE; pe = pe + 1) begin : g_pe
        localparam integer C = pe % COLS;
        localparam integer R = pe / COLS;
        wire [15:0] fwd_a;
        wire [31:0] fwd_b;
        if (C == COLS-1) begin : g_a_edge
            reg [15:0] a_r;
            always @(posedge clk) if (array_advance) a_r <= a_grid_item_r[pe][15:0];
            assign fwd_a = a_r;
        end else begin : g_a_fwd
            assign fwd_a = a_grid_item_r[pe+1][15:0];
        end
        if (R < ROWS-1) begin : g_b_fwd
            assign fwd_b = {b_grid_item_r[(pe+COLS)*2+1][15:0], b_grid_item_r[(pe+COLS)*2][15:0]};
        end else begin : g_b_edge
            reg [31:0] b_r;
            always @(posedge clk) if (array_advance) b_r <= {b_grid_item_r[pe*2+1][15:0], b_grid_item_r[pe*2][15:0]};
            assign fwd_b = b_r;
        end
        wire [A_ITEM_WIDTH-1:0] ai = a_grid_item_r[pe];
        rc_mxfp8_gemu u_gemu (
            .clk                  (clk),
            .rstn                 (rstn),
            .advance              (array_advance),
            .step_valid           ({a_grid_valid_r[pe] && b_grid_valid_r[pe*2+1], a_grid_valid_r[pe] && b_grid_valid_r[pe*2]}),
            .step_numeric_valid   ({ai[A_NUMERIC_VALID] && b_grid_item_r[pe*2+1][B_NUMERIC_VALID],
                                    ai[A_NUMERIC_VALID] && b_grid_item_r[pe*2][B_NUMERIC_VALID]}),
            .step_block_last      (ai[A_BLOCK_LAST]),
            .step_first_block     (ai[A_FIRST_BLOCK]),
            .step_emit_result     (ai[A_EMIT_RESULT]),
            .lane0_a_pair         (ai[15:0]),
            .lane0_b_pair         (b_grid_item_r[pe*2][15:0]),
            .lane0_a_scale        (ai[A_SCALE_MSB:A_SCALE_LSB]),
            .lane0_b_scale        (b_grid_item_r[pe*2][23:16]),
            .lane1_b_pair         (b_grid_item_r[pe*2+1][15:0]),
            .lane1_b_scale        (b_grid_item_r[pe*2+1][23:16]),
            .step_final           ({2{ai[A_FINAL]}}),
            .step_operation       (ai[A_OP_MSB:A_OP_LSB]),
            .forwarded_a_pair     (fwd_a),
            .forwarded_b_pair     (fwd_b),
            .update_request_valid (upd_valid[pe*2 +: 2]),
            .update_integer       (upd_integer[pe*84 +: 84]),
            .update_scales        (upd_scales[pe*18 +: 18]),
            .update_invalid       (upd_invalid[pe*2 +: 2]),
            .update_first         (upd_first[pe]),
            .update_control       (upd_meta[pe]),
            .pipeline_busy        (pipe_busy[pe*2 +: 2])
        );
    end

    // ------------------------------------------------------------------
    // Row export: one PE of a row completes its K32 block per tick (columns one tick apart).
    // Format (Gen2): {second=0, bc64=0, op[1:0], final, end, first, keep1=1, keep0=1, invalid[1:0],
    // scale sums 2x9, INT42 x2}; the vector column applies the attention mask.
    // ------------------------------------------------------------------
    for (genvar r = 0; r < ROWS; r = r + 1) begin : g_export
        reg [112:0] payload;
        reg valid;
        integer c;
        always @* begin
            payload = 0; valid = 0;
            for (c = 0; c < COLS; c = c + 1)
                if (|upd_valid[(r*COLS+c)*2 +: 2]) begin
                    valid = 1'b1;
                    payload = payload | {2'b00, upd_meta[r*COLS+c][3:2], upd_meta[r*COLS+c][1],
                                         &upd_meta[r*COLS+c][1:0], upd_first[r*COLS+c], 2'b11,
                                         upd_invalid[(r*COLS+c)*2 +: 2], upd_scales[(r*COLS+c)*18 +: 18],
                                         upd_integer[(r*COLS+c)*84 +: 84]};
                end
        end
        assign row_valid[r] = valid && array_advance;
        assign row_data[r*113 +: 113] = payload;
    end

    reg grid_busy;
    always @* begin : busy_or
        integer pe;
        grid_busy = |b_delay_busy || |pipe_busy;
        for (pe = 0; pe < NPE; pe = pe + 1) grid_busy = grid_busy | a_grid_valid_r[pe] | b_grid_valid_r[pe*2] |
                                                         b_grid_valid_r[pe*2+1];
    end
    assign busy = grid_busy;
endmodule

`default_nettype wire
