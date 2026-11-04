`timescale 1ns/1ps
`default_nettype none

// Gen3 pull engine: the Gen2 one (rtl/gen2/g2_pull.v) unchanged; in Gen3 its ring is a weight buffer
// next to an array half (gen3_rtl/README.md s5). Streams a P32 operand from
// external memory into a ring owned by its single reader.
//
// Block diagram:
//
//   prog {src, tile_words_r, stride, ntiles, rounds, ring_base_r, ring_ext_r}
//        |
//        v
//   address generator (tile, round, offset) --> burst sizer (<= 16 beats, no 4 KiB crossing)
//                                                     |
//   ring space = ring_ext_r - (issued - released) ----> can_issue --> ar_* (addr, len)
//   r_* beats --> wb_* at ring_base_r + wr_pos_r (wr_pos_r wraps at ring_ext_r)
//   wb_retired --> pulled (words landed, to the walkers)      released <-- walkers (ring freed)
//
// Control:
//
//   idle --start--> run: one AR at a time while issued < total and the ring has space
//        --every word retired--> idle
module g3_pull (
    input  wire          clk,
    input  wire          rstn,
    input  wire          start,
    input  wire [159:0]  prog,
    input  wire [19:0]   released,
    output reg  [19:0]   pulled,
    // AXI read (this channel).
    output reg           ar_valid,
    input  wire          ar_ready,
    output reg  [47:0]   ar_addr,
    output reg  [7:0]    ar_len,
    input  wire          r_valid,
    output wire          r_ready,
    input  wire [255:0]  r_data,
    // UB write beats.
    output wire          wb_valid,
    input  wire          wb_ready,
    output wire [11:0]   wb_addr,
    output wire [255:0]  wb_data,
    input  wire          wb_retired
);
    reg [47:0] src_r;
    reg [13:0] tile_words_r;
    reg [15:0] stride_r;
    reg [7:0]  ntiles_r;
    reg [11:0] ring_base_r;
    reg [12:0] ring_ext_r;
    reg        run_r;
    reg [7:0]  tile_idx_r;
    reg [13:0] tile_off_r;              // words issued within the current tile
    reg [19:0] issued_r, total_r;
    reg [11:0] wr_pos_r;                // physical ring position of the next R beat

    wire [47:0] tile_addr = src_r + (tile_idx_r * stride_r) * 32;
    wire [47:0] cur_addr = tile_addr + tile_off_r * 32;
    wire [13:0] tile_left = tile_words_r - tile_off_r;
    wire [7:0]  to_4k = (13'h1000 - cur_addr[11:0]) >> 5;
    wire [7:0]  bl0 = tile_left > 16 ? 8'd16 : tile_left[7:0];
    wire [7:0]  bl1 = (to_4k != 0 && to_4k < bl0) ? to_4k : bl0;
    wire [19:0] space = ring_ext_r - (issued_r - released);
    wire        can_issue = run_r && !ar_valid && issued_r < total_r && space >= bl1;

    // R beats go straight to the UB (the NI arbitrates the N port).
    assign wb_valid = r_valid && run_r;
    assign r_ready = wb_ready && run_r;
    assign wb_addr = ring_base_r + wr_pos_r;
    assign wb_data = r_data;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            run_r <= 0; ar_valid <= 0; pulled <= 0; issued_r <= 0; total_r <= 0;
            tile_idx_r <= 0; tile_off_r <= 0; wr_pos_r <= 0;
        end else begin
            if (start) begin
                src_r <= prog[118:71]; tile_words_r <= prog[70:57]; stride_r <= prog[56:41];
                ntiles_r <= prog[40:33]; ring_base_r <= prog[24:13];
                ring_ext_r <= prog[12:0];
                total_r <= prog[40:33] * prog[32:25] * prog[70:57];
                run_r <= 1'b1; issued_r <= 0; pulled <= 0; wr_pos_r <= 0;
                tile_idx_r <= 0; tile_off_r <= 0;
            end else if (run_r) begin
                if (ar_valid && ar_ready) ar_valid <= 1'b0;
                if (can_issue) begin
                    ar_valid <= 1'b1; ar_addr <= cur_addr; ar_len <= bl1 - 1'b1;
                    issued_r <= issued_r + bl1;
                    if (tile_off_r + bl1 == tile_words_r) begin
                        tile_off_r <= 0;
                        tile_idx_r <= tile_idx_r + 1 == ntiles_r ? 8'd0 : tile_idx_r + 1'b1;
                    end else tile_off_r <= tile_off_r + bl1;
                end
                if (wb_valid && wb_ready) wr_pos_r <= wr_pos_r + 1 == ring_ext_r ? 12'd0 : wr_pos_r + 1'b1;
                if (wb_retired) pulled <= pulled + 1'b1;
                if (pulled + wb_retired == total_r && total_r != 0 && issued_r == total_r) run_r <= 1'b0;
            end
        end
    end
endmodule

`default_nettype wire
