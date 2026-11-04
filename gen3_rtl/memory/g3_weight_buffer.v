`timescale 1ns/1ps
`default_nettype none

// Gen3 weight buffers (docs/gen3/README.md s5.2, gen3_rtl/README.md s5): one cell ring of 132 words
// (four 33-word P32 cells) next to each array half. STREAM weights go NI pull engine -> ring ->
// array B edge without touching the UB.
//
//   pull engine h --> write port (wb_addr = ring position, always ready, retired next cycle)
//   g3_feed cell reader h --> read port (one-cycle registered read)
//
// G3_USE_TSMC28_MC2 maps each ring to four TSMC28 2-port register files (136 x 64, the smallest legal
// depth >= 132; one write and one read port); otherwise the storage is behavioural (2 x 132 x 256 bit).
module g3_weight_buffer #(
    parameter integer WORDS = 132
) (
    input  wire          clk,
    input  wire          rstn,
    input  wire [1:0]    wr_valid,
    input  wire [23:0]   wr_addr,               // {right, left} x 12
    input  wire [511:0]  wr_data,
    output reg  [1:0]    wr_retired,
    input  wire [1:0]    rd_en,
    input  wire [15:0]   rd_addr,               // {right, left} x 8
    output wire [511:0]  rd_data
);
    for (genvar h = 0; h < 2; h = h + 1) begin : g_ring
`ifdef G3_USE_TSMC28_MC2
        for (genvar l = 0; l < 4; l = l + 1) begin : g_slice
            TS6N28HPCPHVTA136X64M2FWBSO u_macro (
                .CLKW(clk), .AA(wr_addr[h*12 +: 8]), .D(wr_data[h*256 + l*64 +: 64]), .BWEB(64'd0),
                .WEB(~wr_valid[h]), .CLKR(clk), .AB(rd_addr[h*8 +: 8]), .REB(~rd_en[h]),
                .Q(rd_data[h*256 + l*64 +: 64]), .AMA(8'd0), .DM(64'd0), .BWEBM({64{1'b1}}), .WEBM(1'b1),
                .AMB(8'd0), .REBM(1'b1), .BIST(1'b0), .SLP(1'b0), .SD(1'b0));
        end
`else
        reg [255:0] mem [0:WORDS-1];
        reg [255:0] q_r;
        always @(posedge clk) begin
            if (wr_valid[h]) mem[wr_addr[h*12 +: 8]] <= wr_data[h*256 +: 256];
            if (rd_en[h]) q_r <= mem[rd_addr[h*8 +: 8]];
        end
        assign rd_data[h*256 +: 256] = q_r;
`endif
    end
    always @(posedge clk or negedge rstn)
        if (!rstn) wr_retired <= 2'b00;
        else wr_retired <= wr_valid;

    // synthesis translate_off
    always @(posedge clk) if (rstn) begin : collision
        integer h;
        for (h = 0; h < 2; h = h + 1)
            if (wr_valid[h] && rd_en[h] && wr_addr[h*12 +: 12] == {4'd0, rd_addr[h*8 +: 8]})
                $fatal(1, "G3_WEIGHT_BUFFER_COLLISION half=%0d", h);
    end
    // synthesis translate_on
endmodule

`default_nettype wire
