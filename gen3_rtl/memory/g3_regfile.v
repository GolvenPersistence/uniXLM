`timescale 1ns/1ps
`default_nettype none

// Synchronous register file with one write port and one read port (Gen3 vector lanes; copied from
// rtl/gen2/g2_regfile.v). Read data is registered: valid the cycle after ren and
// held until the next read. A read and a write of the same word in one cycle is illegal.
// G3_USE_TSMC28_MC2 maps 64x32, 128x32 and 160x64 to TSMC28 MC2 2-port RF macros.
module g3_regfile #(
    parameter integer DEPTH = 64,
    parameter integer WIDTH = 32
) (
    input  wire                     clk,
    input  wire                     wen,
    input  wire [$clog2(DEPTH)-1:0] wa,
    input  wire [WIDTH-1:0]         wd,
    input  wire                     ren,
    input  wire [$clog2(DEPTH)-1:0] ra,
    output wire [WIDTH-1:0]         rd
);
    localparam integer AW = $clog2(DEPTH);
`ifdef G3_USE_TSMC28_MC2
    if (DEPTH == 64 && WIDTH == 32) begin : g_64x32
        TS6N28HPCPHVTA64X32M2FWBSO u_macro (
            .CLKW(clk), .AA(wa), .D(wd), .BWEB({WIDTH{1'b0}}), .WEB(~wen), .CLKR(clk), .AB(ra), .REB(~ren),
            .Q(rd), .AMA({AW{1'b0}}), .DM({WIDTH{1'b0}}), .BWEBM({WIDTH{1'b1}}), .WEBM(1'b1),
            .AMB({AW{1'b0}}), .REBM(1'b1), .BIST(1'b0), .SLP(1'b0), .SD(1'b0));
    end else if (DEPTH == 128 && WIDTH == 32) begin : g_128x32
        TS6N28HPCPHVTA128X32M2FWBSO u_macro (
            .CLKW(clk), .AA(wa), .D(wd), .BWEB({WIDTH{1'b0}}), .WEB(~wen), .CLKR(clk), .AB(ra), .REB(~ren),
            .Q(rd), .AMA({AW{1'b0}}), .DM({WIDTH{1'b0}}), .BWEBM({WIDTH{1'b1}}), .WEBM(1'b1),
            .AMB({AW{1'b0}}), .REBM(1'b1), .BIST(1'b0), .SLP(1'b0), .SD(1'b0));
    end else if (DEPTH == 160 && WIDTH == 64) begin : g_160x64
        TS6N28HPCPHVTA160X64M2FWBSO u_macro (
            .CLKW(clk), .AA(wa), .D(wd), .BWEB({WIDTH{1'b0}}), .WEB(~wen), .CLKR(clk), .AB(ra), .REB(~ren),
            .Q(rd), .AMA({AW{1'b0}}), .DM({WIDTH{1'b0}}), .BWEBM({WIDTH{1'b1}}), .WEBM(1'b1),
            .AMB({AW{1'b0}}), .REBM(1'b1), .BIST(1'b0), .SLP(1'b0), .SD(1'b0));
    end else begin : g_flops
        reg [WIDTH-1:0] mem [0:DEPTH-1];
        reg [WIDTH-1:0] q;
        always @(posedge clk) begin
            if (wen) mem[wa] <= wd;
            if (ren) q <= mem[ra];
        end
        assign rd = q;
    end
`else
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [WIDTH-1:0] q;
    always @(posedge clk) begin
        if (wen) mem[wa] <= wd;
        if (ren) q <= mem[ra];
    end
    assign rd = q;
`endif
    // synthesis translate_off
    always @(posedge clk) if (wen && ren && wa == ra) $fatal(1, "G3_REGFILE_COLLISION %m");
    // synthesis translate_on
endmodule

`default_nettype wire
