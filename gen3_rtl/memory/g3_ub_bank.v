`timescale 1ns/1ps
`default_nettype none

// One UB bank: 1024 x 256 bit, two synchronous ports, each a read or a byte-masked write per cycle.
// Read data is registered (valid the cycle after the access, held until the port's next read).
// The UB arbiter never puts two accesses to the same word in one cycle.
// G3_USE_TSMC28_MC2 maps the bank to four TSMC28 true dual-port SRAM macros (1024 x 64, as the
// Gen2 physical UB); otherwise the storage is behavioural.
module g3_ub_bank (
    input  wire         clk,
    input  wire [1:0]   en,
    input  wire [1:0]   we,
    input  wire [19:0]  addr,           // {port 1, port 0} x 10
    input  wire [511:0] wdata,          // {port 1, port 0} x 256
    input  wire [63:0]  strb,           // {port 1, port 0} x 32 byte enables
    output wire [511:0] rdata           // {port 1, port 0} x 256
);
`ifdef G3_USE_TSMC28_MC2
    for (genvar l = 0; l < 4; l = l + 1) begin : g_slice
        wire [63:0] mask_a, mask_b;
        for (genvar b = 0; b < 8; b = b + 1) begin : g_mask
            assign mask_a[b*8 +: 8] = {8{~strb[l*8 + b]}};
            assign mask_b[b*8 +: 8] = {8{~strb[32 + l*8 + b]}};
        end
        TSDN28HPCPA1024X64M4MWBASO u_macro (
            .CLKA(clk), .CEBA(~en[0]), .WEBA(~we[0]), .AA(addr[9:0]), .DA(wdata[l*64 +: 64]), .BWEBA(mask_a),
            .QA(rdata[l*64 +: 64]),
            .CLKB(clk), .CEBB(~en[1]), .WEBB(~we[1]), .AB(addr[19:10]), .DB(wdata[256 + l*64 +: 64]), .BWEBB(mask_b),
            .QB(rdata[256 + l*64 +: 64]),
            .SLP(1'b0), .SD(1'b0), .WTSEL(2'b01), .RTSEL(2'b01), .VG(1'b1), .VS(1'b1), .AWT(1'b0), .BIST(1'b0),
            .CLKM(1'b0), .AMA(10'd0), .DMA(64'd0), .BWEBMA({64{1'b1}}), .WEBMA(1'b1), .CEBMA(1'b1),
            .AMB(10'd0), .DMB(64'd0), .BWEBMB({64{1'b1}}), .WEBMB(1'b1), .CEBMB(1'b1));
    end
`else
    reg [255:0] mem [0:1023];
    reg [511:0] q_r;
    always @(posedge clk) begin : ports
        integer p, b;
        for (p = 0; p < 2; p = p + 1)
            if (en[p]) begin
                if (we[p]) begin
                    for (b = 0; b < 32; b = b + 1)
                        if (strb[p*32 + b]) mem[addr[p*10 +: 10]][b*8 +: 8] <= wdata[p*256 + b*8 +: 8];
                end else q_r[p*256 +: 256] <= mem[addr[p*10 +: 10]];
            end
    end
    assign rdata = q_r;
`endif
    // synthesis translate_off
    always @(posedge clk) if (&en && |we && addr[9:0] == addr[19:10]) $fatal(1, "G3_UB_BANK_COLLISION %m");
    // synthesis translate_on
endmodule

`default_nettype wire
