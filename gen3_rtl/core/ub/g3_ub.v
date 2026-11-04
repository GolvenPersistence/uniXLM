`timescale 1ns/1ps
`default_nettype none

// Gen3 Unified Buffer (docs/gen3/README.md s5.1, gen3_rtl/README.md s4): a plain memory with
// sequential streams. The Gen2 UB (rtl/gen2/g2_ub.v) without the CTU and without multicast:
// every read context delivers to its own port.
//
//   seg_* (per port) --> segment FIFOs: NPORT ports x 3 read contexts x SEGQ entries
//                           | walk: start + off[i] + j*s1 (outer-major), wrapped once at ext
//                           v
//   in_* (per port) --> ingress FIFO (4) --+    +-------------------------------------+
//                                          +--> | arbiter (one cycle): writes in port   |
//                                               | order, then reads round robin over    |
//                                               | contexts; <= 2 accesses per bank (one |
//                                               | per bank port), no two accesses to a  |
//                                               | word that is written, one egress beat |
//                                               | per port, credit > 0                  |
//                                               +-----+-------------------+-----------+
//                                                     | bank port k.0/k.1 (granted cycle)
//                                                     v
//                                   4 banks x 1024 x 256 bit (g3_ub_bank), bank = addr[4:3]
//                                                     | registered read data
//                                                     v
//                                   eg_* lane per port (stream = context), one cycle later
//   mark_retired (per port) <-- a marked write reached its bank
//
// Ports: 0 array A, 1 array B (contexts 0 left, 1 right), 2 array C (2 C tiles, 0 old V^T),
// 3 VRU, 4 NI. Watermark gating is done by the sequencers before a segment is pushed.
module g3_ub #(
    parameter integer NPORT = 5,
    parameter integer SEGQ = 4
) (
    input  wire                  clk,
    input  wire                  rstn,
    input  wire [NPORT-1:0]      seg_valid,
    output wire [NPORT-1:0]      seg_ready,
    input  wire [NPORT*128-1:0]  seg_data,
    output reg  [NPORT*3-1:0]    seg_closed,
    output reg  [NPORT-1:0]      eg_valid,
    output reg  [NPORT*2-1:0]    eg_stream,
    output reg  [NPORT*2-1:0]    eg_inner_idx,
    output reg  [NPORT-1:0]      eg_inner_last,
    output reg  [NPORT-1:0]      eg_seg_last,
    output reg  [NPORT*8-1:0]    eg_user,
    output wire [NPORT*256-1:0]  eg_data,
    input  wire [NPORT*3-1:0]    credit_return,
    input  wire [NPORT-1:0]      in_valid,
    output wire [NPORT-1:0]      in_ready,
    input  wire [NPORT*12-1:0]   in_addr,
    input  wire [NPORT*256-1:0]  in_data,
    input  wire [NPORT*32-1:0]   in_strb,
    input  wire [NPORT-1:0]      in_mark,
    output reg  [NPORT-1:0]      mark_retired,
    output reg  [31:0]           stat_read_beats,
    output reg  [31:0]           stat_write_beats,
    output reg  [31:0]           stat_bank_conflicts,
    output wire                  busy
);
    `include "g3_defs.vh"
    localparam integer NCTX = NPORT * 3;
    localparam integer CREDITS = 8;
    function automatic [11:0] f_base (input [127:0] s); f_base  = s[`S_BASE];  endfunction
    function automatic [12:0] f_ext  (input [127:0] s); f_ext   = s[`S_EXT];   endfunction
    function automatic [11:0] f_start(input [127:0] s); f_start = s[`S_START]; endfunction
    function automatic [1:0]  f_n0   (input [127:0] s); f_n0    = s[`S_N0];    endfunction
    function automatic [11:0] f_n1   (input [127:0] s); f_n1    = s[`S_N1];    endfunction
    function automatic [11:0] f_s1   (input [127:0] s); f_s1    = s[`S_S1];    endfunction
    function automatic [7:0]  f_user (input [127:0] s); f_user  = s[`S_USER];  endfunction
    function automatic        f_close(input [127:0] s); f_close = s[`S_CLOSE]; endfunction
    function automatic [11:0] f_off  (input [127:0] s, input [1:0] i);
        f_off = i == 2'd0 ? 12'd0 : i == 2'd1 ? s[`S_O1] : i == 2'd2 ? s[`S_O2] : s[`S_O3];
    endfunction
    function automatic [11:0] seg_off(input [127:0] h, input [11:0] ro, input [11:0] io);
        reg [13:0] sum;
        begin
            sum = {2'd0, f_start(h)} + {2'd0, ro} + {2'd0, io};
            seg_off = sum >= {1'b0, f_ext(h)} ? 12'(sum - {1'b0, f_ext(h)}) : sum[11:0];
        end
    endfunction

    // ------------------------------------------------------------------
    // Read contexts (segment FIFO + walker) and ingress write FIFOs.
    // ------------------------------------------------------------------
    reg [127:0] seg_fifo_mem [0:NCTX-1][0:SEGQ-1];
    reg [1:0]   seg_rd_ptr_r [0:NCTX-1], seg_wr_ptr_r [0:NCTX-1];
    reg [2:0]   seg_level_r [0:NCTX-1];
    reg [1:0]   walk_inner_r [0:NCTX-1];
    reg [11:0]  walk_outer_r [0:NCTX-1];
    reg [11:0]  walk_outer_off_r [0:NCTX-1];
    reg [3:0]   credit_r [0:NCTX-1];
    reg [300:0] wr_fifo_mem [0:NPORT-1][0:3];   // {mark, strb, addr, data}
    reg [1:0]   wr_rd_ptr_r [0:NPORT-1], wr_wr_ptr_r [0:NPORT-1];
    reg [2:0]   wr_level_r [0:NPORT-1];
    wire [127:0] seg_head [0:NCTX-1];
    wire [300:0] wr_head [0:NPORT-1];
    wire [NCTX-1:0] seg_push;
    for (genvar c = 0; c < NCTX; c = c + 1) begin : g_ctx
        assign seg_head[c] = seg_fifo_mem[c][seg_rd_ptr_r[c]];
        assign seg_push[c] = seg_valid[c/3] && seg_ready[c/3] && seg_data[(c/3)*128 +: 2] == c % 3;
    end
    for (genvar p = 0; p < NPORT; p = p + 1) begin : g_in
        assign in_ready[p] = wr_level_r[p] != 3'd4;
        assign seg_ready[p] = seg_level_r[p*3 + seg_data[p*128 +: 2]] != SEGQ;
        assign wr_head[p] = wr_fifo_mem[p][wr_rd_ptr_r[p]];
    end

    // ------------------------------------------------------------------
    // Arbitration (combinational): writes in port order, then reads round robin over contexts.
    // Each granted access takes the next free port of its bank (slot 0, then 1).
    // ------------------------------------------------------------------
    reg [NCTX-1:0]  rd_grant, rd_slot;
    reg [NPORT-1:0] wr_grant;
    reg [3:0]       rr_ptr_r;
    reg             conflict;
    reg [11:0]      addr_c [0:NCTX-1];
    reg [7:0]       b_en, b_we;                 // bank port j = 2*bank + slot
    reg [79:0]      b_addr;
    reg [2047:0]    b_wdata;
    reg [255:0]     b_strb;
    always @* begin : arbitrate
        integer k, p, c, pass, j;
        reg [7:0]   n;                          // accesses granted per bank (2 bits each)
        reg [NPORT-1:0] lane_use;
        reg [9:0]   row;
        reg [300:0] e;
        reg         hit;
        n = 8'd0;
        lane_use = 0; rd_grant = 0; rd_slot = 0; wr_grant = 0; conflict = 1'b0;
        b_en = 0; b_we = 0; b_addr = 0; b_wdata = 0; b_strb = 0;
        for (p = 0; p < NPORT; p = p + 1) begin
            e = wr_head[p];
            k = e[260:259];
            row = {e[267:261], e[258:256]};
            // A second write to a word already written this cycle waits.
            hit = n[k*2 +: 2] == 2'd1 && b_addr[k*20 +: 10] == row;
            if (wr_level_r[p] != 3'd0 && n[k*2 +: 2] != 2'd2 && !hit) begin
                j = 2*k + n[k*2 +: 2];
                wr_grant[p] = 1'b1;
                b_en[j] = 1'b1; b_we[j] = 1'b1; b_addr[j*10 +: 10] = row;
                b_wdata[j*256 +: 256] = e[255:0]; b_strb[j*32 +: 32] = e[299:268];
                n[k*2 +: 2] = n[k*2 +: 2] + 1'b1;
            end else if (wr_level_r[p] != 3'd0) conflict = 1'b1;
        end
        for (c = 0; c < NCTX; c = c + 1) begin
            addr_c[c] = f_base(seg_head[c]) +
                        seg_off(seg_head[c], walk_outer_off_r[c], f_off(seg_head[c], walk_inner_r[c]));
        end
        // Round robin from rr_ptr_r: contexts rr_ptr_r..NCTX-1, then 0..rr_ptr_r-1.
        for (pass = 0; pass < 2; pass = pass + 1)
            for (c = 0; c < NCTX; c = c + 1)
                if ((pass == 0) == (c >= rr_ptr_r) && seg_level_r[c] != 3'd0 &&
                    credit_r[c] != 4'd0 && !lane_use[c/3]) begin
                    k = addr_c[c][4:3];
                    row = {addr_c[c][11:5], addr_c[c][2:0]};
                    // A read of a word written this cycle waits (writes take the bank ports first).
                    hit = n[k*2 +: 2] == 2'd1 && b_we[2*k] && b_addr[k*20 +: 10] == row;
                    if (n[k*2 +: 2] != 2'd2 && !hit) begin
                        j = 2*k + n[k*2 +: 2];
                        rd_grant[c] = 1'b1; rd_slot[c] = n[k*2];
                        b_en[j] = 1'b1; b_addr[j*10 +: 10] = row;
                        n[k*2 +: 2] = n[k*2 +: 2] + 1'b1;
                        lane_use[c/3] = 1'b1;
                    end else conflict = 1'b1;
                end
    end

    // ------------------------------------------------------------------
    // Banks.
    // ------------------------------------------------------------------
    wire [511:0] bank_q [0:3];
    for (genvar k = 0; k < 4; k = k + 1) begin : g_bank
        g3_ub_bank u_bank (
            .clk   (clk),
            .en    (b_en[2*k +: 2]),
            .we    (b_we[2*k +: 2]),
            .addr  (b_addr[k*20 +: 20]),
            .wdata (b_wdata[k*512 +: 512]),
            .strb  (b_strb[k*64 +: 64]),
            .rdata (bank_q[k])
        );
    end

    // ------------------------------------------------------------------
    // Control state.
    // ------------------------------------------------------------------
    reg [NCTX-1:0] rd_pend_r;
    reg [1:0]      rd_bank_r [0:NCTX-1];
    reg            rd_port_r [0:NCTX-1];
    reg [7:0]      rd_user_r [0:NCTX-1];
    reg [1:0]      rd_idx_r [0:NCTX-1];
    reg            rd_ilast_r [0:NCTX-1], rd_slast_r [0:NCTX-1];
    reg [255:0]    eg_data_r [0:NPORT-1];
    for (genvar q = 0; q < NPORT; q = q + 1) begin : g_eg
        assign eg_data[q*256 +: 256] = eg_data_r[q];
    end
    wire [NCTX-1:0] last_inner, last_outer;
    for (genvar c = 0; c < NCTX; c = c + 1) begin : g_last
        assign last_inner[c] = walk_inner_r[c] == f_n0(seg_head[c]);
        assign last_outer[c] = walk_outer_r[c] == f_n1(seg_head[c][127:0]);
    end

    always @(posedge clk or negedge rstn) begin : control
        integer c, p;
        if (!rstn) begin
            for (c = 0; c < NCTX; c = c + 1) begin
                seg_rd_ptr_r[c] <= 0; seg_wr_ptr_r[c] <= 0; seg_level_r[c] <= 0; credit_r[c] <= CREDITS;
                walk_inner_r[c] <= 0; walk_outer_r[c] <= 0; walk_outer_off_r[c] <= 0;
            end
            for (p = 0; p < NPORT; p = p + 1) begin
                wr_rd_ptr_r[p] <= 0; wr_wr_ptr_r[p] <= 0; wr_level_r[p] <= 0;
            end
            rr_ptr_r <= 0; rd_pend_r <= 0; eg_valid <= 0; seg_closed <= 0; mark_retired <= 0;
            stat_read_beats <= 0; stat_write_beats <= 0; stat_bank_conflicts <= 0;
        end else begin
            for (c = 0; c < NCTX; c = c + 1) begin
                credit_r[c] <= credit_r[c] + credit_return[c] - rd_grant[c];
                if (seg_push[c]) seg_wr_ptr_r[c] <= seg_wr_ptr_r[c] + 1'b1;
                seg_closed[c] <= rd_grant[c] && last_inner[c] && last_outer[c] && f_close(seg_head[c]);
                if (rd_grant[c]) begin
                    if (!last_inner[c]) walk_inner_r[c] <= walk_inner_r[c] + 1'b1;
                    else if (!last_outer[c]) begin
                        walk_inner_r[c] <= 0;
                        walk_outer_r[c] <= walk_outer_r[c] + 1'b1;
                        walk_outer_off_r[c] <= walk_outer_off_r[c] + f_s1(seg_head[c]);
                    end else begin
                        seg_rd_ptr_r[c] <= seg_rd_ptr_r[c] + 1'b1;
                        walk_inner_r[c] <= 0; walk_outer_r[c] <= 0; walk_outer_off_r[c] <= 0;
                    end
                end
                seg_level_r[c] <= seg_level_r[c] + seg_push[c] - (rd_grant[c] && last_inner[c] && last_outer[c]);
            end
            rd_pend_r <= rd_grant;
            stat_read_beats <= stat_read_beats + $countones(rd_pend_r);
            stat_write_beats <= stat_write_beats + $countones(wr_grant);
            if (conflict) stat_bank_conflicts <= stat_bank_conflicts + 1;
            for (p = 0; p < NPORT; p = p + 1) begin
                eg_valid[p] <= |(rd_pend_r[p*3 +: 3]);
                mark_retired[p] <= wr_grant[p] && wr_head[p][300];
                if (in_valid[p] && in_ready[p]) wr_wr_ptr_r[p] <= wr_wr_ptr_r[p] + 1'b1;
                if (wr_grant[p]) wr_rd_ptr_r[p] <= wr_rd_ptr_r[p] + 1'b1;
                wr_level_r[p] <= wr_level_r[p] + (in_valid[p] && in_ready[p]) - wr_grant[p];
            end
            rr_ptr_r <= rr_ptr_r == NCTX - 1 ? 4'd0 : rr_ptr_r + 1'b1;
        end
    end

    // Storage and data registers (no reset; every write is qualified by reset state above).
    always @(posedge clk) begin : storage
        integer c, p;
        for (p = 0; p < NPORT; p = p + 1) begin
            if (seg_valid[p] && seg_ready[p])
                seg_fifo_mem[p*3 + seg_data[p*128 +: 2]][seg_wr_ptr_r[p*3 + seg_data[p*128 +: 2]]] <= seg_data[p*128 +: 128];
            if (in_valid[p] && in_ready[p])
                wr_fifo_mem[p][wr_wr_ptr_r[p]] <= {in_mark[p], in_strb[p*32 +: 32], in_addr[p*12 +: 12],
                                                    in_data[p*256 +: 256]};
        end
        for (c = 0; c < NCTX; c = c + 1) begin
            if (rd_grant[c]) begin
                rd_bank_r[c] <= addr_c[c][4:3];
                rd_port_r[c] <= rd_slot[c];
                rd_user_r[c] <= f_user(seg_head[c]);
                rd_idx_r[c] <= walk_inner_r[c];
                rd_ilast_r[c] <= last_inner[c];
                rd_slast_r[c] <= last_inner[c] && last_outer[c];
            end
            if (rd_pend_r[c]) begin
                eg_stream[(c/3)*2 +: 2] <= 2'(c % 3);
                eg_inner_idx[(c/3)*2 +: 2] <= rd_idx_r[c];
                eg_inner_last[c/3] <= rd_ilast_r[c];
                eg_seg_last[c/3] <= rd_slast_r[c];
                eg_user[(c/3)*8 +: 8] <= rd_user_r[c];
                eg_data_r[c/3] <= bank_q[rd_bank_r[c]][rd_port_r[c]*256 +: 256];
            end
        end
    end

    reg any_q;
    always @* begin : busy_any
        integer c, p;
        any_q = |rd_pend_r;
        for (c = 0; c < NCTX; c = c + 1) if (seg_level_r[c] != 3'd0) any_q = 1'b1;
        for (p = 0; p < NPORT; p = p + 1) if (wr_level_r[p] != 3'd0) any_q = 1'b1;
    end
    assign busy = any_q;
endmodule

`default_nettype wire
