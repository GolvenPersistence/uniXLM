`timescale 1ns/1ps
`default_nettype none

// Gen3 ATTN sequencer (gen3_rtl/README.md s6.2): one ATTN parent (ONESHOT) on the single array.
//
// Work items are (KV head, row pass): the q heads of a KV head are stacked as rows (each head padded
// to whole 8-row panels), 32 rows (4 panels) per pass. Per item, over 64-key blocks j:
//   QK(j):  op 1, kg = Dh/32; A = Q panels (scales + payload per Dh group), B = K cells
//           (left: keys 64j..64j+31, right: 64j+32..64j+63)
//   PV(j):  per Dh pass d (64 dims): op 2, kg = 32-key groups of the block; A = P (vector column),
//           B = V^T cells (key group, dims 64d..64d+31 left, +32..+63 right)
//   O:      per Dh pass, an output descriptor (op 3): the vector column writes O rows (P8, like Q)
// Order per item: QK(0) QK(1) PV(0) QK(2) PV(1) ... PV(Jb-1) O; QK(j+1) overlaps softmax(j).
// Causal: row r of the pass is token ptb[r/8] + r%8; key k is visible iff k <= token + delta;
// blocks past the pass's last reachable key are skipped.
// Items alternate two context slots when Dh <= 128: the next item's QK(0) is issued before the
// current item's last PV (it overlaps that block's softmax), and the vector column holds a slot's
// softmax and PV until the O job of the slot's previous item has read l and O. With one slot
// (Dh > 128) an item starts when the previous item has written O (one mark per item).
//
// Descriptor (controller -> here), as Gen2:
//   [7:0] Q idx [15:8] Q gen [27:16] Q base [40:28] Q wm
//   [48:41] K idx [56:49] K gen [104:57] K bases x4 [156:105] K wm x4 [157] K APPEND
//   [165:158] V idx [173:166] V gen [221:174] V bases x4 [273:222] V wm x4 [274] V APPEND
//   [282:275] O idx [290:283] O gen [302:291] O base [315:303] O ext
//   [327:316] tokens [332:328] q_heads [335:333] group [347:336] keys [361:348] delta [365:362] Dh/32
module g3_attn_seq (
    input  wire          clk,
    input  wire          rstn,
    input  wire          start,
    input  wire [511:0]  desc,
    input  wire          ev_valid,
    input  wire [7:0]    ev_idx,
    input  wire [7:0]    ev_gen,
    input  wire [12:0]   ev_wm,
    output reg           cm_valid,
    input  wire          cm_ready,
    output wire [7:0]    cm_idx,
    output wire [7:0]    cm_gen,
    output wire [12:0]   cm_wm,
    output wire          cm_final,
    output reg           sa_valid,
    input  wire          sa_ready,
    output reg  [127:0]  sa,
    output reg           sb_valid,
    input  wire          sb_ready,
    output reg  [127:0]  sb,
    output reg           uop_valid,
    input  wire          uop_ready,
    output reg  [255:0]  uop,
    input  wire          mark_retired,
    output reg           sources_closed,
    output reg           done,
    output reg  [3:0]    done_status,
    output reg           busy
);
    `include "g3_defs.vh"
    function automatic [11:0] block_word(input [15:0] n);
        block_word = 33 * (n >> 2) + 1 + 8 * n[1:0];
    endfunction
    function automatic [11:0] scale_word(input [15:0] n);
        scale_word = 33 * (n >> 2);
    endfunction
    function automatic [127:0] mkseg(input [1:0] ctx, input [11:0] start_, input [1:0] n0m1, input [11:0] o1,
                                     input [11:0] o2, input [11:0] o3, input [11:0] n1, input [11:0] s1,
                                     input [7:0] user);
        mkseg = {18'd0, 1'b0, user, s1, n1 - 12'd1, o3, o2, o1, n0m1, start_, 13'd4096, 12'd0, ctx};
    endfunction
    function automatic [7:0] su(input set_last, input full, input [1:0] dst, input [1:0] slot, input hf);
        su = {1'b1, set_last, full, dst, slot, hf};
    endfunction

    reg [511:0] d_r;
    reg [12:0]  q_wm_r;
    reg [12:0]  k_wm_r [0:3], v_wm_r [0:3];
    assign cm_idx = d_r[282:275];
    assign cm_gen = d_r[290:283];
    assign cm_wm = d_r[315:303];
    assign cm_final = 1'b1;
    wire [11:0] q_base = d_r[27:16];
    wire        k_app = d_r[157], v_app = d_r[274];
    wire [11:0] o_base = d_r[302:291];
    wire [11:0] tokens = d_r[327:316];
    wire [4:0]  q_heads = d_r[332:328];
    wire [2:0]  group = d_r[335:333];
    wire [11:0] keys = d_r[347:336];
    wire signed [13:0] delta = d_r[361:348];
    wire [3:0]  gcnt = d_r[365:362];
    // Shape.
    wire [8:0]  hp = (tokens + 12'd7) >> 3;
    wire [11:0] npan = group * hp;
    wire [11:0] ppk = (npan + 12'd3) >> 2;               // passes per KV head
    wire [3:0]  kvh = q_heads / group;
    wire [11:0] items = kvh * ppk;
    wire [7:0]  jblocks = 8'((keys + 12'd63) >> 6);
    wire        causal = $signed({1'b0, keys}) - 1 > delta;
    wire [2:0]  dpn = (gcnt + 4'd1) >> 1;                // Dh passes of 64 dims
    wire        big = gcnt > 4'd4;
    wire [1:0]  slots = big ? 2'd1 : 2'd2;

    // ------------------------------------------------------------------
    // Item geometry (latched when the item starts).
    // ------------------------------------------------------------------
    reg [11:0] item_r, done_r;    // item_r: next item to start
    wire [3:0]  g_kv = item_r / ppk;
    wire [11:0] g_i = item_r % ppk;
    wire [11:0] g_pp = 4 * g_i;                             // first panel inside the KV head's stack
    wire [11:0] g_np_raw = npan - g_pp;
    wire [2:0]  g_np = g_np_raw > 4 ? 3'd4 : g_np_raw[2:0];
    reg  [31:0] g_rows;
    reg  [43:0] g_ptb;
    reg  signed [15:0] g_tmax;
    always @* begin : geo
        integer p, pin, v, r;
        g_rows = 0; g_ptb = 0; g_tmax = -1;
        for (p = 0; p < 4; p = p + 1) begin
            pin = (g_pp + p) % hp;
            g_ptb[p*11 +: 11] = 11'(8 * pin);
            v = p < g_np ? tokens - 8 * pin : 0;
            if (v > 8) v = 8;
            if (v < 0) v = 0;
            for (r = 0; r < 8; r = r + 1) if (r < v) g_rows[p*8 + r] = 1'b1;
            if (v > 0 && 8 * pin + v - 1 > g_tmax) g_tmax = 8 * pin + v - 1;
        end
    end
    wire signed [15:0] g_reach = g_tmax + delta;
    wire [7:0]  g_jb = !causal ? jblocks : g_reach < 0 ? 8'd1 :
                       ((g_reach >>> 6) + 1 > jblocks ? jblocks : 8'((g_reach >>> 6) + 1));
    reg [3:0]  it_kv_r, nx_kv_r;
    reg [11:0] it_p0_r, nx_p0_r;   // Q panel index of the pass's first panel
    reg [2:0]  it_np_r, nx_np_r;
    reg [31:0] it_rows_r, nx_rows_r;
    reg [43:0] it_ptb_r, nx_ptb_r;
    reg [7:0]  it_jb_r, nx_jb_r;
    reg        it_slot_r, nx_slot_r;
    reg        nx_valid_r;         // the next item started early (its QK(0) is issued)
    reg        early_r;            // the QK being issued is the next item's QK(0)
    // Geometry of the QK being issued: the next item's while early_r.
    wire [3:0]  q_kv = early_r ? nx_kv_r : it_kv_r;
    wire [11:0] q_p0 = early_r ? nx_p0_r : it_p0_r;
    wire [2:0]  q_np = early_r ? nx_np_r : it_np_r;

    // ------------------------------------------------------------------
    // Item sequencer.
    // ------------------------------------------------------------------
    localparam [1:0] I_QK = 0, I_PV = 1, I_OD = 2;
    localparam [3:0] A_IDLE = 0, A_ITEM = 1, A_UOP = 2, A_SEG = 3, A_NEXT = 4, A_DRAIN = 5, A_COMMIT = 6, A_DONE = 7,
                     A_LASTPV = 8;
    reg [3:0] state_r;
    reg [1:0] kind_r;
    reg [7:0] kb_r;               // key block (64 keys)
    reg [1:0] dp_r;
    reg [3:0] c_r;                // QK: Dh group; PV: key group (0, 1)
    reg [2:0] seg_i_r;
    // Block geometry (kb: the block of the uop being issued; the early QK(0) is block 0).
    wire [7:0]  kb = early_r ? 8'd0 : kb_r;
    wire [11:0] kleft = keys - 64 * kb;
    wire [5:0]  nl_k = kleft > 32 ? 6'd32 : kleft[5:0];
    wire [5:0]  nr_k = kleft > 64 ? 6'd32 : kleft > 32 ? 6'(kleft - 32) : 6'd0;
    wire [6:0]  width = nl_k + nr_k;
    wire [1:0]  pv_kg = nr_k != 0 ? 2'd2 : 2'd1;
    wire signed [15:0] limb = delta - 64 * $signed({8'd0, kb}) + 1;
    wire [11:0] dleft = 32 * gcnt - 64 * dp_r;
    wire [5:0]  nl_d = dleft > 32 ? 6'd32 : dleft[5:0];
    wire [5:0]  nr_d = dleft > 64 ? 6'd32 : dleft > 32 ? 6'(dleft - 32) : 6'd0;
    wire [11:0] kbase = d_r[57 + q_kv[1:0]*12 +: 12];
    wire [11:0] vbase = d_r[174 + it_kv_r[1:0]*12 +: 12];
    wire [12:0] kwm = k_wm_r[q_kv[1:0]];
    wire [12:0] vwm = v_wm_r[it_kv_r[1:0]];
    // Watermark needs.
    wire [15:0] q_last = (q_p0 + q_np - 1) * gcnt + gcnt - 1;
    wire [12:0] q_need = 33 * ((q_last >> 2) + 1);
    wire [12:0] k_rows = 64 * (kb + 1) > keys ? keys : 64 * (kb + 1);
    wire [12:0] k_words = 33 * gcnt * ((k_rows + 31) >> 5);
    wire q_ok = q_wm_r == `G3_WM_READY || q_wm_r >= q_need;
    wire k_ok = kwm == `G3_WM_READY || kwm >= (k_app ? k_rows : k_words);
    wire v_ok = vwm == `G3_WM_READY || vwm >= (v_app ? k_rows : k_words);
    wire slot_free = slots == 2'd2 || done_r == item_r;
    wire early_ok = slots == 2'd2 && item_r < items;      // start the next item before the last PV
    // Uops.
    reg [255:0] u_w;
    always @* begin
        u_w = 0;
        u_w[`U_ROWS] = early_r ? nx_rows_r : it_rows_r;
        u_w[`U_EMIT] = 1'b1;
        u_w[`U_SLOT] = early_r ? nx_slot_r : it_slot_r;
        u_w[`U_WIDTH] = width;
        u_w[`U_CAUSAL] = causal;
        u_w[`U_LIMB] = limb[13:0];
        u_w[`U_PTB] = early_r ? nx_ptb_r : it_ptb_r;
        u_w[`U_QKG] = gcnt;
        u_w[`U_BIG] = big;
        u_w[`U_DP] = dp_r;
        case (kind_r)
        I_QK: begin
            u_w[`U_OP] = 2'd1; u_w[`U_KG] = {2'd0, gcnt}; u_w[`U_NL] = nl_k; u_w[`U_NR] = nr_k;
            u_w[`U_CSTART] = kb == 0;
        end
        I_PV: begin
            u_w[`U_OP] = 2'd2; u_w[`U_KG] = {4'd0, pv_kg}; u_w[`U_NL] = nl_d; u_w[`U_NR] = nr_d;
            u_w[`U_ASRC] = 1'b1; u_w[`U_LASTKB] = kb_r + 1 == it_jb_r; u_w[`U_LASTDP] = dp_r + 1 == dpn;
        end
        default: begin
            u_w[`U_OP] = 2'd3;
            u_w[`U_LAYOUT] = `G3_LAYOUT_P8; u_w[`U_GD] = {5'd0, gcnt};
            u_w[`U_H0] = 1'b1; u_w[`U_H1] = 2 * dp_r + 1 < gcnt;
            u_w[`U_BASE0] = o_base; u_w[`U_BASE1] = o_base;
            u_w[`U_PAN0] = it_p0_r; u_w[`U_PAN1] = it_p0_r;
            u_w[`U_G0] = {6'd0, dp_r, 1'b0}; u_w[`U_G1] = {6'd0, dp_r, 1'b1};
            u_w[`U_NPAN] = it_np_r;
            u_w[`U_MARK] = dp_r + 1 == dpn;
        end
        endcase
    end
    // Segments: QK group c: 0..3 Q scale p, 4 Q payload, 5 K left scale, 6 payload, 7 K right scale,
    // (8 = payload, as seg_i 7 with a second phase); PV key group c: 0 V left scale, 1 payload,
    // 2 V right scale, 3 payload.
    reg        seg_port;          // 0 A, 1 B
    reg        seg_here;
    reg [127:0] seg_d;
    reg        seg_last;          // the last segment of the uop's current group
    reg [3:0]  rseg_i_r;          // QK right payload phase
    always @* begin : seg_mux
        reg [15:0] n0, n1, n2, n3, nq;
        reg [11:0] cell_w;
        reg hf;
        seg_port = 0; seg_here = 1'b0; seg_d = 0; seg_last = 1'b0; cell_w = 0; hf = 0;
        n0 = q_p0 * gcnt + c_r; n1 = n0 + gcnt; n2 = n1 + gcnt; n3 = n2 + gcnt;
        nq = n0;
        if (kind_r == I_QK) begin
            case (seg_i_r)
            3'd0, 3'd1, 3'd2, 3'd3: begin
                nq = seg_i_r == 0 ? n0 : seg_i_r == 1 ? n1 : seg_i_r == 2 ? n2 : n3;
                seg_port = 0; seg_here = seg_i_r < q_np;
                seg_d = mkseg(2'd0, q_base + scale_word(nq), 2'd0, 0, 0, 0, 12'd1, 12'd0,
                              su(seg_i_r + 1 == q_np, 1'b0, seg_i_r[1:0], nq[1:0], 1'b0));
            end
            3'd4: begin
                seg_port = 0; seg_here = 1'b1;
                seg_d = mkseg(2'd0, q_base + block_word(n0), 2'(q_np - 1), block_word(n1) - block_word(n0),
                              block_word(n2) - block_word(n0), block_word(n3) - block_word(n0), 12'd8, 12'd1, 8'd0);
            end
            default: begin
                // 5 left scale, 6 left payload, 7 right scale (rseg 0) / right payload (rseg 1).
                hf = seg_i_r == 7;
                cell_w = kbase + 33 * ((2 * kb + hf) * gcnt + c_r);
                seg_port = 1; seg_here = !hf || nr_k != 0;
                if (seg_i_r == 5 || (seg_i_r == 7 && rseg_i_r == 0))
                    seg_d = mkseg({1'b0, hf}, cell_w, 2'd0, 0, 0, 0, 12'd1, 12'd0, su(1'b1, 1'b1, 2'd0, 2'd0, hf));
                else
                    seg_d = mkseg({1'b0, hf}, cell_w + 1, 2'd3, 12'd8, 12'd16, 12'd24, 12'd8, 12'd1, 8'd0);
                seg_last = seg_i_r == 7 && (rseg_i_r == 1 || nr_k == 0);
            end
            endcase
        end else begin
            hf = seg_i_r >= 2;
            cell_w = vbase + 33 * ((2 * kb_r + c_r) * gcnt + 2 * dp_r + hf);
            seg_port = 1; seg_here = !hf || 2 * dp_r + 1 < gcnt;
            if (seg_i_r == 0 || seg_i_r == 2)
                seg_d = mkseg({1'b0, hf}, cell_w, 2'd0, 0, 0, 0, 12'd1, 12'd0, su(1'b1, 1'b1, 2'd0, 2'd0, hf));
            else
                seg_d = mkseg({1'b0, hf}, cell_w + 1, 2'd3, 12'd8, 12'd16, 12'd24, 12'd8, 12'd1, 8'd0);
            seg_last = seg_i_r == 3;
        end
    end
    wire seg_slot = seg_port ? !sb_valid : !sa_valid;
    wire seg_go = state_r == A_SEG && (!seg_here || seg_slot);
    wire grp_cnt_last = kind_r == I_QK ? c_r + 1 == gcnt : c_r + 1 == pv_kg;
    wire uop_gate = !uop_valid && (kind_r != I_QK || (q_ok && k_ok)) && (kind_r != I_PV || v_ok);
    wire item_end = kind_r == I_OD && dp_r + 1 == dpn;
    wire drained = !sa_valid && !sb_valid && !uop_valid && done_r == item_r;

    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state_r <= A_IDLE; cm_valid <= 0; sa_valid <= 0; sb_valid <= 0; uop_valid <= 0; sources_closed <= 0;
            done <= 0; busy <= 0; done_status <= 0; item_r <= 0; done_r <= 0; nx_valid_r <= 0; early_r <= 0;
            kind_r <= 0; kb_r <= 0; dp_r <= 0; c_r <= 0; seg_i_r <= 0; rseg_i_r <= 0;
        end else begin
            sources_closed <= 0; done <= 0;
            if (sa_valid && sa_ready) sa_valid <= 1'b0;
            if (sb_valid && sb_ready) sb_valid <= 1'b0;
            if (uop_valid && uop_ready) uop_valid <= 1'b0;
            if (cm_valid && cm_ready) cm_valid <= 1'b0;
            if (mark_retired && state_r != A_IDLE) done_r <= done_r + 1'b1;
            if (seg_go && seg_here) begin
                if (seg_port) begin sb_valid <= 1'b1; sb <= seg_d; end
                else begin sa_valid <= 1'b1; sa <= seg_d; end
            end
            case (state_r)
            A_IDLE: if (start) begin
                state_r <= A_ITEM; busy <= 1'b1; d_r <= desc; item_r <= 0; done_r <= 0; nx_valid_r <= 0; early_r <= 0;
                q_wm_r <= desc[40:28];
                for (integer e = 0; e < 4; e = e + 1) begin
                    k_wm_r[e] <= desc[105 + e*13 +: 13];
                    v_wm_r[e] <= desc[222 + e*13 +: 13];
                end
            end
            A_ITEM: begin
                if (item_r >= items) state_r <= A_DRAIN;
                else if (slot_free) begin
                    it_kv_r <= g_kv; it_p0_r <= g_kv * npan + g_pp; it_np_r <= g_np; it_rows_r <= g_rows;
                    it_ptb_r <= g_ptb; it_jb_r <= g_jb; it_slot_r <= slots == 2 ? item_r[0] : 1'b0;
                    item_r <= item_r + 1'b1;
                    kind_r <= I_QK; kb_r <= 0; dp_r <= 0; c_r <= 0; seg_i_r <= 0; rseg_i_r <= 0;
                    state_r <= A_UOP;
                end
            end
            A_UOP: if (uop_gate) begin
                uop_valid <= 1'b1; uop <= u_w; seg_i_r <= 0; rseg_i_r <= 0; c_r <= 0;
                state_r <= kind_r == I_OD ? A_NEXT : A_SEG;
            end
            A_SEG: if (seg_go) begin
                if (kind_r == I_QK && seg_i_r == 7 && rseg_i_r == 0 && nr_k != 0) rseg_i_r <= 1;
                else if (!seg_last) seg_i_r <= kind_r == I_PV && seg_i_r == 1 && 2 * dp_r + 1 >= gcnt ? 3'd3 :
                                               seg_i_r + 1'b1;
                else begin
                    seg_i_r <= 0; rseg_i_r <= 0;
                    if (grp_cnt_last) state_r <= A_NEXT;
                    else c_r <= c_r + 1'b1;
                end
            end
            A_NEXT: begin
                state_r <= A_UOP;
                case (kind_r)
                I_QK: if (early_r) begin early_r <= 1'b0; kind_r <= I_PV; end         // back to the last PV
                      else if (kb_r == 0 && it_jb_r > 1) kb_r <= 1;
                      else begin
                          kind_r <= I_PV; dp_r <= 0; kb_r <= kb_r == 0 ? 8'd0 : kb_r - 1'b1;
                          state_r <= A_LASTPV;
                      end
                I_PV: if (dp_r + 1 != dpn) dp_r <= dp_r + 1'b1;
                      else if (kb_r + 2 < it_jb_r) begin kind_r <= I_QK; kb_r <= kb_r + 2; dp_r <= 0; end
                      else if (kb_r + 1 < it_jb_r) begin kb_r <= kb_r + 1'b1; dp_r <= 0; state_r <= A_LASTPV; end
                      else begin kind_r <= I_OD; dp_r <= 0; end
                default: if (item_end && nx_valid_r) begin
                             // Continue the item started early: QK(1), or its single block's PV.
                             it_kv_r <= nx_kv_r; it_p0_r <= nx_p0_r; it_np_r <= nx_np_r; it_rows_r <= nx_rows_r;
                             it_ptb_r <= nx_ptb_r; it_jb_r <= nx_jb_r; it_slot_r <= nx_slot_r; nx_valid_r <= 1'b0;
                             dp_r <= 0;
                             if (nx_jb_r > 1) begin kind_r <= I_QK; kb_r <= 1; end
                             else begin kind_r <= I_PV; kb_r <= 0; state_r <= A_LASTPV; end
                         end else if (item_end) state_r <= A_ITEM;
                         else dp_r <= dp_r + 1'b1;
                endcase
            end
            // Before the first Dh pass of a PV: if it is the item's last block, start the next item.
            A_LASTPV: begin
                state_r <= A_UOP;
                if (kb_r + 1 == it_jb_r && early_ok) begin
                    nx_kv_r <= g_kv; nx_p0_r <= g_kv * npan + g_pp; nx_np_r <= g_np; nx_rows_r <= g_rows;
                    nx_ptb_r <= g_ptb; nx_jb_r <= g_jb; nx_slot_r <= item_r[0];
                    item_r <= item_r + 1'b1; nx_valid_r <= 1'b1;
                    early_r <= 1'b1; kind_r <= I_QK;
                end
            end
            A_DRAIN: if (drained) begin state_r <= A_COMMIT; cm_valid <= 1'b1; end
            A_COMMIT: if (!cm_valid) begin state_r <= A_DONE; sources_closed <= 1'b1; done_status <= `G3_ST_OK; end
            A_DONE: begin state_r <= A_IDLE; done <= 1'b1; busy <= 1'b0; end
            default: state_r <= A_IDLE;
            endcase
            if (ev_valid && state_r != A_IDLE) begin
                if (ev_idx == d_r[7:0] && ev_gen == d_r[15:8] && q_wm_r != `G3_WM_READY && ev_wm > q_wm_r) q_wm_r <= ev_wm;
                for (integer e = 0; e < 4; e = e + 1) begin
                    if (ev_idx == 8'(d_r[48:41] + e) && ev_gen == d_r[56:49] && k_wm_r[e] != `G3_WM_READY && ev_wm > k_wm_r[e])
                        k_wm_r[e] <= ev_wm;
                    if (ev_idx == 8'(d_r[165:158] + e) && ev_gen == d_r[173:166] && v_wm_r[e] != `G3_WM_READY && ev_wm > v_wm_r[e])
                        v_wm_r[e] <= ev_wm;
                end
            end
        end
    end
endmodule

`default_nettype wire
