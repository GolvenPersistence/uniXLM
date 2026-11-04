`timescale 1ns/1ps
`default_nettype none

// Gen3 Node controller (gen3_rtl/README.md s6.3): the Gen2 controller (rtl/gen2/g2_ctrl.v,
// docs/gen2/binding_sync.md, execution_model.md s2) for one array. Changes: one array parent (GEMM
// or ATTN) at a time instead of core leases; BIND [127] = transposed layout (V^T cache); GEMM width
// ignored; V^T appends and ATTN 64-key blocks (gen3_rtl/README.md s2).
//
// Block diagram:
//
//   cmd --> command FIFO (16) --> admission FSM <------> URF (256 entries: valid, gen,
//                                    |                   base, ext, order, transposed, kind, alen,
//                                    |                   wm, readers, writer, poison)
//                                    |                   granule owner map (512 x 8 words)
//                                    |                   stream descriptors (8)
//                                    | start + descriptor (watermark snapshots)
//                                    +--> gemm_* (GEMM sequencer)  +--> attn_* (ATTN sequencer)
//                                    +--> tx_* (MOVE_OUT channels x2)
//                                    +--> vru_* (VECTOR)
//   commit events: GEMM, NI RX, VRU, ATTN --> event arbiter (RR) --> URF apply --> ev_* next cycle
//   parent done / sources closed --> reader release (one binding per cycle) --+
//   WAIT --> waiters (4) -----------------------------------------------------+--> completion
//   admission errors ---------------------------------------------------------+    FIFO (16) -> cpl
//
// Control (admission, in command order):
//
//   S_IDLE -> S_DECODE -+-> S_BIND_SCAN (victims quiescent, R5) -> S_BIND_COMMIT -> S_CPL
//                       +-> S_GEMM | S_ATTN | S_VEC | S_MOVE:
//                       |     binding checks -> error: S_CPL (status)
//                       |                    -> ok: wait for resources -> admit -> S_IDLE
//                       +-> S_FENCE (no parent active) -> S_CPL
//                       +-> S_WAIT (place a waiter) -> S_IDLE
//                       +-> S_UNBIND (quiescent) -> S_CPL;  ABORT -> S_CPL
//   Parent slots: 0 array (GEMM or ATTN), 1-2 TX channels, 3 VRU.
//   The reader release engine never runs in a registration cycle.
module g3_ctrl (
    input  wire          clk,
    input  wire          rstn,
    input  wire          cmd_valid,
    output wire          cmd_ready,
    input  wire [127:0]  cmd_data,
    output wire          cpl_valid,
    input  wire          cpl_ready,
    output wire [7:0]    cpl_tag,
    output wire [3:0]    cpl_status,
    output wire [31:0]   cpl_data,
    // Commit events.
    input  wire          gemm_cm_valid,
    output reg           gemm_cm_ready,
    input  wire [7:0]    gemm_cm_idx,
    input  wire [7:0]    gemm_cm_gen,
    input  wire [12:0]   gemm_cm_wm,
    input  wire          gemm_cm_final,
    input  wire          ni_ev_valid,
    output reg           ni_ev_ready,
    input  wire [7:0]    ni_ev_idx,
    input  wire [7:0]    ni_ev_gen,
    input  wire [12:0]   ni_ev_wm,
    input  wire          ni_ev_final,
    // Broadcast.
    output reg           ev_valid,
    output reg  [7:0]    ev_idx,
    output reg  [7:0]    ev_gen,
    output reg  [12:0]   ev_wm,
    // URF lookup for NI RX.
    input  wire [7:0]    lk_idx,
    output wire          lk_valid,
    output wire [7:0]    lk_gen,
    output wire [11:0]   lk_base,
    output wire [12:0]   lk_ext,
    output wire [1:0]    lk_writer,
    // GEMM sequencer.
    output reg           gemm_start,
    output reg  [383:0]  gemm_pd,
    output reg           gemm_abort,
    input  wire          gemm_sources_closed,
    input  wire          gemm_done,
    input  wire [3:0]    gemm_status,
    // VRU.
    output reg           vru_start,
    output reg  [255:0]  vru_desc,
    input  wire          vru_done,
    input  wire [3:0]    vru_status,
    input  wire          vru_sources_closed,
    input  wire          vru_cm_valid,
    output reg           vru_cm_ready,
    input  wire [7:0]    vru_cm_idx,
    input  wire [7:0]    vru_cm_gen,
    input  wire [12:0]   vru_cm_wm,
    input  wire          vru_cm_final,
    // ATTN parent.
    output reg           attn_start,
    output reg  [511:0]  attn_desc,
    input  wire          attn_done,
    input  wire [3:0]    attn_status,
    input  wire          attn_sources_closed,
    input  wire          attn_cm_valid,
    output reg           attn_cm_ready,
    input  wire [7:0]    attn_cm_idx,
    input  wire [7:0]    attn_cm_gen,
    input  wire [12:0]   attn_cm_wm,
    input  wire          attn_cm_final,
    // MOVE_OUT channels.
    output reg  [1:0]    tx_start,
    output reg  [2*128-1:0] tx_desc,
    input  wire [1:0]    tx_done,
    // Status.
    output wire          idle,
    output reg  [31:0]   stat_admitted,
    output reg  [31:0]   stat_bind_wait,
    output reg  [31:0]   stat_lease_wait
);
    `include "g3_defs.vh"

    // ------------------------------------------------------------------
    // URF v2, stream descriptors, granule owner map.
    // ------------------------------------------------------------------
    reg        urf_valid_mem [0:255];
    reg [7:0]  urf_gen_mem [0:255];
    reg [11:0] urf_base_mem [0:255];
    reg [12:0] urf_ext_mem [0:255];
    reg        urf_order_mem [0:255];
    reg        urf_tr_mem [0:255];               // transposed P32 cells (V^T cache)
    reg [1:0]  urf_kind_mem [0:255];
    reg [3:0]  urf_alen_mem [0:255];
    reg [12:0] urf_wm_mem [0:255];
    reg [3:0]  urf_readers_mem [0:255];
    reg [1:0]  urf_writer_mem [0:255];
    reg        urf_poison_mem [0:255];
    reg [42:0] sd_src_mem [0:7];
    reg [19:0] sd_len_mem [0:7];
    reg        om_valid_mem [0:511];
    reg [7:0]  om_idx_mem [0:511];
    reg [7:0]  om_gen_mem [0:511];

    assign lk_valid = urf_valid_mem[lk_idx];
    assign lk_gen = urf_gen_mem[lk_idx];
    assign lk_base = urf_base_mem[lk_idx];
    assign lk_ext = urf_ext_mem[lk_idx];
    assign lk_writer = urf_writer_mem[lk_idx];

    function automatic ready_bind(input [7:0] i);
        ready_bind = urf_writer_mem[i] == `G3_W_NONE && urf_wm_mem[i] >= urf_ext_mem[i];
    endfunction

    // ------------------------------------------------------------------
    // Command FIFO.
    // ------------------------------------------------------------------
    reg [127:0] cmd_fifo_mem [0:15];
    reg [3:0] cmd_rd_ptr_r, cmd_wr_ptr_r; reg [4:0] cmd_level_r;
    assign cmd_ready = cmd_level_r != 16;
    wire [127:0] cmd_head = cmd_fifo_mem[cmd_rd_ptr_r];

    // ------------------------------------------------------------------
    // Completion FIFO.
    // ------------------------------------------------------------------
    reg [43:0] cpl_fifo_mem [0:15];                  // {tag, status, data}
    reg [3:0] cpl_rd_ptr_r, cpl_wr_ptr_r; reg [4:0] cpl_level_r;
    assign cpl_valid = cpl_level_r != 0;
    assign {cpl_tag, cpl_status, cpl_data} = cpl_fifo_mem[cpl_rd_ptr_r];

    // Parent slots: 0 array (GEMM or ATTN), 1-2 TX channels, 3 VRU.
    reg [3:0] parent_active_r;
    reg       parent_attn_r;                      // slot 0 holds an ATTN
    reg [7:0] parent_tag_r [0:3];
    reg [7:0] parent_in_r [0:3][0:8];             // input binding indices
    reg [3:0] parent_in_count_r [0:3];
    reg [3:0] parent_close_pend_r; // readers to release
    reg [3:0] parent_done_pend_r;  // completion to report
    reg [3:0] parent_status_r [0:3];

    // Waiters.
    reg [3:0] wait_active_r;
    reg [7:0] wait_tag_r [0:3], wait_idx_r [0:3], wait_gen_r [0:3];
    reg wait_mode_r [0:3];

    // ------------------------------------------------------------------
    // Admission FSM.
    // ------------------------------------------------------------------
    localparam [3:0] S_IDLE = 0, S_DECODE = 1, S_BIND_SCAN = 2, S_BIND_COMMIT = 3,
                     S_GEMM = 4, S_MOVE = 5, S_FENCE = 6, S_WAIT = 7, S_UNBIND = 8,
                     S_CPL = 9, S_VEC = 10, S_ATTN = 11;
    reg [3:0] state_r, state_n;
    reg [9:0] bind_granule_r, bind_granule_end_r;                   // granule scan
    reg [3:0] bind_count_r;
    reg [12:0] bind_sub_ext_r;
    reg [3:0] cpl_status_r;
    reg [31:0] cpl_data_r;

    // Decoded fields.
    wire [7:0]  cmd_op = cmd_head[7:0];
    wire [7:0]  cmd_tag = cmd_head[15:8];
    // BIND
    wire [7:0]  b_idx = cmd_head[23:16], b_gen = cmd_head[31:24];
    wire [11:0] b_base = cmd_head[43:32];
    wire [12:0] b_ext = cmd_head[55:44] + 13'd1;
    wire        b_order = cmd_head[56];
    wire        b_tr = cmd_head[127];
    wire [1:0]  b_kind = cmd_head[58:57];
    wire        b_ext_w = cmd_head[59];
    wire [3:0]  b_count = cmd_head[63:60] + 4'd1;
    // GEMM
    wire [7:0]  ga_i = cmd_head[23:16], ga_g = cmd_head[31:24];
    wire [7:0]  gb_i = cmd_head[39:32], gb_g = cmd_head[47:40];
    wire [7:0]  gc_i = cmd_head[55:48], gc_g = cmd_head[63:56];
    wire [7:0]  gd_i = cmd_head[71:64], gd_g = cmd_head[79:72];
    wire [11:0] gm = cmd_head[91:80] + 12'd1;
    wire [11:0] gn = cmd_head[103:92] + 12'd1;
    wire [8:0]  gk = cmd_head[111:104] + 9'd1;
    wire [3:0]  gepi = cmd_head[115:112];
    wire [3:0]  gah = cmd_head[121:118] + 4'd1;
    wire [3:0]  gdh = cmd_head[125:122] + 4'd1;
    wire        gbkn = cmd_head[126];
    // RMSNorm fold (bit 127; eps in the width field [117:116], which Gen3 does not decode otherwise).
    // Producer (epilogue ADD): D idx+1 = x * g, D idx+2 = 1/rms, D idx+3 = g (all with D's gen; g is a
    // source). C, the residual, is often itself a producer's x, with that producer's x * g at its idx+1, so
    // g stays out of C's block. Consumer: 1/rms = A idx+1.
    wire        gfold = cmd_head[127];
    wire [1:0]  geps = cmd_head[117:116];
    wire        g_fp = gfold && gepi == 4'd1;
    wire        g_fc = gfold && gepi != 4'd1;
    wire [7:0]  gx_i = gd_i + 8'd1, gr_i = gd_i + 8'd2, gg_i = gd_i + 8'd3, gq_i = ga_i + 8'd1;
    wire [7:0]  g_rpn = 8'((gm + 31) >> 5);
    wire        has_c = gepi == 4'd1 || gepi == 4'd2 || gepi == 4'd6;
    wire [7:0]  g_nt = 8'((gn + 31) >> 5);
    wire        b_stream = urf_kind_mem[gb_i] == `G3_KIND_STREAM;
    wire [3:0]  d_alen = urf_alen_mem[gd_i];
    wire        d_array = d_alen > 1;
    wire        d_append = urf_kind_mem[gd_i] == `G3_KIND_APPEND;
    wire        d_tr = urf_tr_mem[gd_i];
    wire [12:0] d_arow = urf_wm_mem[gd_i];
    // MOVE_OUT
    wire [7:0]  mo_i = cmd_head[23:16], mo_g = cmd_head[31:24];
    wire [11:0] mo_off = cmd_head[43:32];
    wire [12:0] mo_words = cmd_head[55:44] + 13'd1;
    wire [47:0] mo_dst = cmd_head[103:56];

    // VECTOR (RMSNorm landing): MAIN, AUX (gamma), RESULT.
    wire [7:0]  vm_i = cmd_head[23:16], vm_g = cmd_head[31:24];
    wire [15:0] v_rhs = cmd_head[47:32];
    wire [7:0]  va_i = cmd_head[55:48], va_g = cmd_head[63:56];
    wire [7:0]  vr_i = cmd_head[71:64], vr_g = cmd_head[79:72];
    wire [15:0] v_res2 = cmd_head[95:80];
    wire [11:0] v_rows = cmd_head[107:96] + 12'd1;
    wire [8:0]  v_groups = cmd_head[115:108] + 9'd1;
    wire [3:0]  v_op = cmd_head[119:116];
    wire [1:0]  v_eps = cmd_head[121:120];
    wire        v_whole = cmd_head[122];
    reg [3:0] v_status;
    always @* begin : vector_check
        v_status = `G3_ST_OK;
        if (v_op != 4'd8 || v_rhs != 0 || v_res2 != 0 || v_whole) v_status = `G3_ST_UNSUPPORTED;
        else if (!urf_valid_mem[vm_i] || urf_gen_mem[vm_i] != vm_g || !urf_valid_mem[va_i] ||
                 urf_gen_mem[va_i] != va_g ||
                 !urf_valid_mem[vr_i] || urf_gen_mem[vr_i] != vr_g) v_status = `G3_ST_BINDING;
        else if (urf_kind_mem[vm_i] != `G3_KIND_STATIC || urf_order_mem[vm_i] ||
                 urf_kind_mem[va_i] != `G3_KIND_STATIC ||
                 urf_order_mem[va_i] || urf_kind_mem[vr_i] != `G3_KIND_STATIC || urf_order_mem[vr_i]) v_status = `G3_ST_UNSUPPORTED;
        else if ((urf_writer_mem[vm_i] == `G3_W_NONE && urf_wm_mem[vm_i] < urf_ext_mem[vm_i] &&
                  !urf_poison_mem[vm_i]) ||
                 (urf_writer_mem[va_i] == `G3_W_NONE && urf_wm_mem[va_i] < urf_ext_mem[va_i] &&
                  !urf_poison_mem[va_i]) ||
                 urf_writer_mem[vr_i] != `G3_W_NONE || urf_wm_mem[vr_i] != 0 || urf_readers_mem[vr_i] != 0)
            v_status = `G3_ST_ORDER;
        else if (urf_poison_mem[vm_i] || urf_poison_mem[va_i]) v_status = `G3_ST_POISONED;
    end
    wire v_reg_now = state_r == S_VEC && v_status == `G3_ST_OK && !parent_active_r[3];

    // ATTN (ONESHOT landing): Q, K/V (arrays over KV heads), O.
    wire [7:0]  aq_i = cmd_head[23:16], aq_g = cmd_head[31:24];
    wire [7:0]  ak_i = cmd_head[39:32], ak_g = cmd_head[47:40];
    wire [7:0]  av_i = cmd_head[55:48], av_g = cmd_head[63:56];
    wire [7:0]  ao_i = cmd_head[71:64], ao_g = cmd_head[79:72];
    wire [11:0] a_tokens = cmd_head[90:80] + 12'd1;
    wire [4:0]  a_qh = cmd_head[94:91] + 5'd1;
    wire [2:0]  a_grp = cmd_head[96:95] + 3'd1;
    wire [11:0] a_keys = cmd_head[107:97] + 12'd1;
    wire signed [13:0] a_delta = cmd_head[121:108];
    wire [3:0]  a_g = cmd_head[124:122] + 4'd1;
    wire [1:0]  a_ctx = cmd_head[127:126];
    wire [4:0]  a_kvh = a_qh / a_grp;
    wire [8:0]  a_hp = (a_tokens + 12'd7) >> 3;
    wire [7:0]  a_j = 8'((a_keys + 12'd31) >> 5);
    wire [15:0] a_qwords = 33 * ((a_qh * a_hp * a_g + 16'd3) >> 2);
    wire [15:0] a_kvwords = 33 * a_g * a_j;
    reg [3:0] a_status;
    always @* begin : attn_check
        reg [3:0] s;
        integer x;
        reg [7:0] ki, vi;
        s = `G3_ST_OK;
        if (a_ctx != 0 || a_kvh > 4)
            s = `G3_ST_UNSUPPORTED;
        else if (a_qh % a_grp != 0) s = `G3_ST_RANGE;
        else begin
            if (!urf_valid_mem[aq_i] || urf_gen_mem[aq_i] != aq_g || !urf_valid_mem[ao_i] ||
                urf_gen_mem[ao_i] != ao_g) s = `G3_ST_BINDING;
            for (x = 0; x < 4; x = x + 1) begin
                ki = ak_i + x; vi = av_i + x;
                if (x < a_kvh &&
                    (!urf_valid_mem[ki] || urf_gen_mem[ki] != ak_g || !urf_valid_mem[vi] || urf_gen_mem[vi] != av_g))
                    s = `G3_ST_BINDING;
            end
            if (s == `G3_ST_OK) begin
                if (urf_kind_mem[aq_i] != `G3_KIND_STATIC || urf_order_mem[aq_i] ||
                    urf_kind_mem[ao_i] != `G3_KIND_STATIC ||
                    urf_order_mem[ao_i] || urf_kind_mem[ak_i] == `G3_KIND_STREAM || !urf_order_mem[ak_i] ||
                    urf_tr_mem[ak_i] || !urf_tr_mem[av_i] ||
                    urf_kind_mem[av_i] == `G3_KIND_STREAM || !urf_order_mem[av_i] ||
                    (a_kvh > 1 && (urf_alen_mem[ak_i] != a_kvh || urf_alen_mem[av_i] != a_kvh)))
                    s = `G3_ST_UNSUPPORTED;
                else if (urf_ext_mem[aq_i] < a_qwords || urf_ext_mem[ao_i] < a_qwords) s = `G3_ST_RANGE;
                else begin
                    for (x = 0; x < 4; x = x + 1) begin
                        ki = ak_i + x; vi = av_i + x;
                        if (x < a_kvh && (urf_ext_mem[ki] < a_kvwords || urf_ext_mem[vi] < a_kvwords)) s = `G3_ST_RANGE;
                    end
                    if (s == `G3_ST_OK) begin
                        if (urf_writer_mem[aq_i] == `G3_W_NONE && urf_wm_mem[aq_i] < urf_ext_mem[aq_i] &&
                            !urf_poison_mem[aq_i])
                            s = `G3_ST_ORDER;
                        for (x = 0; x < 4; x = x + 1) begin
                            ki = ak_i + x; vi = av_i + x;
                            if (x < a_kvh) begin
                                if (urf_writer_mem[ki] == `G3_W_NONE && !urf_poison_mem[ki] &&
                                    (urf_kind_mem[ki] == `G3_KIND_APPEND ? urf_wm_mem[ki] < a_keys
                                     : urf_wm_mem[ki] < urf_ext_mem[ki]))
                                    s = `G3_ST_ORDER;
                                if (urf_writer_mem[vi] == `G3_W_NONE && !urf_poison_mem[vi] &&
                                    (urf_kind_mem[vi] == `G3_KIND_APPEND ? urf_wm_mem[vi] < a_keys
                                     : urf_wm_mem[vi] < urf_ext_mem[vi]))
                                    s = `G3_ST_ORDER;
                            end
                        end
                        if (urf_writer_mem[ao_i] != `G3_W_NONE || urf_wm_mem[ao_i] != 0 || urf_readers_mem[ao_i] != 0)
                            s = `G3_ST_ORDER;
                        if (s == `G3_ST_OK) begin
                            if (urf_poison_mem[aq_i] || urf_poison_mem[ao_i]) s = `G3_ST_POISONED;
                            for (x = 0; x < 4; x = x + 1) begin
                                ki = ak_i + x; vi = av_i + x;
                                if (x < a_kvh && (urf_poison_mem[ki] || urf_poison_mem[vi])) s = `G3_ST_POISONED;
                            end
                        end
                    end
                end
            end
        end
        a_status = s;
    end

    // One array parent at a time.
    wire a_resources = !parent_active_r[0];
    wire at_reg_now = state_r == S_ATTN && a_status == `G3_ST_OK && a_resources;
    function automatic [12:0] kv_snap(input [7:0] i);
        kv_snap = urf_kind_mem[i] == `G3_KIND_APPEND ? urf_wm_mem[i] : snap(i);
    endfunction

    // GEMM checks -> status (0 = admit).
    // (An always block, not a function in a continuous assignment: the
    // checks read URF state that is not an argument.)
    reg [3:0] g_status;
    always @* begin : gemm_check
        reg [3:0] s;
        begin
            s = `G3_ST_OK;
            if (gbkn || gepi > 4'd6) s = `G3_ST_UNSUPPORTED;
            // V^T appends: no epilogue; several rows only from a 32-row boundary. Row appends of
            // several rows start at an 8-row panel.
            else if (d_tr && (!d_append || gepi != 4'd0 || (gm != 1 && d_arow[4:0] != 0))) s = `G3_ST_UNSUPPORTED;
            else if (!d_tr && d_append && gm != 1 && d_arow[2:0] != 0) s = `G3_ST_UNSUPPORTED;
            else if (has_c != (cmd_head[63:48] != 0)) s = `G3_ST_RESERVED;
            else if (gk % gah != 0 || g_nt % gdh != 0) s = `G3_ST_RANGE;
            else if (!urf_valid_mem[ga_i] || urf_gen_mem[ga_i] != ga_g || !urf_valid_mem[gb_i] ||
                     urf_gen_mem[gb_i] != gb_g ||
                     !urf_valid_mem[gd_i] || urf_gen_mem[gd_i] != gd_g ||
                     (has_c && (!urf_valid_mem[gc_i] || urf_gen_mem[gc_i] != gc_g))) s = `G3_ST_BINDING;
            else if (urf_kind_mem[ga_i] != `G3_KIND_STATIC || urf_order_mem[ga_i] ||
                     (!b_stream && (urf_kind_mem[gb_i] != `G3_KIND_STATIC || !urf_order_mem[gb_i])) ||
                     (has_c && (urf_kind_mem[gc_i] != `G3_KIND_STATIC || urf_order_mem[gc_i])))
                s = `G3_ST_UNSUPPORTED;
            else if ((urf_writer_mem[ga_i] == `G3_W_NONE && urf_wm_mem[ga_i] < urf_ext_mem[ga_i] &&
                      !urf_poison_mem[ga_i]) ||
                     (!b_stream && urf_writer_mem[gb_i] == `G3_W_NONE && urf_wm_mem[gb_i] < urf_ext_mem[gb_i] &&
                      !urf_poison_mem[gb_i]) ||
                     (b_stream && urf_readers_mem[gb_i] != 0) ||
                     (has_c && urf_writer_mem[gc_i] == `G3_W_NONE && urf_wm_mem[gc_i] < urf_ext_mem[gc_i] &&
                      !urf_poison_mem[gc_i]))
                s = `G3_ST_ORDER;
            else if (!d_append &&
                     (urf_writer_mem[gd_i] != `G3_W_NONE || urf_wm_mem[gd_i] != 0 || urf_readers_mem[gd_i] != 0 ||
                                   urf_kind_mem[gd_i] != `G3_KIND_STATIC))
                s = `G3_ST_ORDER;
            else if (d_array && d_alen != gdh) s = `G3_ST_RANGE;
            // RMSNorm fold: the producer writes one plain P8 tensor (and x * g, 1/rms beside it) in full
            // 64-column passes (its sum of squares replays each pass's right half after the left).
            else if (g_fp && (d_array || d_append || d_tr || gdh != 1 || gn[5:0] != 0)) s = `G3_ST_UNSUPPORTED;
            else if ((g_fp && (!urf_valid_mem[gx_i] || urf_gen_mem[gx_i] != gd_g || !urf_valid_mem[gr_i] ||
                               urf_gen_mem[gr_i] != gd_g || !urf_valid_mem[gg_i] || urf_gen_mem[gg_i] != gd_g)) ||
                     (g_fc && (!urf_valid_mem[gq_i] || urf_gen_mem[gq_i] != ga_g)))
                s = `G3_ST_BINDING;
            else if ((g_fp && (urf_kind_mem[gx_i] != `G3_KIND_STATIC || urf_order_mem[gx_i] ||
                               urf_ext_mem[gx_i] != urf_ext_mem[gd_i] || urf_kind_mem[gr_i] != `G3_KIND_STATIC ||
                               urf_ext_mem[gr_i] < {3'd0, g_rpn, 2'b00} ||
                               urf_kind_mem[gg_i] != `G3_KIND_STATIC || urf_order_mem[gg_i])) ||
                     (g_fc && urf_kind_mem[gq_i] != `G3_KIND_STATIC))
                s = `G3_ST_UNSUPPORTED;
            else if ((g_fp && (urf_writer_mem[gx_i] != `G3_W_NONE || urf_wm_mem[gx_i] != 0 || urf_readers_mem[gx_i] != 0 ||
                               urf_writer_mem[gr_i] != `G3_W_NONE || urf_wm_mem[gr_i] != 0 || urf_readers_mem[gr_i] != 0 ||
                               (urf_writer_mem[gg_i] == `G3_W_NONE && urf_wm_mem[gg_i] < urf_ext_mem[gg_i] &&
                                !urf_poison_mem[gg_i]))) ||
                     (g_fc && urf_writer_mem[gq_i] == `G3_W_NONE && urf_wm_mem[gq_i] < urf_ext_mem[gq_i] &&
                      !urf_poison_mem[gq_i]))
                s = `G3_ST_ORDER;
            else if (urf_poison_mem[ga_i] || urf_poison_mem[gb_i] || (has_c && urf_poison_mem[gc_i]) ||
                     urf_poison_mem[gd_i] || (g_fp && urf_poison_mem[gg_i]) || (g_fc && urf_poison_mem[gq_i]))
                s = `G3_ST_POISONED;
            g_status = s;
        end
    end

    wire g_resources = !parent_active_r[0] && (!d_append || urf_writer_mem[gd_i] == `G3_W_NONE);

    // MOVE_OUT of an APPEND binding (a KV cache) waits for its active
    // appender and then reads the committed rows ungated.
    wire mo_append = urf_kind_mem[mo_i] == `G3_KIND_APPEND;
    wire mo_src_ok = mo_append ? urf_writer_mem[mo_i] == `G3_W_NONE :
                     !(urf_writer_mem[mo_i] == `G3_W_NONE && urf_wm_mem[mo_i] < urf_ext_mem[mo_i]);
    wire mv_reg_now = state_r == S_MOVE && urf_valid_mem[mo_i] && urf_gen_mem[mo_i] == mo_g &&
                      !(mo_off + mo_words > urf_ext_mem[mo_i] || mo_dst[4:0] != 0) &&
                      mo_src_ok && (!parent_active_r[1] || !parent_active_r[2]);

    function automatic [12:0] snap(input [7:0] i);
        snap = ready_bind(i) ? `G3_WM_READY : urf_wm_mem[i];
    endfunction

    // ------------------------------------------------------------------
    // Event arbiter and URF apply (+ admission writes, same always block).
    // ------------------------------------------------------------------
    reg [2:0] ev_rr_ptr_r;
    reg ev_take;
    reg [7:0] ev_t_idx, ev_t_gen; reg [12:0] ev_t_wm; reg ev_t_final;
    integer ei, es;
    always @* begin
        ev_take = 0; gemm_cm_ready = 0; ni_ev_ready = 0; vru_cm_ready = 0; attn_cm_ready = 0;
        ev_t_idx = 0; ev_t_gen = 0; ev_t_wm = 0; ev_t_final = 0;
        for (ei = 0; ei < 7; ei = ei + 1) begin
            es = (ei + ev_rr_ptr_r) % 7;
            if (!ev_take) begin
                if (es == 0 && gemm_cm_valid) begin
                    ev_take = 1; gemm_cm_ready = 1;
                    ev_t_idx = gemm_cm_idx; ev_t_gen = gemm_cm_gen; ev_t_wm = gemm_cm_wm; ev_t_final = gemm_cm_final;
                end else if (es == 4 && ni_ev_valid) begin
                    ev_take = 1; ni_ev_ready = 1;
                    ev_t_idx = ni_ev_idx; ev_t_gen = ni_ev_gen; ev_t_wm = ni_ev_wm; ev_t_final = ni_ev_final;
                end else if (es == 5 && vru_cm_valid) begin
                    ev_take = 1; vru_cm_ready = 1;
                    ev_t_idx = vru_cm_idx; ev_t_gen = vru_cm_gen; ev_t_wm = vru_cm_wm; ev_t_final = vru_cm_final;
                end else if (es == 6 && attn_cm_valid) begin
                    ev_take = 1; attn_cm_ready = 1;
                    ev_t_idx = attn_cm_idx; ev_t_gen = attn_cm_gen; ev_t_wm = attn_cm_wm; ev_t_final = attn_cm_final;
                end
            end
        end
    end

    // Reader release engine: one decrement per cycle.
    reg [2:0] rel_slot_r; reg [3:0] rel_idx_r; reg rel_busy_r;

    // ------------------------------------------------------------------
    // Admission FSM: transition conditions, state register, next state.
    // (Combinational checks inline the URF reads: @* does not see memories
    // read inside functions.)
    // ------------------------------------------------------------------
    wire bind_err = b_gen == 0 || b_base[2:0] != 0 || b_base + b_ext > 13'd4096 ||
                    b_ext % (8 * b_count) != 0 || b_idx + b_count > 9'd256 ||
                    (b_kind == `G3_KIND_STREAM && (b_idx < 8'hf8 || b_count != 1)) ||
                    b_kind == 2'd3 || (b_ext_w && b_kind != `G3_KIND_STATIC);
    // BIND (R5): the target entries and every live owner of the scanned granule are quiescent.
    reg bind_busy_victim;
    always @* begin
        bind_busy_victim = 0;
        for (integer k = 0; k < 16; k = k + 1) begin : victim
            reg [7:0] t;
            t = b_idx + k;
            if (k < bind_count_r && urf_valid_mem[b_idx + k] &&
                !(urf_readers_mem[t] == 0 && urf_writer_mem[t] == `G3_W_NONE)) bind_busy_victim = 1;
        end
        if (bind_granule_r < bind_granule_end_r && om_valid_mem[bind_granule_r] &&
            urf_valid_mem[om_idx_mem[bind_granule_r]] &&
            urf_gen_mem[om_idx_mem[bind_granule_r]] == om_gen_mem[bind_granule_r] &&
            !(urf_readers_mem[om_idx_mem[bind_granule_r]] == 0 &&
              urf_writer_mem[om_idx_mem[bind_granule_r]] == `G3_W_NONE))
            bind_busy_victim = 1;
    end
    wire bind_last    = !(bind_granule_r + 1 < bind_granule_end_r);    // last granule of the range
    wire mo_bind_err  = !urf_valid_mem[mo_i] || urf_gen_mem[mo_i] != mo_g;
    wire mo_range_err = mo_off + mo_words > urf_ext_mem[mo_i] || mo_dst[4:0] != 0;
    wire g_reg_now    = state_r == S_GEMM && g_status == `G3_ST_OK && g_resources;
    wire adm_register = g_reg_now || mv_reg_now || v_reg_now || at_reg_now;   // parent registration

    // URF reader counters: one driver each. A registration adds the parent's source bindings (a binding
    // used twice counts twice); the release walk (never in a registration cycle) removes one per step;
    // BIND clears the rebound range.
    wire       reader_release = !adm_register && rel_busy_r && rel_idx_r < parent_in_count_r[rel_slot_r[1:0]];
    wire [7:0] reader_release_idx = parent_in_r[rel_slot_r[1:0]][rel_idx_r];
    for (genvar ri = 0; ri < 256; ri = ri + 1) begin : g_readers
        // Sources registered this cycle that are binding ri (the four admissions are exclusive).
        wire [3:0] g_add  = 4'(ga_i == ri) + 4'(gb_i == ri) + 4'(has_c && gc_i == ri) +
                            4'(g_fp && gg_i == ri) + 4'(g_fc && gq_i == ri);
        wire [3:0] v_add  = 4'(vm_i == ri) + 4'(va_i == ri);
        wire [3:0] at_add = 4'(aq_i == ri) +
                            4'(a_kvh > 5'd0 && ak_i == ri) + 4'(a_kvh > 5'd0 && av_i == ri) +
                            4'(a_kvh > 5'd1 && 8'(ak_i + 8'd1) == ri) + 4'(a_kvh > 5'd1 && 8'(av_i + 8'd1) == ri) +
                            4'(a_kvh > 5'd2 && 8'(ak_i + 8'd2) == ri) + 4'(a_kvh > 5'd2 && 8'(av_i + 8'd2) == ri) +
                            4'(a_kvh > 5'd3 && 8'(ak_i + 8'd3) == ri) + 4'(a_kvh > 5'd3 && 8'(av_i + 8'd3) == ri);
        wire [3:0] add_count = g_reg_now ? g_add : mv_reg_now ? 4'(mo_i == ri) : v_reg_now ? v_add :
                               at_reg_now ? at_add : 4'd0;
        always @(posedge clk or negedge rstn) begin
            if (!rstn) urf_readers_mem[ri] <= 0;
            else if (state_r == S_BIND_COMMIT && bind_last && ri >= b_idx && ri < b_idx + bind_count_r)
                urf_readers_mem[ri] <= 0;
            else urf_readers_mem[ri] <= urf_readers_mem[ri] + add_count -
                4'(reader_release && reader_release_idx == ri && urf_readers_mem[ri] != 0);
        end
    end
    wire fence_ok     = parent_active_r == 0 && parent_done_pend_r == 0;
    wire wait_place   = state_r == S_WAIT && !(&wait_active_r);             // a free waiter
    wire [7:0] ub_i   = cmd_head[23:16];
    wire unbind_valid = urf_valid_mem[ub_i] && urf_gen_mem[ub_i] == cmd_head[31:24];
    wire unbind_quiet = urf_readers_mem[ub_i] == 0 && urf_writer_mem[ub_i] == `G3_W_NONE;

    // Completion arbiter: a finished parent (lowest slot), else a waiter whose condition holds
    // (lowest first), else the admission completion of S_CPL.
    reg        done_push, wait_push;
    reg [2:0]  done_slot;
    reg [1:0]  wait_slot;
    reg [43:0] wait_entry;                                  // {tag, status, data}
    always @* begin
        done_push = 0; done_slot = 0; wait_push = 0; wait_slot = 0; wait_entry = 0;
        for (integer k = 0; k < 4; k = k + 1)
            if (!done_push && parent_done_pend_r[k] && !parent_close_pend_r[k] &&
                !(rel_busy_r && rel_slot_r == k) && cpl_level_r < 14) begin
                done_push = 1; done_slot = k;
            end
        for (integer k = 0; k < 4; k = k + 1)
            if (!done_push && !wait_push && wait_active_r[k] && cpl_level_r < 14) begin : waiter
                reg [7:0] w;
                w = wait_idx_r[k];
                if (!urf_valid_mem[w] || urf_gen_mem[w] != wait_gen_r[k]) begin
                    wait_push = 1; wait_slot = k; wait_entry = {wait_tag_r[k], `G3_ST_BINDING, 32'd0};
                end else if (urf_poison_mem[w]) begin
                    wait_push = 1; wait_slot = k; wait_entry = {wait_tag_r[k], `G3_ST_POISONED, 19'd0, urf_wm_mem[w]};
                end else if (wait_mode_r[k] ? urf_readers_mem[w] == 0 && urf_writer_mem[w] == `G3_W_NONE
                                            : urf_writer_mem[w] == `G3_W_NONE && urf_wm_mem[w] >= urf_ext_mem[w]) begin
                    wait_push = 1; wait_slot = k; wait_entry = {wait_tag_r[k], `G3_ST_OK, 19'd0, urf_wm_mem[w]};
                end
            end
    end
    wire adm_cpl  = state_r == S_CPL && !done_push && !wait_push && cpl_level_r < 15;
    wire cpl_push = done_push || wait_push || adm_cpl;
    wire adm_pop  = adm_register || wait_place || adm_cpl;       // the head command retires

    always @(posedge clk or negedge rstn)
        if (!rstn) state_r <= S_IDLE;
        else state_r <= state_n;

    always @* begin
        state_n = state_r;
        case (state_r)
        S_IDLE:        if (cmd_level_r != 0) state_n = S_DECODE;
        S_DECODE:
            case (cmd_op)
            `G3_OP_BIND:     state_n = bind_err ? S_CPL : S_BIND_SCAN;
            `G3_OP_GEMM:     state_n = S_GEMM;
            `G3_OP_MOVE_OUT: state_n = S_MOVE;
            `G3_OP_VECTOR:   state_n = S_VEC;
            `G3_OP_ATTN:     state_n = S_ATTN;
            `G3_OP_FENCE:    state_n = S_FENCE;
            `G3_OP_WAIT:     state_n = S_WAIT;
            `G3_OP_UNBIND:   state_n = S_UNBIND;
            default:         state_n = S_CPL;                     // ABORT, unsupported opcodes
            endcase
        S_BIND_SCAN:   if (!bind_busy_victim && bind_last) state_n = S_BIND_COMMIT;
        S_BIND_COMMIT: if (bind_last) state_n = S_CPL;
        S_GEMM:        if (g_status != `G3_ST_OK) state_n = S_CPL;
                       else if (g_resources) state_n = S_IDLE;
        S_MOVE:        if (mo_bind_err || mo_range_err || (!mo_append && !mo_src_ok)) state_n = S_CPL;
                       else if (mv_reg_now) state_n = S_IDLE;
        S_VEC:         if (v_status != `G3_ST_OK) state_n = S_CPL;
                       else if (v_reg_now) state_n = S_IDLE;
        S_ATTN:        if (a_status != `G3_ST_OK) state_n = S_CPL;
                       else if (a_resources) state_n = S_IDLE;
        S_FENCE:       if (fence_ok) state_n = S_CPL;
        S_WAIT:        if (wait_place) state_n = S_IDLE;
        S_UNBIND:      if (!unbind_valid || unbind_quiet) state_n = S_CPL;
        S_CPL:         if (adm_cpl) state_n = S_IDLE;
        default:       state_n = S_IDLE;
        endcase
    end

    // ------------------------------------------------------------------
    // Registered outputs: URF, events, parents, completions, admission actions.
    // ------------------------------------------------------------------
    integer i, e;
    always @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            for (i = 0; i < 256; i = i + 1) begin
                urf_valid_mem[i] <= 0;
                urf_writer_mem[i] <= 0;
                urf_poison_mem[i] <= 0;
                urf_wm_mem[i] <= 0;
                urf_gen_mem[i] <= 0;
                urf_kind_mem[i] <= 0;
                urf_alen_mem[i] <= 0;
                urf_order_mem[i] <= 0;
                urf_tr_mem[i] <= 0;
                urf_base_mem[i] <= 0;
                urf_ext_mem[i] <= 0;
            end
            for (i = 0; i < 512; i = i + 1) om_valid_mem[i] <= 0;
            cmd_rd_ptr_r <= 0;
            cmd_wr_ptr_r <= 0;
            cmd_level_r <= 0;
            cpl_rd_ptr_r <= 0;
            cpl_wr_ptr_r <= 0;
            cpl_level_r <= 0;
            parent_active_r <= 0;
            parent_attn_r <= 0;
            parent_close_pend_r <= 0;
            parent_done_pend_r <= 0;
            wait_active_r <= 0;
            gemm_start <= 0; gemm_abort <= 0; tx_start <= 0; vru_start <= 0; attn_start <= 0;
            ev_valid <= 0; ev_rr_ptr_r <= 0; rel_busy_r <= 0;
            stat_admitted <= 0; stat_bind_wait <= 0; stat_lease_wait <= 0;
            for (i = 0; i < 4; i = i + 1)
                for (e = 0; e < 9; e = e + 1) parent_in_r[i][e] <= 0;
        end else begin
            gemm_start <= 0; gemm_abort <= 0; tx_start <= 0; vru_start <= 0; attn_start <= 0;
            // Unused parent-source slots (TX: one source, VRU: two) are constants.
            for (i = 1; i < 4; i = i + 1)
                for (e = 1; e < 9; e = e + 1)
                    if (i < 3 || e >= 2) parent_in_r[i][e] <= 0;
            // ---------------- command FIFO push
            if (cmd_valid && cmd_ready) begin
                cmd_fifo_mem[cmd_wr_ptr_r] <= cmd_data;
                cmd_wr_ptr_r <= cmd_wr_ptr_r + 1'b1;
            end

            // ---------------- events: apply now, broadcast next cycle
            ev_valid <= ev_take;
            if (ev_take) begin
                ev_idx <= ev_t_idx; ev_gen <= ev_t_gen; ev_wm <= ev_t_wm;
                ev_rr_ptr_r <= ev_rr_ptr_r == 6 ? 0 : ev_rr_ptr_r + 1'b1;
                if (urf_valid_mem[ev_t_idx] && urf_gen_mem[ev_t_idx] == ev_t_gen) begin
                    if (ev_t_wm > urf_wm_mem[ev_t_idx]) urf_wm_mem[ev_t_idx] <= ev_t_wm;
                    if (ev_t_final) urf_writer_mem[ev_t_idx] <= `G3_W_NONE;
                end
            end

            // ---------------- executor completions
            if (gemm_sources_closed || gemm_done || attn_sources_closed || attn_done) parent_close_pend_r[0] <= 1'b1;
            if (gemm_done) begin parent_done_pend_r[0] <= 1'b1; parent_status_r[0] <= gemm_status; end
            if (attn_done) begin parent_done_pend_r[0] <= 1'b1; parent_status_r[0] <= attn_status; end
            for (i = 0; i < 2; i = i + 1)
                if (tx_done[i]) begin
                    parent_close_pend_r[1+i] <= 1'b1;
                    parent_done_pend_r[1+i] <= 1'b1;
                    parent_status_r[1+i] <= `G3_ST_OK;
                end
            if (vru_sources_closed || vru_done) parent_close_pend_r[3] <= 1'b1;
            if (vru_done) begin parent_done_pend_r[3] <= 1'b1; parent_status_r[3] <= vru_status; end

            // ---------------- reader release (never in a registration cycle)
            if (!adm_register) begin
                if (!rel_busy_r) begin
                    for (i = 3; i >= 0; i = i - 1)
                        if (parent_close_pend_r[i]) begin rel_slot_r <= i; rel_idx_r <= 0; end
                    if (|parent_close_pend_r) rel_busy_r <= 1'b1;
                end else begin
                    if (rel_idx_r < parent_in_count_r[rel_slot_r]) begin
                        // The reader count update is owned by g_readers.
                        rel_idx_r <= rel_idx_r + 1'b1;
                    end else begin
                        parent_close_pend_r[rel_slot_r] <= 1'b0; parent_in_count_r[rel_slot_r] <= 0; rel_busy_r <= 1'b0;
                    end
                end
            end

            // ---------------- completions: finished parents, waiters, admission (one per cycle)
            if (done_push) begin
                cpl_fifo_mem[cpl_wr_ptr_r] <= {parent_tag_r[done_slot], parent_status_r[done_slot], 32'd0};
                parent_done_pend_r[done_slot] <= 1'b0; parent_active_r[done_slot] <= 1'b0;
            end else if (wait_push) begin
                cpl_fifo_mem[cpl_wr_ptr_r] <= wait_entry;
                wait_active_r[wait_slot] <= 1'b0;
            end else if (adm_cpl) cpl_fifo_mem[cpl_wr_ptr_r] <= {cmd_tag, cpl_status_r, cpl_data_r};
            if (cpl_push) cpl_wr_ptr_r <= cpl_wr_ptr_r + 1'b1;
            cpl_level_r <= cpl_level_r + cpl_push - (cpl_valid && cpl_ready);
            if (cpl_valid && cpl_ready) cpl_rd_ptr_r <= cpl_rd_ptr_r + 1'b1;

            // ---------------- admission actions (transitions: state_n)
            case (state_r)
            S_DECODE: begin
                case (cmd_op)
                `G3_OP_BIND: begin
                    if (bind_err) begin
                        cpl_status_r <= `G3_ST_RANGE; cpl_data_r <= 0;
                    end else begin
                        bind_granule_r <= b_base[11:3];
                        bind_granule_end_r <= (b_base + b_ext) >> 3;
                        bind_count_r <= b_count;
                        bind_sub_ext_r <= b_ext / b_count;
                    end
                end
                `G3_OP_GEMM, `G3_OP_MOVE_OUT, `G3_OP_VECTOR, `G3_OP_ATTN, `G3_OP_FENCE, `G3_OP_WAIT, `G3_OP_UNBIND: ;
                `G3_OP_ABORT: begin
                    if (parent_active_r[0] && !parent_attn_r && parent_tag_r[0] == cmd_head[23:16]) gemm_abort <= 1'b1;
                    cpl_status_r <= `G3_ST_OK; cpl_data_r <= 0;
                end
                default: begin cpl_status_r <= `G3_ST_UNSUPPORTED; cpl_data_r <= 0; end
                endcase
            end
            S_BIND_SCAN:
                if (bind_busy_victim) stat_bind_wait <= stat_bind_wait + 1;
                else if (!bind_last) bind_granule_r <= bind_granule_r + 1'b1;
                else bind_granule_r <= b_base[11:3];
            S_BIND_COMMIT: begin
                // Evict the live owner of this granule, then claim it.
                if (om_valid_mem[bind_granule_r] && urf_valid_mem[om_idx_mem[bind_granule_r]] &&
                    urf_gen_mem[om_idx_mem[bind_granule_r]] == om_gen_mem[bind_granule_r])
                    urf_valid_mem[om_idx_mem[bind_granule_r]] <= 1'b0;
                om_valid_mem[bind_granule_r] <= 1'b1;
                om_idx_mem[bind_granule_r] <= b_idx + ((bind_granule_r - b_base[11:3]) / (bind_sub_ext_r >> 3));
                om_gen_mem[bind_granule_r] <= b_gen;
                if (!bind_last) bind_granule_r <= bind_granule_r + 1'b1;
                else begin
                    for (e = 0; e < 16; e = e + 1) if (e < bind_count_r) begin
                        urf_valid_mem[b_idx + e] <= 1'b1; urf_gen_mem[b_idx + e] <= b_gen;
                        urf_base_mem[b_idx + e] <= b_base + e * bind_sub_ext_r;
                        urf_ext_mem[b_idx + e] <= bind_sub_ext_r;
                        urf_order_mem[b_idx + e] <= b_order; urf_kind_mem[b_idx + e] <= b_kind;
                        urf_tr_mem[b_idx + e] <= b_tr;
                        urf_alen_mem[b_idx + e] <= e == 0 ? bind_count_r : 4'd1;
                        urf_wm_mem[b_idx + e] <= 0; urf_poison_mem[b_idx + e] <= 0;
                        urf_writer_mem[b_idx + e] <= b_ext_w ? `G3_W_EXT : `G3_W_NONE;
                    end
                    if (b_kind == `G3_KIND_STREAM) begin
                        sd_src_mem[b_idx[2:0]] <= cmd_head[106:64]; sd_len_mem[b_idx[2:0]] <= cmd_head[126:107] + 20'd1;
                    end
                    cpl_status_r <= `G3_ST_OK; cpl_data_r <= 0;
                end
            end
            S_GEMM: begin
                if (g_status != `G3_ST_OK) begin
                    cpl_status_r <= g_status; cpl_data_r <= 0;
                end else if (!g_resources) begin
                    stat_lease_wait <= stat_lease_wait + 1;
                end else begin : admit
                    reg [383:0] pdv;
                    pdv = 0;
                    pdv[`PD_TAG] = cmd_tag;
                    pdv[`PD_A_IDX] = ga_i; pdv[`PD_A_GEN] = ga_g; pdv[`PD_A_BASE] = urf_base_mem[ga_i];
                    pdv[`PD_A_WM] = snap(ga_i);
                    pdv[`PD_B_IDX] = gb_i; pdv[`PD_B_GEN] = gb_g; pdv[`PD_B_BASE] = urf_base_mem[gb_i];
                    pdv[`PD_B_WM] = b_stream ? 13'd0 : snap(gb_i);
                    pdv[`PD_B_STREAM] = b_stream; pdv[`PD_B_SRC] = sd_src_mem[gb_i[2:0]];
                    pdv[`PD_B_LEN] = sd_len_mem[gb_i[2:0]];
                    pdv[`PD_C_IDX] = gc_i; pdv[`PD_C_GEN] = gc_g; pdv[`PD_C_BASE] = urf_base_mem[gc_i];
                    pdv[`PD_C_WM] = has_c ? snap(gc_i) : `G3_WM_READY; pdv[`PD_C_VALID] = has_c;
                    pdv[`PD_D_IDX] = gd_i; pdv[`PD_D_GEN] = gd_g; pdv[`PD_D_BASE] = urf_base_mem[gd_i];
                    pdv[`PD_D_EXT] = urf_ext_mem[gd_i]; pdv[`PD_D_ALEN] = d_alen - 1'b1;
                    pdv[`PD_D_APPEND] = d_append;
                    pdv[`PD_D_AROW] = urf_wm_mem[gd_i];
                    pdv[`PD_D_P32] = urf_order_mem[gd_i];
                    pdv[`PD_D_TR] = urf_tr_mem[gd_i];
                    pdv[`PD_M] = gm; pdv[`PD_N] = gn; pdv[`PD_G] = gk; pdv[`PD_EPI] = gepi;
                    pdv[`PD_AHEADS] = gah - 1'b1; pdv[`PD_DHEADS] = gdh - 1'b1;
                    pdv[`PD_FP] = g_fp; pdv[`PD_FC] = g_fc; pdv[`PD_EPS] = geps;
                    pdv[`PD_XG_BASE] = urf_base_mem[gx_i];
                    pdv[`PD_R_BASE] = g_fp ? urf_base_mem[gr_i] : urf_base_mem[gq_i];
                    pdv[`PD_R_EXT] = urf_ext_mem[gr_i];
                    pdv[`PD_G_BASE] = urf_base_mem[gg_i];
                    pdv[`PD_F_WM] = g_fp ? snap(gg_i) : g_fc ? snap(gq_i) : `G3_WM_READY;
                    gemm_pd <= pdv;
                    gemm_start <= 1'b1;
                    parent_active_r[0] <= 1'b1; parent_attn_r <= 1'b0; parent_tag_r[0] <= cmd_tag;
                    // Sources (a fold input in the slot after C, or after B without C).
                    parent_in_r[0][0] <= ga_i; parent_in_r[0][1] <= gb_i; parent_in_r[0][2] <= has_c ? gc_i : gq_i;
                    parent_in_r[0][3] <= g_fp ? gg_i : gq_i;
                    parent_in_count_r[0] <= (has_c ? 3'd3 : 3'd2) + {2'd0, gfold};
                    for (e = 0; e < 16; e = e + 1)
                        if (e < (d_array ? d_alen : g_fp ? 4'd3 : 4'd1)) urf_writer_mem[gd_i + e] <= `G3_W_PARENT;
                    stat_admitted <= stat_admitted + 1;
                end
            end
            S_MOVE: begin
                if (mo_bind_err) begin
                    cpl_status_r <= `G3_ST_BINDING; cpl_data_r <= 0;
                end else if (mo_range_err) begin
                    cpl_status_r <= `G3_ST_RANGE; cpl_data_r <= 0;
                end else if (!mo_append && !mo_src_ok) begin
                    cpl_status_r <= `G3_ST_ORDER; cpl_data_r <= 0;
                end else if (mo_src_ok && (!parent_active_r[1] || !parent_active_r[2])) begin : mv
                    reg ch;
                    ch = parent_active_r[1];
                    tx_desc[ch*128 +: 128] <= {14'd0, mo_i, mo_g, urf_base_mem[mo_i],
                                               mo_append ? `G3_WM_READY : snap(mo_i), mo_off, mo_words, mo_dst};
                    tx_start[ch] <= 1'b1;
                    parent_active_r[1+ch] <= 1'b1; parent_tag_r[1+ch] <= cmd_tag;
                    parent_in_r[1+ch][0] <= mo_i; parent_in_count_r[1+ch] <= 3'd1;
                    stat_admitted <= stat_admitted + 1;
                end
            end
            S_VEC: begin
                if (v_status != `G3_ST_OK) begin
                    cpl_status_r <= v_status; cpl_data_r <= 0;
                end else if (!parent_active_r[3]) begin
                    vru_desc <= {110'd0, v_eps, v_groups, v_rows,
                                 urf_ext_mem[vr_i], urf_base_mem[vr_i], vr_g, vr_i,
                                 snap(va_i), urf_base_mem[va_i], va_g, va_i,
                                 snap(vm_i), urf_base_mem[vm_i], vm_g, vm_i};
                    vru_start <= 1'b1;
                    parent_active_r[3] <= 1'b1; parent_tag_r[3] <= cmd_tag;
                    parent_in_r[3][0] <= vm_i; parent_in_r[3][1] <= va_i; parent_in_count_r[3] <= 3'd2;
                    urf_writer_mem[vr_i] <= `G3_W_PARENT;
                    stat_admitted <= stat_admitted + 1;
                end
            end
            S_ATTN: begin
                if (a_status != `G3_ST_OK) begin
                    cpl_status_r <= a_status; cpl_data_r <= 0;
                end else if (!a_resources) begin
                    stat_lease_wait <= stat_lease_wait + 1;
                end else begin : at_admit
                    reg [511:0] dv;
                    reg [7:0] ki, vi;
                    dv = 0;
                    dv[7:0] = aq_i; dv[15:8] = aq_g; dv[27:16] = urf_base_mem[aq_i]; dv[40:28] = snap(aq_i);
                    dv[48:41] = ak_i; dv[56:49] = ak_g; dv[157] = urf_kind_mem[ak_i] == `G3_KIND_APPEND;
                    dv[165:158] = av_i; dv[173:166] = av_g; dv[274] = urf_kind_mem[av_i] == `G3_KIND_APPEND;
                    for (e = 0; e < 4; e = e + 1) begin
                        ki = ak_i + e; vi = av_i + e;
                        dv[57 + e*12 +: 12] = urf_base_mem[ki]; dv[105 + e*13 +: 13] = kv_snap(ki);
                        dv[174 + e*12 +: 12] = urf_base_mem[vi]; dv[222 + e*13 +: 13] = kv_snap(vi);
                    end
                    dv[282:275] = ao_i;
                    dv[290:283] = ao_g;
                    dv[302:291] = urf_base_mem[ao_i];
                    dv[315:303] = urf_ext_mem[ao_i];
                    dv[327:316] = a_tokens; dv[332:328] = a_qh; dv[335:333] = a_grp; dv[347:336] = a_keys;
                    dv[361:348] = a_delta; dv[365:362] = a_g;
                    attn_desc <= dv; attn_start <= 1'b1;
                    parent_active_r[0] <= 1'b1; parent_attn_r <= 1'b1; parent_tag_r[0] <= cmd_tag;
                    parent_in_r[0][0] <= aq_i; parent_in_count_r[0] <= 4'd1 + 2 * a_kvh[3:0];
                    for (e = 0; e < 4; e = e + 1) if (e < a_kvh) begin
                        ki = ak_i + e; vi = av_i + e;
                        parent_in_r[0][1 + 2*e] <= ki; parent_in_r[0][2 + 2*e] <= vi;
                    end
                    urf_writer_mem[ao_i] <= `G3_W_PARENT;
                    stat_admitted <= stat_admitted + 1;
                end
            end
            S_FENCE: if (fence_ok) begin
                cpl_status_r <= `G3_ST_OK; cpl_data_r <= 0;
            end
            S_WAIT: begin : wt
                reg placed;
                placed = 0;
                for (i = 0; i < 4; i = i + 1)
                    if (!placed && !wait_active_r[i]) begin
                        placed = 1; wait_active_r[i] <= 1'b1; wait_tag_r[i] <= cmd_tag;
                        wait_idx_r[i] <= cmd_head[23:16];
                        wait_gen_r[i] <= cmd_head[31:24];
                        wait_mode_r[i] <= cmd_head[32];
                    end
            end
            S_UNBIND: begin
                if (!unbind_valid) begin
                    cpl_status_r <= `G3_ST_BINDING; cpl_data_r <= 0;
                end else if (unbind_quiet) begin
                    for (e = 0; e < 16; e = e + 1)
                        if (e < urf_alen_mem[cmd_head[23:16]]) urf_valid_mem[cmd_head[23:16] + e] <= 1'b0;
                    cpl_status_r <= `G3_ST_OK; cpl_data_r <= 0;
                end
            end
            default: ;
            endcase
            if (adm_pop) cmd_rd_ptr_r <= cmd_rd_ptr_r + 1'b1;
            cmd_level_r <= cmd_level_r + (cmd_valid && cmd_ready) - adm_pop;
        end
    end
    assign idle = state_r == S_IDLE && cmd_level_r == 0 && parent_active_r == 0 && cpl_level_r == 0 && wait_active_r == 0;
endmodule

`default_nettype wire
