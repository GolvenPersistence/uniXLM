// Shared Gen3 numeric helper functions (included inside modules); copied from rtl/gen2/g2_numeric_functions.vh.
// decode_element: MXFP8 E4M3 x E8M0 -> FP32, exact (Gen1 VRU decoder).
// choose_scale:   FP32 row maximum -> E8M0 scale (Gen1 accumulator rule).
// max_value:      Gen1 row-maximum rule (unsigned magnitude or signed).
// quant_e4m3:     FP32 x E8M0 -> E4M3 code, RNE, saturating (rc_mxfp8_requant2).

function automatic [31:0] decode_element;
    input [7:0] value;
    input [7:0] scale;
    reg [17:0] magnitude;
    reg [23:0] normalized;
    reg [22:0] fraction;
    integer msb;
    integer exponent;
    integer shift;
    integer bit_i;
    begin
        magnitude = 18'd0;
        normalized = 24'd0;
        fraction = 23'd0;
        msb = 0;
        exponent = 0;
        shift = 0;
        if (scale == 8'hff || value[6:0] == 7'h7f) begin
            decode_element = 32'h7fc0_0000;
        end else if (value[6:0] == 0) begin
            decode_element = {value[7], 31'd0};
        end else begin
            magnitude = value[6:3] == 0 ? value[2:0] :
                ({14'd0, 1'b1, value[2:0]} << (value[6:3] - 1'b1));
            for (bit_i = 0; bit_i < 18; bit_i = bit_i + 1)
                if (magnitude[bit_i]) msb = bit_i;
            exponent = msb + scale - 136;
            if (exponent > 127) begin
                decode_element = {value[7], 8'hff, 23'd0};
            end else if (exponent >= -126) begin
                normalized = {6'd0, magnitude} << (23 - msb);
                fraction = normalized[22:0];
                decode_element = {value[7], 8'(exponent + 127), fraction};
            end else begin
                shift = scale + 13;
                fraction = magnitude << shift;
                decode_element = {value[7], 8'd0, fraction};
            end
        end
    end
endfunction

function automatic [7:0] choose_scale;
    input [31:0] a;
    integer s;
    begin
        if (a[30:23] == 255 && a[22:0] != 0) choose_scale = 255;
        else if (a[30:0] == 0) choose_scale = 127;
        else if (a[30:23] == 255) choose_scale = 254;
        else begin
            s = int'({1'b0, a[30:23]}) - 8 + (a[22:0] > 23'h600000);
            choose_scale = s < 0 ? 0 : s > 254 ? 254 : s[7:0];
        end
    end
endfunction

function automatic [31:0] max_value;
    input [31:0] a, b;
    input signed_mode;
    begin
        if ((a[30:23] == 255 && a[22:0] != 0) || (b[30:23] == 255 && b[22:0] != 0))
            max_value = 32'h7fc00000;
        else if (!signed_mode) max_value = a[30:0] >= b[30:0] ? {1'b0, a[30:0]} : {1'b0, b[30:0]};
        else if (a[30:0] == 0 && b[30:0] == 0) max_value = a[31] && !b[31] ? b : a;
        else if (a[31] != b[31]) max_value = a[31] ? b : a;
        else if (a[31]) max_value = a[30:0] <= b[30:0] ? a : b;
        else max_value = a[30:0] >= b[30:0] ? a : b;
    end
endfunction

// SiLU / GELU piecewise-linear coefficients (Gen1 vru_compute, unchanged).
function automatic [7:0] silu64_code(input [31:0] value);
    reg [5:0] index;
    begin
        case(value[30:23])
            123:index=6'd1;
            124:index={5'b00001,value[22]};
            125:index={4'b0001,value[22:21]};
            126:index={3'b001,value[22:20]};
            127:index={2'b01,value[22:19]};
            128:index={2'b10,value[22:19]};
            129:index={2'b11,value[22:19]};
            default:index=0;
        endcase
        silu64_code={value[31],value[30:0]>=31'h41000000,index};
    end
endfunction

function automatic [63:0] silu64_coeff(input [7:0] code);
    reg [95:0] entry;
    begin
        entry=0;
        case(code[5:0])
            6'd0:entry=96'h000000003ef800ab3f03ffab;
            6'd1:entry=96'hbaff6aed3ee809fc3f0bfb02;
            6'd2:entry=96'hbbbed1503ed82b293f13ea6b;
            6'd3:entry=96'hbc3db4d93ec873cc3f1bc61a;
            6'd4:entry=96'hbc9cdd7c3eb8f3083f23867c;
            6'd5:entry=96'hbce907b93ea9b7623f2b244f;
            6'd6:entry=96'hbd213e3d3e9ace973f3298b4;
            6'd7:entry=96'hbd541e263e8c45793f39dd43;
            6'd8:entry=96'hbd864a653e7c4fa03f40ec18;
            6'd9:entry=96'hbda503673e6100823f47bfdf;
            6'd10:entry=96'hbdc5e7783e46b0753f4e53e3;
            6'd11:entry=96'hbde8a0623e2d6fca3f54a40d;
            6'd12:entry=96'hbe066adc3e154c3b3f5aacf1;
            6'd13:entry=96'hbe1917163dfca1c03f606bc8;
            6'd14:entry=96'hbe2c286a3dd10c703f65de72;
            6'd15:entry=96'hbe3f732d3da7e46a3f6b0373;
            6'd16:entry=96'hbe52cd0c3d8130ab3f6fd9eb;
            6'd17:entry=96'hbe660d863d39e7123f74618f;
            6'd18:entry=96'hbe790e583cecabdd3f789aa1;
            6'd19:entry=96'hbe85d5eb3c5e87043f7c85e4;
            6'd20:entry=96'hbe8ee298ba123f073f801248;
            6'd21:entry=96'hbe979e58bc5e119b3f81bc23;
            6'd22:entry=96'hbe9ffbe9bcd060d73f834183;
            6'd23:entry=96'hbea7efb1bd1471fc3f84a390;
            6'd24:entry=96'hbeaf6fbfbd3c72483f85e392;
            6'd25:entry=96'hbeb673c7bd605df73f8702f0;
            6'd26:entry=96'hbebcf519bd80323b3f880324;
            6'd27:entry=96'hbec2ee8fbd8e5b9e3f88e5ba;
            6'd28:entry=96'hbec85c7dbd9ac49b3f89ac4a;
            6'd29:entry=96'hbecd3c9bbda587123f8a5871;
            6'd30:entry=96'hbed18de7bdaebd093f8aebd1;
            6'd31:entry=96'hbed55090bdb680763f8b6807;
            6'd32:entry=96'hbed9d08abdbf806b3f8bf807;
            6'd33:entry=96'hbede1accbdc793b93f8c793c;
            6'd34:entry=96'hbee05c77bdcb96b13f8cb96b;
            6'd35:entry=96'hbee0bce3bdcc39163f8cc391;
            6'd36:entry=96'hbedf69afbdca1a5c3f8ca1a6;
            6'd37:entry=96'hbedc9418bdc5c8b33f8c5c8b;
            6'd38:entry=96'hbed86ec2bdbfc0f03f8bfc0f;
            6'd39:entry=96'hbed32bedbdb86f163f8b86f1;
            6'd40:entry=96'hbeccfc1abdb02f523f8b02f5;
            6'd41:entry=96'hbec60d0cbda74f403f8a74f4;
            6'd42:entry=96'hbebe891abd9e0f513f89e0f5;
            6'd43:entry=96'hbeb696c9bd94a4473f894a44;
            6'd44:entry=96'hbeae5898bd8b38a13f88b38a;
            6'd45:entry=96'hbea5ecfabd81ee043f881ee0;
            6'd46:entry=96'hbe9d6e6fbd71bd023f878de8;
            6'd47:entry=96'hbe94f3b3bd603b7e3f8701dc;
            6'd48:entry=96'hbe889193bd47773d3f863bba;
            6'd49:entry=96'hbe7103e1bd293bb13f8549de;
            6'd50:entry=96'hbe52f3ebbd0e82d63f847417;
            6'd51:entry=96'hbe375656bcee83023f83ba0c;
            6'd52:entry=96'hbe1e58f0bcc6872d3f831a1d;
            6'd53:entry=96'hbe0802eabca47dfe3f8291f8;
            6'd54:entry=96'hbde88164bc87c0963f821f02;
            6'd55:entry=96'hbdc5da8abc5f4b1e3f81be96;
            6'd56:entry=96'hbda7b26bbc3715a03f816e2b;
            6'd57:entry=96'hbd8d9fb4bc15b5fd3f812b6c;
            6'd58:entry=96'hbd6e6bf8bbf4405c3f80f440;
            6'd59:entry=96'hbd4815a4bbc6d0913f80c6d1;
            6'd60:entry=96'hbd27721abba183613f80a183;
            6'd61:entry=96'hbd0bc590bb82f9f53f8082fa;
            6'd62:entry=96'hbce8c9bfbb5414933f806a0a;
            6'd63:entry=96'hbcc16c89bb2b724b3f8055b9;
        endcase
        silu64_coeff=code[6] ? {32'd0,(code[7]?32'd0:32'h3f800000)} :
            {entry[95:64],(code[7]?entry[63:32]:entry[31:0])};
    end
endfunction

// GELU tanh PWL, fixed 1/16 spacing on |x|<4. The classification
// travels through the same register as the original activation segment.
function automatic [7:0] gelu64_code(input [31:0] value);
    reg [5:0] index;
    begin
        case(value[30:23])
            123:index=6'd1;
            124:index={5'b00001,value[22]};
            125:index={4'b0001,value[22:21]};
            126:index={3'b001,value[22:20]};
            127:index={2'b01,value[22:19]};
            128:index={1'b1,value[22:18]};
            default:index=0;
        endcase
        gelu64_code={value[31],value[30:0]>=31'h40800000,index};
    end
endfunction

function automatic [63:0] gelu64_coeff(input [7:0] code);
    reg [95:0] entry;
    begin
        entry=0;
        case(code[5:0])
            6'd0:entry=96'h000000003ef33e003f066100;
            6'd1:entry=96'hbb4b53503ed9d3963f131635;
            6'd2:entry=96'hbc174d523ec0b5773f1fa545;
            6'd3:entry=96'hbc9588193ea814fc3f2bf582;
            6'd4:entry=96'hbcf556ea3e9021483f37ef5c;
            6'd5:entry=96'hbd346e823e720cec3f437cc5;
            6'd6:entry=96'hbd76bb403e45d9c33f4e898f;
            6'd7:entry=96'hbda009463e1bf1063f5903bf;
            6'd8:entry=96'hbdc7696f3de921b83f62dbc9;
            6'd9:entry=96'hbdf0a1f53d9fd9ae3f6c04ca;
            6'd10:entry=96'hbe0d689e3d38b5aa3f7474a5;
            6'd11:entry=96'hbe228b133c76fa983f7c2416;
            6'd12:entry=96'hbe374addbc43ab8b3f818757;
            6'd13:entry=96'hbe4b4038bd132ca63f849965;
            6'd14:entry=96'hbe5e0b77bd6916ea3f8748b7;
            6'd15:entry=96'hbe6f5702bd9970e33f89970e;
            6'd16:entry=96'hbe7ed8fabdb874d23f8b874d;
            6'd17:entry=96'hbe862a33bdd1d59e3f8d1d5a;
            6'd18:entry=96'hbe8bccffbde5df8e3f8e5df9;
            6'd19:entry=96'hbe90443bbdf4ea743f8f4ea7;
            6'd20:entry=96'hbe93863bbdff57413f8ff574;
            6'd21:entry=96'hbe95900fbe02c6cb3f9058d9;
            6'd22:entry=96'hbe96651fbe03fcb43f907f96;
            6'd23:entry=96'hbe960ea6be0384643f90708c;
            6'd24:entry=96'hbe949b07be0194e63f90329d;
            6'd25:entry=96'hbe921d1dbdfcc8bc3f8fcc8c;
            6'd26:entry=96'hbe8eab69bdf44e643f8f44e6;
            6'd27:entry=96'hbe8a5f4ebdea1ebb3f8ea1ec;
            6'd28:entry=96'hbe85543fbdde97be3f8de97c;
            6'd29:entry=96'hbe7f4df5bdd210883f8d2108;
            6'd30:entry=96'hbe72e99dbdc4d8b23f8c4d8b;
            6'd31:entry=96'hbe65b5d8bdb737e73f8b737e;
            6'd32:entry=96'hbe57eb9bbda96da93f8a96db;
            6'd33:entry=96'hbe49c15fbd9bb1503f89bb15;
            6'd34:entry=96'hbe3b6a44bd8e32283f88e322;
            6'd35:entry=96'hbe2d155abd8117b93f88117c;
            6'd36:entry=96'hbe1eed1abd6904543f874823;
            6'd37:entry=96'hbe111703bd5115793f8688ac;
            6'd38:entry=96'hbe03b367bd3a88a83f85d445;
            6'd39:entry=96'hbdedbaadbd2578243f852bc1;
            6'd40:entry=96'hbdd5555bbd11f3e33f848f9f;
            6'd41:entry=96'hbdbe587fbd0002d33f840017;
            6'd42:entry=96'hbda8dc2dbcdf48353f837d21;
            6'd43:entry=96'hbd94ef98bcc1a0a53f830683;
            6'd44:entry=96'hbd8299d9bca6f5613f829bd6;
            6'd45:entry=96'hbd63b59cbc8f24403f823c91;
            6'd46:entry=96'hbd45580dbc74091c3f81e812;
            6'd47:entry=96'hbd2a0371bc4ed1d03f819da4;
            6'd48:entry=96'hbd119765bc2e41bf3f815c83;
            6'd49:entry=96'hbcf7d916bc11f4e23f8123ea;
            6'd50:entry=96'hbcd1aca0bbf30d043f80f30d;
            6'd51:entry=96'hbcb0490ebbc926a83f80c927;
            6'd52:entry=96'hbc9349d1bba5765c3f80a576;
            6'd53:entry=96'hbc749177bb8744bf3f808745;
            6'd54:entry=96'hbc49c09bbb5bcada3f806de5;
            6'd55:entry=96'hbc255b34bb3170c83f8058b8;
            6'd56:entry=96'hbc06a3a4bb0e55da3f80472b;
            6'd57:entry=96'hbbd9cddebae2e0033f8038b8;
            6'd58:entry=96'hbbaefcf3bab3a1343f802ce8;
            6'd59:entry=96'hbb8ba25cba8d479d3f802352;
            6'd60:entry=96'hbb5d506aba5cbdd73f801b98;
            6'd61:entry=96'hbb2e2a6dba2b463f3f801569;
            6'd62:entry=96'hbb081881ba03f9f03f80107f;
            6'd63:entry=96'hbad32d3ab9c9f82a3f800ca0;
        endcase
        gelu64_coeff=code[6] ? {32'd0,(code[7]?32'd0:32'h3f800000)} :
            {entry[95:64],(code[7]?entry[63:32]:entry[31:0])};
    end
endfunction

// E4M3 RNE saturating quantisation of one FP32 value with an E8M0 scale
// (same arithmetic as rc_mxfp8_requant2).
function automatic [7:0] quant_e4m3(input [31:0] v, input [7:0] scale);
    reg sign; reg [7:0] ef; reg [22:0] fr; integer se, te, msb, i3;
    reg [23:0] sig; reg [24:0] rounded; reg [7:0] code;
    begin
        sign = v[31]; ef = v[30:23]; fr = v[22:0];
        if (ef == 8'hff && fr != 0) quant_e4m3 = 8'h7f;
        else if (v[30:0] == 0) quant_e4m3 = {sign, 7'd0};
        else if (scale == 8'hff) quant_e4m3 = 8'h7f;
        else if (ef == 8'hff) quant_e4m3 = {sign, 7'h7e};
        else begin
            if (ef == 0) begin
                msb = 0; for (i3 = 0; i3 < 23; i3 = i3 + 1) if (fr[i3]) msb = i3;
                se = msb - 149; sig = {1'b0, fr} << (23 - msb);
            end else begin se = ef - 127; sig = {1'b1, fr}; end
            te = se - scale + 127;
            if (te > 8) code = 8'h7e;
            else if (te >= -6) begin
                rounded = rne(sig, 20);
                if (rounded >= 16) begin rounded = 8; te = te + 1; end
                if (te > 8 || (te == 8 && rounded > 14)) code = 8'h7e;
                else code = (te + 7) * 8 + rounded - 8;
            end else begin
                rounded = rne(sig, 14 - te);
                code = rounded >= 8 ? 8'h08 : rounded[7:0];
            end
            quant_e4m3 = {sign, code[6:0]};
        end
    end
endfunction
function automatic [24:0] rne(input [23:0] value, input integer shift);
    reg [47:0] q, remv, halfv;
    begin
        if (shift <= 0) rne = {1'b0, value} << -shift;
        else if (shift > 30) rne = 0;
        else begin
            q = value >> shift; remv = value & ((48'd1 << shift) - 1); halfv = 48'd1 << (shift - 1);
            if (remv > halfv || (remv == halfv && q[0])) q = q + 1;
            rne = q[24:0];
        end
    end
endfunction
