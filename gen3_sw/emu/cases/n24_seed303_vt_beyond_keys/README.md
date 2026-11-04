# n24_attn_causal_odd, seed 303: stored V^T codes vs the Python oracle

`program.hex` is `tb/gen3/run_node_tb.py` case `n24_attn_causal_odd` built with seed 303
(`python gen3_sw/emu/tools/oracle_check.py --seed 303 --cases n24_attn_causal_odd`). keys = 150, so the last V^T cell
(keys 128..159) holds random codes at keys 150..159 that no query sees.

- `hmem_emu_stored_codes.hex`: `g3emu run` (default): attention uses the stored codes and scales, as the RTL does
  (`g3f_lane_bf.sv`: PV exponent = P scale + stored V^T scale - 278).
- `hmem_emu_oracle_requant.hex`: `g3emu run --attn-requantize-mask 4`: V^T keys >= keys zeroed and re-quantised, as
  `g3lib.attention` / `Gen3Attention.run` do. This equals the Python oracle.

They differ in two O words (byte address 0x2000_05e0 byte 11 and 0x2000_0ba0 byte 3: code 0x81 vs 0x80, values near zero),
because a lower re-quantised scale changes the truncation of `(pdot * c) >> 2`. Running this program on the RTL
settles which one the hardware produces; the emulator expects `hmem_emu_stored_codes.hex`. Caches written by the Node
itself keep keys beyond the token count at code 0, where both agree.
