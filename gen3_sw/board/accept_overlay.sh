#!/usr/bin/env bash
# Run after loading a statically checked overlay. Arguments: IMAGE_DIR CORE_MHZ.
set -euo pipefail
output=${1:?usage: accept_overlay.sh IMAGE_DIR CORE_MHZ}
expected=${2:?usage: accept_overlay.sh IMAGE_DIR CORE_MHZ}
exec 9>/tmp/gen3_board_accept.lock
flock -n 9
cd "$output"
sha256sum -c SHA256SUMS
python3 - qualification.json "$expected" <<'PY'
import json, sys
q = json.load(open(sys.argv[1]))
assert q['static_timing_passed'] is True
assert q['core_mhz'] == float(sys.argv[2]) and q['axi_mhz'] == 300
assert q['wns_ns'] >= 0 and q['whs_ns'] >= 0 and q['bus_skew_min_slack_ns'] >= 0
PY
cd /home/xilinx/gen3_300
model=/home/xilinx/llama.cpp_mod/models/SmolVLM2-256M-Video-Instruct-f16.gguf
printf '%s  %s\n' 59068440ef6ddda7e9731fb29de5d5ab475ea3fe69ba4fc54d4ec3f7bb6d02a8 "$model" | sha256sum -c -
sha256sum -c <<'HASHES'
12b5ad7d87c95660751c5f7dad7629595ad353afb8a37ac046a268a0b103fef7  ref_prefill_8.bin
36e4c0dc5e901e2a2d364664215aa76e5da07000769354ba0a9c211057762f3d  ref_prefill_64.bin
HASHES
for n in 8 64; do
  sudo -n env XILINX_XRT=/usr OMP_NUM_THREADS=4 timeout --kill-after=5s 900s stdbuf -oL \
    ./gen3_sw/gen3rt/build/gen3rt_llm -m "$model" -n "$n" --ctx 256 \
    --prefill --board --profile --compare "ref_prefill_${n}.bin" \
    --dump "$output/board_prefill_${n}.bin" > "$output/board_prefill_${n}.log" 2>&1
  cmp "ref_prefill_${n}.bin" "$output/board_prefill_${n}.bin"
  python3 - "$output/board_prefill_${n}.log" "$expected" <<'PY'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text()
clock = re.search(r'board: core clock ([0-9.]+) MHz', text)
expected = float(sys.argv[2])
assert clock and abs(float(clock.group(1)) - expected) <= expected * .01, 'Measured core frequency mismatch'
print(clock.group(0))
PY
  sha256sum "$output/board_prefill_${n}.bin"
  tail -n 8 "$output/board_prefill_${n}.log"
  echo "BOARD_PREFILL_${n}_PASS"
done
echo BOARD_ACCEPTANCE_PASS
