#!/bin/sh
# On PYNQ-ZU (~/gen3_v3): the AICAS evaluation with the Gen3 Node backend. llama-server (built in
# gen3_sw/llama.cpp-build-board with GGML_GEN3_BOARD) runs SmolVLM2 on the Node; AICAS's own clients measure it:
# throughput_eval.py (one image, a long prompt) and acc_eval.py (OCRBench sampled, 100 items).
#   sh gen3_sw/board/run_eval.sh BIT [throughput|accuracy|both] [model] [mmproj]
# Defaults: the model AICAS reports with (SmolVLM2-500M-Video-Instruct-Q8_0-pruned, the same text model shape as
# the 256M one). Results in ~/gen3_v3/runs/eval_<time>/.
set -e
BIT=${1:-$HOME/gen3_v3/bit/gen3_node_v3.bit}
MODE=${2:-both}
MODELS=$HOME/llama.cpp_mod/models
MODEL=${3:-$MODELS/SmolVLM2-500M-Video-Instruct-Q8_0-pruned.gguf}
MMPROJ=${4:-$MODELS/mmproj-SmolVLM2-500M-Video-Instruct-Q8_0-pruned.gguf}
BIN=$HOME/gen3_v3/gen3_sw/llama.cpp-build-board/bin
AICAS=$HOME/llama.cpp_mod/AICAS
DATA=$HOME/llama.cpp_mod/data
RUN=$HOME/gen3_v3/runs/eval_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN"
sudo -n env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 $HOME/gen3_v3/gen3_sw/board/load_overlay.py \
    "$BIT" > "$RUN/overlay.log" 2>&1
sudo -n pkill -x llama-server 2>/dev/null || true
sudo -n env GGML_GEN3_DEVICE=board ${GGML_GEN3_PROFILE:+GGML_GEN3_PROFILE=1} $BIN/llama-server -m "$MODEL" \
    --mmproj "$MMPROJ" --host 127.0.0.1 --port 8080 -c 2048 -np 1 -ngl 99 -fa off -t 4 \
    --slot-prompt-similarity 0 > "$RUN/server.log" 2>&1 &
cleanup() { sudo -n pkill -x llama-server 2>/dev/null || true; }
trap cleanup EXIT
i=0
until curl -fsS http://127.0.0.1:8080/health > "$RUN/health.json" 2>/dev/null; do
    i=$((i + 1))
    if [ $i -gt 600 ]; then tail -60 "$RUN/server.log" >&2; exit 3; fi
    sleep 1
done
if [ "$MODE" != accuracy ]; then
    "$AICAS/.venv/bin/python" "$AICAS/throughput_eval.py" -i "$AICAS/image.png" -o "$RUN/throughput_metrics.json" \
        > "$RUN/throughput_stdout.log" 2>&1
    cat "$RUN/throughput_metrics.json"
fi
if [ "$MODE" != throughput ]; then
    "$AICAS/.venv/bin/python" "$AICAS/acc_eval.py" --image_folder "$DATA" --output_folder "$RUN" \
        --OCRBench_file "$AICAS/sampled.json" --save_name SmolVLM2_GEN3 > "$RUN/accuracy_stdout.log" 2>&1
    tail -30 "$RUN/accuracy_stdout.log"
fi
echo "$RUN"
