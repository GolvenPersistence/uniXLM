#!/bin/sh
# On PYNQ-ZU (~/gen3_v3): load the v3 overlay, then run SmolVLM2 through llama.cpp on the Gen3 Node.
#   sh gen3_sw/board/run_board.sh BIT [cli|mtmd|dump] [prompt] [image]
# dump: the Node's logits of every step to ~/gen3_v3/dump.bin; on the PC, gen3rt_llm --compare dump.bin runs the same
# steps on the emulator and checks them bit for bit (the emulator is too slow for the board's CPU).
set -e
BIT=${1:-$HOME/gen3_v3/bit/gen3_node_v3.bit}
MODE=${2:-cli}
PROMPT=${3:-"The capital of France is"}
MODELS=$HOME/llama.cpp_mod/models
BIN=$HOME/gen3_v3/gen3_sw/llama.cpp-build-board/bin
sudo -n env XILINX_XRT=/usr /usr/local/share/pynq-venv/bin/python3 $HOME/gen3_v3/gen3_sw/board/load_overlay.py "$BIT"
case $MODE in
cli)
    sudo -n GGML_GEN3_DEVICE=board $BIN/llama-cli -m $MODELS/SmolVLM2-256M-Video-Instruct-f16.gguf -p "$PROMPT" \
        -n 16 -ngl 99 -fa off --temp 0 -no-cnv -c 512 -t 4 ;;
mtmd)
    sudo -n GGML_GEN3_DEVICE=board $BIN/llama-mtmd-cli -m $MODELS/SmolVLM2-256M-Video-Instruct-f16.gguf \
        --mmproj $MODELS/mmproj-SmolVLM2-256M-Video-Instruct-f16.gguf --image "$4" -p "$PROMPT" \
        -n 16 -ngl 99 -fa off --temp 0 -c 2048 -t 4 ;;
dump)
    sudo -n $HOME/gen3_v3/gen3_sw/gen3rt/build/gen3rt_llm -m $MODELS/SmolVLM2-256M-Video-Instruct-f16.gguf \
        -p "$PROMPT" -n 8 --ctx 256 --prefill --board --dump $HOME/gen3_v3/dump.bin ;;
esac
