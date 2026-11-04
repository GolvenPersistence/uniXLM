#!/usr/bin/env bash
# Source this on KV260 only to select the promoted P66 package and artifacts.
# Runtime/performance knobs are applied by the P66 llama-server; do not
# re-export TPU_* or GGML_* tuning flags here.

export KV260_BASELINE_BIN_DIR="${KV260_BASELINE_BIN_DIR:-/home/xilinx/workspace_gyq/bin_p66_opdesc_from_p40_p51_userland_20260517}"

export P66_BASELINE_BITSTREAM_BIN="${P66_BASELINE_BITSTREAM_BIN:-/home/xilinx/workspace_gyq/bitstreams/p66_bpack2_pipeline_dot8dual_300m_20260519_gyqworkspace/tpu_soc_wrapper.bit.bin}"
export P66_BASELINE_BITSTREAM_SHA256="${P66_BASELINE_BITSTREAM_SHA256:-1c1dde2c8d78d7493dd3e97ab8c4fae8b2e57bd7b44fd46a048529a9a87d7f77}"
export P66_BASELINE_R5_FIRMWARE="${P66_BASELINE_R5_FIRMWARE:-r5-ctrl-seq-run-p51-20260514.elf}"
