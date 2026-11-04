"""Install the A53 Q8_0 GEMV hook in the pinned AICAS llama.cpp CPU backend.

Usage: python gen3_sw/patches/apply_cpu_a53.py <llama.cpp checkout>
Apply before building llama.cpp. GGML_GEN3_CPU_A53=0 restores its original path.
This patch is separate from the FPGA backend and does not change model weights.
"""
from pathlib import Path
import argparse
import shutil


def apply(root: Path) -> None:
    source = root / "ggml/src/ggml-cpu/ggml-cpu.c"
    text = source.read_text(encoding="utf-8")
    include = '#include "gen3-a53.h"'
    call = "    if (g3_a53_q8_gemv(params, dst)) return;"
    hook = ("void ggml_compute_forward_mul_mat(\n"
            "        const struct ggml_compute_params * params,\n"
            "              struct ggml_tensor * dst) {")
    if text.count(hook) != 1:
        raise ValueError("Unsupported CPU source: MUL_MAT hook is not unique")
    if text.count(include) > 1 or text.count(call) > 1:
        raise ValueError("Duplicate A53 hook")
    if include not in text:
        anchor = "static bool ggml_compute_forward_mul_mat_regular_attn_fast("
        if anchor not in text:
            anchor = "static void ggml_compute_forward_mul_mat_one_chunk("
        if text.count(anchor) != 1:
            raise ValueError("Unsupported CPU source: include anchor is not unique")
        text = text.replace(anchor, include + "\n\n" + anchor, 1)
    if call not in text:
        text = text.replace(hook, hook + "\n" + call, 1)
    # Validate before writing either file; applying twice produces identical bytes.
    header = Path(__file__).with_name("gen3-a53.h")
    shutil.copyfile(header, source.with_name(header.name))
    source.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("llama_root", type=Path)
    apply(parser.parse_args().llama_root)
    print("Applied A53 CPU GEMV optimization")
