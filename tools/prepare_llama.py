"""Create a working llama.cpp tree and connect the Gen3 backend."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="New working tree; defaults to gen3_sw/llama.cpp")
    parser.add_argument("--cpu-a53", action="store_true", help="Apply the optional Cortex-A53 patch")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    target = args.output.resolve() if args.output else root / "gen3_sw/llama.cpp"
    if target.exists():
        parser.error("The output directory must not exist.")
    shutil.copytree(root / "software/kv260", target)
    patches = root / "gen3_sw/patches"
    subprocess.run([sys.executable, str(patches / "apply_ggml_gen3.py"), str(target)], check=True)
    if args.cpu_a53:
        subprocess.run([sys.executable, str(patches / "apply_cpu_a53.py"), str(target)], check=True)
    print(f"Prepared Gen3 llama.cpp working tree: {target}")


if __name__ == "__main__":
    main()
