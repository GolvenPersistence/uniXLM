"""SmolVLM-256M text layer (32-token prefill, then one decode step) on the emulator, checked against the Python oracle.

Builds the Node program of tb/gen3/run_layer_smolvlm.py (build() only: no simulator), runs `g3emu run`, and compares
every checked tensor (x3 of both steps, x3 * g0 and 1/rms when the down GEMM folds, the final K and V^T caches) with
the chained Python reference (G3_ACC=bf48), byte masks as that script uses them.

    python gen3_sw/emu/tools/layer_check.py [--seed 2026] [--norm0 fold|vru] [--norm1 fold|vru]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ["G3_ACC"] = "bf48"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EMU = HERE.parent
REPO = EMU.parents[1]
sys.path.insert(0, str(REPO / "tb" / "gen2"))
sys.path.insert(0, str(REPO / "tb" / "gen3"))             # first: tb/gen2 has a Gen2 run_layer_smolvlm.py too
import run_layer_smolvlm as layer  # noqa: E402
assert Path(layer.__file__).resolve().parent.name == "gen3", layer.__file__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--norm0", choices=("fold", "vru"), default="fold")
    ap.add_argument("--norm1", choices=("fold", "vru"), default="fold")
    ap.add_argument("--out", default=str(EMU / "out" / "layer"))
    ap.add_argument("--emu", default=str(EMU / "build" / ("g3emu.exe" if sys.platform == "win32" else "g3emu")))
    args = ap.parse_args()
    layer.Model.fold_norm0 = args.norm0 == "fold"
    layer.Model.fold_norm1 = args.norm1 == "fold"
    t0 = time.time()
    prog, checks, tags = layer.build(np.random.default_rng(args.seed))
    out = Path(args.out)
    prog.write(out)
    print(f"program + Python oracle built in {time.time() - t0:.1f} s ({len(prog.recs)} records)")
    t0 = time.time()
    r = subprocess.run([args.emu, "run", str(out), "--out", str(out / "emu_out")], capture_output=True, text=True)
    print(r.stdout.strip(), r.stderr.strip())
    print(f"emulator wall time {time.time() - t0:.2f} s")
    if r.returncode:
        return 1
    hmem = {}
    for line in (out / "emu_out" / "hmem_out.hex").read_text().splitlines():
        a, d = line.split()
        hmem[int(a, 16)] = int(d, 16)
    ok = True
    for name, addr, words, masks in checks:
        bad = sum(1 for j, w in enumerate(words) if (hmem.get((addr >> 5) + j, 0) ^ w) & masks[j])
        ok &= bad == 0
        print(f"{'PASS' if bad == 0 else 'FAIL'} {name:16s} {len(words):5d} words, {bad} differ")
    st = {}
    for line in (out / "emu_out" / "completions.txt").read_text().splitlines():
        _, tag, s, _ = (int(x) for x in line.split())
        if s:
            st[tag] = s
    if st:
        ok = False
        print("non-OK completions:", st)
    print("layer:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
