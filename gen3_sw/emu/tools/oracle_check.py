"""Every case of tb/gen3/run_node_tb.py on the emulator, checked against the Python oracle (no RTL needed).

Builds each case's program exactly as the RTL regression does (same seeds), runs `g3emu run`, and checks the words the
regression checks (the oracle's expected tensors at their host addresses) and the expected completion statuses.
Useful for cases without RTL results (e.g. n41_gelu, n42_gelu_decode) and after changes to the Python references.

    python gen3_sw/emu/tools/oracle_check.py [--cases n41_gelu ...] [--seed 11]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ["G3_ACC"] = "bf48"

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
EMU = HERE.parent
REPO = EMU.parents[1]
sys.path.insert(0, str(REPO / "tb" / "gen2"))
sys.path.insert(0, str(REPO / "tb" / "gen3"))
import g2lib  # noqa: E402
import run_node_tb as cases_mod  # noqa: E402

assert Path(cases_mod.__file__).resolve().parent.name == "gen3", cases_mod.__file__


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="*", default=list(cases_mod.CASES))
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=str(EMU / "out" / "oracle"))
    ap.add_argument("--emu", default=str(EMU / "build" / ("g3emu.exe" if sys.platform == "win32" else "g3emu")))
    ap.add_argument("--attn-requantize", action="store_true", help="attention inputs as the Python oracle sees them")
    args = ap.parse_args()
    extra = ["--attn-requantize"] if args.attn_requantize else []
    names = list(cases_mod.CASES)
    failed = 0
    for case in args.cases:
        rng = np.random.default_rng(args.seed + names.index(case))
        prog, checks = cases_mod.CASES[case](rng)
        vec = Path(args.out) / case
        prog.write(vec)
        r = subprocess.run([args.emu, "run", str(vec), "--out", str(vec / "emu_out"), "--quiet"] + extra,
                           capture_output=True, text=True)
        if r.returncode:
            print("FAIL", case, r.stderr.strip()[-300:])
            failed += 1
            continue
        hmem = {}
        for line in (vec / "emu_out" / "hmem_out.hex").read_text().splitlines():
            a, d = line.split()
            hmem[int(a, 16)] = int(d, 16)
        bad = [(hex(addr), j) for addr, words in checks for j, w in enumerate(words) if hmem.get((addr >> 5) + j) != w]
        cpl = {}
        for line in (vec / "emu_out" / "completions.txt").read_text().splitlines():
            _, tag, st, _ = (int(x) for x in line.split())
            cpl[tag] = st
        st_bad = [(t, g2lib.STATUS[s]) for t, s in cpl.items() if g2lib.STATUS[s] != prog.expected_status.get(t, "OK")]
        missing = [t for t in prog.tags if t not in cpl]
        ok = not bad and not st_bad and not missing
        failed += not ok
        words = sum(len(w) for _, w in checks)
        print(("PASS" if ok else "FAIL"), f"{case:24s} checked words {words:5d}",
              "" if ok else f"bad {len(bad)} first {bad[:3]} status {st_bad} missing {missing}",
              r.stderr.strip()[:200])
    print(f"{len(args.cases) - failed}/{len(args.cases)} cases equal to the Python oracle")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
