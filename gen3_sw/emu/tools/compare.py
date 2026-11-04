"""Run the emulator on every RTL reference case and compare bit for bit.

For each case directory <ref>/<case>/ holding program.hex, hmem_out.hex and completions.txt (written by the RTL
regression, tb/gen3/run_node_tb.py), runs `g3emu run` and checks
  (1) every external memory word of hmem_out.hex (words that are X placeholders, all 'f', are ignored) is present in
      the emulator's dump with the same value, and the emulator wrote no word the RTL did not;
  (2) per tag, the sequence of completion statuses (and their data fields, the watermark of a WAIT) equals
      completions.txt.

    python gen3_sw/emu/tools/compare.py [--ref .codex_tmp/node_v3_reg] [--cases n1_single ...] [--attn-requantize]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EMU = HERE.parent
REPO = EMU.parents[1]
STATUS = ["OK", "UNSUPPORTED", "RESERVED_NONZERO", "RANGE", "BINDING", "ORDER", "CAPACITY", "POISONED", "ABORTED",
          "EXECUTION", "QUIESCED", "TIMEOUT", "CONTEXT"]


def read_hmem(path: Path) -> tuple[dict, int]:
    words, xs = {}, 0
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        # X / Z words: the RTL bench writes them as 65 'f' (one more digit than a word); raw x / z digits too
        if len(parts[1]) != 64 or any(ch in "xXzZ" for ch in parts[1]):
            xs += 1
            continue
        words[int(parts[0], 16)] = int(parts[1], 16)
    return words, xs


def read_cpl(path: Path) -> dict:
    by_tag = defaultdict(list)
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 4:
            by_tag[int(parts[1])].append((int(parts[2]), int(parts[3])))
    return dict(by_tag)


def case_key(name: str):
    m = re.match(r"n(\d+)_", name)
    return (int(m.group(1)) if m else 10**6, name)


def compare_case(case_dir: Path, out_dir: Path, emu: Path, extra_args: list[str]) -> dict:
    r = subprocess.run([str(emu), "run", str(case_dir), "--out", str(out_dir), "--quiet"] + extra_args,
                       capture_output=True, text=True)
    res = dict(case=case_dir.name, ok=False, run_ok=r.returncode == 0, warnings=[w for w in r.stderr.splitlines() if w])
    if r.returncode != 0:
        res["error"] = r.stderr.strip()[-500:]
        return res
    ref_words, ref_x = read_hmem(case_dir / "hmem_out.hex")
    emu_words, _ = read_hmem(out_dir / "hmem_out.hex")
    bad = [a for a in sorted(ref_words) if emu_words.get(a) != ref_words[a]]
    extra = [a for a in sorted(emu_words) if a not in ref_words]
    ref_cpl = read_cpl(case_dir / "completions.txt")
    emu_cpl = read_cpl(out_dir / "completions.txt")
    st_bad = []
    for tag in sorted(set(ref_cpl) | set(emu_cpl)):
        a = [STATUS[s] + (f"/data {d}" if d else "") for s, d in ref_cpl.get(tag, [])]
        b = [STATUS[s] + (f"/data {d}" if d else "") for s, d in emu_cpl.get(tag, [])]
        if a != b:
            st_bad.append((tag, a, b))
    res.update(words=len(ref_words), x_words=ref_x, words_bad=len(bad), words_extra=len(extra),
               completions=sum(len(v) for v in ref_cpl.values()), status_bad=st_bad)
    if bad:
        a = bad[0]
        res["first_bad"] = dict(word_addr=hex(a), byte_addr=hex(a << 5), ref=f"{ref_words[a]:064x}",
                                emu=f"{emu_words[a]:064x}" if a in emu_words else None,
                                bad_bytes=[i for i in range(32) if a not in emu_words or
                                           (ref_words[a] >> (8 * i)) & 0xFF != (emu_words[a] >> (8 * i)) & 0xFF])
        res["bad_ranges"] = ranges(bad)
    if extra:
        res["extra_ranges"] = ranges(extra)
    res["ok"] = not bad and not extra and not st_bad
    return res


def ranges(addrs: list[int], limit: int = 6) -> list[str]:
    out, start, prev = [], None, None
    for a in addrs:
        if start is None:
            start = prev = a
        elif a == prev + 1:
            prev = a
        else:
            out.append(f"{start:x}-{prev:x}")
            start = prev = a
    if start is not None:
        out.append(f"{start:x}-{prev:x}")
    return out[:limit] + ([f"... {len(out) - limit} more"] if len(out) > limit else [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=str(REPO / ".codex_tmp" / "node_v3_reg"))
    ap.add_argument("--cases", nargs="*")
    ap.add_argument("--emu", default=str(EMU / "build" / ("g3emu.exe" if sys.platform == "win32" else "g3emu")))
    ap.add_argument("--out", default=str(EMU / "out" / "cases"))
    ap.add_argument("--attn-requantize", action="store_true", help="attention inputs as the Python oracle sees them")
    ap.add_argument("--json", help="write the per-case results here")
    args = ap.parse_args()
    ref = Path(args.ref)
    names = args.cases or sorted((p.name for p in ref.iterdir()
                                  if (p / "program.hex").exists() and (p / "hmem_out.hex").exists()
                                  and (p / "completions.txt").exists()), key=case_key)
    extra = ["--attn-requantize"] if args.attn_requantize else []
    results, failed = [], 0
    for name in names:
        res = compare_case(ref / name, Path(args.out) / name, Path(args.emu), extra)
        results.append(res)
        failed += not res["ok"]
        if res["ok"]:
            print(f"PASS {name:24s} words {res['words']:6d}  completions {res['completions']:3d}"
                  + (f"  ({res['x_words']} X words ignored)" if res["x_words"] else ""))
        else:
            keys = ("error", "words_bad", "words_extra", "status_bad", "first_bad", "bad_ranges", "extra_ranges")
            print(f"FAIL {name:24s} " + json.dumps({k: res[k] for k in keys if res.get(k)}))
        for w in res["warnings"][:3]:
            print("     " + w)
    print(f"{len(names) - failed}/{len(names)} cases bit-exact")
    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
