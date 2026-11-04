"""Verify distributed source files and report external RTL dependencies."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true", help="Also fail if external RTL files are absent")
    parser.add_argument("--json", type=Path, help="Save the check report")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    failed = []
    checked = 0
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        path = root / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failed.append(name)
        checked += 1
    syntax_errors = []
    scripts = [*root.joinpath("tools").glob("*.py"), *root.joinpath("gen3_sw").rglob("*.py")]
    for path in scripts:
        try:
            compile(path.read_bytes(), str(path), "exec")
        except SyntaxError as error:
            syntax_errors.append({"path": path.relative_to(root).as_posix(), "line": error.lineno, "error": error.msg})
    filelist = root / "gen3_rtl/filelists/g3_node.f"
    rtl_paths = [line.strip() for line in filelist.read_text(encoding="utf-8").splitlines()
                 if line.strip() and not line.lstrip().startswith(("#", "//", "+"))]
    missing = [name for name in rtl_paths if not (root / name).is_file()]
    report = {"hash_files_checked": checked, "hash_failures": failed, "python_files_checked": len(scripts),
              "python_syntax_errors": syntax_errors, "rtl_filelist_entries": len(rtl_paths),
              "missing_external_rtl": missing,
              "partial_history": ["mtmd_p225_image_encode_cache"],
              "status": "pass" if not failed and not syntax_errors else "failed"}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if failed or syntax_errors else (2 if args.strict and missing else 0))


if __name__ == "__main__":
    main()
