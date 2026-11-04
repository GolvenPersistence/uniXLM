"""Restore a historical source snapshot from the shared base and changed files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_join(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or any(":" in p for p in path.parts):
        raise ValueError(f"Invalid snapshot path: {name}")
    return root.joinpath(*path.parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=["production", "server_p280_rms_norm_mul_fuse", "mtmd_p225_image_encode_cache"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true", help="Restore only recovered files of the truncated snapshot")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    variant = root / "legacy/kv260/variants" / args.variant
    manifest = json.loads((variant / "snapshot.json").read_text(encoding="utf-8"))
    if manifest["partial_archive_subtree"] and not args.allow_partial:
        parser.error("This snapshot is partial. Use --allow-partial to restore its recovered subset.")
    target = args.output.resolve()
    if target.exists():
        parser.error("The output directory must not exist.")
    sources = {}
    for name, expected in manifest["files"].items():
        delta = safe_join(variant / "files", name)
        source = delta if delta.is_file() else safe_join(root / manifest["base"], name)
        if not source.is_file() or digest(source) != expected:
            raise RuntimeError(f"Missing or damaged snapshot file: {name}")
        sources[name] = source
    target.mkdir(parents=True)
    for name, source in sources.items():
        destination = safe_join(target, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    print(f"Restored {len(sources)} files to {target}")
    if manifest["partial_archive_subtree"]:
        print("Status: recovered subset of an incomplete input archive")


if __name__ == "__main__":
    main()
