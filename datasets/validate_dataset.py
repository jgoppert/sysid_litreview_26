"""Validate a contributed benchmark dataset manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .registry import load_manifest, validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="dataset id, or path to a dataset directory / dataset.json")
    return parser.parse_args()


def _load(dataset: str) -> dict:
    path = Path(dataset)
    if path.is_file():
        return json.loads(path.read_text())
    if path.is_dir():
        return json.loads((path / "dataset.json").read_text())
    return load_manifest(dataset)


def main() -> int:
    args = parse_args()
    manifest = _load(args.dataset)
    errors = validate_manifest(manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"dataset manifest ok: {manifest['id']} ({manifest['status']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

