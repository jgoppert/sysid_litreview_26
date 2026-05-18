"""Validate a benchmark method plugin without running the full suite."""

from __future__ import annotations

import argparse
from pathlib import Path

from .method_api import load_method_class, load_method_metadata, validate_method_class


def smoke_plugin(plugin_dir: Path) -> None:
    metadata = load_method_metadata(plugin_dir)
    method_class = load_method_class(plugin_dir)
    validate_method_class(method_class)
    print(f"ok: {metadata.name} ({metadata.entry_point})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plugin_dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    smoke_plugin(args.plugin_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
