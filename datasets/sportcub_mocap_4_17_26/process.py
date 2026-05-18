"""Run the Sport Cub raw-data preparation pipeline."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORK_DATA = ROOT / "work" / "data"
DEFAULT_DATA_ROOT = WORK_DATA / "sportcub_mocap_4_17_26" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--steps", default="1,2,3")
    parser.add_argument("--only-cases", default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.data_root.exists():
        raise SystemExit(f"missing Sport Cub data root: {args.data_root}")
    pipeline_data_root = args.data_root
    nested_data_root = args.data_root / "Sports_Cub_Data_17April"
    pipeline = args.data_root / "v3_sportcub_pipeline.py"
    nested_pipeline = nested_data_root / "v3_sportcub_pipeline.py"
    external_pipeline = Path.home() / "git" / "sport_cub_processing" / "v3_sportcub_pipeline.py"
    if not pipeline.exists() and nested_pipeline.exists():
        pipeline = nested_pipeline
        pipeline_data_root = nested_data_root
    if not pipeline.exists() and external_pipeline.exists():
        pipeline = external_pipeline
        if nested_data_root.exists():
            pipeline_data_root = nested_data_root
    if not pipeline.exists():
        raise SystemExit(
            "missing raw Sport Cub processing pipeline in the ignored data root; "
            "only canonical NPZ artifacts are committed"
        )
    command = [sys.executable, str(pipeline), "--data-root", str(pipeline_data_root), "--steps", args.steps]
    if args.only_cases:
        command.extend(["--only-cases", args.only_cases])
    if args.no_plots:
        command.append("--no-plots")
    print("+", " ".join(command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
