#!/usr/bin/env python3
"""Run method benchmarks and generate paper-support figures."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "simulate": ROOT / "simulation" / "generate_dataset.py",
    "suite": ROOT / "comparison_suite.py",
    "rates": ROOT / "observation_rate_study.py",
    "oem": ROOT / "output_error" / "oem_benchmark.py",
    "sindy": ROOT / "sindy" / "sindy_benchmark.py",
    "frequency": ROOT / "frequency_domain" / "frequency_benchmark.py",
    "pinn": ROOT / "pinn" / "pinn_benchmark.py",
    "ude": ROOT / "neural_residual" / "ude_benchmark.py",
    "compare": ROOT / "compare.py",
}
TORCH_TARGETS = {"suite", "pinn", "ude"}


def run_command(name: str, extra_args: list[str]) -> int:
    script = COMMANDS[name]
    python = ROOT / ".venv" / "bin" / "python"
    executable = str(python) if name in TORCH_TARGETS and python.exists() else sys.executable
    command = [executable, str(script), *extra_args]
    print("+", " ".join(command))
    return subprocess.run(command, cwd=ROOT.parent).returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target",
        choices=["all", *COMMANDS.keys()],
        help="benchmark target to run",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="extra arguments passed to the selected benchmark",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = [name for name in COMMANDS.keys() if name != "simulate"] if args.target == "all" else [args.target]
    for target in targets:
        returncode = run_command(target, args.args)
        if returncode != 0:
            return returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
