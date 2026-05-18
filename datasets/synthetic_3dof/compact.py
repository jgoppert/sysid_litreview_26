"""Write compact 3DOF longitudinal benchmark NPZ datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from simulation.longitudinal import (
    CONTROL_NAMES,
    DIRECT_STATE_NAMES,
    FORMAT_VERSION,
    POSE_NAMES,
    SimulationConfig,
    control_from_command,
    generate_split,
    pose_from_mocap,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "work" / "data" / "longitudinal_3dof_nonlinear_open_loop"


def compact_split(trials: list[object], dataset_id: str, split_name: str) -> dict[str, np.ndarray]:
    time_s = np.broadcast_to(trials[0].t, (len(trials), len(trials[0].t))).copy()
    return {
        "time_s": time_s,
        "valid_mask": np.ones(time_s.shape, dtype=bool),
        "control_meas": np.asarray([control_from_command(trial.u_cmd) for trial in trials]),
        "pose_meas": np.asarray([pose_from_mocap(trial.mocap_meas) for trial in trials]),
        "direct_state_meas": np.asarray([trial.y_meas for trial in trials]),
        "segment_names": np.asarray([f"trial_{index:03d}" for index in range(len(trials))]),
        "control_names": CONTROL_NAMES,
        "pose_names": POSE_NAMES,
        "direct_state_names": DIRECT_STATE_NAMES,
        "truth_available": np.asarray(True),
        "system_dof": np.asarray(3),
        "dataset_id": np.asarray(dataset_id),
        "split_name": np.asarray(split_name),
        "sample_period_s": np.asarray(float(np.median(np.diff(trials[0].t)))),
        "format_version": np.asarray(FORMAT_VERSION),
    }


def write_compact_dataset(output: Path, config: SimulationConfig) -> None:
    output.mkdir(parents=True, exist_ok=True)
    train = generate_split("train", config.train_trials, config, config.seed)
    validation = generate_split("validation", config.validation_trials, config, config.seed + 10_000)
    np.savez_compressed(output / "train.npz", **compact_split(train, output.name, "train"))
    np.savez_compressed(output / "validation.npz", **compact_split(validation, output.name, "validation"))
    print(f"Wrote compact 3DOF dataset to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-trials", type=int, default=64)
    parser.add_argument("--validation-trials", type=int, default=16)
    parser.add_argument("--duration", type=float, default=40.0)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--dataset-mode",
        choices=[
            "open_loop",
            "sine_sweep",
            "aggressive",
            "trim_grid",
            "safe_loop",
            "open_loop_safe",
            "sine_sweep_safe",
            "aggressive_safe",
            "proprietary_autopilot",
        ],
        default="open_loop",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SimulationConfig(
        duration=args.duration,
        dt=args.dt,
        train_trials=args.train_trials,
        validation_trials=args.validation_trials,
        seed=args.seed,
        dataset_mode=args.dataset_mode,
    )
    write_compact_dataset(args.output, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
