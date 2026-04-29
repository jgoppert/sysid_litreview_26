#!/usr/bin/env python3
"""Generate longitudinal 3-DOF aircraft training and validation trials."""

from __future__ import annotations

import argparse
from pathlib import Path

from longitudinal import SimulationConfig, write_dataset


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "data" / "longitudinal_3dof_nonlinear"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="directory for generated dataset files")
    parser.add_argument("--train-trials", type=int, default=64, help="number of training trajectories")
    parser.add_argument("--validation-trials", type=int, default=16, help="number of validation trajectories")
    parser.add_argument("--duration", type=float, default=40.0, help="trajectory duration in seconds")
    parser.add_argument("--dt", type=float, default=0.01, help="locked mocap sample time in seconds; must be 0.01")
    parser.add_argument("--seed", type=int, default=7, help="random seed")
    parser.add_argument(
        "--dataset-mode",
        choices=["open_loop", "sine_sweep", "safe_loop", "open_loop_safe", "sine_sweep_safe", "proprietary_autopilot"],
        default="open_loop",
        help="experiment design used to generate commands and inputs",
    )
    parser.add_argument(
        "--aero-variation",
        type=float,
        default=0.0,
        help="per-trial std dev for hidden aero residual scale; keep 0 for a single common true model",
    )
    parser.add_argument("--mocap-position-noise", type=float, default=0.002, help="1-sigma position noise in meters")
    parser.add_argument("--mocap-attitude-noise", type=float, default=0.0015, help="1-sigma pitch attitude noise in radians")
    parser.add_argument("--mocap-smoothing-window", type=int, default=21, help="samples used for derived-state smoothing")
    parser.add_argument("--process-disturbance", action="store_true", help="enable small colored process disturbances")
    parser.add_argument("--no-process-disturbance", action="store_true", help="disable small colored process disturbances")
    parser.add_argument("--no-plot", action="store_true", help="skip preview plot generation")
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
        mocap_position_noise=args.mocap_position_noise,
        mocap_attitude_noise=args.mocap_attitude_noise,
        mocap_smoothing_window=args.mocap_smoothing_window,
        process_disturbance=args.process_disturbance and not args.no_process_disturbance,
        aero_variation=args.aero_variation,
    )
    write_dataset(args.output, config, make_plot=not args.no_plot)
    print(f"Wrote dataset to {args.output}")
    print(f"  train:      {args.output / 'train.npz'}")
    print(f"  validation: {args.output / 'validation.npz'}")
    print(f"  metadata:   {args.output / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
