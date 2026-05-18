"""Convert Sport Cub step-3 segment CSVs into compact benchmark NPZ files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
WORK_DATA = ROOT / "work" / "data"
DATASET_ID = "sportcub_mocap_4_17_26"
DEFAULT_DATA_ROOT = WORK_DATA / "sportcub_mocap_4_17_26" / "raw"
DEFAULT_OUTPUT = ROOT / "data"

STATE_NAMES = ("x_n", "y_e", "z_d", "u", "v", "w", "q_w", "q_x", "q_y", "q_z", "p", "q", "r")
INPUT_NAMES = ("throttle", "elevator", "aileron", "rudder")
CONTROL_NAMES = ("thrust", "aileron", "elevator", "rudder")
POSE_NAMES = ("x_e", "y_n", "z_u", "q_w", "q_x", "q_y", "q_z")
MOCAP_NAMES = ("x_n", "y_e", "z_d", "q_w", "q_x", "q_y", "q_z")
EULER_NAMES = ("roll", "pitch", "yaw")
NED_TO_ENU = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

TRAIN_SEGMENTS = (
    ("elev_3211_1__elev_001", "elev_3211_1", "elevator", 1),
    ("trim_2__elev_001", "trim_2", "elevator", 1),
    ("elev_3211_3__elev_001", "elev_3211_3", "elevator", 1),
    ("aileron3211__ail_001", "aileron3211", "aileron", 1),
    ("aileron3211__ail_002", "aileron3211", "aileron", 2),
    ("aileron_doub__ail_001", "aileron_doub", "aileron", 1),
    ("rudder_3211__rud_001", "rudder_3211", "rudder", 1),
    ("rudder_3211__rud_002", "rudder_3211", "rudder", 2),
    ("throttle_cub1__thr_001", "throttle_cub1", "throttle", 1),
)
VALIDATION_SEGMENTS = (
    ("elev_3211_3__elev_002", "elev_3211_3", "elevator", 2),
    ("aileron3211__ail_003", "aileron3211", "aileron", 3),
    ("aileron_doub__ail_002", "aileron_doub", "aileron", 2),
    ("rudder_dublet_2__rud_001", "rudder_dublet_2", "rudder", 1),
    ("throttle_cub1__mix_001", "throttle_cub1", "mixed", 1),
)


def segment_path(data_root: Path, case: str, cls: str, index: int) -> Path:
    short = {"elevator": "elev", "aileron": "ail", "rudder": "rud", "throttle": "thr", "mixed": "mix"}[cls]
    return data_root / case / "step3_segments" / cls / f"{case}_{short}_seg_{index:03d}.csv"


def euler_xyz_to_quat_wxyz(euler: np.ndarray) -> np.ndarray:
    """Convert roll/pitch/yaw columns to scalar-first quaternions."""

    half = 0.5 * euler
    cr, cp, cy = np.cos(half[:, 0]), np.cos(half[:, 1]), np.cos(half[:, 2])
    sr, sp, sy = np.sin(half[:, 0]), np.sin(half[:, 1]), np.sin(half[:, 2])
    quat = np.column_stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-12)
    return quat


def rotation_body_to_ned(quat_wxyz: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = quat_wxyz / max(float(np.linalg.norm(quat_wxyz)), 1e-12)
    return np.array(
        [
            [1.0 - 2.0 * (q2**2 + q3**2), 2.0 * (q1 * q2 - q0 * q3), 2.0 * (q1 * q3 + q0 * q2)],
            [2.0 * (q1 * q2 + q0 * q3), 1.0 - 2.0 * (q1**2 + q3**2), 2.0 * (q2 * q3 - q0 * q1)],
            [2.0 * (q1 * q3 - q0 * q2), 2.0 * (q2 * q3 + q0 * q1), 1.0 - 2.0 * (q1**2 + q2**2)],
        ]
    )


def quat_wxyz_from_rotation(matrix: np.ndarray) -> np.ndarray:
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return quat


def pose_ned_to_enu(mocap_ned: np.ndarray) -> np.ndarray:
    pose = np.empty_like(mocap_ned)
    pose[:, 0] = mocap_ned[:, 1]
    pose[:, 1] = mocap_ned[:, 0]
    pose[:, 2] = -mocap_ned[:, 2]
    for index, quat_ned in enumerate(mocap_ned[:, 3:7]):
        pose[index, 3:7] = quat_wxyz_from_rotation(NED_TO_ENU @ rotation_body_to_ned(quat_ned))
    return pose


def read_segment_csv(path: Path, columns: list[str]) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        missing = [column for column in columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path} missing columns: {missing}")
        values = {column: [] for column in columns}
        for row in reader:
            try:
                parsed = [float(row[column]) for column in columns]
            except (TypeError, ValueError):
                continue
            if not all(np.isfinite(parsed)):
                continue
            for column, value in zip(columns, parsed):
                values[column].append(value)
    if not values[columns[0]]:
        raise ValueError(f"{path} has no finite samples")
    return {column: np.asarray(value, dtype=np.float64) for column, value in values.items()}


def load_segment(data_root: Path, spec: tuple[str, str, str, int]) -> dict[str, np.ndarray | str]:
    name, case, cls, index = spec
    path = segment_path(data_root, case, cls, index)
    if not path.exists():
        raise FileNotFoundError(path)
    required = [
        "time_from_start",
        "x_smooth",
        "y_smooth",
        "z_smooth",
        "u",
        "v",
        "w",
        "p",
        "q",
        "r",
        "phi",
        "theta",
        "psi",
        "throttle",
        "elevator",
        "aileron",
        "rudder",
    ]
    data = read_segment_csv(path, required)
    time_s = data["time_from_start"]
    time_s = time_s - time_s[0]
    pos_ned = np.column_stack([data["x_smooth"], data["y_smooth"], data["z_smooth"]])
    pos_ned = pos_ned - pos_ned[0]
    euler = np.column_stack([data["phi"], data["theta"], data["psi"]])
    quat_wxyz = euler_xyz_to_quat_wxyz(euler)
    x = np.full((len(time_s), len(STATE_NAMES)), np.nan, dtype=np.float64)
    x[:, 0:3] = pos_ned
    x[:, 3:6] = np.column_stack([data["u"], data["v"], data["w"]])
    x[:, 6:10] = quat_wxyz
    x[:, 10:13] = np.column_stack([data["p"], data["q"], data["r"]])
    u_cmd = np.column_stack([data["throttle"], data["elevator"], data["aileron"], data["rudder"]])
    control_meas = np.column_stack([data["throttle"], data["aileron"], data["elevator"], data["rudder"]])
    mocap = np.column_stack([pos_ned, quat_wxyz])
    pose = pose_ned_to_enu(mocap)
    return {
        "name": name,
        "time_s": time_s,
        "x": x,
        "u_cmd": u_cmd,
        "control_meas": control_meas,
        "mocap": mocap,
        "pose": pose,
        "euler": euler,
    }


def stack_segments(data_root: Path, specs: tuple[tuple[str, str, str, int], ...], split_name: str) -> dict[str, np.ndarray]:
    segments = [load_segment(data_root, spec) for spec in specs]
    n_segments = len(segments)
    max_len = max(len(segment["time_s"]) for segment in segments)
    time_s = np.full((n_segments, max_len), np.nan, dtype=np.float64)
    valid_mask = np.zeros((n_segments, max_len), dtype=bool)
    x = np.full((n_segments, max_len, len(STATE_NAMES)), np.nan, dtype=np.float64)
    u_cmd = np.full((n_segments, max_len, len(INPUT_NAMES)), np.nan, dtype=np.float64)
    control_meas = np.full((n_segments, max_len, len(CONTROL_NAMES)), np.nan, dtype=np.float64)
    mocap = np.full((n_segments, max_len, len(MOCAP_NAMES)), np.nan, dtype=np.float64)
    pose = np.full((n_segments, max_len, len(POSE_NAMES)), np.nan, dtype=np.float64)
    euler = np.full((n_segments, max_len, len(EULER_NAMES)), np.nan, dtype=np.float64)
    segment_names = []
    for segment_index, segment in enumerate(segments):
        n = len(segment["time_s"])
        segment_names.append(str(segment["name"]))
        time_s[segment_index, :n] = segment["time_s"]
        valid_mask[segment_index, :n] = True
        x[segment_index, :n, :] = segment["x"]
        u_cmd[segment_index, :n, :] = segment["u_cmd"]
        control_meas[segment_index, :n, :] = segment["control_meas"]
        mocap[segment_index, :n, :] = segment["mocap"]
        pose[segment_index, :n, :] = segment["pose"]
        euler[segment_index, :n, :] = segment["euler"]
    return {
        "time_s": time_s,
        "valid_mask": valid_mask,
        "x_meas": x,
        "y_meas": x,
        "direct_state_meas": x,
        "mocap_meas": mocap,
        "pose_meas": pose,
        "u_cmd": u_cmd,
        "control_meas": control_meas,
        "segment_names": np.asarray(segment_names),
        "state_names": np.asarray(STATE_NAMES),
        "direct_state_names": np.asarray(STATE_NAMES),
        "input_names": np.asarray(INPUT_NAMES),
        "control_names": np.asarray(CONTROL_NAMES),
        "pose_names": np.asarray(POSE_NAMES),
        "mocap_names": np.asarray(MOCAP_NAMES),
        "euler_names": np.asarray(EULER_NAMES),
        "euler_meas": euler,
        "truth_available": np.asarray(False),
        "system_dof": np.asarray(6),
        "dataset_id": np.asarray(DATASET_ID),
        "split_name": np.asarray(split_name),
        "sample_period_s": np.asarray(0.01),
        "format_version": np.asarray("sysid.timeseries.ragged.v1"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_root = args.data_root
    nested_data_root = data_root / "Sports_Cub_Data_17April"
    if not (data_root / TRAIN_SEGMENTS[0][1]).exists() and nested_data_root.exists():
        data_root = nested_data_root
    train = stack_segments(data_root, TRAIN_SEGMENTS, "train")
    validation = stack_segments(data_root, VALIDATION_SEGMENTS, "validation")
    np.savez_compressed(args.output / f"{DATASET_ID}_train.npz", **train)
    np.savez_compressed(args.output / f"{DATASET_ID}_validation.npz", **validation)
    print(f"Wrote compact Sport Cub dataset to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
