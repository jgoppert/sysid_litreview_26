"""Validate compact benchmark dataset arrays committed under data/."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
REQUIRED_SPLIT_KEYS = {
    "time_s",
    "valid_mask",
    "control_meas",
    "pose_meas",
    "segment_names",
    "control_names",
    "pose_names",
    "system_dof",
    "format_version",
    "dataset_id",
    "split_name",
    "sample_period_s",
    "truth_available",
}
SPLIT_NAMES = ("train", "validation")
POSE_NAMES_BY_DOF = {
    3: ("x_e", "z_u", "theta"),
    6: ("x_e", "y_n", "z_u", "q_w", "q_x", "q_y", "q_z"),
}
DIRECT_STATE_NAMES_BY_DOF = {
    3: ("V", "alpha", "gamma", "q"),
    6: ("x_n", "y_e", "z_d", "u", "v", "w", "q_w", "q_x", "q_y", "q_z", "p", "q", "r"),
}
CONTROL_NAMES = ("thrust", "aileron", "elevator", "rudder")
OPTIONAL_CHANNEL_GROUPS = {
    "accel_meas": ("accel_names", ("a_x_body", "a_y_body", "a_z_body")),
    "gyro_meas": ("gyro_names", ("p", "q", "r")),
    "mag_meas": ("mag_names", ("m_x_body", "m_y_body", "m_z_body")),
    "onboard_pose_est": ("onboard_pose_names", POSE_NAMES_BY_DOF[6]),
}
CURRENT_FORMAT_VERSION = "sysid.timeseries.ragged.v1"


def _string_value(array: np.ndarray) -> str:
    value = np.asarray(array)
    if value.shape == ():
        return str(value.item())
    if value.size == 1:
        return str(value.reshape(-1)[0])
    return ""


def _string_list(array: np.ndarray) -> list[str]:
    return [str(value) for value in np.asarray(array).reshape(-1)]


def _int_value(data: np.lib.npyio.NpzFile, key: str, default: int) -> int:
    if key not in data.files:
        return default
    value = np.asarray(data[key])
    if value.shape == ():
        return int(value.item())
    if value.size == 1:
        return int(value.reshape(-1)[0])
    return default


def _split_path(dataset_id: str, split_name: str) -> Path:
    return DATA_ROOT / f"{dataset_id}_{split_name}.npz"


def _dataset_id_from_flat_path(path: Path) -> str | None:
    for split_name in SPLIT_NAMES:
        suffix = f"_{split_name}.npz"
        if path.name.endswith(suffix):
            return path.name[: -len(suffix)]
    return None


def _split_name_from_flat_path(path: Path) -> str | None:
    for split_name in SPLIT_NAMES:
        if path.name.endswith(f"_{split_name}.npz"):
            return split_name
    return None


def _validate_split(path: Path, expected_dataset_id: str | None = None, expected_split_name: str | None = None) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing split file: {path}"]
    data = np.load(path, allow_pickle=False)
    missing = sorted(REQUIRED_SPLIT_KEYS - set(data.files))
    if missing:
        errors.append(f"{path}: missing keys {missing}")
        return errors
    time_s = data["time_s"]
    valid_mask = data["valid_mask"]
    control_meas = data["control_meas"]
    pose_meas = data["pose_meas"]
    system_dof = _int_value(data, "system_dof", 6)
    pose_names = POSE_NAMES_BY_DOF.get(system_dof)
    direct_state_names = DIRECT_STATE_NAMES_BY_DOF.get(system_dof)
    if pose_names is None:
        errors.append(f"{path}: system_dof must be one of {sorted(POSE_NAMES_BY_DOF)}, got {system_dof}")
        pose_names = POSE_NAMES_BY_DOF[6]
    if direct_state_names is None:
        direct_state_names = DIRECT_STATE_NAMES_BY_DOF[6]
    format_version = _string_value(data["format_version"])
    if format_version != CURRENT_FORMAT_VERSION:
        errors.append(f"{path}: format_version must be {CURRENT_FORMAT_VERSION!r}, got {format_version!r}")
    dataset_id = _string_value(data["dataset_id"])
    split_name = _string_value(data["split_name"])
    if expected_dataset_id is not None and dataset_id != expected_dataset_id:
        errors.append(f"{path}: dataset_id must be {expected_dataset_id!r}, got {dataset_id!r}")
    if expected_split_name is not None and split_name != expected_split_name:
        errors.append(f"{path}: split_name must be {expected_split_name!r}, got {split_name!r}")
    if split_name not in SPLIT_NAMES:
        errors.append(f"{path}: split_name must be one of {list(SPLIT_NAMES)}, got {split_name!r}")
    if time_s.ndim != 2:
        errors.append(f"{path}: time_s must be 2D [segment, sample]")
    if valid_mask.shape != time_s.shape:
        errors.append(f"{path}: valid_mask shape {valid_mask.shape} must match time_s {time_s.shape}")
    if valid_mask.dtype != np.bool_:
        errors.append(f"{path}: valid_mask must be boolean")
    if control_meas.ndim != 3 or control_meas.shape[:2] != time_s.shape:
        errors.append(f"{path}: control_meas must be 3D with leading shape {time_s.shape}")
    if pose_meas.ndim != 3 or pose_meas.shape[:2] != time_s.shape:
        errors.append(f"{path}: pose_meas must be 3D with leading shape {time_s.shape}")
    if _string_list(data["pose_names"]) != list(pose_names):
        errors.append(f"{path}: pose_names must be {list(pose_names)} for system_dof={system_dof}")
    if _string_list(data["control_names"]) != list(CONTROL_NAMES):
        errors.append(f"{path}: control_names must be {list(CONTROL_NAMES)}")
    if "direct_state_meas" in data.files:
        direct_state_meas = data["direct_state_meas"]
        if "direct_state_names" not in data.files:
            errors.append(f"{path}: direct_state_meas requires companion direct_state_names")
        elif _string_list(data["direct_state_names"]) != list(direct_state_names):
            errors.append(f"{path}: direct_state_names must be {list(direct_state_names)} for system_dof={system_dof}")
        if direct_state_meas.ndim != 3 or direct_state_meas.shape[:2] != time_s.shape:
            errors.append(f"{path}: direct_state_meas must be 3D with leading shape {time_s.shape}")
    for array_key, (names_key, expected_names) in OPTIONAL_CHANNEL_GROUPS.items():
        if array_key not in data.files:
            continue
        values = data[array_key]
        if names_key not in data.files:
            errors.append(f"{path}: {array_key} requires companion {names_key}")
            continue
        names = data[names_key]
        if values.ndim != 3 or values.shape[:2] != time_s.shape:
            errors.append(f"{path}: {array_key} must be 3D with leading shape {time_s.shape}")
        if names.ndim != 1 or values.ndim == 3 and names.shape[0] != values.shape[2]:
            errors.append(f"{path}: {names_key} length must match {array_key} channel count")
        elif _string_list(names) != list(expected_names):
            errors.append(f"{path}: {names_key} must be {list(expected_names)}")
    if np.any(valid_mask) and not np.all(np.isfinite(time_s[valid_mask])):
        errors.append(f"{path}: finite time_s values are required where valid_mask is true")
    for segment_index in range(time_s.shape[0]):
        times = time_s[segment_index, valid_mask[segment_index]]
        if len(times) < 2:
            continue
        diffs = np.diff(times)
        if np.any(diffs <= 0.0):
            errors.append(f"{path}: time_s must be strictly increasing in segment {segment_index}")
            continue
        sample_period = float(np.median(diffs))
        if not np.allclose(diffs, sample_period, rtol=1e-4, atol=max(1e-9, sample_period * 1e-6)):
            errors.append(f"{path}: time_s must be fixed-rate within segment {segment_index}")
    return errors


def validate_dataset(path: Path | str) -> list[str]:
    errors: list[str] = []
    path = Path(path)
    if path.exists() and path.is_file():
        dataset_id = _dataset_id_from_flat_path(path)
        split_name = _split_name_from_flat_path(path)
        if dataset_id is None or split_name is None:
            return [f"{path}: committed compact files must be named <dataset_id>_<split>.npz"]
        errors.extend(_validate_split(path, dataset_id, split_name))
        return errors
    if path.exists() and path.is_dir():
        return [f"{path}: committed compact datasets must use flat data/<dataset_id>_<split>.npz files"]
    dataset_id = path.name
    for split in SPLIT_NAMES:
        errors.extend(_validate_split(_split_path(dataset_id, split), dataset_id, split))
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset",
        nargs="*",
        help="dataset id or flat NPZ path; defaults to every data/*_{train,validation}.npz pair",
    )
    parser.add_argument("--allow-empty", action="store_true", help="succeed when no compact datasets are present")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.dataset:
        paths = [Path(item) if Path(item).exists() else DATA_ROOT / item for item in args.dataset]
    else:
        dataset_ids: set[str] = set()
        if DATA_ROOT.exists():
            for path in DATA_ROOT.glob("*.npz"):
                dataset_id = _dataset_id_from_flat_path(path)
                if dataset_id:
                    dataset_ids.add(dataset_id)
        paths = sorted(DATA_ROOT / dataset_id for dataset_id in dataset_ids)
    if not paths and args.allow_empty:
        print("No compact datasets found.")
        return 0
    if not paths:
        print(f"ERROR: no compact datasets found under {DATA_ROOT}")
        return 1
    all_errors: list[str] = []
    for path in paths:
        errors = validate_dataset(path)
        if errors:
            all_errors.extend(errors)
        else:
            print(f"compact dataset ok: {path}")
    for error in all_errors:
        print(f"ERROR: {error}")
    return 1 if all_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
