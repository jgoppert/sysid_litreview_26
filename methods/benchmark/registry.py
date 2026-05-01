"""Registry bridge for built-in and plugin benchmark methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .method_api import MethodMetadata, load_method_metadata


BUILTIN_METHOD_TRAINING = {
    "Nominal": "open_loop",
    "Linear-SS": "trim_grid",
    "Model-Stitching": "trim_grid",
    "Subspace-Hankel": "trim_grid",
    "Frequency-Welch": "trim_grid",
    "Frequency-Stitching": "trim_grid",
    "Koopman-EDMD": "aggressive",
    "EquationError-LS": "aggressive",
    "EKF-ParamID": "aggressive",
    "Fisher-UQ": "aggressive",
    "OEM-SS": "aggressive",
    "OEM-MocapOutput": "aggressive",
    "Variational-Mocap": "aggressive",
    "SINDy": "aggressive",
    "Symbolic-Stepwise": "aggressive",
    "GP-CoeffClosure": "aggressive",
    "UDE-Residual": "aggressive",
    "PINN-CoeffClosure": "aggressive",
    "NN-CoeffSurrogate": "aggressive",
    "OEM-HiddenController": "safe_loop",
    "UDE-HiddenControl": "safe_loop",
    "PINN-HiddenElevator": "safe_loop",
}

GPU_BUILTINS = {
    "UDE-Residual",
    "PINN-CoeffClosure",
    "UDE-HiddenControl",
    "PINN-HiddenElevator",
    "NN-CoeffSurrogate",
}

MOCAP_ONLY_BUILTINS = {
    "OEM-MocapOutput",
    "Variational-Mocap",
}

SIX_DOF_BUILTINS = {
    "6DOF-Nominal": ("direct", "mocap"),
    "6DOF-LinearSS": ("direct", "mocap"),
    "6DOF-Model-Stitching": ("direct", "mocap"),
    "6DOF-Subspace-Hankel": ("direct", "mocap"),
    "6DOF-Frequency-Welch": ("direct", "mocap"),
    "6DOF-Frequency-Stitching": ("direct", "mocap"),
    "6DOF-Koopman-EDMD": ("direct", "mocap"),
    "6DOF-EquationError-LS": ("direct", "mocap"),
    "6DOF-EKF-ParamID": ("direct", "mocap"),
    "6DOF-Fisher-UQ": ("direct", "mocap"),
    "6DOF-OEM-SS": ("direct", "mocap"),
    "6DOF-RidgeResidual": ("direct", "mocap"),
    "6DOF-OEM-MocapOutput": ("mocap",),
    "6DOF-Variational-Mocap": ("direct", "mocap"),
    "6DOF-SINDy": ("direct", "mocap"),
    "6DOF-Symbolic-Stepwise": ("direct", "mocap"),
    "6DOF-GP-RBF": ("direct", "mocap"),
    "6DOF-UDE-Residual": ("direct", "mocap"),
    "6DOF-PINN-Closure": ("direct", "mocap"),
    "6DOF-NN-Surrogate": ("direct", "mocap"),
}


def builtin_method_metadata() -> list[MethodMetadata]:
    """Return plugin-style metadata for methods still implemented in comparison_suite.py."""

    methods = []
    for name, training_scenario in BUILTIN_METHOD_TRAINING.items():
        observation_types = ("mocap",) if name in MOCAP_ONLY_BUILTINS else ("direct", "mocap")
        methods.append(
            MethodMetadata(
                name=name,
                entry_point="comparison_suite:builtin",
                model_families=("aircraft3dof",),
                observation_types=observation_types,
                training_scenarios=(training_scenario,),
                requires_gpu=name in GPU_BUILTINS,
                description="Built-in method currently dispatched by methods/comparison_suite.py.",
            )
        )
    for name, observation_types in SIX_DOF_BUILTINS.items():
        methods.append(
            MethodMetadata(
                name=name,
                entry_point="models.aircraft6dof.comparison_suite:builtin",
                model_families=("aircraft6dof",),
                observation_types=observation_types,
                training_scenarios=("aircraft_6dof_mixed",),
                requires_gpu=False,
                description="Built-in 6DOF baseline dispatched by methods/models/aircraft6dof/comparison_suite.py.",
            )
        )
    return methods


def discover_plugin_metadata(plugin_root: Path) -> list[MethodMetadata]:
    """Load metadata for contributor plugins with method.json files."""

    if not plugin_root.exists():
        return []
    methods = []
    for path in sorted(plugin_root.glob("*/method.json")):
        methods.append(load_method_metadata(path.parent))
    return methods


def all_method_metadata(plugin_root: Path) -> list[MethodMetadata]:
    """Return built-in method metadata plus contributor plugin metadata."""

    by_name = {method.name: method for method in builtin_method_metadata()}
    for method in discover_plugin_metadata(plugin_root):
        by_name[method.name] = method
    return [by_name[name] for name in sorted(by_name)]


def metadata_to_dict(method: MethodMetadata) -> dict[str, object]:
    return {
        "name": method.name,
        "entry_point": method.entry_point,
        "model_families": list(method.model_families),
        "observation_types": list(method.observation_types),
        "training_scenarios": list(method.training_scenarios),
        "requires_gpu": method.requires_gpu,
        "description": method.description,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plugin-root", type=Path, default=Path(__file__).resolve().parents[1] / "plugins")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = [metadata_to_dict(method) for method in all_method_metadata(args.plugin_root)]
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
