"""Public method plugin API for benchmark contributors."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


REQUIRED_METADATA_FIELDS = {
    "name",
    "entry_point",
    "model_families",
    "observation_types",
    "training_scenarios",
}


@dataclass(frozen=True)
class MethodMetadata:
    name: str
    entry_point: str
    model_families: tuple[str, ...]
    observation_types: tuple[str, ...]
    training_scenarios: tuple[str, ...]
    requires_gpu: bool = False
    heavy: bool = False
    description: str = ""


@dataclass(frozen=True)
class DatasetView:
    """Minimal data object passed to plugin methods.

    The current suite still uses its existing `SplitData` class internally. This
    view is the stable contributor-facing shape that future runners should pass
    to plugins.
    """

    scenario: str
    model_family: str
    observation_type: str
    t: Any
    u_pilot: Any
    y_obs: Any
    x_true: Any | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class FittedMethod:
    """Container returned by a plugin fit call."""

    method_name: str
    model: Any
    metadata: dict[str, Any]


class BenchmarkMethod(Protocol):
    """Protocol implemented by benchmark method plugins."""

    metadata: MethodMetadata

    def fit(self, train_data: DatasetView, config: dict[str, Any]) -> FittedMethod:
        """Fit the method on one training dataset."""

    def rollout(
        self,
        fitted: FittedMethod,
        validation_data: DatasetView,
        config: dict[str, Any],
    ) -> Any:
        """Predict validation states or observations using validation inputs only."""


def load_method_metadata(plugin_dir: Path) -> MethodMetadata:
    """Load and validate `method.json` from a plugin directory."""

    path = plugin_dir / "method.json"
    if not path.exists():
        raise FileNotFoundError(f"missing plugin metadata: {path}")
    raw = json.loads(path.read_text())
    missing = sorted(REQUIRED_METADATA_FIELDS - set(raw))
    if missing:
        raise ValueError(f"{path} is missing required fields: {', '.join(missing)}")
    for key in ["model_families", "observation_types", "training_scenarios"]:
        if not isinstance(raw[key], list) or not all(isinstance(item, str) for item in raw[key]):
            raise TypeError(f"{path}: {key} must be a list of strings")
    return MethodMetadata(
        name=str(raw["name"]),
        entry_point=str(raw["entry_point"]),
        model_families=tuple(raw["model_families"]),
        observation_types=tuple(raw["observation_types"]),
        training_scenarios=tuple(raw["training_scenarios"]),
        requires_gpu=bool(raw.get("requires_gpu", False)),
        heavy=bool(raw.get("heavy", False)),
        description=str(raw.get("description", "")),
    )


def load_method_class(plugin_dir: Path) -> type:
    """Import the plugin entry-point class declared in `method.json`."""

    metadata = load_method_metadata(plugin_dir)
    module_name, _, class_name = metadata.entry_point.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"{plugin_dir}/method.json entry_point must use module:ClassName")
    module_path = plugin_dir / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"missing plugin module: {module_path}")
    spec = importlib.util.spec_from_file_location(f"benchmark_plugin_{metadata.name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import plugin module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    method_class = getattr(module, class_name, None)
    if method_class is None:
        raise AttributeError(f"{module_path} does not define {class_name}")
    return method_class


def validate_method_class(method_class: type) -> None:
    """Check that a plugin class has the minimum benchmark methods."""

    for name in ["fit", "rollout"]:
        value = getattr(method_class, name, None)
        if value is None or not callable(value):
            raise TypeError(f"{method_class.__name__} must define callable {name}()")
