"""Discover and validate contributed benchmark dataset manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DATASETS_ROOT = Path(__file__).resolve().parent
REQUIRED_FIELDS = {
    "id",
    "title",
    "status",
    "model_family",
    "source_type",
    "observation_type",
    "sample_rate_hz",
    "inputs",
    "outputs",
}
ALLOWED_STATUS = {"provisional", "archived"}


def dataset_dir(dataset_id: str) -> Path:
    return DATASETS_ROOT / dataset_id


def manifest_path(dataset_id: str) -> Path:
    return dataset_dir(dataset_id) / "dataset.json"


def load_manifest(dataset_id: str) -> dict[str, Any]:
    path = manifest_path(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"missing dataset manifest: {path}")
    return json.loads(path.read_text())


def discover_manifests(root: Path = DATASETS_ROOT) -> list[dict[str, Any]]:
    manifests = []
    for path in sorted(root.glob("*/dataset.json")):
        manifests.append(json.loads(path.read_text()))
    return manifests


def source_url(manifest: dict[str, Any]) -> str | None:
    canonical = manifest.get("canonical_archive") or {}
    temporary = manifest.get("temporary_source") or {}
    return canonical.get("url") or temporary.get("url")


def validate_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(manifest))
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    status = manifest.get("status")
    if status not in ALLOWED_STATUS:
        errors.append(f"status must be one of {sorted(ALLOWED_STATUS)}, got {status!r}")
    if status == "archived" and not manifest.get("canonical_archive"):
        errors.append("archived datasets must define canonical_archive")
    if status == "provisional" and not manifest.get("temporary_source"):
        errors.append("provisional datasets must define temporary_source")
    if not source_url(manifest):
        errors.append("dataset must define a downloadable source URL")
    for key in ("inputs", "outputs"):
        if key in manifest and not isinstance(manifest[key], list):
            errors.append(f"{key} must be a list")
    return errors

