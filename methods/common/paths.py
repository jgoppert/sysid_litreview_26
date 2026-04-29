"""Common output paths."""

from __future__ import annotations

from pathlib import Path


METHODS_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = METHODS_ROOT / "fig"
RESULTS_DIR = METHODS_ROOT / "results"
TABLE_DIR = METHODS_ROOT / "tables"

