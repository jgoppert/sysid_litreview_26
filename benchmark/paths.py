"""Shared repository paths used by benchmark orchestration."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METHOD_CODE = ROOT / "methods"
WORK = ROOT / "work"
WORK_DATA = WORK / "data"
DATA = ROOT / "data"
DATASETS = ROOT / "datasets"
RESULTS = ROOT / "results"
LATEX = ROOT / "latex"
LATEX_FIG = LATEX / "fig"
LATEX_TABLES = LATEX / "tables"
LATEX_GENERATED = LATEX / "generated"
SITE_DATA = ROOT / "site" / "public" / "data"

SPORTCUB_DATASET_ID = "sportcub_mocap_4_17_26"
