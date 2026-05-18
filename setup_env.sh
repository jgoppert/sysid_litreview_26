#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e ".[gpu]" --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple
.venv/bin/python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
PY
