"""Plotting helpers for method comparison figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, path_base: Path, dpi: int = 250) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".svg"))
    fig.savefig(path_base.with_suffix(".png"), dpi=dpi)
    plt.close(fig)

