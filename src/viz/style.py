"""Shared figure style: one semantic color mapping, fixed sizes, save() helper."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import ROOT  # noqa: E402

OUT = ROOT / "docs" / "figures"
SUMMARIES = ROOT / "reports" / "summaries"

# validated categorical slots (dataviz reference palette, light mode), fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
GRAY, LIGHTGRAY = "#8a8985", "#c9c8c3"

C = {"model1": BLUE, "model3": ORANGE, "persistence": AQUA, "climatology": GRAY, "observed": LIGHTGRAY,
     "train": BLUE, "val": ORANGE, "test": AQUA,
     "MI": BLUE, "OH": ORANGE, "IN": AQUA}
LS = {"model": "-", "persistence": ":", "climatology": "--"}
FULL, HALF = 8.0, 4.0
DPI = 200


def lead_color(lead: int) -> str:
    return C["model1"] if int(lead) == 1 else C["model3"]


def apply() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelcolor": INK2, "axes.edgecolor": LIGHTGRAY, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2.0,
        "legend.frameon": False, "legend.fontsize": 9, "figure.dpi": 100,
    })


def save(fig, name: str, out: Path | None = None) -> Path:
    out = Path(out or OUT)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    if path.stat().st_size > 1_200_000:
        raise RuntimeError(f"{name} is {path.stat().st_size / 1e6:.1f} MB (> 1.2 MB budget)")
    print(f"  wrote {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}", flush=True)
    return path
