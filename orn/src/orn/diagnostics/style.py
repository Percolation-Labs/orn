"""Uniform plot styling.

Single entry point `apply_style()` sets matplotlib rcParams so every figure
in the repo has the same typography and palette. Called at the top of every
plot function in `diagnostics.plots`. Zero new runtime dependencies: this
is standard matplotlib styling, tuned for readable web / paper output.

Design goals:
- Readable at Medium's default image size (roughly 700 px wide).
- Works in black-and-white print (figures relied on shape + position, not
  colour alone).
- Matches a seaborn-like muted palette without the seaborn dependency.
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Muted seaborn-style categorical palette (colorblind-safe).
PALETTE = [
    "#4C72B0", "#DD8452", "#55A467", "#C44E52",
    "#8172B2", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]

# Sequential colormap for heatmaps and spectra.
CMAP_SEQ = "viridis"


_APPLIED = False


def apply_style(reset: bool = False) -> None:
    """Apply the shared matplotlib rcParams. Idempotent."""
    global _APPLIED
    if _APPLIED and not reset:
        return
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",

        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans", "Helvetica Neue", "Helvetica", "Arial", "sans-serif",
        ],
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10,
        "axes.labelweight": "regular",
        "axes.edgecolor": "#555",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "axes.grid": True,
        "axes.axisbelow": True,

        "grid.color": "#bfbfbf",
        "grid.linestyle": "-",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.35,

        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.title_fontsize": 9,

        "xtick.color": "#333",
        "ytick.color": "#333",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })
    _APPLIED = True
