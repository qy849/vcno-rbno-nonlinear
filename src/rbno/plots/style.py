"""Plot settings shared by the paper figures (taken from the original repositories).

Ratio, loss-history panels: 8 x 6 in canvas, dpi 300, title 25 / label 25 / tick 22 /
legend 15 pt (``vcno.eval_metrics`` and ``plot_loss_history.ipynb``). Field pictures:
6 x 5 in, ``tricontourf`` with 100 levels and the ``turbo`` colormap, rcParams of
``visualize_solutions.py``. Stability: 11 x 6 in, ``plot_stability.py``.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIGSIZE = (8, 6)
DPI = 300
TITLE_FS = 25
LABEL_FS = 25
TICK_FS = 22
LEGEND_FS = 15

BLUE = "#4C72B0"
BLUE_EDGE = "#1F3B63"
ORANGE = "#DD8452"
ORANGE_EDGE = "#8C4A1E"
GREEN = "#3f9b6e"
RED = "#C44E52"

PANEL_TITLES = {
    "semilinear_k0": r"Heat conduction, $\kappa=0$",
    "semilinear_k1000": r"Semilinear, $\kappa=10^3$",
    "semilinear": "Semilinear Diffusion-reaction",
    "quasilinear": "Quasilinear Diffusion",
    "hyperelastic": "Compressible Hyperelasticity",
}


def field_rcparams() -> None:
    plt.rcParams.update({
        "axes.titlesize": 25, "axes.labelsize": 16, "xtick.labelsize": 19, "ytick.labelsize": 19,
        "figure.titlesize": 22, "legend.fontsize": 16, "font.size": 16,
    })
