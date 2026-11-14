"""Training records of the baselines (``fig:baseline_training_*``).

Port of ``notebooks/loss_history_plot/plot_{pca_net,fno,rbno_l2_data,rbno_hdiv_h1_data,fc_pino}
_loss_history.ipynb`` of the original repository.

* PCA-Net, FNO, and the two RBNO data-loss ablations (``loss_history.npy``: epoch, training
  loss, validation loss): one 8 x 6 in panel with the training and validation loss per epoch
  on a log axis, green crosses at the running minima of the validation loss (best model
  saved), the final training loss and the lowest validation loss marked and labelled in red
  a few points from the marker (labels are mirrored or the y limits widened to keep them
  inside the axes).
* FC-PINO (``loss_history_data.npy``: step, training, validation data loss;
  ``loss_history_phys.npy``: step, total, residual, boundary, data, validation total,
  validation physics, validation data): a row of three panels, the stage-1 data loss, the
  stage-2 objective, and the stage-2 training components, with the best checkpoint of each
  stage marked.

    python -m rbno.plots.baseline_history quasilinear --out figures/training/baselines/quasilinear
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np

from ..utils import REPO_ROOT
from . import style
from .style import plt

METHODS = {
    "pca_net": "PCA-Net",
    "fno": "FNO",
    "rbno_l2_data": r"RBNO ($\mathbb{L}$ data loss)",
    "rbno_hdiv_h1_data": r"RBNO ($\mathbb{H}$ data loss)",
}
TITLES = {
    "semilinear_k1000": "Semilinear Diffusion-reaction",
    "quasilinear": "Quasilinear Diffusion",
    "hyperelastic": "Compressible Hyperelasticity",
}
TITLE_FS, PANEL_TITLE_FS, ROW_TITLE_FS = 22, 20, 26
TITLE_FS_METHOD = {"rbno_l2_data": 18, "rbno_hdiv_h1_data": 18}  # longer titles, as in the original notebooks
LABEL_FS, TICK_FS, LEGEND_FS, ANNOT_FS = 22, 18, 14, 14
TRAIN_OFFSET, VAL_OFFSET = (-14, -22), (-14, 22)
PAD_FRAC = 0.03
FC_HEIGHT = 5.0  # FC-PINO rows: flatter panels, so a full-width row is not taller than a row of three single panels


def _annotate(ax, value: float, xy, offset, box: bool = False):
    dx, dy = offset
    return ax.annotate(
        f"{value:.2e}", xy=xy, xytext=offset, textcoords="offset points", color="red", fontsize=ANNOT_FS,
        ha="right" if dx < 0 else "left", va="top" if dy < 0 else "bottom",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=1.5) if box else None,
        arrowprops=dict(arrowstyle="->", color="red", lw=1.5, shrinkA=2, shrinkB=8.0))


def _fit_annotations(ax, annotations) -> None:
    """Mirror labels that overrun a side spine, then widen the y limits until all fit."""
    if not annotations:
        return
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    axes_box = ax.get_window_extent(renderer)
    for a in annotations:
        b = a.get_window_extent(renderer)
        dx, dy = a.xyann
        if (b.x0 < axes_box.x0 and dx < 0) or (b.x1 > axes_box.x1 and dx > 0):
            a.set_position((-dx, dy))
            a.set_ha("left" if dx < 0 else "right")
    fig.canvas.draw()
    inv = ax.transData.inverted()
    lo, hi = ax.get_ylim()
    new_lo, new_hi = lo, hi
    for a in annotations:
        b = a.get_window_extent(renderer)
        edges = [float(inv.transform((0.0, y))[1]) for y in (b.y0, b.y1)]
        if all(np.isfinite(edges)) and min(edges) > 0:
            new_lo, new_hi = min(new_lo, *edges), max(new_hi, *edges)
    if new_lo < lo or new_hi > hi:
        span = np.log10(new_hi) - np.log10(new_lo)
        ax.set_ylim(10.0 ** (np.log10(new_lo) - PAD_FRAC * span), 10.0 ** (np.log10(new_hi) + PAD_FRAC * span))


def _finish(ax, xlabel: str, annotations=()) -> None:
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel("Loss", fontsize=LABEL_FS)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, which="both", alpha=0.3)
    _fit_annotations(ax, list(annotations))
    lo, hi = ax.get_ylim()
    if np.log10(hi / lo) < 2.0:  # short range: major ticks at 1, 2, 5 instead of crowded minor labels
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_formatter(ticker.NullFormatter())
        ax.set_ylim(lo, hi)
    ax.legend(fontsize=LEGEND_FS, loc="upper right")


def running_minima(values: np.ndarray) -> list[int]:
    out, best = [], np.inf
    for i, v in enumerate(values):
        if np.isfinite(v) and v < best:
            best = v
            out.append(i)
    return out


def single(name: str, method: str, out: Path) -> None:
    h = np.load(REPO_ROOT / "results" / name / method / "loss_history.npy")
    ep, tr, va = h[:, 0], h[:, 1], h[:, 2]
    saves = running_minima(va)
    fig, ax = plt.subplots(figsize=style.FIGSIZE)
    ax.plot(ep, tr, color="blue", lw=2.5, label="Training loss")
    ax.plot(ep, va, color="orange", lw=2.5, label="Validation loss")
    ax.plot(ep[-1], tr[-1], "o", color="blue", ms=10, label="Final training loss")
    ax.plot(ep[saves], va[saves], "gx", ms=5, label="Best model saved")
    ax.plot(ep[saves[-1]], va[saves[-1]], "go", ms=10, label="Lowest validation loss")
    # the lower marker gets the label below it, the upper one the label above
    t_off, v_off = (TRAIN_OFFSET, VAL_OFFSET) if tr[-1] <= va[saves[-1]] else (VAL_OFFSET, TRAIN_OFFSET)
    ann = [_annotate(ax, tr[-1], (ep[-1], tr[-1]), t_off, box=True),
           _annotate(ax, va[saves[-1]], (ep[saves[-1]], va[saves[-1]]), v_off, box=True)]
    ax.set_title(f"{METHODS[method]}: {TITLES.get(name, name)}", fontsize=TITLE_FS_METHOD.get(method, TITLE_FS))
    _finish(ax, "Epoch", ann)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=style.DPI)
    plt.close(fig)


def fc_pino(name: str, out: Path) -> None:
    run = REPO_ROOT / "results" / name / "fc_pino"
    d = np.load(run / "loss_history_data.npy")
    p = np.load(run / "loss_history_phys.npy")
    i1, i2 = int(np.argmin(d[:, 2])), int(np.argmin(p[:, 5]))
    fig, axes = plt.subplots(1, 3, figsize=(style.FIGSIZE[0] * 3, FC_HEIGHT))

    ax = axes[0]
    ax.plot(d[:, 0], d[:, 1], color="tab:blue", lw=2.0, label="Train loss")
    ax.plot(d[:, 0], d[:, 2], color="tab:orange", lw=2.0, marker="o", ms=3, label="Validation loss")
    ax.plot(d[i1, 0], d[i1, 2], "go", ms=9, label="Best model saved")
    ax.set_title("Stage 1: data loss", fontsize=PANEL_TITLE_FS)
    _finish(ax, "Stage-1 iteration", [_annotate(ax, d[i1, 2], (d[i1, 0], d[i1, 2]), TRAIN_OFFSET, box=True)])

    ax = axes[1]
    ax.plot(p[:, 0], p[:, 1], color="tab:blue", lw=2.0, label="Train loss")
    ax.plot(p[:, 0], p[:, 5], color="tab:orange", lw=2.0, marker="o", ms=3, label="Validation loss")
    ax.plot(p[i2, 0], p[i2, 5], "go", ms=9, label="Best model saved")
    ax.set_title("Stage 2: data loss + physics loss", fontsize=PANEL_TITLE_FS)
    _finish(ax, "Stage-2 iteration", [_annotate(ax, p[i2, 5], (p[i2, 0], p[i2, 5]), TRAIN_OFFSET, box=True)])

    ax = axes[2]
    for col, color, label in ((2, "tab:red", "Residual loss"), (3, "tab:purple", "Boundary loss"), (4, "tab:green", "Data loss")):
        ax.plot(p[:, 0], p[:, col], color=color, lw=2.0, label=label)
    ax.set_title("Stage 2 training loss components", fontsize=PANEL_TITLE_FS)
    _finish(ax, "Stage-2 iteration")

    fig.suptitle(f"FC-PINO: {TITLES.get(name, name)}", fontsize=ROW_TITLE_FS)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=style.DPI, bbox_inches="tight")
    plt.close(fig)


def make(name: str, out: Path) -> None:
    for method in METHODS:
        if (REPO_ROOT / "results" / name / method / "loss_history.npy").exists():
            single(name, method, out / f"{method}.png")
    if (REPO_ROOT / "results" / name / "fc_pino" / "loss_history_phys.npy").exists():
        fc_pino(name, out / "fc_pino.png")
    print(f"baseline loss histories ({name}) -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    make(args.experiment, out if out.is_absolute() else REPO_ROOT / out)


if __name__ == "__main__":
    main()
