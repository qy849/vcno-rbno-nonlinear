"""Residual--error figures (``fig:ratios``).

For each benchmark, from ``results/<experiment>/<method>/metrics.npz``:

* ``loss_excess_vs_error.png`` -- ``(eta^2 - eta_h^2)^{1/2}`` against ``E_h`` on log-log axes,
  one colour per experiment; dashed lines have the smallest and largest observed slope
  ``1/chi`` of each experiment;
* ``chi_hist.png``             -- histograms of ``chi``;
* ``eta_vs_error.png``, ``ratio_hist.png`` -- the plain residual ``eta`` against ``E_h`` and the
  histogram of ``E_h / eta`` (format of the recorded figures), for reference.

No titles; axis labels in the notation of eq. (numerical_metrics).

    python -m rbno.plots.ratios semilinear_k0 semilinear_k1000 --out figures/ratios/semilinear
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullFormatter

from ..utils import REPO_ROOT
from . import style
from .style import plt

COLORS = [(style.BLUE, style.BLUE_EDGE), (style.ORANGE, style.ORANGE_EDGE)]


def load(name: str, method: str) -> dict:
    z = np.load(REPO_ROOT / "results" / name / method / "metrics.npz")
    m = z["has_truth"].astype(bool)
    return {k: z[k][m] for k in ("E_h", "eta", "loss_excess", "chi", "ratio_E_over_eta")}


def _finish(ax, xlabel, ylabel, legend: bool):
    ax.set_xlabel(xlabel, fontsize=style.LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=style.LABEL_FS)
    ax.tick_params(axis="both", which="both", labelsize=style.TICK_FS)
    if legend:
        ax.legend(fontsize=style.LEGEND_FS, loc="best")


def scatter(groups, labels, out: Path, *, y_key: str, ylabel: str, ratio_key: str):
    fig, ax = plt.subplots(figsize=style.FIGSIZE, constrained_layout=True)
    xs_all = []
    for (face, edge), d, label in zip(COLORS, groups, labels):
        y = d["eta"] if y_key == "eta" else np.sqrt(np.clip(d["loss_excess"], 0, None))
        ok = np.isfinite(d["E_h"]) & np.isfinite(y) & (y > 0)
        x, y = d["E_h"][ok], y[ok]
        xs_all.append(x)
        ax.scatter(x, y, s=45, color=face, edgecolor=edge, linewidths=0.6, alpha=0.75,
                   label=label if len(groups) > 1 else None)
        r = d[ratio_key][np.isfinite(d[ratio_key]) & (d[ratio_key] > 0)]
        xl = np.array([x.min() / 1.5, x.max() * 1.5])
        for slope in (1.0 / r.max(), 1.0 / r.min()):
            ax.plot(xl, slope * xl, ls="--", lw=2.0, color=face)
    ax.set_xscale("log")
    ax.set_yscale("log")
    for axis in (ax.xaxis, ax.yaxis):  # labels at 1, 2, 5 x 10^k only: no collisions on narrow ranges
        axis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        axis.set_major_formatter(LogFormatterSciNotation(labelOnlyBase=False))
        axis.set_minor_formatter(NullFormatter())
    _finish(ax, r"$E_h$", ylabel, legend=len(groups) > 1)
    fig.savefig(out, dpi=style.DPI)
    plt.close(fig)


def histogram(groups, labels, out: Path, *, key: str, xlabel: str):
    fig, ax = plt.subplots(figsize=style.FIGSIZE, constrained_layout=True)
    vals = [d[key][np.isfinite(d[key])] for d in groups]
    lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
    bins = np.linspace(lo, hi, 11 if len(groups) == 1 else 21)
    for (face, _), v, label in zip(COLORS, vals, labels):
        ax.hist(v, bins=bins, color=face, edgecolor="black", alpha=0.85 if len(groups) == 1 else 0.6,
                label=label if len(groups) > 1 else None)
    _finish(ax, xlabel, "Frequency", legend=len(groups) > 1)
    fig.savefig(out, dpi=style.DPI)
    plt.close(fig)


def make(names, labels, out_dir: Path, method: str = "rbno") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [load(n, method) for n in names]
    scatter(groups, labels, out_dir / "loss_excess_vs_error.png", y_key="excess",
            ylabel=r"$(\eta^2-\eta_h^2)^{1/2}$", ratio_key="chi")
    histogram(groups, labels, out_dir / "chi_hist.png", key="chi", xlabel=r"$\chi$")
    scatter(groups, labels, out_dir / "eta_vs_error.png", y_key="eta", ylabel=r"$\eta$",
            ratio_key="ratio_E_over_eta")
    histogram(groups, labels, out_dir / "ratio_hist.png", key="ratio_E_over_eta", xlabel=r"$E_h/\eta$")
    print(f"ratio figures -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", default="rbno")
    args = ap.parse_args()
    labels = args.labels or [style.PANEL_TITLES.get(n, n) for n in args.experiments]
    out = Path(args.out)
    make(args.experiments, labels, out if out.is_absolute() else REPO_ROOT / out, args.method)


if __name__ == "__main__":
    main()
