"""Stability figure (``fig:stability``).

Three columns: finite element solutions on the meshes of the sweep (blue ramp,
coarse -> fine), Gauss-Newton iterates on the finest mesh, RBNO predictions. Horizontal bars
mark the minimum of each group, the dashed red line the minimum over all states.
Only parameters whose finest-mesh solve converged enter columns two and three.

    python -m rbno.plots.stability hyperelastic --out figures/stability/hyperelastic.png
    python -m rbno.plots.stability semilinear_k1000 quasilinear hyperelastic --row \
        --out figures/stability/row.png     # fig:stability: one row, one legend below
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext

from ..utils import REPO_ROOT
from . import style
from .style import plt

RES_COLORS = {32: "#6baed6", 64: "#2171b5", 128: "#08306b"}
NEWTON_COLOR, RBNO_COLOR, MARGIN_COLOR = "#DD8452", "#3f9b6e", "#C44E52"


def _scatter(ax, xs, vals, color):
    ax.scatter(xs, vals, s=16, color=color, alpha=0.45, edgecolor="none", zorder=2)


def _panel(ax, name: str, rng, *, fs: dict) -> tuple[float, list[int]]:
    """Draw the stability panel of one experiment into ``ax``; returns (minimum, resolutions)."""
    df = pd.read_csv(REPO_ROOT / "analysis" / name / "stability" / "lambda_min.csv")
    df = df[np.isfinite(df.lam_min)]
    shown = []
    res = df[df.population == "resolution"]
    resolutions = sorted(int(n) for n in res.n.unique())
    offs = np.linspace(-0.24, 0.24, len(resolutions)) if len(resolutions) > 1 else np.zeros(1)
    n_hf = 0
    for n, off in zip(resolutions, offs):
        v = res[(res.n == n) & (res.converged == 1)].lam_min.to_numpy()
        color = RES_COLORS.get(n, matplotlib.colormaps["Blues"](0.6))
        n_hf += v.size
        if v.size:
            shown.extend(v)
            _scatter(ax, off + rng.uniform(-0.07, 0.07, v.size), v, color)
            ax.scatter(off, v.min(), marker="_", s=fs["bar_small"], color=color, linewidths=2.2, zorder=4)
    top = max(resolutions)
    ok_rows = set(res[(res.n == top) & (res.converged == 1)].row)
    counts = []
    for col, (pop, color) in enumerate((("newton", NEWTON_COLOR), ("rbno", RBNO_COLOR)), start=1):
        v = df[(df.population == pop) & df.row.isin(ok_rows)].lam_min.to_numpy()
        counts.append(v.size)
        if v.size:
            shown.extend(v)
            _scatter(ax, col + rng.uniform(-0.18, 0.18, v.size), v, color)
            ax.scatter(col, v.min(), marker="_", s=fs["bar_large"], color=color, linewidths=2.4, zorder=4)
    c_star = float(np.min(shown))
    ax.set_yscale("log")
    ax.axhline(c_star, color=MARGIN_COLOR, ls="--", lw=1.6, zorder=3)
    lo, hi = ax.get_ylim()
    decades = [10.0 ** e for e in range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
               if lo <= 10.0 ** e <= hi]
    # Drop decade ticks too close to the minimum so their labels do not collide.
    ticks = sorted([t for t in decades if abs(np.log10(t / c_star)) > 0.12] + [c_star])
    e = int(np.floor(np.log10(c_star)))
    fmt = LogFormatterMathtext()
    ax.set_yticks(ticks)
    ax.set_yticklabels([rf"${c_star / 10 ** e:.2f}\times10^{{{e}}}$" if t == c_star else fmt(t) for t in ticks])
    for lbl, t in zip(ax.get_yticklabels(), ticks):
        if t == c_star:
            lbl.set_color(MARGIN_COLOR)
            lbl.set_verticalalignment("bottom")
    ax.set_xlim(-0.6, 2.4)
    ax.set_xticks([0, 1, 2])
    names = (("FE solutions", "Gauss–Newton iterates", "RBNO prediction") if not fs.get("stack")
             else ("FE\nsolutions", "Gauss–Newton\niterates", "RBNO\npredictions"))
    ax.set_xticklabels([f"{t}\n(n={c})" for t, c in zip(names, (n_hf, *counts))], fontsize=fs["xtick"])
    # minor tick labels (shown on axes spanning less than about a decade) must not collide with
    # the label of the minimum
    minor = [t for t in ax.yaxis.get_minorticklocs() if lo <= t <= hi and abs(np.log10(t / c_star)) > 0.1]
    ax.set_yticks(minor, minor=True)
    ax.grid(axis="y", which="major", ls=":", lw=0.6, alpha=0.5)
    ax.tick_params(axis="y", which="both", labelsize=fs["ytick"])
    return c_star, resolutions


def _legend_handles(resolutions: list[int]) -> list[Line2D]:
    handles = [Line2D([0], [0], marker="o", ls="none", color=RES_COLORS.get(n, "C0"), markersize=8,
                      label=rf"$n_x={n}$")
               for n in resolutions]
    handles.append(Line2D([0], [0], marker="_", ls="none", color="0.35", markersize=13,
                          markeredgewidth=2.4, label="minimum of a group"))
    handles.append(Line2D([0], [0], ls="--", lw=1.6, color=MARGIN_COLOR, label="minimum over all states"))
    return handles


SINGLE_FS = {"bar_small": 260, "bar_large": 520, "xtick": 17, "ytick": 22}
ROW_FS = {"bar_small": 160, "bar_large": 320, "xtick": 15, "ytick": 17, "stack": True}


def make(name: str, out: Path, seed: int = 0, title: bool = False) -> float:
    """One experiment, legend on the right."""
    fig, ax = plt.subplots(figsize=(11.0, 6.0), constrained_layout=True)
    c_star, resolutions = _panel(ax, name, np.random.default_rng(seed), fs=SINGLE_FS)
    if title:
        ax.set_title(style.PANEL_TITLES.get(name, name), fontsize=26)
    ax.set_ylabel(r"$\lambda_{\min,h}$", fontsize=26)
    fig.legend(handles=_legend_handles(resolutions), loc="outside center right", fontsize=16, frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"stability figure -> {out} (min {c_star:.3e})")
    return c_star


def make_row(names: list[str], out: Path, seed: int = 0) -> list[float]:
    """Several experiments side by side in one row, one legend below all panels."""
    fig, axes = plt.subplots(1, len(names), figsize=(6.0 * len(names), 5.6), constrained_layout=True)
    mins, resolutions = [], []
    for k, (ax, name) in enumerate(zip(np.atleast_1d(axes), names)):
        c_star, resolutions = _panel(ax, name, np.random.default_rng(seed), fs=ROW_FS)
        mins.append(c_star)
        if k == 0:
            ax.set_ylabel(r"$\lambda_{\min,h}$", fontsize=22)
    fig.legend(handles=_legend_handles(resolutions), loc="outside lower center", ncol=len(resolutions) + 2,
               fontsize=17, frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"stability row figure -> {out} (minima {', '.join(f'{m:.3e}' for m in mins)})")
    return mins


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", action="store_true")
    ap.add_argument("--row", action="store_true", help="all experiments in one row, one legend below")
    args = ap.parse_args()
    out = Path(args.out)
    out = out if out.is_absolute() else REPO_ROOT / out
    if args.row:
        make_row(args.experiments, out)
    else:
        for name in args.experiments:
            make(name, out, title=args.title)


if __name__ == "__main__":
    main()
