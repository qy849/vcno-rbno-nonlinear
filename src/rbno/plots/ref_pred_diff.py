"""Reference, RBNO prediction, and difference for one test sample (``fig:ref_pred_diff_*``).

Port of ``notebooks/evals_plot/{diffusion_reaction,nonlinear_poisson,nonlinear_elasticity}
_ref_pred_diff.ipynb`` of the original repository. The finite element solution (reference)
and the RBNO prediction of the homogeneous parts ``u^o`` and ``sigma^o`` are evaluated at
the mesh vertices and drawn with ``tricontourf`` (100 levels, turbo); reference and
prediction share one colorbar, the difference has its own (plain decimals). One figure per
component, with panels at fixed positions so that all figures of a benchmark have the same
geometry:

* square domains: Reference | Prediction | Difference side by side on an 18 x 5 in canvas;
* the 2 x 1 hyperelastic domain: the three panels stacked on a canvas 10 in wide whose
  height follows from the domain aspect; titles 25 pt for ``u``, 40 pt for ``sigma``.

    python -m rbno.plots.ref_pred_diff configs/quasilinear.yaml --row 8000 --out figures/ref_pred_diff/quasilinear
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np

from ..config import Experiment
from ..problems import build_problem
from ..utils import REPO_ROOT, evaluate_at_points
from .style import plt


def _side_by_side(x, y, ref, pred, diff, name, fmt, levels=100, tick_fs=15, cbar_fs=15, figsize=(18, 5)):
    """Reference | Prediction (shared colorbar) | Difference (own colorbar), fixed positions in inches."""
    fig_w, fig_h = figsize
    LEFT, BOTTOM, TOP, RIGHT = 0.67, 0.16, 0.315, 1.10  # BOTTOM: room for the lowest colorbar label
    GAP_PANELS, GAP_CBAR, CBAR_W = 0.747, 0.105, 0.23
    panel = fig_h - BOTTOM - TOP
    gap_mid = fig_w - (LEFT + 3 * panel + GAP_PANELS + 2 * (GAP_CBAR + CBAR_W) + RIGHT)
    x0 = LEFT
    x1 = x0 + panel + GAP_PANELS
    cb0 = x1 + panel + GAP_CBAR
    x2 = cb0 + CBAR_W + gap_mid
    cb1 = x2 + panel + GAP_CBAR

    def box(x_in, w_in):
        return [x_in / fig_w, BOTTOM / fig_h, w_in / fig_w, panel / fig_h]

    vmin, vmax = min(ref.min(), pred.min()), max(ref.max(), pred.max())
    fig = plt.figure(figsize=figsize)
    axs = [fig.add_axes(box(v, panel)) for v in (x0, x1, x2)]
    axs[0].tricontourf(x, y, ref, levels=levels, cmap="turbo", vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Reference {name}", fontsize=18)
    cf1 = axs[1].tricontourf(x, y, pred, levels=levels, cmap="turbo", vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Prediction {name}", fontsize=18)
    cb = fig.colorbar(cf1, cax=fig.add_axes(box(cb0, CBAR_W)), format=fmt)
    cb.ax.tick_params(labelsize=cbar_fs)
    cb.locator = ticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    cf2 = axs[2].tricontourf(x, y, diff, levels=levels, cmap="turbo")
    axs[2].set_title(f"Difference {name}", fontsize=18)
    cd = fig.colorbar(cf2, cax=fig.add_axes(box(cb1, CBAR_W)))
    cd.ax.tick_params(labelsize=cbar_fs)
    cd.formatter = ticker.ScalarFormatter(useOffset=False)
    cd.formatter.set_scientific(False)
    cd.locator = ticker.MaxNLocator(nbins=5)
    cd.update_ticks()
    for ax in axs:
        ax.tick_params(axis="both", which="major", labelsize=tick_fs, left=False, bottom=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect("equal", adjustable="box")
    return fig


# stacked layout (inches); the canvas height is derived from the domain aspect
FIG_W, LEFT_IN, CBAR_PAD_IN, CBAR_W_IN, LABEL_RESERVE_IN = 10.0, 0.08, 0.16, 0.26, 1.95
TOP_IN, TITLE_IN, GAP_IN, BOTTOM_IN = 0.06, 0.65, 0.35, 0.32


def _stacked(x, y, ref, pred, diff, name, fmt, aspect, levels=100, title_fs=25, tick_fs=22, cbar_fs=22):
    """Reference / Prediction / Difference stacked 3 x 1; Reference and Prediction share a colorbar."""
    vmin, vmax = min(ref.min(), pred.min()), max(ref.max(), pred.max())
    panel_w = FIG_W - LEFT_IN - CBAR_PAD_IN - CBAR_W_IN - LABEL_RESERVE_IN
    panel_h = panel_w / aspect
    fig_h = TOP_IN + 3 * (TITLE_IN + panel_h) + 2 * GAP_IN + BOTTOM_IN
    fig = plt.figure(figsize=(FIG_W, fig_h))
    axs, y_in = [], fig_h - TOP_IN
    for _ in range(3):
        y_in -= TITLE_IN
        axs.append(fig.add_axes([LEFT_IN / FIG_W, (y_in - panel_h) / fig_h, panel_w / FIG_W, panel_h / fig_h]))
        y_in -= panel_h + GAP_IN
    axs[0].tricontourf(x, y, ref, levels=levels, cmap="turbo", vmin=vmin, vmax=vmax)
    axs[0].set_title(f"Reference {name}", fontsize=title_fs)
    cf1 = axs[1].tricontourf(x, y, pred, levels=levels, cmap="turbo", vmin=vmin, vmax=vmax)
    axs[1].set_title(f"Prediction {name}", fontsize=title_fs)
    cf2 = axs[2].tricontourf(x, y, diff, levels=levels, cmap="turbo")
    axs[2].set_title(f"Difference {name}", fontsize=title_fs)
    for ax in axs:
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis="both", which="major", labelsize=tick_fs, left=False, bottom=False)
    left = (LEFT_IN + panel_w + CBAR_PAD_IN) / FIG_W
    width = CBAR_W_IN / FIG_W
    b = axs[1].get_position().y0
    cb = fig.colorbar(cf1, cax=fig.add_axes([left, b, width, axs[0].get_position().y1 - b]), format=fmt)
    cb.ax.tick_params(labelsize=cbar_fs)
    cb.locator = ticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    p2 = axs[2].get_position()
    cd = fig.colorbar(cf2, cax=fig.add_axes([left, p2.y0, width, p2.height]))
    cd.ax.tick_params(labelsize=cbar_fs)
    cd.locator = ticker.MaxNLocator(nbins=5)
    cd.update_ticks()
    fig.canvas.draw()  # fail instead of silently clipping on the fixed canvas
    tb = fig.get_tightbbox(fig.canvas.get_renderer())
    w, h = fig.get_size_inches()
    if tb.x0 < 0 or tb.y0 < 0 or tb.x1 > w or tb.y1 > h:
        raise RuntimeError(f"{name}: content would be clipped by the fixed canvas")
    return fig


def reference_and_prediction(exp: Experiment, row: int) -> tuple[np.ndarray, np.ndarray]:
    valid = np.load(exp.test_dir / "valid_indices.npy")
    k = np.flatnonzero(valid == row)
    if k.size != 1:
        raise KeyError(f"row {row} has no stored test solution in {exp.test_dir}")
    ref = np.asarray(np.load(exp.test_dir / "sigma_u_dof.npy", mmap_mode="r")[k[0]], dtype=np.float64)
    run = exp.results_dir("rbno")
    j = np.flatnonzero(np.load(run / "pred_indices.npy") == row)
    if j.size != 1:
        raise KeyError(f"row {row} is not among the RBNO predictions")
    pred = np.asarray(np.load(run / "pred_sigma_u_dof.npy", mmap_mode="r")[j[0]], dtype=np.float64)
    return ref, pred


def make(exp: Experiment, row: int, out: Path, fmt_u: str, fmt_sigma: str) -> None:
    problem = build_problem(exp)
    mesh = problem.domain
    xy = mesh.geometry.x[:, :2]
    x, y = xy[:, 0], xy[:, 1]
    ref, pred = reference_and_prediction(exp, row)
    fields = {}
    for tag, state in (("ref", ref), ("pred", pred)):
        sigma, u = problem.split_state(state)
        fields[tag] = (evaluate_at_points(mesh, u, xy), evaluate_at_points(mesh, sigma, xy))
    (u_ref, s_ref), (u_pred, s_pred) = fields["ref"], fields["pred"]
    aspect = problem.length_x / problem.length_y
    stacked = aspect > 1.5
    if u_ref.shape[1] == 1:
        u_names = [("u", r"$u^{\circ}$")]
        s_names = [("sigma1", r"$\sigma_1^{\circ}$"), ("sigma2", r"$\sigma_2^{\circ}$")]
    else:
        u_names = [("u1", r"$u_1^{\circ}$"), ("u2", r"$u_2^{\circ}$")]
        s_names = [(f"sigma_{c}", rf"$\sigma_{{{c}}}^{{\circ}}$") for c in ("11", "12", "21", "22")]
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for (vals_ref, vals_pred, names, fmt, sizes) in ((u_ref, u_pred, u_names, fmt_u, (25, 22, 22)),
                                                     (s_ref, s_pred, s_names, fmt_sigma, (40, 35, 35))):
        for c, (fname, label) in enumerate(names):
            r, p = vals_ref[:, c], vals_pred[:, c]
            if stacked:
                fig = _stacked(x, y, r, p, p - r, label, fmt, aspect, title_fs=sizes[0], tick_fs=sizes[1],
                               cbar_fs=sizes[2])
                fig.savefig(out / f"{fname}_ref_pred_diff_{row}.png", dpi=300)
            else:
                fig = _side_by_side(x, y, r, p, p - r, label, fmt)
                fig.savefig(out / f"{fname}_ref_pred_diff_{row}.png", dpi=300)
            plt.close(fig)
            summary.append(f"{fname}: max|ref| {np.abs(r).max():.3e}, max|diff| {np.abs(p - r).max():.3e}")
    print(f"reference/prediction/difference of row {row} -> {out}\n  " + "\n  ".join(summary))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--row", type=int, required=True, help="global parameter row of a test sample")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt-u", default="%.1f")
    ap.add_argument("--fmt-sigma", default="%.1f")
    args = ap.parse_args()
    out = Path(args.out)
    make(Experiment.load(args.config), args.row, out if out.is_absolute() else REPO_ROOT / out, args.fmt_u, args.fmt_sigma)


if __name__ == "__main__":
    main()
