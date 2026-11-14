"""Parameter sample and finite element solution pictures (``fig:benchmarks``).

Writes ``p_<row>.png``, ``u_<row>[_k].png`` and ``sigma_<row>_<k>.png`` (homogeneous parts
``u0`` and ``sigma0``) in the style of ``scripts/visualize_{parameters,solutions}.py`` of
the original repository: 6 x 5 in panels, ``tricontourf`` on the mesh vertices with 100
levels and the turbo colormap, one colorbar per panel.

    python -m rbno.plots.fields configs/semilinear_k1000.yaml --row 5000 --out figures/benchmarks/semilinear
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.ticker as ticker
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ..config import Experiment
from ..problems import build_problem
from ..utils import REPO_ROOT, evaluate_at_points
from . import style
from .style import plt


def _panel(xy, values, title: str, fmt: str):
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    cf = ax.tricontourf(xy[:, 0], xy[:, 1], values, levels=100, cmap="turbo")
    ax.set_title(title, pad=12)
    ax.set_aspect(1.0, adjustable="box")
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(cf, cax=cax, format=fmt)
    cb.locator = ticker.MaxNLocator(nbins=5)
    cb.update_ticks()
    ax.tick_params(left=False, bottom=False)
    return fig


def make(exp: Experiment, row: int, out: Path, fmt_p: str, fmt_u: str, fmt_sigma: str, dpi: int = 800) -> None:
    style.field_rcparams()
    out.mkdir(parents=True, exist_ok=True)
    problem = build_problem(exp)
    mesh = problem.domain
    xy = mesh.geometry.x[:, :2]
    p_dof = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")[row]
    if exp.problem_type == "semilinear":
        img = problem.parameter_images(p_dof)[0, 0]  # vertex values on the closed squares
        ix = np.rint(xy[:, 0] / problem.length_x * problem.num_x).astype(int)
        iy = np.rint(xy[:, 1] / problem.length_y * problem.num_y).astype(int)
        p_vals = img[iy, ix]
    else:
        p_vals = evaluate_at_points(mesh, problem.p_function(p_dof), xy)[:, 0]
    fig = _panel(xy, p_vals, r"Parameter $\mathfrak{p}$", fmt_p)
    fig.savefig(out / f"p_{row}.png", bbox_inches="tight", pad_inches=0.1, dpi=300)
    plt.close(fig)

    for store in (exp.test_dir, exp.snapshot_dir):
        valid = list(np.load(store / "valid_indices.npy"))
        if row in valid:
            state = np.load(store / "sigma_u_dof.npy", mmap_mode="r")[valid.index(row)]
            break
    else:
        raise KeyError(f"row {row} has no converged solution")
    sigma, u = problem.split_state(state)
    uvals = evaluate_at_points(mesh, u, xy)
    if uvals.shape[1] == 1:
        names = [("", r"Reference solution $u^{\circ}$")]
    else:
        names = [(f"_{k + 1}", rf"Reference solution $u^{{\circ}}_{{{k + 1}}}$") for k in range(uvals.shape[1])]
    for k, (suffix, title) in enumerate(names):
        fig = _panel(xy, uvals[:, k], title, fmt_u)
        fig.savefig(out / f"u_{row}{suffix}.png", bbox_inches="tight", pad_inches=0.1, dpi=dpi)
        plt.close(fig)
    svals = evaluate_at_points(mesh, sigma, xy)
    labels = ["11", "12", "21", "22"] if svals.shape[1] == 4 else ["1", "2"]
    for k, lab in enumerate(labels):
        fig = _panel(xy, svals[:, k], rf"Reference solution $\sigma^{{\circ}}_{{{lab}}}$", fmt_sigma)
        fig.savefig(out / f"sigma_{row}_{lab}.png", bbox_inches="tight", pad_inches=0.1, dpi=dpi)
        plt.close(fig)
    print(f"field pictures of row {row} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--row", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fmt-p", default="%.1f")
    ap.add_argument("--fmt-u", default="%.2f")
    ap.add_argument("--fmt-sigma", default="%.1f")
    args = ap.parse_args()
    out = Path(args.out)
    make(Experiment.load(args.config), args.row, out if out.is_absolute() else REPO_ROOT / out,
         args.fmt_p, args.fmt_u, args.fmt_sigma)


if __name__ == "__main__":
    main()
