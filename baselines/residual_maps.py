"""Pointwise residual densities of FC-PINO and the RBNO on one test sample (``fig:residual_maps``).

Port of ``notebooks/pointwise_residual_plot/{fc_pino,rbno}_pointwise_residual_visualization.ipynb``
of the original repository. Three pictures, each with the densities ``r_1^2`` and ``r_2^2``:

* ``fc_pino_grid.png``: the residual FC-PINO is trained on, from the Fourier-continuation
  derivatives of the network on the uniform grid (``GridPhysics.pointwise``; its grid mean
  is the loss in the rows marked with a dagger of ``tab:baseline_fe``);
* ``fc_pino_fe.png`` and ``rbno_fe.png``: the finite element residual ``residual_terms`` of
  the prediction in RT_1 x CG_2, ``inner(r_i, r_i)`` interpolated into CG_1 (sampled at the
  mesh vertices) and placed on the grid.

Gray colormap, colour limits from the 1% and 99% quantiles of each image, mean of the image
in the title; the panels sit side by side on the unit square and are stacked for the 2 x 1
hyperelastic domain. The assembled loss of each prediction is written to ``summary.json``.

    python -m residual_maps semilinear_k1000 --row 5000 --out figures/residual/semilinear_k1000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker as ticker  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import ufl  # noqa: E402
from dolfinx import fem  # noqa: E402

from common import build_context, grid_cache, load_baseline_config, parameter_images, select_rows  # noqa: E402
from fc_pino.__main__ import build_model, coords  # noqa: E402
from fc_pino.residual import GridPhysics, _ip  # noqa: E402
from rbno.utils import REPO_ROOT, save_json  # noqa: E402

QUANTILE_CLIP = (0.01, 0.99)
STACKED = dict(nrows=2, ncols=1, figsize=(11, 11),
               adjust=dict(left=0.04, right=0.9, bottom=0.05, top=0.9, hspace=0.28),
               title_fs=20, title_pad=10, cbar_fraction=0.025, tick_fs=16)
SIDE_BY_SIDE = dict(nrows=1, ncols=2, figsize=(16, 7),
                    adjust=dict(left=0.04, right=0.92, bottom=0.06, top=0.88, wspace=0.06),
                    title_fs=24, title_pad=12, cbar_fraction=0.046, tick_fs=20)


def fc_pino_grid_residual(exp, ctx, row: int, device: str) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """``(r1^2, r2^2)`` images of FC-PINO's own residual, the boundary penalty, and the grid prediction."""
    _, cfg = load_baseline_config(exp.name, "fc_pino")
    torch.set_default_dtype(torch.float64)
    dev = torch.device(device)
    _, rows = grid_cache(exp, "snapshots")
    sup = select_rows(cfg["train"], rows)
    p_sup = torch.from_numpy(parameter_images(ctx, sup).astype(np.float64))
    mean = p_sup.mean(dim=(0, 2, 3), keepdim=True)  # input normalization of training
    std = p_sup.std(dim=(0, 2, 3), keepdim=True) + 1e-8
    model = build_model(ctx, cfg, p_sup.shape[1], dev)
    model.load_state_dict(torch.load(exp.checkpoint_dir("fc_pino") / "best_val.pt", map_location=dev, weights_only=False))
    model.eval()
    physics = GridPhysics(ctx, dev, detf_floor=float(cfg.get("detf_floor", 1e-3)))
    p = torch.from_numpy(parameter_images(ctx, np.array([row])).astype(np.float64))
    with torch.no_grad():
        x = torch.cat([(p - mean) / std, coords(ctx, 1, "cpu")], 1).to(dev)
        pred, d = model(x, derivs_to_compute=["dx", "dy"])
        r1, r2 = physics.pointwise(p.to(dev), pred, d)
        bc = physics.boundary(pred)
    return r1[0, 0].cpu().numpy(), r2[0, 0].cpu().numpy(), float(bc), pred[0].cpu().numpy()


def fe_residual(ctx, state: np.ndarray, p_dof: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    """``(r1^2, r2^2)`` of the FE residual sampled at the mesh vertices, and the assembled loss."""
    P = ctx.problem
    sigma, u = P.split_state(state)
    p = P.p_function(p_dof)
    r1, r2 = P.residual_terms(sigma, u, p)
    ny, nx = ctx.shape
    images = []
    for expr in (ufl.inner(r1, r1), ufl.inner(r2, r2)):
        f = fem.Function(ctx.V1)
        f.interpolate(fem.Expression(expr, _ip(ctx.V1)))
        img = np.zeros(ny * nx)
        img[ctx.flat] = f.x.array
        images.append(img.reshape(ny, nx))
    return images[0], images[1], P.loss_components(sigma, u, p)


def position(exp, method: str, row: int) -> int:
    """Position of the global parameter row ``row`` in the stored predictions of ``method``."""
    hit = np.flatnonzero(np.load(exp.results_dir(method) / "pred_indices.npy") == row)
    if hit.size != 1:
        raise KeyError(f"row {row} is not (uniquely) among the {method} predictions")
    return int(hit[0])


def prediction(exp, method: str, row: int) -> np.ndarray:
    k = position(exp, method, row)
    return np.asarray(np.load(exp.results_dir(method) / "pred_sigma_u_dof.npy", mmap_mode="r")[k], dtype=np.float64)


def quantile_limits(values, lower: float, upper: float) -> tuple[float, float]:
    vmin, vmax = np.quantile(values[np.isfinite(values)], [lower, upper])
    if np.isclose(vmin, vmax):
        delta = max(abs(float(vmin)) * 1e-6, 1e-12)
        vmin, vmax = vmin - delta, vmax + delta
    return float(vmin), float(vmax)


def plot_pair(r1, r2, prefix: str, extent, out: Path) -> None:
    lay = STACKED if extent[1] - extent[0] > 1.5 * (extent[3] - extent[2]) else SIDE_BY_SIDE
    fig, axs = plt.subplots(lay["nrows"], lay["ncols"], figsize=lay["figsize"])
    fig.subplots_adjust(**lay["adjust"])
    for ax, img, label in zip(axs, (r1, r2), (r"$r_1^2$", r"$r_2^2$")):
        vmin, vmax = quantile_limits(img, *QUANTILE_CLIP)
        im = ax.imshow(img, extent=extent, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"{prefix} {label}\n(mean={float(np.mean(img)):.2e})", fontsize=lay["title_fs"], pad=lay["title_pad"])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, fraction=lay["cbar_fraction"], pad=0.02)
        cb.locator = ticker.MaxNLocator(nbins=4)
        fmt = ticker.ScalarFormatter(useMathText=False)
        fmt.set_scientific(True)
        fmt.set_powerlimits((0, 0))
        cb.formatter = fmt
        cb.update_ticks()
        cb.ax.tick_params(labelsize=lay["tick_fs"])
        cb.ax.yaxis.get_offset_text().set_fontsize(lay["tick_fs"])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    from rbno.config import Experiment

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("--row", type=int, required=True, help="global parameter row of a test sample")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    exp = Experiment.load(REPO_ROOT / "configs" / f"{args.experiment}.yaml")
    ctx = build_context(exp)
    P = ctx.problem
    extent = [0.0, P.length_x, 0.0, P.length_y]
    p_dof = np.asarray(np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")[args.row], dtype=np.float64)

    summary = {"experiment": exp.name, "row": args.row, "quantile_clip": list(QUANTILE_CLIP)}
    g1, g2, bc, grid_pred = fc_pino_grid_residual(exp, ctx, args.row, args.device)
    # FC-PINO and the RBNO are shown on the same sample: the recomputed FC-PINO prediction must be
    # the stored one of this row, and the row must be among the RBNO predictions.
    k = position(exp, "fc_pino", args.row)
    stored = np.load(exp.results_dir("fc_pino") / "pred_grid.npy", mmap_mode="r")[k]
    mismatch = float(np.abs(grid_pred - stored).max() / np.abs(stored).max())
    if mismatch > 1e-4:
        raise RuntimeError(f"recomputed FC-PINO prediction differs from the stored one of row {args.row}: {mismatch:.1e}")
    summary["positions"] = {"fc_pino": k, "rbno": position(exp, "rbno", args.row)}
    summary["fc_pino_recompute_rel_diff"] = mismatch
    plot_pair(g1, g2, "FC-PINO residual", extent, args.out / "fc_pino_grid.png")
    summary["fc_pino_grid"] = {"r1_mean": float(g1.mean()), "r2_mean": float(g2.mean()),
                               "loss": float(g1.mean() + g2.mean()), "bc_penalty": bc}
    for method, prefix in (("fc_pino", "FC-PINO FE residual"), ("rbno", "RBNO residual")):
        f1, f2, (l1, l2) = fe_residual(ctx, prediction(exp, method, args.row), p_dof)
        plot_pair(f1, f2, prefix, extent, args.out / f"{method}_fe.png")
        summary[f"{method}_fe"] = {"r1_vertex_mean": float(f1.mean()), "r2_vertex_mean": float(f2.mean()),
                                   "r1_assembled": l1, "r2_assembled": l2, "loss": l1 + l2}
    save_json(args.out / "summary.json", summary)
    print(f"row {args.row}, positions {summary['positions']}, FC-PINO recompute diff {mismatch:.1e}:",
          {k: v["loss"] for k, v in summary.items() if isinstance(v, dict) and "loss" in v})


if __name__ == "__main__":
    main()
