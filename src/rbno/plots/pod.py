"""POD figure (``fig:pod_record``): eigenvalues ``lambda_r`` and the mean projection error
``||(I - P_r) s_h||_H^2`` on the test set (log2 x-axis as in the original
``plot_reduced_basis_combined.py``); the marker shows the rank ``r`` used. With one experiment
and ``analysis/<exp>/reduced_opt/sweep.npz`` (``rbno.reduced_opt``) the panel also shows, over
the parameters of the sweep, the squared error ``||s_r^dagger - s_h||_H^2`` of the reduced
minimizer, the losses of ``P_r s_h`` and of ``s_r^dagger``, and the floor ``J(s_h)``.

    python -m rbno.plots.pod quasilinear --out figures/pod/quasilinear.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..config import Experiment
from ..utils import REPO_ROOT, load_json
from . import style
from .style import plt


def pow2_ticks(nmax: int) -> list[int]:
    ticks = [2 ** k for k in range(0, int(np.floor(np.log2(nmax))) + 1, 2)]
    if nmax / ticks[-1] < 4:
        ticks[-1] = int(nmax)
    else:
        ticks.append(int(nmax))
    return ticks


def make(names: list[str], out: Path, labels: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    colors = ["#1f77b4", "#ff7f0e"]
    nmax = 1
    for k, name in enumerate(names):
        exp = Experiment.load(REPO_ROOT / "configs" / f"{name}.yaml")
        z = np.load(exp.pod_dir / "projection_errors.npz")
        lam, r_all, err = z["eigenvalues"], z["r"], z["test_mean"]
        r_used = int(exp.section("pod")["num_basis"])
        c = colors[k % 2]
        tag = f" ({labels[k]})" if labels else ""
        ax.plot(np.arange(1, lam.size + 1), lam, color=c, lw=1.8, label=r"$\lambda_r$" + tag)
        ax.plot(r_all, err, color=c, lw=2.2, ls="--",
                label=r"$\|(I-P_r)s_h\|_{\mathbb{H}}^2$" + tag)
        ax.plot([r_used], [err[r_used - 1]], "o", color=c, ms=9)
        nmax = max(nmax, lam.size)
        sweep = exp.analysis_dir / "reduced_opt" / "sweep.npz"
        if len(names) == 1 and sweep.exists():
            w = np.load(sweep)
            rs = w["r"]
            ax.plot(rs, np.nanmean(w["opt_err2"], axis=1), "s-", color="#d62728", lw=1.8, ms=6,
                    label=r"$\|s_r^\dagger-s_h\|_{\mathbb{H}}^2$")
            ax.plot(rs, np.nanmean(w["proj_loss"], axis=1), "o:", color="#2ca02c", lw=2.0, ms=6,
                    label=r"$J_{h,q}(P_r s_h)$")
            ax.plot(rs, np.nanmean(w["opt_loss"], axis=1), "^:", color="#9467bd", lw=2.0, ms=7,
                    label=r"$J_{h,q}(s_r^\dagger)$")
            floor = exp.results_dir("reduced_opt") / "metrics_summary.json"
            if floor.exists():
                ax.axhline(load_json(floor)["loss_h"]["mean"], color="black", lw=1.2, ls="-.",
                           label=r"$J_{h,q}(s_h)$")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ticks = pow2_ticks(nmax)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_xlabel(r"$r$", fontsize=24)
    ax.tick_params(axis="both", labelsize=22)
    ax.legend(fontsize=19, ncol=2, loc="lower left", columnspacing=1.0, handlelength=1.6)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=style.DPI)
    plt.close(fig)
    print(f"POD figure -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    make(args.experiments, out if out.is_absolute() else REPO_ROOT / out, args.labels)


if __name__ == "__main__":
    main()
