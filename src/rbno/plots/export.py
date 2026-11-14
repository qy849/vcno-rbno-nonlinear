"""Copy the figures to ``paper_figures/`` under the paths used in the LaTeX sources.

The recorded quasilinear and hyperelastic images of the draft keep their file names (old
equation names ``nonlinear_poisson``/``nonlinear_elasticity``); new semilinear images get
``semilinear`` paths.

    python -m rbno.plots.export
"""

from __future__ import annotations

import shutil

from ..utils import REPO_ROOT

F = REPO_ROOT / "figures"
T = "training_and_inference"
MAP = {
    # fig:benchmarks
    f"{T}/semilinear/p_1.png": "benchmarks/semilinear/p_5000.png",
    f"{T}/semilinear/u_1.png": "benchmarks/semilinear/u_5000.png",
    f"{T}/semilinear/sigma_1_1.png": "benchmarks/semilinear/sigma_5000_1.png",
    f"{T}/semilinear/sigma_1_2.png": "benchmarks/semilinear/sigma_5000_2.png",
    f"{T}/nonlinear_poisson/p_1.png": "benchmarks/quasilinear/p_8000.png",
    f"{T}/nonlinear_poisson/u_1.png": "benchmarks/quasilinear/u_8000.png",
    f"{T}/nonlinear_poisson/sigma_1_1.png": "benchmarks/quasilinear/sigma_8000_1.png",
    f"{T}/nonlinear_poisson/sigma_1_2.png": "benchmarks/quasilinear/sigma_8000_2.png",
    f"{T}/nonlinear_elasticity/p_1.png": "benchmarks/hyperelastic/p_5000.png",
    f"{T}/nonlinear_elasticity/u_1_1.png": "benchmarks/hyperelastic/u_5000_1.png",
    f"{T}/nonlinear_elasticity/u_1_2.png": "benchmarks/hyperelastic/u_5000_2.png",
    **{f"{T}/nonlinear_elasticity/sigma_1_{c}.png": f"benchmarks/hyperelastic/sigma_5000_{c}.png"
       for c in ("11", "12", "21", "22")},
    # fig:ratios (format of the top row for all three benchmarks)
    f"{T}/semilinear/loss_excess_vs_error.png": "ratios/semilinear/loss_excess_vs_error.png",
    f"{T}/semilinear/chi_hist.png": "ratios/semilinear/chi_hist.png",
    f"{T}/nonlinear_poisson/loss_excess_vs_error.png": "ratios/quasilinear/loss_excess_vs_error.png",
    f"{T}/nonlinear_poisson/chi_hist.png": "ratios/quasilinear/chi_hist.png",
    f"{T}/nonlinear_elasticity/loss_excess_vs_error.png": "ratios/hyperelastic/loss_excess_vs_error.png",
    f"{T}/nonlinear_elasticity/chi_hist.png": "ratios/hyperelastic/chi_hist.png",
    # fig:pod_record
    "reduced_basis/semilinear/pod.png": "pod/semilinear_k1000.png",
    "reduced_basis/semilinear/pod_k0.png": "pod/semilinear_k0.png",
    "reduced_basis/nonlinear_poisson/pod.png": "pod/quasilinear.png",
    "reduced_basis/nonlinear_elasticity/pod.png": "pod/hyperelastic.png",
    # fig:training_record
    f"{T}/semilinear/loss_history_semilinear_k0.png": "training/semilinear_k0.png",
    f"{T}/semilinear/loss_history_semilinear_k1000.png": "training/semilinear_k1000.png",
    f"{T}/nonlinear_poisson/loss_history.png": "training/quasilinear.png",
    f"{T}/nonlinear_elasticity/loss_history.png": "training/hyperelastic.png",
    # fig:baseline_training_* (PCA-Net, FNO, RBNO data-loss ablations, FC-PINO)
    **{f"{T}/baselines/{e}/{m}.png": f"training/baselines/{e}/{m}.png"
       for e in ("semilinear_k1000", "quasilinear", "hyperelastic")
       for m in ("pca_net", "fno", "rbno_l2_data", "rbno_hdiv_h1_data", "fc_pino")},
    # fig:ref_pred_diff_* (RBNO, test rows of fig:benchmarks)
    **{f"ref_pred_diff/{b}/{f}_ref_pred_diff_{r}.png": f"ref_pred_diff/{e}/{f}_ref_pred_diff_{r}.png"
       for b, e, r, fs in (("semilinear", "semilinear_k1000", 5000, ("u", "sigma1", "sigma2")),
                           ("nonlinear_poisson", "quasilinear", 8000, ("u", "sigma1", "sigma2")),
                           ("nonlinear_elasticity", "hyperelastic", 5000,
                            ("u1", "u2", "sigma_11", "sigma_12", "sigma_21", "sigma_22")))
       for f in fs},
    # fig:stability
    "discrete_stability/exp_stability_row.png": "stability/row.png",   # one row, one legend
    "discrete_stability/semilinear/exp_stability_semilinear.png": "stability/semilinear_k1000.png",
    "discrete_stability/nonlinear_poisson/exp_stability_nonlinear_poisson.png": "stability/quasilinear.png",
    "discrete_stability/nonlinear_elasticity/exp_stability_nonlinear_elasticity.png": "stability/hyperelastic.png",
    # fig:residual_maps (FC-PINO grid residual and RBNO FE residual; fc_pino_fe.png is not used)
    **{f"residual_maps/{b}/{m}.png": f"residual/{e}/{m}.png"
       for b, e in (("semilinear", "semilinear_k1000"), ("nonlinear_poisson", "quasilinear"),
                    ("nonlinear_elasticity", "hyperelastic"))
       for m in ("fc_pino_grid", "rbno_fe")},
}


def main() -> None:
    out = REPO_ROOT / "paper_figures" / "figures"
    for dst, src in MAP.items():
        target = out / dst
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(F / src, target)
    print(f"copied {len(MAP)} figures -> {out}")


if __name__ == "__main__":
    main()
