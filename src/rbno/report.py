"""Collect all results into ``docs/numbers.json`` and LaTeX fragments in ``docs/tex/``.

Fragments (drop-in replacements for the tables of ``sections/06-numerical-experiments.tex``
and ``tables/``): ``table_accuracy.tex``, ``table_ratios.tex``, ``table_cost.tex``,
``table_baseline_fe.tex`` (with the former native-grid table merged in), and
``text_snippets.tex`` with the sentences that replace the ``\\tbd{...}`` entries.

    python -m rbno.report
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import Experiment
from .utils import REPO_ROOT, save_json

EXPS = ["semilinear_k0", "semilinear_k1000", "quasilinear", "hyperelastic"]
BASELINE_EXPS = ["semilinear_k1000", "quasilinear", "hyperelastic"]
METHODS = [("pca_net", "PCA-Net"), ("fno", "FNO"), ("fc_pino", "FC-PINO"), ("rbno", "RBNO"),
           ("rbno_l2_data", r"RBNO ($\LL$ data loss)"), ("rbno_hdiv_h1_data", r"RBNO ($\HH$ data loss)"),
           ("fe_nodal", "Pointwise reconstruction")]
ROW_NAMES = {"semilinear_k0": r"Heat conduction, $\kappa=0$", "semilinear_k1000": r"Semilinear, $\kappa=10^3$",
             "quasilinear": "Quasilinear", "hyperelastic": "Hyperelastic"}
BLOCK_NAMES = {"semilinear_k1000": r"Semilinear diffusion--reaction, $\kappa=10^3$",
               "quasilinear": "Quasilinear diffusion", "hyperelastic": "Neo-Hookean elasticity"}


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def num(x, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return rf"\num{{{x:.{digits}e}}}"


def fix(x, digits: int = 2) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def exp_of(name: str) -> Experiment:
    return Experiment.load(REPO_ROOT / "configs" / f"{name}.yaml")


def collect() -> dict:
    out = {}
    for name in EXPS:
        exp = exp_of(name)
        d = {"metrics": {}, "native": {}, "train": {}}
        for m, _ in METHODS + [("projection", "")]:
            d["metrics"][m] = _load(exp.results_dir(m) / "metrics_summary.json")
            d["native"][m] = _load(exp.results_dir(m) / "native_grid_metrics.json")
            d["train"][m] = _load(exp.results_dir(m) / "train_summary.json")
        d["timing"] = _load(exp.analysis_dir / "timing.json")
        d["stability"] = _load(exp.analysis_dir / "stability" / "summary.json")
        d["pod"] = _load(exp.pod_dir / "meta.json")
        d["test_store"] = _load(exp.test_dir / "meta.json")
        d["snapshot_store"] = _load(exp.snapshot_dir / "meta.json")
        d["parameters"] = _load(exp.parameter_dir / "meta.json")
        stats = exp.test_dir / "solve_stats.npz"
        if stats.exists():
            z = np.load(stats)
            ok = z["success"] > 0
            d["test_solves"] = {
                "median_iterations": float(np.median(z["iterations"][ok])),
                "median_rel_change": float(np.nanmedian(z["rel_change"][ok])),
                "min_rel_change": float(np.nanmin(z["rel_change"][ok])),
                "max_rel_change": float(np.nanmax(z["rel_change"][ok])),
                "mean_elapsed_s": float(np.mean(z["elapsed_s"][ok])),
                "attempts": {str(int(a)): int(np.sum(z["attempt"] == a)) for a in np.unique(z["attempt"])},
            }
        d["check_kappa0"] = _load(exp.analysis_dir / "check_kappa0.json")
        out[name] = d
    return out


# ---------------------------------------------------------------------------------------
def table_accuracy(R) -> str:
    lines = []
    for name in EXPS:
        m = R[name]["metrics"]["rbno"]
        r = exp_of(name).section("pod")["num_basis"]
        if m is None:
            lines.append(f"{ROW_NAMES[name]} & ${r}$ & -- & -- & -- & -- & --\\\\")
            continue
        lines.append(
            f"{ROW_NAMES[name]} & ${r}$ & {num(m['rel_H']['mean'])} ({num(m['rel_H']['std'])}) & "
            f"{num(m['rel_basis_part']['mean'])} & {num(m['rel_coef_part']['mean'])} & "
            f"{num(m['loss_on_truth_cohort']['mean'])} & {num(m['loss_h']['mean'])}\\\\")
    return "\n".join(lines)


def table_ratios(R) -> str:
    lines = []
    for name in EXPS:
        m = R[name]["metrics"]["rbno"]
        if m is None:
            lines.append(f"{ROW_NAMES[name]} & -- & -- & -- & --\\\\")
            continue
        q, c = m["ratio_E_over_eta"], m["chi"]
        lines.append(f"{ROW_NAMES[name]} & ${fix(q['median'])}$ & ${fix(q['min'])}$--${fix(q['max'])}$ & "
                     f"${fix(c['median'])}$ & ${fix(c['min'])}$--${fix(c['max'])}$\\\\")
    return "\n".join(lines)


def table_cost(R) -> str:
    lines = []
    for name in ("semilinear_k1000", "quasilinear", "hyperelastic"):
        t = R[name]["timing"]
        if t is None:
            lines.append(f"{ROW_NAMES[name]} & -- & -- & -- & --\\\\")
            continue
        lines.append(f"{ROW_NAMES[name]} & ${fix(t.get('training_h'), 1)}$ & ${fix(t['loss_grad_s'], 3)}$ & "
                     f"${fix(t.get('prediction_s'), 4)}$ & ${fix(t['solve_total_s'], 1)}$\\\\")
    return "\n".join(lines)


GRID = ("pca_net", "fno", "fc_pino")       # methods shown in both representations
CG, RT = r"[CG$_1\times$CG$_1$]", r"[RT$_1\times$CG$_2$]"


def _pair(d, key=None, mark: str = "") -> str:
    r"""``mean & std`` for the S columns of the merged baseline table; ``mark`` is a macro
    (``\bmnum``, ``\uwnum``) applied to both numbers."""
    if d is None:
        return "{--} & {--}"
    mean, std = (d[key + "_mean"], d[key + "_std"]) if key else (d["mean"], d["std"])
    if mark:
        return f"{mark}{{{mean:.2e}}} & {mark}{{{std:.2e}}}"
    return f"{mean:.2e} & {std:.2e}"


def table_baseline_fe(R) -> str:
    """Rows of the merged baseline table (``tab:baseline_fe``): grid methods twice, as
    CG1 x CG1 functions against the sampled FE solution and in RT1 x CG2 against the FE solution;
    columns relative LL error, relative HH error, loss (mean & std each).

    Per benchmark the smallest mean relative LL and HH errors are bold and the smallest residual
    loss (the FC-PINO grid loss included) has a wavy underline. The FE solution sampled at the grid
    nodes is a control, not a method: it is not ranked and is set off by a rule."""
    rows = []
    for name in BASELINE_EXPS:
        entries = []   # (label, rel_L2, rel_H, loss, loss_comparable, control)
        for m, label in METHODS:
            fe = R[name]["metrics"][m]
            if m in GRID:
                nat, tr = R[name]["native"][m], R[name]["train"][m]
                if nat is not None:
                    l2 = {"mean": nat["rel_L2_mean"], "std": nat["rel_L2_std"]}
                    hh = {"mean": nat["rel_H_mean"], "std": nat["rel_H_std"]}
                    if m == "fc_pino" and tr is not None:   # its own spectral residual on the grid
                        entries.append((f"{label} {CG}" + r"$^\dagger$", l2, hh,
                                        {"mean": tr["grid_loss_test_mean"], "std": tr["grid_loss_test_std"]},
                                        False, False))
                    else:
                        entries.append((f"{label} {CG}", l2, hh, fe and fe["loss"], True, False))
                label = f"{label} {RT}"
            if fe is not None:
                entries.append((label, fe["rel_L2"], fe["rel_H"], fe["loss_on_truth_cohort"], True,
                                m == "fe_nodal"))
        best = {k: min(e[k]["mean"] for e in entries if not e[5] and e[k] is not None) for k in (1, 2, 3)}
        rows.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\textit{{{BLOCK_NAMES[name]}}}}}\\")
        for e in entries:
            cells = []
            for k, mark in ((1, r"\bmnum"), (2, r"\bmnum"), (3, r"\uwnum")):
                win = not e[5] and e[k] is not None and e[k]["mean"] == best[k]
                cells.append(_pair(e[k], mark=mark if win else ""))
            if e[5]:
                rows.append(r"\cmidrule{1-7}")
            rows.append(f"{e[0]} & " + " & ".join(cells) + "\\\\")
        rows.append(r"\cmidrule{1-7}\morecmidrules\cmidrule{1-7}")   # double rule between benchmarks
    return "\n".join(rows[:-1])


def snippets(R) -> str:
    out = []
    p = {n: R[n]["parameters"] for n in ("quasilinear", "hyperelastic")}
    out.append("% tab:benchmarks caption: largest value of the sampled fields m")
    for n in p:
        if p[n]:
            out.append(f"%   {n}: hat m = {p[n]['m_hat']:.3f}  (p_max = {p[n]['p_max']:.3f})")
    s = R["semilinear_k1000"].get("test_solves")
    if s:
        out.append("% Strength of the reaction (subsec:problem_configs)")
        out.append(f"Newton's method needs a median of ${s['median_iterations']:.0f}$ Gauss--Newton iterations "
                   f"from the solution for $\\kappa=0$, and the state changes by a median of "
                   f"${100 * s['median_rel_change']:.0f}\\%$ (range ${100 * s['min_rel_change']:.0f}$--"
                   f"${100 * s['max_rel_change']:.0f}\\%$) in the relative $\\HH$ norm over the test parameters.")
    c = R["semilinear_k0"].get("check_kappa0")
    if c:
        out.append("% kappa = 0 regression check")
        out.append(f"%   max relative state difference (H) {c['max_rel_state_difference_H']:.2e}; "
                   f"max relative loss difference {c['max_rel_loss_difference_qcd_functional']:.2e}")
    for n in ("quasilinear", "hyperelastic", "semilinear_k0", "semilinear_k1000"):
        m = R[n]["metrics"]["rbno"]
        if m:
            out.append(f"% {n}: chi median {m['chi']['median']:.2f} range {m['chi']['min']:.2f}-{m['chi']['max']:.2f}; "
                       f"E/eta median {m['ratio_E_over_eta']['median']:.2f} range "
                       f"{m['ratio_E_over_eta']['min']:.2f}-{m['ratio_E_over_eta']['max']:.2f}; "
                       f"eta<=eta_h: {m['n_loss_below_floor']}; n={m['n_with_truth']}/{m['n_total']}")
    for n in ("semilinear_k1000", "quasilinear", "hyperelastic"):
        st = R[n]["stability"]
        if st:
            g = st["groups"]
            out.append(f"% stability {n}: " + ", ".join(f"{k}: min {v['min']:.3e} (n={v['n']})" for k, v in g.items())
                       + f"; overall {st['overall_min']:.3e} over {st['n_states']} states; chi bound {st['chi_bound']:.2f}")
    for n in BASELINE_EXPS:
        comps = []
        for m in ("pca_net", "fc_pino", "fe_nodal", "rbno"):
            s = R[n]["metrics"][m]
            if s:
                e = s["error_components"]
                comps.append(f"{m}: |e_u|_H1^2 {s['err_H1_u2']['mean']:.3e}, |e_s|^2 {e['err_sigma_l2']['mean']:.3e}, "
                             f"|div e_s|^2 {e['err_div_sigma_l2']['mean']:.3e}")
        if comps:
            out.append(f"% error split {n}: " + " | ".join(comps))
    return "\n".join(out)


def main() -> None:
    R = collect()
    docs = REPO_ROOT / "docs"
    save_json(docs / "numbers.json", R)
    tex = docs / "tex"
    tex.mkdir(parents=True, exist_ok=True)
    for name, fn in (("table_accuracy", table_accuracy), ("table_ratios", table_ratios),
                     ("table_cost", table_cost), ("table_baseline_fe", table_baseline_fe),
                     ("text_snippets", snippets)):
        (tex / f"{name}.tex").write_text(fn(R) + "\n")
    print(f"wrote {docs / 'numbers.json'} and {tex}/*.tex")


if __name__ == "__main__":
    main()
