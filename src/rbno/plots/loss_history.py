"""Training and validation loss of the RBNO (``fig:training_record``).

Style of ``notebooks/loss_history_plot/plot_loss_history.ipynb``: training and validation
loss per epoch on a log axis, green crosses where the best model was saved, the final
training loss and the lowest validation loss annotated in red. The dashed horizontal
line is the reduced reference loss: the mean loss of the reduced minimizers ``s_r^dagger`` of
the test parameters (``results/<experiment>/reduced_opt``, ``rbno.reduced_opt``); without them,
the mean loss of the projections ``P_r s_h``.

    python -m rbno.plots.loss_history quasilinear --out figures/training/quasilinear.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..utils import REPO_ROOT, load_json
from . import style
from .style import plt


def make(name: str, out: Path, method: str = "rbno", title: bool = False) -> None:
    run = REPO_ROOT / "results" / name / method
    recs = [json.loads(line) for line in (run / "loss_history.jsonl").read_text().splitlines() if line.strip()]
    epoch_recs = [r for r in recs if r.get("epoch_end") or r["step"] == 0]
    x = np.array([r["epoch"] for r in epoch_recs], dtype=float)
    tr = np.array([r["train_loss"] for r in epoch_recs])
    va = np.array([r["val_loss"] for r in epoch_recs])
    step_recs = [r for r in recs if not r.get("epoch_end") and r["step"] > 0]
    if step_recs:  # validation every step: mark the minimum over steps at its epoch
        vbest = min(step_recs, key=lambda r: r["val_loss"])
        best_x, best_y = vbest["step"] / max(1, step_recs[-1]["step"]) * x.max(), vbest["val_loss"]
    else:
        i = int(np.argmin(va))
        best_x, best_y = x[i], va[i]
    saves, cur = [], np.inf
    for xi, vi in zip(x, va):
        if vi < cur:
            cur = vi
            saves.append((xi, vi))

    fig, ax = plt.subplots(figsize=style.FIGSIZE)
    ax.plot(x, tr, color="blue", lw=2.5, label="Training loss")
    ax.plot(x, va, color="orange", lw=2.5, label="Validation loss")
    for ref_method, ref_label in (("reduced_opt", r"Loss of $s_r^\dagger$"), ("projection", r"Loss of $P_r s_h$")):
        ref_path = REPO_ROOT / "results" / name / ref_method / "metrics_summary.json"
        if ref_path.exists():
            break
    if ref_path.exists():
        ref = load_json(ref_path)["loss"]["mean"]
        ax.axhline(ref, color="black", ls="--", lw=2.0, label=ref_label)
        ax.text(0.02, ref, f"{ref:.2e}", transform=ax.get_yaxis_transform(), fontsize=14, va="bottom",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    ax.plot(*zip(*saves), "gx", ms=5, label="Best model saved")
    ax.plot(x[-1], tr[-1], "o", color="blue", ms=10, label="Final training loss")
    ax.plot(best_x, best_y, "go", ms=10, label="Lowest validation loss")
    top = max(tr[len(tr) * 3 // 4:].max(), va[len(va) * 3 // 4:].max()) * 2.0
    span = x.max() - x.min()
    ax.annotate(f"{tr[-1]:.2e}", xy=(x[-1], tr[-1]), xytext=(x.min() + 0.89 * span, top), color="red",
                fontsize=14, va="center", arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.annotate(f"{best_y:.2e}", xy=(best_x, best_y), xytext=(x.min() + 0.70 * span, top), color="red",
                fontsize=14, va="center", arrowprops=dict(arrowstyle="->", color="red", lw=1.5))
    ax.set_yscale("log")
    ax.set_xlabel("Epoch", fontsize=style.LABEL_FS)
    ax.set_ylabel("Loss", fontsize=style.LABEL_FS)
    if title:
        ax.set_title(style.PANEL_TITLES.get(name, name), fontsize=style.TITLE_FS)
    ax.tick_params(axis="both", labelsize=style.TICK_FS)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=style.LEGEND_FS)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=style.DPI)
    plt.close(fig)
    print(f"loss history -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    make(args.experiment, out if out.is_absolute() else REPO_ROOT / out, title=args.title)


if __name__ == "__main__":
    main()
