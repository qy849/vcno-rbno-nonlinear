"""Table of the hyperparameter search of one experiment (``scripts/07_tune.sh``).

Reads ``results/<exp>/tune/*/{config.yaml,train_summary.json}`` and writes
``results/<exp>/tune/summary.csv`` sorted by the best validation loss.

    python -m rbno.tune_summary configs/semilinear_k1000.yaml
"""

from __future__ import annotations

import argparse
import json

from .config import Experiment
from .utils import load_yaml


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    exp = Experiment.load(ap.parse_args().config)
    import pandas as pd

    root = exp.results_dir("tune")
    rows = []
    for run in sorted(p for p in root.iterdir() if (p / "train_summary.json").exists()):
        s = json.loads((run / "train_summary.json").read_text())
        t = load_yaml(run / "config.yaml")["train"]
        rows.append(dict(id=run.name, lr=t["optimizer"]["lr"], eta_min=t["scheduler"].get("eta_min"),
                         epochs=s["epochs"], batch_size=s["batch_size"], steps=s["steps"],
                         best_val=s["best_val_loss"], best_epoch=s["best_epoch"],
                         final_train=s["final_train_loss"], final_val=s["final_val_loss"],
                         hours=s["train_hours"]))
    df = pd.DataFrame(rows).sort_values("best_val")
    df.to_csv(root / "summary.csv", index=False)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3e}"))


if __name__ == "__main__":
    main()
