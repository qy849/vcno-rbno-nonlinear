"""Control row of ``tab:baseline_fe``: the finite element solution sampled at the grid
vertices and treated like a grid prediction (CG_1 x CG_1 read back into RT_1 x CG_2),
without any learning.

    python -m fe_nodal quasilinear
"""

from __future__ import annotations

import argparse

from common import grid_cache, save_grid_predictions
from rbno.config import Experiment
from rbno.utils import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    args = ap.parse_args()
    exp = Experiment.load(REPO_ROOT / "configs" / f"{args.experiment}.yaml")
    imgs, rows = grid_cache(exp, "test")
    out = save_grid_predictions(exp, "fe_nodal", rows, imgs[:])
    print(f"[{exp.name}/fe_nodal] {rows.size} sampled solutions -> {out}")


if __name__ == "__main__":
    main()
