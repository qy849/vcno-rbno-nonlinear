"""Native grid errors (``tab:native_grid``) of a grid-based method.

    python -m native_grid quasilinear fno
"""

from __future__ import annotations

import argparse

from common import native_grid_metrics
from rbno.config import Experiment
from rbno.utils import REPO_ROOT


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("method")
    args = ap.parse_args()
    native_grid_metrics(Experiment.load(REPO_ROOT / "configs" / f"{args.experiment}.yaml"), args.method)


if __name__ == "__main__":
    main()
