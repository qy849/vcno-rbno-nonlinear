"""Experiment configuration: one YAML file per experiment under ``configs/``.

An *experiment* is one benchmark at one setting (``semilinear_k0``, ``semilinear_k1000``,
``quasilinear``, ``hyperelastic``). Its YAML file fixes the discretization, the parameter
distribution, the index blocks (training, validation, test, snapshots), the solver, the
reduced basis, the network and the optimizer. Every pipeline step takes the path of this
file and derives all input and output locations from it, so a run can be repeated by
calling the same step with the same file.

Directory layout (relative to the repository root)::

    data/<parameters.dir>/p_dof.npy, p_grid.npy, ...     parameter samples (may be shared)
    data/<name>/snapshots/                                labelled solutions (POD, baselines)
    data/<name>/test/                                     test solutions
    data/<name>/pod/                                      POD basis and spectrum
    results/<name>/<method>/                              predictions, metrics, histories
    checkpoints/<name>/<method>/                          network weights
    analysis/<name>/                                      stability test, timing, figures
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .utils import REPO_ROOT, load_yaml


@dataclass
class Experiment:
    """Thin accessor around the experiment YAML dictionary."""

    cfg: dict
    path: Path

    @classmethod
    def load(cls, path: str | Path) -> "Experiment":
        path = Path(path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            alt = REPO_ROOT / "configs" / f"{path.stem}.yaml"
            if alt.exists():
                path = alt
        return cls(cfg=load_yaml(path), path=path)

    # -- identity ----------------------------------------------------------------
    @property
    def name(self) -> str:
        return str(self.cfg["name"])

    @property
    def problem_type(self) -> str:
        return str(self.cfg["problem"]["type"]).lower()

    @property
    def title(self) -> str:
        return str(self.cfg.get("title", self.name))

    # -- directories -------------------------------------------------------------
    def _abs(self, rel: str | Path) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else REPO_ROOT / p

    @property
    def data_dir(self) -> Path:
        return self._abs(self.cfg.get("data_dir", f"data/{self.name}"))

    @property
    def parameter_dir(self) -> Path:
        return self._abs(self.cfg["parameters"].get("dir", f"data/{self.name}/parameters"))

    @property
    def snapshot_dir(self) -> Path:
        return self.data_dir / "snapshots"

    @property
    def test_dir(self) -> Path:
        return self.data_dir / "test"

    @property
    def pod_dir(self) -> Path:
        return self.data_dir / "pod"

    def results_dir(self, method: str = "rbno") -> Path:
        return REPO_ROOT / "results" / self.name / method

    def checkpoint_dir(self, method: str = "rbno") -> Path:
        return REPO_ROOT / "checkpoints" / self.name / method

    @property
    def analysis_dir(self) -> Path:
        return REPO_ROOT / "analysis" / self.name

    # -- index blocks ------------------------------------------------------------
    def block(self, key: str) -> np.ndarray:
        """Parameter rows of a named block; a block is ``[start, stop)`` or a list of them."""
        spec = self.cfg["splits"][key]
        if isinstance(spec[0], (list, tuple)):
            return np.concatenate([np.arange(int(a), int(b)) for a, b in spec]).astype(np.int64)
        return np.arange(int(spec[0]), int(spec[1]), dtype=np.int64)

    @property
    def train_rows(self) -> np.ndarray:
        return self.block("train")

    @property
    def val_rows(self) -> np.ndarray:
        return self.block("val")

    @property
    def test_rows(self) -> np.ndarray:
        return self.block("test")

    @property
    def snapshot_rows(self) -> np.ndarray:
        return self.block("snapshots")

    # -- sections ------------------------------------------------------------------
    def section(self, key: str) -> dict:
        return dict(self.cfg.get(key, {}) or {})

    @property
    def mesh(self) -> dict:
        return dict(self.cfg["mesh"])
