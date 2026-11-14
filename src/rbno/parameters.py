"""Parameter distributions of the three benchmarks.

* ``mini_squares_qcd`` -- 16 exponents ``mu_i ~ U(-1, 1)``, conductivity ``10^{mu_i}`` on the
  mini-squares. The samples of QCD are reproduced exactly: QCD drew them with 50 MPI
  ranks, rank ``r`` seeding ``np.random.seed(r)`` and drawing its contiguous share of
  the ``num_train + num_test`` samples in order. The rows are stored in QCD's order
  (rank 0 first), so row ``i`` here is row ``i`` of QCD's train set for ``i < 5000`` and
  row ``i - 5000`` of its test set otherwise.
* ``shifted_lognormal`` -- ``p = shift + exp(m)`` with ``m ~ N(0, (delta I - gamma Laplace)^{-2})``
  realized by the hIPPYlib-X BiLaplacian prior (Robin boundary condition), sampled on
  CG1 with a fixed seed, as in the original repository.
"""

from __future__ import annotations

import numpy as np


def qcd_rank_split(num_samples: int, num_ranks: int) -> list[int]:
    q, r = divmod(int(num_samples), int(num_ranks))
    return [q + 1 if i < r else q for i in range(num_ranks)]


def sample_mini_square_exponents(num_samples: int, num_ranks: int = 50) -> np.ndarray:
    """Exponents ``mu`` of shape ``(num_samples, 16)``, identical to QCD's draw."""
    rows = []
    for rank, count in enumerate(qcd_rank_split(num_samples, num_ranks)):
        np.random.seed(rank)
        rows.extend(np.random.uniform(-1.0, 1.0, 16) for _ in range(count))
    return np.asarray(rows, dtype=np.float64)


def sample_shifted_lognormal(
    num_samples: int,
    Vh,
    *,
    gamma: float,
    delta: float,
    shift: float,
    seed: int,
    robin_bc: bool = True,
    progress: bool = True,
) -> np.ndarray:
    """``(num_samples, n_dofs)`` CG1 dofs of ``shift + exp(m)``."""
    import hippylibx
    import ufl
    from dolfinx import fem
    from tqdm import tqdm

    mean = fem.Function(Vh)
    prior = hippylibx.BiLaplacianPrior(
        Vh=Vh, gamma=gamma, delta=delta, mean=mean.x.petsc_vec, robin_bc=robin_bc, solver_type="lu"
    )
    rng = hippylibx.Random(seed=seed)
    noise = prior.M.createVecRight()
    prior.init_vector(noise, "noise")
    sample = prior.M.createVecRight()

    m = fem.Function(Vh)
    p = fem.Function(Vh)
    points = Vh.element.interpolation_points
    points = points() if callable(points) else points
    expr = fem.Expression(ufl.exp(m) + float(shift), points)
    out = np.empty((num_samples, m.x.array.size), dtype=np.float64)
    for i in tqdm(range(num_samples), desc="sample p", disable=not progress):
        rng.normal(1.0, noise)
        prior.sample(noise, sample)
        m.x.array[:] = sample.array
        m.x.scatter_forward()
        p.interpolate(expr)
        out[i] = p.x.array
    return out


def lognormal_field_max(p_dof: np.ndarray, shift: float) -> float:
    """Largest value of ``m = log(p - shift)`` over all sampled nodal values (``hat m``)."""
    return float(np.log(np.max(p_dof) - float(shift)))
