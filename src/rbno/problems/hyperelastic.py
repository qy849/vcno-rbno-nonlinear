"""Compressible Neo-Hookean elasticity: the clamped body of QCD under traction.

    div P(I + grad u; p) + f = 0  in (0,2) x (0,1),   f = 0,
    u = 0 on the left edge,  P n = t on the right edge,  P n = 0 on top and bottom,
    t = (0.6 exp(-(x2 - 0.5)^2 / 4), 0.3 (1 + x2 / 10)).

The Young's modulus is ``p = 1 + exp(m)`` (CG1) and the Poisson ratio ``nu = 0.4``.
The first Piola--Kirchhoff stress is the derivative of

    psi(F) = mu/2 (F:F - 2) - mu ln J + lambda/2 (ln J)^2,   J = max(det F, 1e-3),

which coincides with the Neo-Hookean law wherever ``det F >= 1e-3`` (the floor only
protects the logarithm during Newton iterations). The stress is approximated row-wise
in RT_1 and the displacement componentwise in CG_2. With the Dirichlet lift ``w = 0``
and the traction lift ``z = grad q`` (rows), ``q`` solving a vector Laplace problem with
the traction as Neumann datum, the residual is

    r1 = sigma0 + lambda_load z - P(I + grad u0; p),     r2 = div sigma0 + f,

where ``lambda_load`` is the load factor used by the continuation (1 for the problem).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import basix.ufl
import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh

from .base import FOSLSProblem, SolveError, const
from .grid import cg1_grid_images

POISSON_RATIO = 0.4
DET_FLOOR = 1e-3


@dataclass
class HyperelasticProblem(FOSLSProblem):
    length_x: float = 2.0
    length_y: float = 1.0
    quadrature_degree: int = 8
    _load: fem.Constant | None = field(default=None, init=False, repr=False)

    name = "hyperelastic"
    vector_valued = True

    def parameter_element(self, cell: str):
        return basix.ufl.element("CG", cell, 1)

    def is_dirichlet(self, x):
        return np.isclose(x[0], 0.0)

    def is_neumann(self, x):
        return np.isclose(x[0], self.length_x) | np.isclose(x[1], 0.0) | np.isclose(x[1], self.length_y)

    def is_right(self, x):
        return np.isclose(x[0], self.length_x)

    # -- shapes ---------------------------------------------------------------------
    def sigma_field(self, sigma):
        shape = sigma.ufl_shape
        if shape == (2, 2):
            return sigma
        if shape == (4,):
            return ufl.as_tensor(((sigma[0], sigma[1]), (sigma[2], sigma[3])))
        raise TypeError(f"Unexpected stress shape {shape}.")

    def u_field(self, u):
        if u.ufl_shape != (2,):
            raise TypeError(f"Unexpected displacement shape {u.ufl_shape}.")
        return u

    # -- data ---------------------------------------------------------------------
    @property
    def load(self) -> fem.Constant:
        if self._load is None:
            self._load = const(self.domain, 1.0)
        return self._load

    def traction(self):
        x = ufl.SpatialCoordinate(self.domain)
        y = x[1] / self.length_y
        return ufl.as_vector((0.6 * ufl.exp(-0.25 * (y - 0.5) ** 2), 0.3 * (1.0 + 0.1 * y)))

    def lame(self, p):
        nu = POISSON_RATIO
        return p * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), p / (2.0 * (1.0 + nu))

    def solve_lifts(self):
        V = self.Vh["u"]
        fdim = self.domain.topology.dim - 1
        dofs_D = fem.locate_dofs_topological(
            V, fdim, mesh.locate_entities_boundary(self.domain, fdim, self.is_dirichlet)
        )
        trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
        a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * ufl.dx(domain=self.domain)
        ds = self.boundary_measure({1: self.is_right})
        q = self._linear_solve(a, ufl.inner(self.traction(), test) * ds(1),
                               [fem.dirichletbc(fem.Function(V), dofs_D)], "lift_q_")
        w = fem.Function(V)  # homogeneous Dirichlet datum
        w.name, q.name = "w", "q"
        return w, q

    def z(self):
        q0, q1 = ufl.split(self.q)
        return ufl.as_tensor((ufl.grad(q0), ufl.grad(q1)))

    def deformation_gradient(self, u):
        return ufl.Identity(2) + ufl.grad(self.u_field(u) + self.w)

    def stress(self, u, p):
        """First Piola--Kirchhoff stress with the determinant floor."""
        lam, mu = self.lame(p)
        F = ufl.variable(self.deformation_gradient(u))
        J = ufl.max_value(ufl.det(F), default_scalar_type(DET_FLOOR))
        psi = (mu / 2.0) * (ufl.tr(F.T * F) - 2.0) - mu * ufl.ln(J) + (lam / 2.0) * ufl.ln(J) ** 2
        return ufl.diff(psi, F)

    def residual_terms(self, sigma, u, p):
        s = self.sigma_field(sigma)
        r1 = s + self.load * self.z() - self.stress(u, p)
        r2 = ufl.div(s)
        return r1, r2

    def linearized_residual(self, tau, v, p, ubar):
        """Consistent tangent of the Neo-Hookean law (no floor; states have det F >> 1e-3)."""
        lam, mu = self.lame(p)
        F = self.deformation_gradient(ubar)
        Finv = ufl.inv(F)
        FinvT = Finv.T
        lnJ = ufl.ln(ufl.det(F))
        H = ufl.grad(self.u_field(v))
        DP = (mu * H + (mu - lam * lnJ) * (FinvT * H.T * FinvT) + lam * ufl.tr(Finv * H) * FinvT)
        t = self.sigma_field(tau)
        return t - DP, ufl.div(t)

    def parameter_images(self, p_dof: np.ndarray) -> np.ndarray:
        return cg1_grid_images(self, p_dof)

    # -- load continuation ------------------------------------------------------------
    def continuation_solve(
        self,
        p: fem.Function,
        seed: np.ndarray | None,
        loads: np.ndarray,
        *,
        snes_max_it: int = 120,
        capture_iterates: bool = False,
    ) -> fem.Function:
        """Walk the load factor through ``loads``, warm-starting each step from the last.

        ``solve_info`` aggregates all steps: total iterations and time, plus per-step records.
        The Newton iterates of the last step are kept when ``capture_iterates`` is set.
        """
        steps = []
        state_dof = seed
        try:
            for k, lam in enumerate(loads):
                self.load.value = float(lam)
                last = k == len(loads) - 1
                state = self.newton_solve(p, state_dof, snes_max_it=snes_max_it,
                                          capture_iterates=capture_iterates and last)
                steps.append({**self.solve_info, "load": float(lam)})
                state_dof = state.x.array.copy()
        except SolveError as exc:
            steps.append({**exc.info, "load": float(lam)})
            info = _aggregate(steps, success=False)
            self.solve_info = info
            raise SolveError(f"continuation failed at load {lam:.3f}", info) from exc
        finally:
            self.load.value = 1.0
        self.solve_info = _aggregate(steps, success=True)
        return state


def _aggregate(steps: list[dict], success: bool) -> dict:
    return {
        "success": success,
        "elapsed_s": float(sum(s["elapsed_s"] for s in steps)),
        "iterations": int(sum(s["iterations"] for s in steps)),
        "reason": int(steps[-1]["reason"]),
        "reason_name": steps[-1]["reason_name"],
        "error": steps[-1].get("error"),
        "steps": steps,
    }
