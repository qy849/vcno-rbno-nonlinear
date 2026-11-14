"""Quasilinear diffusion: Darcy flow of QCD with a solution-dependent permeability.

    -div(p (1 + u^2) grad u) = f   in (0,1)^2,
    u = 1 - x1 on Gamma_D = {x1 = 0} u {x1 = 1},   flux . n = 0 on Gamma_N = {x2 = 0} u {x2 = 1},

with ``p = 0.01 + exp(m)`` (CG1) and nine Gaussian wells
``f = sum_i 100 exp(-(|x - c_i| / w)^2)``, ``c_i = (j/4, k/4)``, ``w = 1/32``.
The Neumann datum vanishes, so ``q = 0`` and ``z = 0``; the lifted residual is

    r1 = sigma0 - p (1 + (u0 + w)^2) grad(u0 + w),     r2 = div sigma0 + f .
"""

from __future__ import annotations

from dataclasses import dataclass

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh

from .base import FOSLSProblem, const
from .grid import cg1_grid_images


@dataclass
class QuasilinearProblem(FOSLSProblem):
    quadrature_degree: int = 8

    name = "quasilinear"

    def parameter_element(self, cell: str):
        return basix.ufl.element("CG", cell, 1)

    def is_dirichlet(self, x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], self.length_x)

    def is_neumann(self, x):
        return np.isclose(x[1], 0.0) | np.isclose(x[1], self.length_y)

    def f2(self):
        x = ufl.SpatialCoordinate(self.domain)
        width = 1.0 / 32.0
        expr = 0
        for cx, cy in [(m / 4.0, n / 4.0) for m in (1, 2, 3) for n in (1, 2, 3)]:
            expr += 100.0 * ufl.exp(-((ufl.sqrt((x[0] - cx) ** 2 + (x[1] - cy) ** 2) / width) ** 2))
        return expr

    def solve_lifts(self):
        V = self.Vh["u"]
        fdim = self.domain.topology.dim - 1
        dofs_D = fem.locate_dofs_topological(
            V, fdim, mesh.locate_entities_boundary(self.domain, fdim, self.is_dirichlet)
        )
        trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
        dx = ufl.dx(domain=self.domain)
        uD = fem.Function(V)
        uD.interpolate(lambda x: 1.0 - x[0])
        w = self._linear_solve(
            ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx,
            ufl.inner(const(self.domain, 0.0), test) * dx,
            [fem.dirichletbc(uD, dofs_D)], "lift_w_",
        )
        q = fem.Function(V)  # zero Neumann datum
        w.name, q.name = "w", "q"
        return w, q

    def residual_terms(self, sigma, u, p):
        uw = u + self.w
        r1 = sigma - (p * (1.0 + uw**2) * ufl.grad(uw) - ufl.grad(self.q))
        r2 = ufl.div(sigma) + self.f2()
        return r1, r2

    def linearized_residual(self, tau, v, p, ubar):
        U = ubar + self.w
        flux = p * (1.0 + U**2) * ufl.grad(v) + 2.0 * p * U * ufl.grad(U) * v
        return tau - flux, ufl.div(tau)

    def parameter_images(self, p_dof: np.ndarray) -> np.ndarray:
        return cg1_grid_images(self, p_dof)
