"""Semilinear diffusion--reaction: heat conduction of QCD with a cubic reaction.

    -div(p grad u) + kappa u^3 = f   in (0,1)^2,
    u = u_D = 0.1 (1 - x1) sin(4 pi x2)            on Gamma_D = {x1 = 0} u {x1 = 1},
    p grad u . n = g = 0.1 (1 - x2) cos(2 pi x1)    on Gamma_N = {x2 = 0} u {x2 = 1},

with the flux-free source ``f(v) = (f2, v) - (f1, grad v)``, ``f2 = 1`` and
``f1 = (0.5, -0.5) chi`` on the union of the 16 mini-squares
``[j/8 - 1/16, j/8 + 1/16] x [k/8 - 1/16, k/8 + 1/16]``, ``j, k in {1, 3, 5, 7}``.
The conductivity is piecewise constant (DG0): ``p = 10^{mu_i}`` on square ``i`` and 1
elsewhere. For ``kappa = 0`` this is exactly the heat-conduction benchmark of QCD,
including its lifts and residual (``z = grad q`` with ``q`` in CG2, and ``div z`` taken as
zero, as in QCD).

With ``sigma = p grad u + f1`` and the lifts ``u = u0 + w``, ``sigma = sigma0 + z``:

    r1 = sigma0 - (p grad(u0 + w) - z + f1),
    r2 = div sigma0 + f2 - kappa (u0 + w)^3 .

All integrands are polynomials on each triangle (r2 has degree 6), so quadrature of
degree 12 integrates the loss and its gradient exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import basix.ufl
import numpy as np
import ufl
from dolfinx import fem, mesh

from .base import FOSLSProblem, const

SQUARE_CENTERS = np.array([(m / 8.0, n / 8.0) for m in (1, 3, 5, 7) for n in (1, 3, 5, 7)])
SQUARE_HALF_WIDTH = 1.0 / 16.0


def square_index(points: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Index of the closed mini-square containing each point (``-1`` if none).

    The order of the squares follows QCD: the x-center runs in the outer loop.
    """
    pts = np.asarray(points)[:, :2]
    out = np.full(pts.shape[0], -1, dtype=np.int64)
    for j, (cx, cy) in enumerate(SQUARE_CENTERS):
        inside = (np.abs(pts[:, 0] - cx) <= SQUARE_HALF_WIDTH + tol) & (
            np.abs(pts[:, 1] - cy) <= SQUARE_HALF_WIDTH + tol
        )
        out[inside & (out < 0)] = j
    return out


@dataclass
class SemilinearProblem(FOSLSProblem):
    kappa: float = 0.0
    quadrature_degree: int = 12
    _kappa: fem.Constant | None = field(default=None, init=False, repr=False)
    _chi: fem.Function | None = field(default=None, init=False, repr=False)
    _vertex_cells: np.ndarray | None = field(default=None, init=False, repr=False)

    name = "semilinear"

    # -- geometry -----------------------------------------------------------------
    def parameter_element(self, cell: str):
        return basix.ufl.element("DG", cell, 0)

    def is_dirichlet(self, x):
        return np.isclose(x[0], 0.0) | np.isclose(x[0], self.length_x)

    def is_neumann(self, x):
        return np.isclose(x[1], 0.0) | np.isclose(x[1], self.length_y)

    # -- data ---------------------------------------------------------------------
    @property
    def kappa_constant(self) -> fem.Constant:
        if self._kappa is None:
            self._kappa = const(self.domain, float(self.kappa))
        return self._kappa

    def set_kappa(self, value: float) -> None:
        """Change the reaction strength in every form built from this problem."""
        self.kappa = float(value)
        self.kappa_constant.value = float(value)

    @property
    def chi(self) -> fem.Function:
        """DG0 indicator of the union of the mini-squares (cell midpoint test, as in QCD)."""
        if self._chi is None:
            chi = fem.Function(self.Vh["p"])
            chi.interpolate(lambda x: (square_index(x.T) >= 0).astype(np.float64))
            self._chi = chi
        return self._chi

    def f1(self):
        return ufl.as_vector((0.5 * self.chi, -0.5 * self.chi))

    def f2(self):
        return const(self.domain, 1.0)

    def u_dirichlet(self, x):
        return 0.1 * (1.0 - x[0]) * np.sin(4.0 * np.pi * x[1])

    def g(self):
        x = ufl.SpatialCoordinate(self.domain)
        return 0.1 * (1.0 - x[1]) * ufl.cos(2.0 * ufl.pi * x[0])

    def solve_lifts(self):
        V = self.Vh["u"]
        fdim = self.domain.topology.dim - 1
        facets_D = mesh.locate_entities_boundary(self.domain, fdim, self.is_dirichlet)
        dofs_D = fem.locate_dofs_topological(V, fdim, facets_D)
        trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
        dx = ufl.dx(domain=self.domain)
        a = ufl.inner(ufl.grad(trial), ufl.grad(test)) * dx

        uD = fem.Function(V)
        uD.interpolate(self.u_dirichlet)
        w = self._linear_solve(a, ufl.inner(const(self.domain, 0.0), test) * dx,
                               [fem.dirichletbc(uD, dofs_D)], "lift_w_")

        ds = self.boundary_measure({1: self.is_neumann})
        zero = fem.Function(V)
        q = self._linear_solve(a, ufl.inner(self.g(), test) * ds(1),
                               [fem.dirichletbc(zero, dofs_D)], "lift_q_")
        w.name, q.name = "w", "q"
        return w, q

    # -- residual -------------------------------------------------------------------
    def residual_terms(self, sigma, u, p):
        z = ufl.grad(self.q)
        uw = u + self.w
        r1 = sigma - (p * ufl.grad(uw) - z + self.f1())
        r2 = ufl.div(sigma) + self.f2() - self.kappa_constant * uw**3
        return r1, r2

    def linearized_residual(self, tau, v, p, ubar):
        c = 3.0 * self.kappa_constant * (ubar + self.w) ** 2
        return tau - p * ufl.grad(v), ufl.div(tau) - c * v

    # -- parameters -------------------------------------------------------------------
    def conductivity_dofs(self, mu: np.ndarray) -> np.ndarray:
        """DG0 dofs of ``p`` for exponents ``mu`` of shape ``(N, 16)``."""
        coords = self.Vh["p"].tabulate_dof_coordinates()
        idx = square_index(coords)
        mu = np.atleast_2d(np.asarray(mu, dtype=np.float64))
        out = np.ones((mu.shape[0], coords.shape[0]), dtype=np.float64)
        inside = idx >= 0
        out[:, inside] = 10.0 ** mu[:, idx[inside]]
        return out

    def _vertex_cell_map(self) -> np.ndarray:
        """For each grid vertex (row-major ``(iy, ix)``) one adjacent cell whose value it takes.

        A vertex on a closed mini-square takes the value of an adjacent cell inside that
        square, every other vertex the value 1 of an adjacent background cell; this is
        the vertex image of ``p`` on the closed squares used as network input in QCD.
        """
        if self._vertex_cells is None:
            dom = self.domain
            tdim = dom.topology.dim
            dom.topology.create_connectivity(0, tdim)
            v2c = dom.topology.connectivity(0, tdim)
            n_vertices = dom.topology.index_map(0).size_local
            nodes = mesh.entities_to_geometry(dom, 0, np.arange(n_vertices, dtype=np.int32)).reshape(-1)
            x = dom.geometry.x[nodes]
            ix = np.rint(x[:, 0] / self.length_x * self.num_x).astype(np.int64)
            iy = np.rint(x[:, 1] / self.length_y * self.num_y).astype(np.int64)
            cell_in = square_index(self.Vh["p"].tabulate_dof_coordinates()) >= 0
            cell_dof = self.Vh["p"].dofmap.list.reshape(-1).astype(np.int64)
            out = np.empty((self.num_y + 1) * (self.num_x + 1), dtype=np.int64)
            for vertex in range(n_vertices):
                dofs = cell_dof[v2c.links(vertex)]
                inside = dofs[cell_in[dofs]]
                out[iy[vertex] * (self.num_x + 1) + ix[vertex]] = inside[0] if inside.size else dofs[0]
            self._vertex_cells = out
        return self._vertex_cells

    def parameter_images(self, p_dof: np.ndarray) -> np.ndarray:
        p_dof = np.atleast_2d(np.asarray(p_dof))
        img = p_dof[:, self._vertex_cell_map()]
        return img.reshape(p_dof.shape[0], 1, self.num_y + 1, self.num_x + 1).astype(np.float32)

    def describe(self) -> dict:
        out = super().describe()
        out["kappa"] = float(self.kappa)
        return out
