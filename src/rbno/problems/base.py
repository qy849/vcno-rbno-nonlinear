"""Common FOSLS machinery shared by the three benchmarks.

A benchmark is a first-order system for the flux or stress ``sigma`` and the primal
variable ``u`` on a structured triangulation, discretized in the conforming pair
RT_1 x CG_2 (``RT`` degree 2 and ``CG`` degree 2 in the basix convention; row-wise and
componentwise for elasticity). Inhomogeneous boundary data are removed by the lifts
``w`` (Dirichlet) and ``z = grad q`` (Neumann), so the unknown is the homogeneous state
``s = [sigma0, u0]`` and the loss is

    J_{h,q}(s; p) = || r1 ||^2 + || r2 ||^2,

with the residual components ``(r1, r2)`` of the subclass and every integral evaluated
with a fixed quadrature rule of degree ``q``. The same ``dx`` measure is used by the
Newton solver, the training loss and its gradient, the evaluator and the stability
test, so all of them see one functional.

Subclasses provide the parameter space, the boundary markers, the lifts, the residual
and its linearization; this module provides the rest: function spaces, boundary
conditions, Newton solves, cached assembly of the loss and its gradient, and the
H(div) x H^1 norm with its components.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem, mesh
from dolfinx.fem import petsc as fem_petsc
from mpi4py import MPI
from petsc4py import PETSc

from ..utils import assign_array, collapse_with_map, num_dofs, snes_reason_name

SNES_DEFAULTS = {
    "snes_type": "newtonls",
    "snes_linesearch_type": "bt",
    "snes_max_it": 100,
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "ksp_type": "preonly",
    "pc_type": "lu",
    "pc_factor_mat_solver_type": "mumps",
}


class SolveError(RuntimeError):
    """Raised when the Newton solve does not converge; carries the solve record."""

    def __init__(self, message: str, info: dict):
        super().__init__(message)
        self.info = info


def collect_bc_dofs(bcs) -> np.ndarray:
    dofs = [np.asarray(bc.dof_indices()[0], dtype=np.int64) for bc in bcs]
    dofs = [d for d in dofs if d.size]
    return np.unique(np.concatenate(dofs)) if dofs else np.empty(0, dtype=np.int64)


@dataclass
class _LossCache:
    sigma: fem.Function
    u: fem.Function
    p: fem.Function
    J: fem.Form
    dJ_dsigma: fem.Form
    dJ_du: fem.Form
    J1: fem.Form
    J2: fem.Form
    bc_sigma: np.ndarray
    bc_u: np.ndarray


@dataclass
class FOSLSProblem(ABC):
    comm: MPI.Intracomm
    num_x: int
    num_y: int
    length_x: float = 1.0
    length_y: float = 1.0
    quadrature_degree: int = 8

    _domain: Any = field(default=None, init=False, repr=False)
    _Vh: dict | None = field(default=None, init=False, repr=False)
    _bcs: dict | None = field(default=None, init=False, repr=False)
    _ds: Any = field(default=None, init=False, repr=False)
    _w: fem.Function | None = field(default=None, init=False, repr=False)
    _q: fem.Function | None = field(default=None, init=False, repr=False)
    _cache: _LossCache | None = field(default=None, init=False, repr=False)
    jacobian: str = "newton"
    solve_info: dict | None = field(default=None, init=False, repr=False)
    newton_iterates: list = field(default_factory=list, init=False, repr=False)

    # ---------------------------------------------------------------------------
    # subclass interface
    # ---------------------------------------------------------------------------
    name: ClassVar[str] = "fosls"
    vector_valued: ClassVar[bool] = False

    @abstractmethod
    def parameter_element(self, cell: str):
        """basix element of the coefficient field ``p``."""

    @abstractmethod
    def is_dirichlet(self, x) -> np.ndarray:
        """Boolean mask of boundary points on Gamma_D (primal variable prescribed)."""

    @abstractmethod
    def is_neumann(self, x) -> np.ndarray:
        """Boolean mask of boundary points on Gamma_N (normal flux prescribed)."""

    @abstractmethod
    def solve_lifts(self) -> tuple[fem.Function, fem.Function]:
        """Return the lifts ``(w, q)`` with ``z = grad q``."""

    @abstractmethod
    def residual_terms(self, sigma, u, p):
        """UFL residual components ``(r1, r2)`` of the homogeneous state ``[sigma, u]``."""

    @abstractmethod
    def linearized_residual(self, tau, v, p, ubar):
        """Derivative ``A_{ubar,p}[tau, v]`` of the residual at primal state ``ubar``."""

    # ---------------------------------------------------------------------------
    # mesh, spaces, boundary conditions
    # ---------------------------------------------------------------------------
    @property
    def domain(self):
        if self._domain is None:
            self._domain = mesh.create_rectangle(
                self.comm,
                points=((0.0, 0.0), (self.length_x, self.length_y)),
                n=(self.num_x, self.num_y),
                cell_type=mesh.CellType.triangle,
            )
        return self._domain

    @property
    def cell_name(self) -> str:
        return "triangle"

    def _sigma_element(self):
        import basix.ufl

        rt = basix.ufl.element("RT", self.cell_name, 2)
        return basix.ufl.mixed_element([rt, rt]) if self.vector_valued else rt

    def _u_element(self):
        import basix.ufl

        cg = basix.ufl.element("CG", self.cell_name, 2)
        return basix.ufl.mixed_element([cg, cg]) if self.vector_valued else cg

    @property
    def Vh(self) -> dict:
        if self._Vh is None:
            import basix.ufl

            sig, u = self._sigma_element(), self._u_element()
            self._Vh = {
                "p": fem.functionspace(self.domain, self.parameter_element(self.cell_name)),
                "sigma": fem.functionspace(self.domain, sig),
                "u": fem.functionspace(self.domain, u),
                "sigma_u": fem.functionspace(self.domain, basix.ufl.mixed_element([sig, u])),
                "u_scalar": fem.functionspace(self.domain, basix.ufl.element("CG", self.cell_name, 2)),
            }
        return self._Vh

    @property
    def n_sigma(self) -> int:
        return num_dofs(self.Vh["sigma"])

    @property
    def n_u(self) -> int:
        return num_dofs(self.Vh["u"])

    @property
    def n_state(self) -> int:
        return num_dofs(self.Vh["sigma_u"])

    @property
    def n_param(self) -> int:
        return num_dofs(self.Vh["p"])

    @property
    def bcs(self) -> dict:
        """Homogeneous essential conditions of the state.

        ``sigma0``/``u0`` act on the collapsed spaces, ``state`` on the mixed space.
        """
        if self._bcs is None:
            Vh = self.Vh
            fdim = self.domain.topology.dim - 1
            facets_N = mesh.locate_entities_boundary(self.domain, fdim, self.is_neumann)
            facets_D = mesh.locate_entities_boundary(self.domain, fdim, self.is_dirichlet)
            zero_sigma = fem.Function(Vh["sigma"])
            zero_u = fem.Function(Vh["u"])
            dofs_s = fem.locate_dofs_topological(Vh["sigma"], fdim, facets_N)
            dofs_u = fem.locate_dofs_topological(Vh["u"], fdim, facets_D)
            mixed_s = fem.locate_dofs_topological((Vh["sigma_u"].sub(0), Vh["sigma"]), fdim, facets_N)
            mixed_u = fem.locate_dofs_topological((Vh["sigma_u"].sub(1), Vh["u"]), fdim, facets_D)
            self._bcs = {
                "sigma0": [fem.dirichletbc(zero_sigma, dofs_s)],
                "u0": [fem.dirichletbc(zero_u, dofs_u)],
                "state": [
                    fem.dirichletbc(zero_sigma, mixed_s, Vh["sigma_u"].sub(0)),
                    fem.dirichletbc(zero_u, mixed_u, Vh["sigma_u"].sub(1)),
                ],
            }
        return self._bcs

    def boundary_measure(self, markers: dict[int, Any]):
        """``ds`` with integer tags for the given ``{tag: locator}`` boundary pieces."""
        fdim = self.domain.topology.dim - 1
        ids, tags = [], []
        for tag, locator in markers.items():
            f = mesh.locate_entities_boundary(self.domain, fdim, locator)
            ids.append(f)
            tags.append(np.full(f.size, tag, dtype=np.int32))
        ids = np.hstack(ids).astype(np.int32)
        tags = np.hstack(tags)
        order = np.argsort(ids)
        mt = mesh.meshtags(self.domain, fdim, ids[order], tags[order])
        return ufl.Measure("ds", domain=self.domain, subdomain_data=mt)

    # ---------------------------------------------------------------------------
    # lifts
    # ---------------------------------------------------------------------------
    @property
    def w(self) -> fem.Function:
        if self._w is None:
            self._w, self._q = self.solve_lifts()
        return self._w

    @property
    def q(self) -> fem.Function:
        if self._q is None:
            self._w, self._q = self.solve_lifts()
        return self._q

    def _linear_solve(self, a, L, bcs, prefix: str) -> fem.Function:
        problem = fem_petsc.LinearProblem(
            a, L, bcs=bcs, petsc_options_prefix=prefix,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps"},
        )
        return problem.solve()

    # ---------------------------------------------------------------------------
    # loss
    # ---------------------------------------------------------------------------
    @property
    def dx(self):
        return ufl.Measure("dx", domain=self.domain, metadata={"quadrature_degree": int(self.quadrature_degree)})

    def ls_functional(self, sigma, u, p):
        r1, r2 = self.residual_terms(sigma, u, p)
        return (ufl.inner(r1, r1) + ufl.inner(r2, r2)) * self.dx

    def loss_components(self, sigma, u, p) -> tuple[float, float]:
        """Assembled ``(||r1||^2, ||r2||^2)`` of a state given as Functions/UFL splits."""
        r1, r2 = self.residual_terms(sigma, u, p)
        return (
            self._assemble(ufl.inner(r1, r1) * self.dx),
            self._assemble(ufl.inner(r2, r2) * self.dx),
        )

    def _assemble(self, form) -> float:
        value = fem.assemble_scalar(fem.form(form))
        return float(self.domain.comm.allreduce(value, op=MPI.SUM))

    # -- dof-level helpers -----------------------------------------------------------
    def p_function(self, p_dof) -> fem.Function:
        p = fem.Function(self.Vh["p"])
        assign_array(p, p_dof)
        return p

    def state_function(self, state_dof) -> fem.Function:
        s = fem.Function(self.Vh["sigma_u"])
        assign_array(s, state_dof)
        return s

    def split_state(self, state_dof) -> tuple[fem.Function, fem.Function]:
        """Mixed dof vector -> collapsed ``(sigma, u)`` Functions."""
        s = self.state_function(state_dof)
        return s.sub(0).collapse(), s.sub(1).collapse()

    def loss_of_state(self, state_dof, p_dof) -> tuple[float, float]:
        """``(||r1||^2, ||r2||^2)`` of a mixed dof vector."""
        sigma, u = self.split_state(state_dof)
        return self.loss_components(sigma, u, self.p_function(p_dof))

    # -- cached loss and gradient used in training -------------------------------------
    def _build_cache(self) -> _LossCache:
        Vh = self.Vh
        sigma, u, p = fem.Function(Vh["sigma"]), fem.Function(Vh["u"]), fem.Function(Vh["p"])
        r1, r2 = self.residual_terms(sigma, u, p)
        J1 = ufl.inner(r1, r1) * self.dx
        J2 = ufl.inner(r2, r2) * self.dx
        J = J1 + J2
        return _LossCache(
            sigma=sigma, u=u, p=p,
            J=fem.form(J),
            dJ_dsigma=fem.form(ufl.derivative(J, sigma, ufl.TestFunction(Vh["sigma"]))),
            dJ_du=fem.form(ufl.derivative(J, u, ufl.TestFunction(Vh["u"]))),
            J1=fem.form(J1), J2=fem.form(J2),
            bc_sigma=collect_bc_dofs(self.bcs["sigma0"]),
            bc_u=collect_bc_dofs(self.bcs["u0"]),
        )

    def _load_cache(self, y_concat: np.ndarray, p_dof: np.ndarray) -> _LossCache:
        if self._cache is None:
            self._cache = self._build_cache()
        c = self._cache
        y = np.asarray(y_concat, dtype=np.float64)
        ns = self.n_sigma
        c.sigma.x.array[:] = y[:ns]
        c.sigma.x.array[c.bc_sigma] = 0.0
        c.u.x.array[:] = y[ns:]
        c.u.x.array[c.bc_u] = 0.0
        c.p.x.array[:] = p_dof
        for f in (c.sigma, c.u, c.p):
            f.x.scatter_forward()
        return c

    def loss(self, y_concat: np.ndarray, p_dof: np.ndarray) -> float:
        """Loss ``J`` of the coefficients ``[sigma, u]`` (no gradient)."""
        return float(fem.assemble_scalar(self._load_cache(y_concat, p_dof).J))

    def loss_and_grad(self, y_concat: np.ndarray, p_dof: np.ndarray) -> tuple[float, np.ndarray]:
        """Loss ``J`` and its gradient with respect to the coefficients ``[sigma, u]``.

        The gradient is the assembled linear form ``DJ(s)[psi_j]`` (symbolic derivative of
        the loss form, same quadrature), with the entries of constrained dofs set to zero:
        the state lives in the homogeneous space, so those coefficients are fixed at zero.
        """
        c = self._load_cache(y_concat, p_dof)
        ns = self.n_sigma
        J = fem.assemble_scalar(c.J)
        gs = fem_petsc.assemble_vector(c.dJ_dsigma)
        gu = fem_petsc.assemble_vector(c.dJ_du)
        for vec in (gs, gu):
            vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        g = np.concatenate([gs.array.copy(), gu.array.copy()])
        gs.destroy()
        gu.destroy()
        g[c.bc_sigma] = 0.0
        g[ns + c.bc_u] = 0.0
        return float(J), g

    # ---------------------------------------------------------------------------
    # Newton solve of the discrete normal equations
    # ---------------------------------------------------------------------------
    def solver_options(self, **overrides) -> dict:
        opts = dict(SNES_DEFAULTS)
        opts.update({k: v for k, v in overrides.items() if v is not None})
        return opts

    def newton_solve(
        self,
        p: fem.Function,
        initial_guess: np.ndarray | None = None,
        *,
        snes_max_it: int | None = None,
        capture_iterates: bool = False,
        prefix: str = "state_",
    ) -> fem.Function:
        """Solve ``DJ(s)[t] = 0`` for all ``t`` in the homogeneous space by Newton's method.

        Backtracking line search, relative and absolute tolerances 1e-8, MUMPS. With
        ``jacobian='newton'`` the Jacobian is the second derivative of the loss form; with
        ``'gauss_newton'`` the term with the second derivative of the residual is dropped.
        Both use the loss quadrature and converge to the same discrete solution.
        Raises :class:`SolveError` if SNES does not report convergence.
        """
        Vh = self.Vh
        state = fem.Function(Vh["sigma_u"])
        if initial_guess is not None:
            assign_array(state, initial_guess)
            state.x.array[collect_bc_dofs(self.bcs["state"])] = 0.0
            state.x.scatter_forward()
        sigma, u = ufl.split(state)
        J = self.ls_functional(sigma, u, p)
        F = ufl.derivative(J, state, ufl.TestFunction(Vh["sigma_u"]))
        if self.jacobian == "gauss_newton":
            # Normal-equation Jacobian without the second-order term (r, D^2 R[., .]):
            # 2 (A_s ds, A_s t) with the linearized residual A_s at the current state.
            ds, dv = ufl.split(ufl.TrialFunction(Vh["sigma_u"]))
            ts, tv = ufl.split(ufl.TestFunction(Vh["sigma_u"]))
            a1, a2 = self.linearized_residual(ds, dv, p, u)
            b1, b2 = self.linearized_residual(ts, tv, p, u)
            dF = 2.0 * (ufl.inner(a1, b1) + ufl.inner(a2, b2)) * self.dx
        elif self.jacobian == "newton":
            dF = ufl.derivative(F, state, ufl.TrialFunction(Vh["sigma_u"]))
        else:
            raise ValueError(f"Unknown jacobian {self.jacobian!r} (newton | gauss_newton).")
        problem = fem_petsc.NonlinearProblem(
            F, state, bcs=self.bcs["state"], J=dF,
            petsc_options=self.solver_options(snes_max_it=snes_max_it),
            petsc_options_prefix=prefix,
        )
        self.newton_iterates = []
        if capture_iterates:
            problem.solver.setMonitor(
                lambda snes, it, norm: self.newton_iterates.append(np.array(snes.getSolution().array_r))
            )
        t0 = time.perf_counter()
        error = None
        try:
            problem.solve()
        except Exception as exc:  # PETSc errors in the line search, NaNs, ...
            error = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - t0
        reason = int(problem.solver.getConvergedReason())
        iters = int(problem.solver.getIterationNumber())
        self.solve_info = {
            "success": error is None and reason > 0,
            "elapsed_s": float(elapsed),
            "iterations": iters,
            "reason": reason,
            "reason_name": snes_reason_name(reason),
            "error": error,
        }
        if not self.solve_info["success"]:
            raise SolveError(f"Newton solve failed: {self.solve_info}", dict(self.solve_info))
        return state

    # ---------------------------------------------------------------------------
    # norms
    # ---------------------------------------------------------------------------
    def sigma_field(self, sigma):
        """``sigma`` as a vector (scalar problems) or 2x2 tensor (elasticity)."""
        return sigma

    def u_field(self, u):
        return u

    def norm_components(self, sigma, u) -> dict[str, float]:
        """Squared ``||sigma||, ||div sigma||, ||u||, ||grad u||`` (L2) of a state."""
        s, v = self.sigma_field(sigma), self.u_field(u)
        dx = ufl.dx(domain=self.domain)
        return {
            "sigma_l2": self._assemble(ufl.inner(s, s) * dx),
            "div_sigma_l2": self._assemble(ufl.inner(ufl.div(s), ufl.div(s)) * dx),
            "u_l2": self._assemble(ufl.inner(v, v) * dx),
            "grad_u_l2": self._assemble(ufl.inner(ufl.grad(v), ufl.grad(v)) * dx),
        }

    def norm_matrices(self) -> dict:
        """CSR matrices on the mixed space of the four squared norm components:
        ``sigma_l2``, ``div_sigma_l2``, ``u_l2``, ``grad_u_l2`` (``x^T M x`` for a dof vector)."""
        import scipy.sparse as sp

        V = self.Vh["sigma_u"]
        sigma, u = ufl.TrialFunctions(V)
        tau, v = ufl.TestFunctions(V)
        s, t = self.sigma_field(sigma), self.sigma_field(tau)
        a, b = self.u_field(u), self.u_field(v)
        dx = ufl.dx(domain=self.domain)
        forms = {
            "sigma_l2": ufl.inner(s, t) * dx,
            "div_sigma_l2": ufl.inner(ufl.div(s), ufl.div(t)) * dx,
            "u_l2": ufl.inner(a, b) * dx,
            "grad_u_l2": ufl.inner(ufl.grad(a), ufl.grad(b)) * dx,
        }
        out = {}
        for key, form in forms.items():
            A = fem_petsc.assemble_matrix(fem.form(form))
            A.assemble()
            ai, aj, av = A.getValuesCSR()
            out[key] = sp.csr_matrix((av, aj, ai), shape=A.getSize()).astype(np.float64)
            A.destroy()
        return out

    def h_gram_matrix(self, *, l2_only: bool = False):
        """Gram matrix ``X_h`` of the H(div) x H^1 inner product on the mixed space (CSR).

        With ``l2_only`` the L^2 x L^2 mass matrix is returned instead.
        """
        import scipy.sparse as sp

        V = self.Vh["sigma_u"]
        sigma, u = ufl.TrialFunctions(V)
        tau, v = ufl.TestFunctions(V)
        s, t = self.sigma_field(sigma), self.sigma_field(tau)
        a, b = self.u_field(u), self.u_field(v)
        dx = ufl.dx(domain=self.domain)
        form = ufl.inner(s, t) * dx + ufl.inner(a, b) * dx
        if not l2_only:
            form += ufl.inner(ufl.div(s), ufl.div(t)) * dx + ufl.inner(ufl.grad(a), ufl.grad(b)) * dx
        A = fem_petsc.assemble_matrix(fem.form(form))
        A.assemble()
        ai, aj, av = A.getValuesCSR()
        out = sp.csr_matrix((av, aj, ai), shape=A.getSize()).astype(np.float64)
        A.destroy()
        return out

    # ---------------------------------------------------------------------------
    # stability test pencil
    # ---------------------------------------------------------------------------
    def stability_forms(self, p, ubar):
        """Bilinear forms ``(K, M)``: ``K = (A t, A s)_{L2}`` with the loss quadrature and
        ``M`` the H(div) x H^1 inner product, on the mixed space."""
        V = self.Vh["sigma_u"]
        sigma, u = ufl.TrialFunctions(V)
        tau, v = ufl.TestFunctions(V)
        A1s, A2s = self.linearized_residual(sigma, u, p, ubar)
        A1t, A2t = self.linearized_residual(tau, v, p, ubar)
        K = (ufl.inner(A1s, A1t) + ufl.inner(A2s, A2t)) * self.dx
        s, t = self.sigma_field(sigma), self.sigma_field(tau)
        a, b = self.u_field(u), self.u_field(v)
        dxm = ufl.dx(domain=self.domain)
        M = (ufl.inner(s, t) + ufl.inner(ufl.div(s), ufl.div(t)) + ufl.inner(a, b)
             + ufl.inner(ufl.grad(a), ufl.grad(b))) * dxm
        return K, M

    # ---------------------------------------------------------------------------
    def describe(self) -> dict:
        return {
            "problem": self.name,
            "mesh": [self.num_x, self.num_y],
            "lengths": [self.length_x, self.length_y],
            "quadrature_degree": int(self.quadrature_degree),
            "n_state": self.n_state,
            "n_param": self.n_param,
        }


def const(domain, value) -> fem.Constant:
    return fem.Constant(domain, default_scalar_type(value))


def sub_maps(Vh) -> tuple[np.ndarray, np.ndarray]:
    """Mixed-space dof indices of the ``sigma`` and ``u`` blocks."""
    _, sigma_map = collapse_with_map(Vh["sigma_u"].sub(0))
    _, u_map = collapse_with_map(Vh["sigma_u"].sub(1))
    return sigma_map, u_map

