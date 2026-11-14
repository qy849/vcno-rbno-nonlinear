"""Grid FOSLS residual of FC-PINO (stage 2), evaluated with Fourier-continuation derivatives.

Images are ``(B, C, ny+1, nx+1)``, axis 2 is ``x2`` and axis 3 is ``x1``; the network returns
``derivs[0] = d/dx2`` and ``derivs[1] = d/dx1``. Channels: ``[s1, s2, u]`` (scalar problems)
or ``[s11, s12, s21, s22, u1, u2]``. The lifted residual is the one of the finite element
loss, with every coefficient and lift sampled at the grid vertices (CG1 interpolation of
the FE lifts); its square is averaged over the grid (rectangle rule). Homogeneous
boundary conditions are imposed by a penalty (mean squared boundary values).
"""

from __future__ import annotations

import numpy as np
import torch
import ufl
from dolfinx import fem


def _ip(V):
    pts = V.element.interpolation_points
    return pts() if callable(pts) else pts


def _image(ctx, expr, vector: bool):
    import basix.ufl

    P = ctx.problem
    V = fem.functionspace(P.domain, basix.ufl.element("CG", P.cell_name, 1, shape=(2,))) if vector else ctx.V1
    f = fem.Function(V)
    f.interpolate(fem.Expression(expr, _ip(V)))
    ny, nx = ctx.shape
    vals = f.x.array.reshape(-1, 2) if vector else f.x.array[:, None]
    out = np.zeros((vals.shape[1], ny * nx))
    out[:, ctx.flat] = vals.T
    return out.reshape(vals.shape[1], ny, nx)


class GridPhysics:
    def __init__(self, ctx, device, dtype=torch.float64, detf_floor: float = 1e-3):
        self.ctx, self.kind = ctx, ctx.exp.problem_type
        P = ctx.problem
        t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)[None]  # noqa: E731
        self.floor = float(detf_floor)
        if self.kind in ("semilinear", "quasilinear"):
            self.w = t(_image(ctx, P.w, False))
            self.gw = t(_image(ctx, ufl.grad(P.w), True))
            self.gq = t(_image(ctx, ufl.grad(P.q), True))
            if self.kind == "semilinear":
                ny, nx = ctx.shape
                X, Y = np.meshgrid(np.linspace(0, P.length_x, nx), np.linspace(0, P.length_y, ny))
                from rbno.problems.semilinear import square_index

                chi = (square_index(np.stack([X.ravel(), Y.ravel()], 1)) >= 0).reshape(ny, nx)
                self.f1 = t(np.stack([0.5 * chi, -0.5 * chi]))
                self.kappa = float(P.kappa)
            else:
                self.f2 = t(_image(ctx, P.f2(), False))
        elif self.kind == "hyperelastic":
            q0, q1 = ufl.split(P.q)
            self.z = t(np.concatenate([_image(ctx, ufl.grad(q0), True), _image(ctx, ufl.grad(q1), True)]))
        else:
            raise ValueError(self.kind)

    # derivative accessors
    @staticmethod
    def dx(d, c):
        return d[1][:, c:c + 1]

    @staticmethod
    def dy(d, c):
        return d[0][:, c:c + 1]

    def pointwise(self, p, pred, d):
        """Pointwise squared residual components ``(r1^2, r2^2)``, each ``(B, 1, ny, nx)``."""
        if self.kind == "hyperelastic":
            return self._elasticity(p, pred, d)
        s1, s2, u = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3]
        div = self.dx(d, 0) + self.dy(d, 1)
        if self.kind == "semilinear":
            ux, uy = self.dx(d, 2) + self.gw[:, 0:1], self.dy(d, 2) + self.gw[:, 1:2]
            r1x = s1 - (p * ux - self.gq[:, 0:1] + self.f1[:, 0:1])
            r1y = s2 - (p * uy - self.gq[:, 1:2] + self.f1[:, 1:2])
            r2 = div + 1.0 - self.kappa * (u + self.w) ** 3
        else:
            uw = u + self.w
            c = p * (1.0 + uw ** 2)
            r1x = s1 - (c * (self.dx(d, 2) + self.gw[:, 0:1]) - self.gq[:, 0:1])
            r1y = s2 - (c * (self.dy(d, 2) + self.gw[:, 1:2]) - self.gq[:, 1:2])
            r2 = div + self.f2
        return r1x ** 2 + r1y ** 2, r2 ** 2

    def _elasticity(self, E, pred, d):
        nu = 0.4
        lam, mu = E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))
        F00, F01 = 1.0 + self.dx(d, 4), self.dy(d, 4)
        F10, F11 = self.dx(d, 5), 1.0 + self.dy(d, 5)
        J = F00 * F11 - F01 * F10
        Js = torch.clamp(J, min=self.floor)
        inv = (J > self.floor).to(J.dtype) / Js  # derivative of the floored energy (as in the FE loss)
        a = lam * torch.log(Js) - mu
        P = [mu * F00 + a * inv * F11, mu * F01 - a * inv * F10, mu * F10 - a * inv * F01, mu * F11 + a * inv * F00]
        r1 = sum((pred[:, k:k + 1] + self.z[:, k:k + 1] - P[k]) ** 2 for k in range(4))
        r2 = (self.dx(d, 0) + self.dy(d, 1)) ** 2 + (self.dx(d, 2) + self.dy(d, 3)) ** 2
        return r1, r2

    def residual(self, p, pred, d):
        r1, r2 = self.pointwise(p, pred, d)
        return r1.flatten(1).mean(1), r2.flatten(1).mean(1)

    def boundary(self, pred):
        sq = lambda t: t.square().mean()  # noqa: E731
        if self.kind == "hyperelastic":
            u0, u1 = pred[:, 4:5], pred[:, 5:6]
            loss = sq(u0[..., 0]) + sq(u1[..., 0])                               # u = 0 left
            loss = loss + sq(pred[:, 0:1, :, -1]) + sq(pred[:, 2:3, :, -1])       # s_i1 n = 0 right
            for c in (1, 3):                                                      # s_i2 = 0 top/bottom
                loss = loss + sq(pred[:, c:c + 1, 0, :]) + sq(pred[:, c:c + 1, -1, :])
            return loss
        u, s2 = pred[:, 2:3], pred[:, 1:2]
        return sq(u[..., 0]) + sq(u[..., -1]) + sq(s2[:, :, 0, :]) + sq(s2[:, :, -1, :])
