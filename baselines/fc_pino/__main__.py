"""FC-PINO baseline: FC-FNO (Fourier continuation, double precision) trained in two stages.

Stage 1: grouped relative L2 data loss on ``N_1`` labelled pairs.
Stage 2: from the best stage-1 weights, ``w_phys * (||r1||^2 + ||r2||^2)_grid + w_bc * BC penalty
         + w_data * data loss``, with the residual on ``N_2`` unlabelled parameters and the
         data loss on a separate batch of the ``N_1`` pairs.
AdamW, StepLR every 10% of the iterations (factor 0.8), batch 10, gradient clipping.
The checkpoint with the smallest validation loss of each stage is kept; the stage-2 one
is used. Test predictions are mapped to RT_1 x CG_2 for the common evaluator; the mean
grid (spectral) loss of the test predictions is recorded as FC-PINO's own loss.

    python -m fc_pino quasilinear --device cuda:1
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn.functional as F
from neuralop.layers.fourier_continuation import FCLegendre

from common import (build_context, grid_cache, load_baseline_config, parameter_images, physics_rows,
                    save_grid_predictions, select_rows)
from fno import GroupedRelativeL2, channel_groups
from rbno.utils import save_json

from .fc_fno import FC_FNO
from .residual import GridPhysics


class Sampler:
    """Batches without replacement, reshuffled after every pass."""

    def __init__(self, n, gen):
        self.n, self.gen, self.order, self.pos = n, gen, torch.randperm(n, generator=gen), 0

    def next(self, k):
        out = []
        while k > 0:
            take = min(k, self.n - self.pos)
            out.append(self.order[self.pos:self.pos + take])
            self.pos += take
            k -= take
            if self.pos >= self.n:
                self.order, self.pos = torch.randperm(self.n, generator=self.gen), 0
        return torch.cat(out)


def coords(ctx, b, dev):
    ny, nx = ctx.shape
    lx, ly = ctx.lengths
    yy, xx = torch.meshgrid(torch.linspace(0, ly, ny, dtype=torch.float64),
                            torch.linspace(0, lx, nx, dtype=torch.float64), indexing="ij")
    return torch.stack([xx, yy])[None].expand(b, -1, -1, -1).to(dev)


def build_model(ctx, cfg, n_param_channels: int, dev) -> FC_FNO:
    m = cfg["model"]
    ny, nx = ctx.shape
    lx, ly = ctx.lengths
    return FC_FNO(in_channels=n_param_channels + 2, out_channels=ctx.n_channels,
                  Lengths=(ly * ny / (ny - 1), lx * nx / (nx - 1)),  # vertex grid: spacing L / (N - 1)
                  n_modes=tuple(m["modes"]), hidden_channels=int(m["hidden_channels"]), n_layers=int(m["n_layers"]),
                  FC_obj=FCLegendre(d=int(m["fc_degree"]), n_additional_pts=int(m["fc_cont_points"])).to(dev),
                  positional_embedding=None, non_linearity=F.gelu, projection_nonlinearity=F.tanh).to(dev)


def run(experiment: str, device: str, method: str = "fc_pino") -> None:
    exp, cfg = load_baseline_config(experiment, method)
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)
    dev = torch.device(device)
    ctx = build_context(exp)
    imgs, rows = grid_cache(exp, "snapshots")
    pos = {int(r): k for k, r in enumerate(rows)}
    sup = select_rows(cfg["train"], rows)
    val = select_rows(cfg["val"], rows, exclude=sup)
    phys = physics_rows(exp, int(cfg["num_physics"]), exclude=val) if "num_physics" in cfg else \
        np.arange(*cfg["physics_range"])
    p_sup, p_val, p_phys = (torch.from_numpy(parameter_images(ctx, r).astype(np.float64)) for r in (sup, val, phys))
    mean = p_sup.mean(dim=(0, 2, 3), keepdim=True)
    std = p_sup.std(dim=(0, 2, 3), keepdim=True) + 1e-8
    y_sup = torch.from_numpy(np.stack([imgs[pos[int(r)]] for r in sup]).astype(np.float64))
    y_val = torch.from_numpy(np.stack([imgs[pos[int(r)]] for r in val]).astype(np.float64))

    model = build_model(ctx, cfg, p_sup.shape[1], dev)
    n_params = sum(p.numel() for p in model.parameters())
    physics = GridPhysics(ctx, dev, detf_floor=float(cfg.get("detf_floor", 1e-3)))
    data_loss = GroupedRelativeL2(channel_groups(ctx))
    t = cfg["optim"]
    it1, it2, bs = int(t["data_iterations"]), int(t["physics_iterations"]), int(t["batch_size"])
    w = cfg["weights"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, round(0.1 * (it1 + it2))), gamma=float(t["lr_gamma"]))
    clip = float(t.get("grad_clip", 1.0))
    every = int(t.get("validate_every", 100))
    gen = torch.Generator().manual_seed(seed)
    s_sup, s_phys = Sampler(sup.size, gen), Sampler(phys.size, gen)

    def inp(p):
        return torch.cat([(p - mean) / std, coords(ctx, p.shape[0], "cpu")], 1).to(dev)

    def val_data():
        model.eval()
        with torch.no_grad():
            return sum(data_loss(model(inp(p_val[s:s + 20])), y_val[s:s + 20].to(dev)).item() * len(p_val[s:s + 20])
                       for s in range(0, val.size, 20)) / val.size

    def val_phys():
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for s in range(0, val.size, 20):
                p = p_val[s:s + 20]
                pred, d = model(inp(p), derivs_to_compute=["dx", "dy"])
                r1, r2 = physics.residual(p.to(dev), pred, d)
                tot += (w["physics"] * (r1.mean() + r2.mean()) + w["bc"] * physics.boundary(pred)).item() * len(p)
        return tot / val.size

    print(f"[{exp.name}/{method}] FC-PINO {n_params / 1e6:.1f}M params, N1={sup.size}, val={val.size}, "
          f"N2={phys.size}, iterations {it1}+{it2}", flush=True)
    hist = {"data": [], "phys": []}
    best1 = best2 = np.inf
    state1 = state2 = None
    skipped = 0
    t0 = time.perf_counter()
    for step in range(1, it1 + 1):
        model.train()
        idx = s_sup.next(bs)
        loss = data_loss(model(inp(p_sup[idx])), y_sup[idx].to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        sched.step()
        if step == 1 or step % every == 0 or step == it1:
            v = val_data()
            hist["data"].append((step, loss.item(), v))
            if v < best1:
                best1, state1 = v, copy.deepcopy(model.state_dict())
            print(f"[data] {step}: train {loss.item():.3e} val {v:.3e} best {best1:.3e}", flush=True)
    model.load_state_dict(state1)
    for step in range(1, it2 + 1):
        model.train()
        ip, idd = s_phys.next(bs), s_sup.next(bs)
        pred, d = model(inp(p_phys[ip]), derivs_to_compute=["dx", "dy"])
        r1, r2 = physics.residual(p_phys[ip].to(dev), pred, d)
        res, bc = r1.mean() + r2.mean(), physics.boundary(pred)
        dl = data_loss(model(inp(p_sup[idd])), y_sup[idd].to(dev))
        total = w["physics"] * res + w["bc"] * bc + w["data"] * dl
        opt.zero_grad(set_to_none=True)
        if not torch.isfinite(total):
            skipped += 1
            sched.step()
            continue
        total.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        if not torch.isfinite(gn):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            sched.step()
            continue
        opt.step()
        sched.step()
        if step == 1 or step % every == 0 or step == it2:
            vp, vd = val_phys(), val_data()
            v = vp + w["data"] * vd
            hist["phys"].append((step, total.item(), res.item(), bc.item(), dl.item(), v, vp, vd))
            if v < best2:
                best2, state2 = v, copy.deepcopy(model.state_dict())
            print(f"[phys] {step}: total {total.item():.3e} res {res.item():.3e} bc {bc.item():.3e} "
                  f"data {dl.item():.3e} | val {v:.3e} best {best2:.3e}", flush=True)
    seconds = time.perf_counter() - t0
    model.load_state_dict(state2 if state2 is not None else state1)
    ckpt = exp.checkpoint_dir(method)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt / "best_val.pt")

    test_rows = exp.test_rows
    p_te = torch.from_numpy(parameter_images(ctx, test_rows).astype(np.float64))
    preds, own = [], []
    model.eval()
    with torch.no_grad():
        for s in range(0, test_rows.size, 10):
            p = p_te[s:s + 10]
            pred, d = model(inp(p), derivs_to_compute=["dx", "dy"])
            r1, r2 = physics.residual(p.to(dev), pred, d)
            preds.append(pred.cpu().numpy())
            own.append((r1 + r2).cpu().numpy())
    own = np.concatenate(own)
    out = save_grid_predictions(exp, method, test_rows, np.concatenate(preds))
    np.save(out / "grid_loss.npy", own)
    np.save(out / "loss_history_data.npy", np.array(hist["data"]))
    np.save(out / "loss_history_phys.npy", np.array(hist["phys"]))
    save_json(out / "train_summary.json", {
        "method": method, "num_parameters": int(n_params), "N1": int(sup.size), "N2": int(phys.size),
        "num_val": int(val.size), "physics_rows": [int(phys[0]), int(phys[-1])],
        "best_data_val": best1, "best_phys_val": best2, "skipped_steps": skipped,
        "train_seconds": seconds, "weights": w, "config": cfg,
        "grid_loss_test_mean": float(own.mean()), "grid_loss_test_std": float(own.std())})
    print(f"[{exp.name}/{method}] done in {seconds / 3600:.2f} h, grid loss {own.mean():.3e} -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    run(args.experiment, args.device)


if __name__ == "__main__":
    main()
