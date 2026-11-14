"""FNO baseline: supervised regression of the vertex values of the flux and the state.

``neuralop.models.FNO`` with grid positional embedding (coordinates appended), AdamW and
a grouped relative L2 loss (flux block + state block), trained on labelled snapshot
pairs; the checkpoint with the smallest validation loss is kept. Test predictions are
mapped to RT_1 x CG_2 (``common.save_grid_predictions``) and evaluated by
``rbno.evaluate``; their native grid errors by ``common.native_grid_metrics``.

    python -m fno quasilinear --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
from neuralop.models import FNO
from torch import nn

from common import (build_context, grid_cache, load_baseline_config, parameter_images,
                    save_grid_predictions, select_rows)
from rbno.utils import save_json


class GroupedRelativeL2(nn.Module):
    """Sum over channel groups of the batch-mean relative L2 error."""

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, pred, label):
        total = pred.new_zeros(())
        for a, b in self.groups:
            d = (pred[:, a:b] - label[:, a:b]).flatten(1).norm(dim=1)
            total = total + (d / (label[:, a:b].flatten(1).norm(dim=1) + 1e-10)).mean()
        return total


def channel_groups(ctx):
    return [(0, ctx.n_sigma_channels), (ctx.n_sigma_channels, ctx.n_channels)]


def scheduler(opt, cfg, epochs):
    if cfg.get("scheduler", "cosine") == "steplr":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=int(cfg["lr_step_size"]), gamma=float(cfg["lr_gamma"]))
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=float(cfg.get("eta_min", 1e-6)))


def run(experiment: str, device: str, method: str = "fno") -> None:
    exp, cfg = load_baseline_config(experiment, method)
    torch.manual_seed(int(cfg.get("seed", 0)))
    dev = torch.device(device)
    ctx = build_context(exp)
    imgs, rows = grid_cache(exp, "snapshots")
    pos = {int(r): k for k, r in enumerate(rows)}
    tr = select_rows(cfg["train"], rows)
    va = select_rows(cfg["val"], rows, exclude=tr)
    x_tr, x_va = parameter_images(ctx, tr), parameter_images(ctx, va)
    mean = x_tr.mean(axis=(0, 2, 3), keepdims=True)
    std = x_tr.std(axis=(0, 2, 3), keepdims=True) + 1e-8

    def X(a):
        return torch.from_numpy(((a - mean) / std).astype(np.float32))

    y_tr = torch.from_numpy(np.stack([imgs[pos[int(r)]] for r in tr]))
    y_va = torch.from_numpy(np.stack([imgs[pos[int(r)]] for r in va]))
    x_tr_t, x_va_t = X(x_tr), X(x_va)
    m = cfg["model"]
    model = FNO(n_modes=tuple(m["modes"]), in_channels=x_tr.shape[1], out_channels=ctx.n_channels,
                hidden_channels=int(m["hidden_channels"]), n_layers=int(m["n_layers"]),
                projection_channel_ratio=int(m.get("projection_channel_ratio", 2))).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    loss_fn = GroupedRelativeL2(channel_groups(ctx))
    t = cfg["optim"]
    epochs, bs = int(t["epochs"]), int(t["batch_size"])
    opt = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    sched = scheduler(opt, t, epochs)
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    print(f"[{exp.name}/{method}] FNO {n_params / 1e6:.1f}M params, train {tr.size}, val {va.size}", flush=True)
    hist, best, best_state = [], np.inf, None
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(tr.size, generator=gen)
        losses = []
        for s in range(0, tr.size, bs):
            idx = perm[s:s + bs]
            loss = loss_fn(model(x_tr_t[idx].to(dev)), y_tr[idx].to(dev))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = sum(loss_fn(model(x_va_t[s:s + 20].to(dev)), y_va[s:s + 20].to(dev)).item() * len(x_va_t[s:s + 20])
                     for s in range(0, va.size, 20)) / va.size
        hist.append((ep + 1, float(np.mean(losses)), vl))
        if vl < best:
            best, best_state = vl, copy.deepcopy(model.state_dict())
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"epoch {ep + 1}: train {np.mean(losses):.4e} val {vl:.4e} best {best:.4e}", flush=True)
    seconds = time.perf_counter() - t0
    model.load_state_dict(best_state)
    ckpt = exp.checkpoint_dir(method)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt / "best_val.pt")

    test_rows = exp.test_rows
    x_te = X(parameter_images(ctx, test_rows))
    model.eval()
    with torch.no_grad():
        preds = np.concatenate([model(x_te[s:s + 20].to(dev)).cpu().numpy() for s in range(0, test_rows.size, 20)])
    out = save_grid_predictions(exp, method, test_rows, preds)
    np.save(out / "loss_history.npy", np.array(hist))
    save_json(out / "train_summary.json", {
        "method": method, "num_parameters": int(n_params), "num_train": int(tr.size), "num_val": int(va.size),
        "train_rows": [int(tr[0]), int(tr[-1])], "val_rows": [int(va[0]), int(va[-1])],
        "epochs": epochs, "best_val": best, "train_seconds": seconds, "config": cfg})
    print(f"[{exp.name}/{method}] done in {seconds / 60:.1f} min -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run(args.experiment, args.device)


if __name__ == "__main__":
    main()
