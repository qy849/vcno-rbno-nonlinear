"""PCA-Net baseline: Euclidean PCA of the parameter images and of the solution images
(all channels together) of the training pairs, and an MLP (SELU) between the
coefficients, trained with AdamW, a step schedule and an MSE loss (optionally on
standardized coefficients, as in QCD's heat-conduction notebook).

    python -m pca_net quasilinear --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
from torch import nn

from common import (build_context, grid_cache, load_baseline_config, parameter_images,
                    save_grid_predictions, select_rows)
from rbno.utils import save_json


def pca(snapshots: np.ndarray, k: int) -> np.ndarray:
    """Leading ``k`` Euclidean PCA modes ``(D, k)`` from the snapshot Gram matrix."""
    lam, V = np.linalg.eigh(snapshots @ snapshots.T)
    lam, V = lam[::-1][:k], V[:, ::-1][:, :k]
    return (snapshots.T @ V) / np.sqrt(np.clip(lam, 1e-300, None))[None, :]


def mlp(n_in, n_out, hidden, act="selu"):
    acts = {"selu": nn.SELU, "relu": nn.ReLU, "gelu": nn.GELU}
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), acts[act]()]
        prev = h
    return nn.Sequential(*layers, nn.Linear(prev, n_out))


def run(experiment: str, device: str, method: str = "pca_net") -> None:
    exp, cfg = load_baseline_config(experiment, method)
    torch.manual_seed(int(cfg.get("seed", 0)))
    dev = torch.device(device)
    ctx = build_context(exp)
    imgs, rows = grid_cache(exp, "snapshots")
    pos = {int(r): k for k, r in enumerate(rows)}
    tr = select_rows(cfg["train"], rows)
    va = select_rows(cfg["val"], rows, exclude=tr)
    flat = lambda a: a.reshape(a.shape[0], -1).astype(np.float64)  # noqa: E731
    P_tr, P_va = flat(parameter_images(ctx, tr)), flat(parameter_images(ctx, va))
    Y_tr = flat(np.stack([imgs[pos[int(r)]] for r in tr]))
    Y_va = flat(np.stack([imgs[pos[int(r)]] for r in va]))
    m = cfg["model"]
    B_in, B_out = pca(P_tr, int(m["num_input_basis"])), pca(Y_tr, int(m["num_output_basis"]))
    a_tr, a_va = P_tr @ B_in, P_va @ B_in
    c_tr, c_va = Y_tr @ B_out, Y_va @ B_out
    std_in, std_out = bool(cfg.get("standardize_input", False)), bool(cfg.get("standardize_output", False))
    mi, si = a_tr.mean(0), a_tr.std(0) + 1e-8
    mo, so = c_tr.mean(0), c_tr.std(0) + 1e-8
    fin = (lambda a: (a - mi) / si) if std_in else (lambda a: a)
    fout = (lambda c: (c - mo) / so) if std_out else (lambda c: c)
    inv = (lambda c: c * so + mo) if std_out else (lambda c: c)
    T = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(dev)  # noqa: E731
    x_tr, y_tr, x_va, y_va = T(fin(a_tr)), T(fout(c_tr)), T(fin(a_va)), T(fout(c_va))
    net = mlp(B_in.shape[1], B_out.shape[1], m["hidden_dims"], m.get("activation", "selu")).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    t = cfg["optim"]
    epochs = int(t["epochs"])
    bs = int(t.get("batch_size") or tr.size)
    opt = torch.optim.AdamW(net.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=int(t["lr_step_size"]), gamma=float(t["lr_gamma"]))
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    print(f"[{exp.name}/{method}] PCA-Net {n_params / 1e6:.2f}M params, K_in={B_in.shape[1]}, "
          f"K_out={B_out.shape[1]}, train {tr.size}, val {va.size}", flush=True)
    hist, best, best_state = [], np.inf, None
    t0 = time.perf_counter()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(tr.size, generator=gen).to(dev)
        losses = []
        for s in range(0, tr.size, bs):
            idx = perm[s:s + bs]
            loss = loss_fn(net(x_tr[idx]), y_tr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = loss_fn(net(x_va), y_va).item()
        hist.append((ep + 1, float(np.mean(losses)), vl))
        if vl < best:
            best, best_state = vl, copy.deepcopy(net.state_dict())
        if ep % 250 == 0 or ep == epochs - 1:
            print(f"epoch {ep + 1}: train {np.mean(losses):.4e} val {vl:.4e} best {best:.4e}", flush=True)
    seconds = time.perf_counter() - t0
    net.load_state_dict(best_state)
    ckpt = exp.checkpoint_dir(method)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, ckpt / "best_val.pt")
    np.savez(ckpt / "pca_bases.npz", B_in=B_in, B_out=B_out)

    test_rows = exp.test_rows
    a_te = flat(parameter_images(ctx, test_rows)) @ B_in
    with torch.no_grad():
        c_te = inv(net(T(fin(a_te))).double().cpu().numpy())
    preds = (c_te @ B_out.T).reshape(test_rows.size, ctx.n_channels, *ctx.shape)
    out = save_grid_predictions(exp, method, test_rows, preds)
    np.save(out / "loss_history.npy", np.array(hist))
    save_json(out / "train_summary.json", {
        "method": method, "num_parameters": int(n_params), "num_train": int(tr.size), "num_val": int(va.size),
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
