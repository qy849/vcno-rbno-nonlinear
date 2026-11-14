"""RBNO ablations trained on data instead of the residual.

Identical to the RBNO of the experiment (encoder, input normalization, frozen POD basis
``Pi_r``, optimizer and schedule from ``configs/<experiment>.yaml``); only the loss differs:
the relative squared error ``||Pi_r c - s_h||_W^2 / ||s_h||_W^2`` of the labelled snapshots,
with ``W`` the H(div) x H^1 Gram matrix (``--norm hdiv_h1``) or the L2 x L2 mass matrix
(``--norm l2``). Since ``Pi_r c`` lies in the reduced space, the loss is a quadratic form
in ``c`` with ``G = Pi^T W Pi``, ``b = Pi^T W s_h`` and ``k = s_h^T W s_h`` precomputed.

    python -m rbno_data_loss quasilinear --norm hdiv_h1 --device cuda:0
"""

from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch

from common import load_baseline_config, select_rows
from rbno.models import build_model
from rbno.optim import build_optimizer, build_scheduler
from rbno.pod import load_basis
from rbno.problems import build_problem
from rbno.train import InputEncoder, make_input_cache
from rbno.utils import save_json


def run(experiment: str, norm: str, device: str) -> None:
    method = f"rbno_{norm}_data"
    exp, cfg = load_baseline_config(experiment, "rbno_data_loss")
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)
    dev = torch.device(device)
    problem = build_problem(exp)
    W = problem.h_gram_matrix(l2_only=(norm == "l2"))
    basis = load_basis(exp)
    G = basis.T @ (W @ basis)
    store = exp.snapshot_dir
    srows = np.load(store / "valid_indices.npy")
    data = np.load(store / "sigma_u_dof.npy", mmap_mode="r")
    pos = {int(r): k for k, r in enumerate(srows)}
    tr = select_rows(cfg["train"], srows)
    va = select_rows(cfg["val"], srows, exclude=tr)

    def targets(rows):
        b, k = [], []
        for s in range(0, len(rows), 100):
            Y = np.stack([np.asarray(data[pos[int(r)]]) for r in rows[s:s + 100]])
            WY = (W @ Y.T)
            b.append((basis.T @ WY).T)
            k.append(np.einsum("ij,ji->i", Y, WY))
        return (torch.from_numpy(np.concatenate(b)).to(dev), torch.from_numpy(np.concatenate(k)).to(dev))

    b_tr, k_tr = targets(tr)
    b_va, k_va = targets(va)
    Gt = torch.from_numpy(G).to(dev)
    p_all = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")
    tcfg = exp.section("train")
    enc = InputEncoder(problem, p_all, exp.train_rows, bool(tcfg.get("normalize_input", True)), dev)
    x_tr, x_va = make_input_cache(enc, tr), make_input_cache(enc, va)
    r = basis.shape[1]
    net = build_model(exp.section("model"), x_tr.shape[1], tuple(x_tr.shape[2:]), r).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    ocfg = cfg.get("optimizer", tcfg["optimizer"])
    opt = build_optimizer(net, ocfg)
    sched = build_scheduler(opt, cfg.get("scheduler", tcfg.get("scheduler", {"type": "none"})))
    epochs, bs = int(cfg["epochs"]), int(cfg["batch_size"])

    def loss_fn(c, b, k):
        c = c.double()
        return (((c @ Gt) * c).sum(1) - 2 * (c * b).sum(1) + k) / k

    def val_loss():
        net.eval()
        with torch.no_grad():
            return float(torch.cat([loss_fn(net(x_va[s:s + 100]), b_va[s:s + 100], k_va[s:s + 100])
                                    for s in range(0, va.size, 100)]).mean())

    print(f"[{exp.name}/{method}] {n_params / 1e6:.1f}M params, train {tr.size}, val {va.size}", flush=True)
    gen = torch.Generator().manual_seed(seed)
    hist, best, state = [], np.inf, None
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(tr.size, generator=gen).to(dev)
        tot = 0.0
        for s in range(0, tr.size, bs):
            idx = perm[s:s + bs]
            loss = loss_fn(net(x_tr[idx]), b_tr[idx], k_tr[idx]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        if sched is not None:
            sched.step()
        v = val_loss()
        hist.append((ep, tot / tr.size, v))
        if v < best:
            best, state = v, copy.deepcopy(net.state_dict())
        if ep % max(1, epochs // 20) == 0 or ep == epochs:
            print(f"epoch {ep}: train {tot / tr.size:.4e} val {v:.4e} best {best:.4e}", flush=True)
    seconds = time.perf_counter() - t0
    net.load_state_dict(state)
    ckpt = exp.checkpoint_dir(method)
    ckpt.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt / "best_val.pt")

    rows = exp.test_rows
    x_te = make_input_cache(enc, rows)
    net.eval()
    bt = torch.from_numpy(basis).to(dev)
    with torch.no_grad():
        pred = torch.cat([net(x_te[s:s + 100]).double() @ bt.T for s in range(0, rows.size, 100)]).cpu().numpy()
    out = exp.results_dir(method)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_sigma_u_dof.npy", pred)
    np.save(out / "pred_indices.npy", rows)
    np.save(out / "loss_history.npy", np.array(hist))
    save_json(out / "train_summary.json", {"method": method, "num_parameters": int(n_params), "num_train": int(tr.size),
                                           "num_val": int(va.size), "epochs": epochs, "best_val": best,
                                           "train_seconds": seconds, "config": cfg})
    print(f"[{exp.name}/{method}] done in {seconds / 60:.1f} min -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment")
    ap.add_argument("--norm", choices=["hdiv_h1", "l2"], required=True)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    run(args.experiment, args.norm, args.device)


if __name__ == "__main__":
    main()
