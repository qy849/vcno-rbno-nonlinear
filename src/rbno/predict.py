"""RBNO predictions on the test set.

Writes to ``results/<experiment>/<method>/``::

    pred_sigma_u_dof.npy   (n_test, n_state)  s_r(p_i) = Pi_r c_i, mixed dof order
    pred_coefficients.npy  (n_test, r)        reduced coefficients c_i (network output)
    pred_indices.npy       test parameter rows
    pred_meta.json         checkpoint, timing

The recorded prediction time per parameter is the GPU time of a batched forward pass
plus the reconstruction ``Pi_r c`` (double precision, on the GPU), synchronized, averaged
over the test set after one warm-up batch; it excludes reading the parameters.

    python -m rbno.predict configs/quasilinear.yaml --device cuda:0
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from .config import Experiment
from .models import build_model
from .pod import load_basis
from .problems import build_problem
from .train import InputEncoder, make_input_cache
from .utils import save_json


def load_network(exp: Experiment, method: str, device, in_channels: int, image_shape) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(exp.checkpoint_dir(method) / "best_val.pt", map_location=device, weights_only=False)
    r = int(exp.section("pod")["num_basis"])
    net = build_model(exp.section("model"), in_channels, image_shape, r).to(device)
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    return net, ckpt


def predict(exp: Experiment, device: str, method: str = "rbno") -> None:
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    problem = build_problem(exp)
    p_all = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")
    rows = exp.test_rows
    ckpt = torch.load(exp.checkpoint_dir(method) / "best_val.pt", map_location="cpu", weights_only=False)
    stats = None
    if ckpt.get("input_mean") is not None:
        stats = (ckpt["input_mean"].to(dev), ckpt["input_std"].to(dev))
    encoder = InputEncoder(problem, p_all, exp.train_rows, False, dev, stats=stats)
    x = make_input_cache(encoder, rows)
    net, _ = load_network(exp, method, dev, x.shape[1], tuple(x.shape[2:]))
    basis = torch.from_numpy(load_basis(exp)).to(dev)
    bs = int(exp.section("eval").get("batch_size", 100))

    def run(lo, hi):
        with torch.inference_mode():
            c = net(x[lo:hi]).double()
            return c, c @ basis.T

    run(0, min(bs, rows.size))  # warm-up
    if dev.type == "cuda":
        torch.cuda.synchronize()
    coeffs, states = [], []
    t0 = time.perf_counter()
    for s in range(0, rows.size, bs):
        c, y = run(s, s + bs)
        coeffs.append(c)
        states.append(y)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    out = exp.results_dir(method)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "pred_sigma_u_dof.npy", torch.cat(states).cpu().numpy())
    np.save(out / "pred_coefficients.npy", torch.cat(coeffs).cpu().numpy())
    np.save(out / "pred_indices.npy", rows)
    save_json(out / "pred_meta.json", {
        "checkpoint": str(exp.checkpoint_dir(method) / "best_val.pt"), "best_epoch": ckpt.get("epoch"),
        "best_val_loss": ckpt.get("val_loss"), "num_test": int(rows.size), "batch_size": bs,
        "device": str(dev), "seconds": seconds, "seconds_per_parameter": seconds / rows.size,
    })
    print(f"[{exp.name}/{method}] predicted {rows.size} states, {seconds / rows.size * 1e3:.2f} ms per parameter")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--method", default="rbno")
    args = ap.parse_args()
    predict(Experiment.load(args.config), args.device, args.method)


if __name__ == "__main__":
    main()
