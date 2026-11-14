"""Train the RBNO on the finite element residual (Algorithm 1 of the paper).

Rank 0 runs the network on the GPU in single precision. Ranks ``1..W`` are CPU workers,
each with its own copy of the finite element problem (``COMM_SELF``) and of the POD
matrix ``Pi_r``. One training step:

1. rank 0 evaluates the reduced coefficients ``c_i = s_r(p_i; theta)`` of the batch;
2. the coefficients (``r`` numbers per sample) are sent to the workers, which form the
   finite element coefficients ``s_i = Pi_r c_i`` in double precision, assemble the loss
   ``J(s_i; p_i)`` and the vector ``g_i`` of its derivative form, and return ``J_i`` and
   ``Pi_r^T g_i``;
3. rank 0 back-propagates ``(1/N_b) Pi_r^T g_i`` through the network and updates theta.

No Gram matrix and no solve enter the step. The checkpoint with the smallest validation
loss is kept.

    mpirun -n 61 python -m rbno.train configs/semilinear_k1000.yaml --device cuda:0
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time

import numpy as np
import torch
from mpi4py import MPI

from .config import Experiment
from .models import build_model
from .optim import build_optimizer, build_scheduler
from .pod import load_basis
from .problems import build_problem
from .utils import save_json, save_yaml, set_seed, sigma_u_permutation

TAG_WORK, TAG_RESULT = 11, 12


# ---------------------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------------------
class ResidualWorker:
    """Evaluates ``J(Pi_r c; p)`` and ``Pi_r^T dJ`` for given coefficients and parameter rows."""

    def __init__(self, exp: Experiment):
        self.problem = build_problem(exp)
        perm = sigma_u_permutation(self.problem.Vh)
        self.basis_concat = np.ascontiguousarray(load_basis(exp)[perm])  # rows in [sigma, u] order
        self.p_all = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")

    def evaluate(self, rows, coeffs, with_grad: bool):
        losses = np.empty(len(rows))
        grads = np.empty((len(rows), self.basis_concat.shape[1])) if with_grad else None
        for k, (row, c) in enumerate(zip(rows, coeffs)):
            y = self.basis_concat @ np.asarray(c, dtype=np.float64)
            if with_grad:
                losses[k], g = self.problem.loss_and_grad(y, np.asarray(self.p_all[row]))
                grads[k] = self.basis_concat.T @ g
            else:
                losses[k] = self.problem.loss(y, np.asarray(self.p_all[row]))
        return losses, grads


def worker_loop(comm, exp: Experiment) -> None:
    worker = ResidualWorker(exp)
    while True:
        msg = comm.recv(source=0, tag=TAG_WORK)
        if msg is None:
            return
        losses, grads = worker.evaluate(msg["rows"], msg["coeffs"], msg["grad"])
        comm.send({"pos": msg["pos"], "losses": losses, "grads": grads}, dest=0, tag=TAG_RESULT)


class WorkerPool:
    """Rank-0 side of the residual evaluation."""

    def __init__(self, comm):
        self.comm = comm
        self.n = comm.Get_size() - 1
        if self.n < 1:
            raise RuntimeError("Training needs at least one residual worker (mpirun -n >= 2).")

    def __call__(self, rows: np.ndarray, coeffs: np.ndarray, with_grad: bool = True):
        positions = np.arange(len(rows))
        chunks = np.array_split(positions, self.n)
        for w, pos in enumerate(chunks, start=1):
            self.comm.send({"pos": pos, "rows": rows[pos], "coeffs": coeffs[pos], "grad": with_grad},
                           dest=w, tag=TAG_WORK)
        losses = np.empty(len(rows))
        grads = np.empty_like(coeffs, dtype=np.float64) if with_grad else None
        for w in range(1, self.n + 1):
            res = self.comm.recv(source=w, tag=TAG_RESULT)
            losses[res["pos"]] = res["losses"]
            if with_grad:
                grads[res["pos"]] = res["grads"]
        return losses, grads

    def stop(self):
        for w in range(1, self.n + 1):
            self.comm.send(None, dest=w, tag=TAG_WORK)


# ---------------------------------------------------------------------------------------
# network input
# ---------------------------------------------------------------------------------------
class InputEncoder:
    """Parameter rows -> (optionally standardized) images on the device."""

    def __init__(self, problem, p_all, train_rows, normalize: bool, device, stats=None):
        self.problem, self.p_all, self.device = problem, p_all, device
        self.mean = self.std = None
        if stats is not None:
            self.mean, self.std = stats
        elif normalize:
            imgs = self.images(train_rows, raw=True)
            self.mean = imgs.mean(dim=0, keepdim=True)
            self.std = imgs.std(dim=0, keepdim=True).clamp_min(1e-12)

    def images(self, rows, raw: bool = False) -> torch.Tensor:
        img = self.problem.parameter_images(np.asarray(self.p_all[np.asarray(rows)]))
        t = torch.from_numpy(img).to(self.device)
        if not raw and self.mean is not None:
            t = (t - self.mean) / self.std
        return t


def make_input_cache(encoder: InputEncoder, rows: np.ndarray) -> torch.Tensor:
    out = []
    for s in range(0, rows.size, 1000):
        out.append(encoder.images(rows[s:s + 1000]))
    return torch.cat(out)


# ---------------------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------------------
def train(exp: Experiment, device: str, comm=MPI.COMM_WORLD, run_tag: str = "rbno") -> None:
    tcfg = exp.section("train")
    seed = int(tcfg.get("seed", 0))
    set_seed(seed)
    out_dir, ckpt_dir = exp.results_dir(run_tag), exp.checkpoint_dir(run_tag)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(out_dir / "config.yaml", exp.cfg)
    history_path = out_dir / "loss_history.jsonl"
    history_path.write_text("")

    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    torch.set_float32_matmul_precision(str(tcfg.get("float32_matmul_precision", "high")))
    problem = build_problem(exp)
    p_all = np.load(exp.parameter_dir / "p_dof.npy", mmap_mode="r")
    train_rows, val_rows = exp.train_rows, exp.val_rows
    r = int(exp.section("pod")["num_basis"])

    encoder = InputEncoder(problem, p_all, train_rows, bool(tcfg.get("normalize_input", True)), dev)
    x_train = make_input_cache(encoder, train_rows)
    x_val = make_input_cache(encoder, val_rows)
    model = build_model(exp.section("model"), x_train.shape[1], tuple(x_train.shape[2:]), r).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    net = torch.compile(model) if tcfg.get("compile", False) else model
    optimizer = build_optimizer(model, tcfg["optimizer"])
    scheduler = build_scheduler(optimizer, tcfg.get("scheduler", {"type": "none"}))
    pool = WorkerPool(comm)
    batch = int(tcfg["batch_size"])
    epochs = int(tcfg["epochs"])
    eval_bs = int(exp.section("eval").get("batch_size", 200))
    validate_every_step = bool(tcfg.get("validate_every_step", False))
    print(f"[{exp.name}] RBNO: {n_params / 1e6:.2f}M parameters, r={r}, train={train_rows.size}, "
          f"val={val_rows.size}, workers={pool.n}, device={dev}", flush=True)

    def coefficients(x: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return np.concatenate([net(x[s:s + eval_bs]).double().cpu().numpy()
                                   for s in range(0, x.shape[0], eval_bs)])

    def mean_loss(x: torch.Tensor, rows: np.ndarray) -> float:
        net.eval()
        losses, _ = pool(rows, coefficients(x), with_grad=False)
        net.train()
        return float(losses.mean())

    def log(record: dict) -> None:
        with open(history_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def save(path, epoch, step, val):
        torch.save({"model_state": model.state_dict(), "epoch": epoch, "step": step, "val_loss": val,
                    "input_mean": encoder.mean.cpu() if encoder.mean is not None else None,
                    "input_std": encoder.std.cpu() if encoder.std is not None else None,
                    "num_parameters": n_params, "config": exp.cfg}, path)

    val0 = mean_loss(x_val, val_rows)
    train0 = mean_loss(x_train, train_rows)
    log({"epoch": 0, "step": 0, "train_loss": train0, "val_loss": val0, "lr": optimizer.param_groups[0]["lr"]})
    print(f"epoch 0: train {train0:.4e} | val {val0:.4e} (untrained)", flush=True)

    best = {"val": float("inf"), "state": None, "epoch": 0, "step": 0}
    step = 0
    gen = torch.Generator().manual_seed(seed)
    t0 = time.perf_counter()
    for epoch in range(1, epochs + 1):
        net.train()
        order = train_rows[torch.randperm(train_rows.size, generator=gen).numpy()]
        pos_of = {int(row): k for k, row in enumerate(train_rows)}
        epoch_losses = []
        for s in range(0, order.size, batch):
            rows = order[s:s + batch]
            idx = torch.as_tensor([pos_of[int(rw)] for rw in rows], device=dev)
            coeff = net(x_train[idx])
            losses, grads = pool(rows, coeff.detach().double().cpu().numpy(), with_grad=True)
            optimizer.zero_grad(set_to_none=True)
            coeff.backward(torch.from_numpy(grads / len(rows)).to(dev, dtype=coeff.dtype))
            optimizer.step()
            step += 1
            loss = float(losses.mean())
            epoch_losses.append(loss)
            if validate_every_step:
                val = mean_loss(x_val, val_rows)
                log({"epoch": epoch, "step": step, "train_loss": loss, "val_loss": val,
                     "lr": optimizer.param_groups[0]["lr"], "time_s": time.perf_counter() - t0})
                if val < best["val"]:
                    best.update(val=val, epoch=epoch, step=step,
                                state={k: v.detach().clone() for k, v in model.state_dict().items()})
        if not validate_every_step:
            val = mean_loss(x_val, val_rows)
            if val < best["val"]:
                best.update(val=val, epoch=epoch, step=step,
                            state={k: v.detach().clone() for k, v in model.state_dict().items()})
        train_loss = float(np.mean(epoch_losses))
        log({"epoch": epoch, "step": step, "train_loss": train_loss, "val_loss": val, "epoch_end": True,
             "lr": optimizer.param_groups[0]["lr"], "time_s": time.perf_counter() - t0})
        if scheduler is not None:
            scheduler.step()
        if best["state"] is not None and best["epoch"] == epoch:
            torch.save(best["state"], ckpt_dir / "best_state.tmp")
        if epoch % max(1, epochs // 50) == 0 or epoch == epochs:
            print(f"epoch {epoch}/{epochs}: train {train_loss:.4e} | val {val:.4e} | best {best['val']:.4e} "
                  f"(epoch {best['epoch']}) | {time.perf_counter() - t0:.0f}s", flush=True)
    train_seconds = time.perf_counter() - t0

    final_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    save(ckpt_dir / "last.pt", epochs, step, val)
    model.load_state_dict(best["state"])
    save(ckpt_dir / "best_val.pt", best["epoch"], best["step"], best["val"])
    (ckpt_dir / "best_state.tmp").unlink(missing_ok=True)
    model.load_state_dict(final_state)
    pool.stop()
    save_json(out_dir / "train_summary.json", {
        "experiment": exp.name, "method": run_tag, "num_parameters": int(n_params), "num_basis": r,
        "num_train": int(train_rows.size), "num_val": int(val_rows.size), "epochs": epochs, "steps": step,
        "batch_size": batch, "workers": pool.n, "device": str(dev),
        "best_val_loss": best["val"], "best_epoch": best["epoch"], "best_step": best["step"],
        "final_train_loss": train_loss, "final_val_loss": val,
        "train_seconds": train_seconds, "train_hours": train_seconds / 3600.0,
        "validate_every_step": validate_every_step,
    })
    print(f"done: {train_seconds / 3600:.2f} h, best val {best['val']:.4e} at epoch {best['epoch']}", flush=True)


def apply_override(cfg: dict, item: str) -> None:
    """``a.b.c=value``: set ``cfg["a"]["b"]["c"]`` to the YAML value (dicts are created)."""
    import yaml

    key, value = item.split("=", 1)
    *path, last = key.split(".")
    node = cfg
    for k in path:
        node = node.setdefault(k, {})
    value = yaml.safe_load(value)
    if isinstance(value, str):  # YAML 1.1 reads "3e-3" as a string
        with contextlib.suppress(ValueError):
            value = float(value)
    node[last] = value


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config")
    ap.add_argument("--device", default=None, help="overrides train.device")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--tag", default="rbno", help="output under results/<exp>/<tag> (e.g. tune/<id>)")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config entries, e.g. train.optimizer.lr=3e-3 (values parsed as YAML)")
    args = ap.parse_args()
    exp = Experiment.load(args.config)
    if args.epochs is not None:
        exp.cfg["train"]["epochs"] = args.epochs
    for item in args.set:
        apply_override(exp.cfg, item)
    comm = MPI.COMM_WORLD
    try:
        if comm.Get_rank() == 0:
            train(exp, args.device or exp.section("train").get("device", "cuda:0"), comm, run_tag=args.tag)
        else:
            worker_loop(comm, exp)
    except Exception:
        import traceback

        traceback.print_exc()
        comm.Abort(1)


if __name__ == "__main__":
    main()
