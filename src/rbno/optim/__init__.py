"""Optimizers (SOAP, Muon with auxiliary Adam) and learning-rate schedules."""

from __future__ import annotations

import torch


def build_optimizer(model: torch.nn.Module, cfg: dict) -> torch.optim.Optimizer:
    kind = str(cfg.get("type", "adamw")).lower()
    lr = float(cfg.get("lr", 1e-3))
    wd = float(cfg.get("weight_decay", 0.0))
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if kind == "soap":
        from .soap import SOAP

        kwargs = dict(cfg.get("kwargs", {}))
        if "betas" in kwargs:
            kwargs["betas"] = tuple(kwargs["betas"])
        return SOAP(model.parameters(), lr=lr, weight_decay=wd, **kwargs)
    if kind == "muon_auxadam":
        from .muon import SingleDeviceMuonWithAuxAdam

        min_ndim = int(cfg.get("muon_min_ndim", 2))
        muon = [p for p in model.parameters() if p.ndim >= min_ndim]
        adam = [p for p in model.parameters() if p.ndim < min_ndim]
        mk = dict(cfg.get("muon_kwargs", {}))
        ak = dict(cfg.get("adam_kwargs", {}))
        if "betas" in ak:
            ak["betas"] = tuple(ak["betas"])
        return SingleDeviceMuonWithAuxAdam([
            dict(params=muon, use_muon=True, lr=float(cfg.get("muon_lr", lr)), weight_decay=wd, **mk),
            dict(params=adam, use_muon=False, lr=float(cfg.get("adam_lr", lr)), weight_decay=wd, **ak),
        ])
    raise ValueError(f"Unknown optimizer {kind!r} (adamw | soap | muon_auxadam).")


def build_scheduler(optimizer, cfg: dict):
    """Schedulers are stepped once per epoch."""
    kind = str(cfg.get("type", "none")).lower()
    if kind == "steplr":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(cfg["step_size"]), gamma=float(cfg["gamma"]))
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(cfg["t_max"]), eta_min=float(cfg["eta_min"]))
    if kind == "none":
        return None
    raise ValueError(f"Unknown scheduler {kind!r} (steplr | cosine | none).")
