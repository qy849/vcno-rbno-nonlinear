"""Encoders mapping parameter images ``(B, C, ny+1, nx+1)`` to reduced coefficients."""

from __future__ import annotations

import torch

from .cnn import CNNEncoder, QCDConvNet


def build_model(model_cfg: dict, in_channels: int, image_shape: tuple[int, int], out_dim: int) -> torch.nn.Module:
    kind = str(model_cfg.get("type", "cnn")).lower()
    if kind == "qcd_cnn":
        return QCDConvNet(in_channels, image_shape, out_dim,
                          activation=model_cfg.get("activation", "leakyrelu"),
                          init=model_cfg.get("init", "xavier_uniform"))
    if kind == "cnn":
        return CNNEncoder(in_channels, image_shape, out_dim,
                          width=int(model_cfg.get("width", 512)),
                          hidden_channels=model_cfg.get("hidden_channels", [128, 256, 512]),
                          activation=model_cfg.get("activation", "gelu"))
    raise ValueError(f"Unknown model type {kind!r} (qcd_cnn | cnn).")


__all__ = ["build_model", "CNNEncoder", "QCDConvNet"]
