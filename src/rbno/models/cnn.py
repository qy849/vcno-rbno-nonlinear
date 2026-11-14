"""Convolutional encoders.

``QCDConvNet``  -- the heat-conduction network of QCD: four convolutions (kernels 5, 5, 5, 3;
strides 2, 2, 3, 1; 64 to 512 channels), a dense layer of width 512 and a linear output,
LeakyReLU activations and Xavier-uniform initialization (15.1M parameters for 129x129
inputs and r = 128).

``CNNEncoder``  -- the encoder of the quasilinear and hyperelastic RBNO in the original
repository: 3x3 convolutions with the given channels, 2x2 average pooling after every
convolution but the first, then ``Linear(flat, width) -> act -> Linear(width, out)``;
PyTorch default (Kaiming-uniform) initialization.
"""

from __future__ import annotations

import torch
from torch import nn


def activation(name: str) -> nn.Module:
    key = str(name).lower().replace("-", "").replace("_", "")
    if key == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    if key == "gelu":
        return nn.GELU()
    if key == "relu":
        return nn.ReLU()
    raise ValueError(f"Unknown activation {name!r}.")


def _flat_dim(features: nn.Module, in_channels: int, image_shape) -> int:
    with torch.no_grad():
        return int(features(torch.zeros(1, in_channels, *image_shape)).numel())


class QCDConvNet(nn.Module):
    def __init__(self, in_channels: int, image_shape, out_dim: int, activation: str = "leakyrelu",
                 init: str = "xavier_uniform"):
        super().__init__()
        act = globals()["activation"]
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, stride=2), act(activation),
            nn.Conv2d(64, 128, kernel_size=5, stride=2), act(activation),
            nn.Conv2d(128, 256, kernel_size=5, stride=3), act(activation),
            nn.Conv2d(256, 512, kernel_size=3, stride=1), act(activation),
            nn.Flatten(),
        )
        flat = _flat_dim(self.features, in_channels, image_shape)
        self.head = nn.Sequential(nn.Linear(flat, 512), act(activation), nn.Linear(512, out_dim))
        if init == "xavier_uniform":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(self.features(x))


class CNNEncoder(nn.Module):
    def __init__(self, in_channels: int, image_shape, out_dim: int, width: int = 512,
                 hidden_channels=(128, 256, 512), activation: str = "gelu"):
        super().__init__()
        act = globals()["activation"]
        layers, ch = [], in_channels
        for k, out_ch in enumerate(hidden_channels):
            layers += [nn.Conv2d(ch, int(out_ch), kernel_size=3, padding=1), act(activation)]
            if k > 0:
                layers.append(nn.AvgPool2d(2))
            ch = int(out_ch)
        layers.append(nn.Flatten())
        self.features = nn.Sequential(*layers)
        flat = _flat_dim(self.features, in_channels, image_shape)
        self.head = nn.Sequential(nn.Linear(flat, width), act(activation), nn.Linear(width, out_dim))

    def forward(self, x):
        return self.head(self.features(x))
