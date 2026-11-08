"""Small 1D ResNet for PTB-XL multi-label superclass classification."""

from __future__ import annotations

import torch
from torch import nn

CLASSES = ("NORM", "MI", "STTC", "CD", "HYP")


class ResidualBlock(nn.Module):
    """Two-convolution residual block for one-dimensional ECG signals."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.activation(x + residual)


class ResNet1D(nn.Module):
    """Compact ResNet that returns one independent logit per superclass."""

    def __init__(self, in_leads: int, n_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_leads, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.stage1 = self._make_stage(32, 32, blocks=2, stride=1)
        self.stage2 = self._make_stage(32, 64, blocks=2, stride=2)
        self.stage3 = self._make_stage(64, 128, blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(128, n_classes)

    @staticmethod
    def _make_stage(
        in_channels: int, out_channels: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride=stride)]
        layers.extend(ResidualBlock(out_channels, out_channels) for _ in range(blocks - 1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(self.dropout(x))


def build_resnet1d(in_leads: int = 1, n_classes: int = len(CLASSES)) -> ResNet1D:
    """Build the small classifier used for clean and corrupted ECG experiments."""
    if in_leads < 1 or n_classes < 1:
        raise ValueError("in_leads and n_classes must be positive")
    return ResNet1D(in_leads=in_leads, n_classes=n_classes)
