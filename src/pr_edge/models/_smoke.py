"""A deliberately tiny network used only by smoke mode.

Not part of the assignment. It exists so that ``--smoke`` can exercise the full
config -> train -> evaluate -> profile -> artifacts path without depending on
Model A (#9), Model B (#10) or the SOTA harness (#19) being implemented yet,
and without CI spending a minute on a real architecture.

Keep it trivial. If this ever grows features, something has gone wrong.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig


class SmokeNet(nn.Module):
    """One conv, global average pool, one linear. Returns raw logits."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(8, cfg.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)
