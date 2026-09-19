"""Model A - the standard CNN baseline (issue #9).

Interleaved standard 2D convolutions and max-pooling, then a classifier head,
matching the topology in figure 1 of the assignment brief. This is the heavy
baseline that Model B's depthwise separable variant is measured against, so it
should stay deliberately conventional.

Model A is allowed a Flatten -> Dense head. That is part of what makes it the
heavy baseline, and the contrast with Model B's global average pooling is a
real finding for the report.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig


class ModelA(nn.Module):
    """Standard Conv/ReLU/MaxPool stack. Returns raw logits, never softmax.

    Args:
        cfg: model section of the run config. ``channels`` gives the width of
            each conv stage, ``head`` selects ``flatten`` or ``gap``.

    Implemented by #9. See docs/slides/CNNs.pdf for convolution arithmetic,
    padding, stride and pooling.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError("ModelA is issue #9")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, num_classes) logits."""
        raise NotImplementedError("ModelA is issue #9")
