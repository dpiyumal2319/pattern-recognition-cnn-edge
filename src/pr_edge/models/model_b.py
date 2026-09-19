"""Model B - the depthwise separable CNN (issue #10).

Model A's structure with every standard convolution replaced by a depthwise
separable block, per docs/slides/CNNs.pdf and the MobileNets paper.

HARD CONSTRAINT: total trainable parameters must not exceed 100,000. The single
most effective lever is the classifier head - replacing Flatten -> Dense with
global average pooling followed by one small Linear. Plan the channel widths
around that rather than shaving convolutions, which costs the feature extractor
the capacity the accuracy actually comes from.

``tests/test_model_b.py`` asserts the budget and CI enforces it, so a violation
cannot be merged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 followed by pointwise 1x1.

    Kept as two separately inspectable stages so the per-layer parameter and
    MAC derivation in #11 can name each one, mirroring the worked example in
    CNNs.pdf:

        regular 3x3, 256 -> 256 :  9 x 256 x 256  ~= 590,000 ops
        depthwise 3x3           :  9 x 256        ~=   2,300 ops
        pointwise 1x1, 256->256 :      256 x 256  ~=  65,500 ops
        total                   :                 ~=  67,800 ops  (8.7x speedup)

    Implemented by #10.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        raise NotImplementedError("DepthwiseSeparableBlock is issue #10")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("DepthwiseSeparableBlock is issue #10")


class ModelB(nn.Module):
    """Lightweight CNN, <= 100k trainable parameters. Returns raw logits.

    Same input and output shapes as :class:`~pr_edge.models.model_a.ModelA` so
    the comparison is like-for-like. Implemented by #10.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        raise NotImplementedError("ModelB is issue #10")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, H, W) -> (N, num_classes) logits."""
        raise NotImplementedError("ModelB is issue #10")
