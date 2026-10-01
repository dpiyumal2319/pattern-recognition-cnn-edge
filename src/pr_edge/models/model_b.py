"""Model B - the depthwise separable CNN (issue #10).

Model A's structure with every standard convolution replaced by a depthwise
separable block, per docs/slides/CNNs.pdf p43-p45 and the MobileNets paper.

This is a pure substitution: the stage widths ([32, 64, 128]), valid padding,
2x2 pooling, dense head (128 -> 64 -> 10) and dropout are Model A's, and only
two things change - each 3x3 convolution becomes a depthwise separable block,
and Flatten becomes global average pooling (``head: gap`` in the config). Any
difference between the two models in the report is therefore attributable to
those two levers alone.

HARD CONSTRAINT: total trainable parameters must not exceed 100,000. With the
committed config Model B has 36,968 (Model A: 692,106) and needs 3,725,196
multiply-accumulates per image (Model A: 30,038,016).

``tests/test_model_b.py`` asserts the budget and CI enforces it, so a violation
cannot be merged.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .model_a import KERNEL_SIZE, ModelA


class DepthwiseSeparableBlock(nn.Module):
    """Depthwise 3x3 followed by pointwise 1x1.

    Kept as two separately inspectable stages so the per-layer parameter and
    MAC derivation in #11 can name each one, mirroring the worked example in
    CNNs.pdf p44:

        regular 3x3, 256 -> 256 :  9 x 256 x 256  ~= 590,000 ops
        depthwise 3x3           :  9 x 256        ~=   2,300 ops
        pointwise 1x1, 256->256 :      256 x 256  ~=  65,500 ops
        total                   :                 ~=  67,800 ops  (8.7x speedup)

    p44 draws the block as those two steps with nothing between them, so there
    is no activation inside it: the block is a drop-in replacement for one
    standard convolution, and the ReLU that follows a convolution in Model A
    follows the whole block here.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        # Step 1: one 3x3 filter per input channel. groups=in_ch is p45's
        # group-wise convolution with G = C_in. Unpadded, like Model A, so
        # both models produce the same spatial sizes stage by stage.
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, kernel_size=KERNEL_SIZE, stride=stride, groups=in_ch,
        )
        # Step 2: a 1x1 convolution mixing the channels, C_in -> C_out.
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ModelB(ModelA):
    """Lightweight CNN, <= 100k trainable parameters. Returns raw logits.

    Same input and output shapes as :class:`~pr_edge.models.model_a.ModelA` so
    the comparison is like-for-like. It inherits Model A's construction,
    validation and 64x64 input check, overriding only the per-stage
    convolution - the rest of the difference (global average pooling instead
    of Flatten) comes from ``head: gap`` in ``configs/model_b.yaml``.
    """

    def _convolution(self, in_ch: int, out_ch: int) -> nn.Module:
        return DepthwiseSeparableBlock(in_ch, out_ch)
