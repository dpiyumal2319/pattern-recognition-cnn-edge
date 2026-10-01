"""Model A - the standard CNN baseline (issue #9).

Interleaved standard 2D convolutions and max-pooling, then a classifier head,
matching the topology in figure 1 of the assignment brief. This is the heavy
baseline that Model B's depthwise separable variant is measured against, so it
should stay deliberately conventional.

The template is the lecturer's own example CNN in docs/slides/CNNs.pdf p54:

    Conv2D(32, 3x3, relu) -> MaxPool 2x2 -> Conv2D(64, 3x3, relu) -> MaxPool 2x2
    -> Flatten -> Dense(128, relu) -> Dense(64, relu) -> Dense(10, softmax)

Model A keeps that shape - valid (unpadded) convolutions, no batch
normalisation - with one more conv stage for the larger 64x64x3 input. Stage
widths and dense widths come from the config, so the parameter arithmetic in
#11 can be stated from the config alone.

Model A is allowed a Flatten -> Dense head. That is part of what makes it the
heavy baseline, and the contrast with Model B's global average pooling is a
real finding for the report: as p54 itself notes, the majority of parameters
reside within the fully-connected dense layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ..config import ModelConfig

# 3x3 kernels and 2x2/stride-2 max pooling, as in the CNNs.pdf p54 example.
# p34 gives F = 2, S = 2 as the usual pooling choice.
KERNEL_SIZE = 3
POOL_SIZE = 2

HEADS = ("flatten", "gap")


def _conv_output_size(size: int) -> int:
    """Spatial size after one valid 3x3 convolution and one 2x2 max pool.

    CNNs.pdf p32: an unpadded K x K filter maps W to W - K + 1. Pooling then
    halves it, flooring odd sizes (29 -> 14), exactly as p54's 11 -> 5 does.
    """
    return (size - KERNEL_SIZE + 1) // POOL_SIZE


class ModelA(nn.Module):
    """Standard Conv/ReLU/MaxPool stack. Returns raw logits, never softmax.

    Args:
        cfg: model section of the run config. ``channels`` gives the width of
            each conv stage, ``fc_units`` the hidden dense widths, ``head``
            selects ``flatten`` (as in p54) or ``gap`` (global average pooling),
            and ``dropout`` is applied before each hidden dense layer.

    With the committed config (channels [32, 64, 128], fc_units [128, 64]) the
    shapes are 64 -> 62 -> 31 -> 29 -> 14 -> 12 -> 6, a 128 x 6 x 6 = 4608-wide
    flatten, and 692,106 parameters, 86.5% of them in the dense head.
    """

    # The brief caps input resolution at 64x64 RGB. These are class attributes
    # rather than constructor arguments because the frozen signature is
    # ModelA(cfg) and the model section cannot see data.input_size; the flatten
    # width must nonetheless be known before the first forward pass.
    IN_CHANNELS = 3
    INPUT_SIZE = 64

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        channels = list(cfg.channels)
        fc_units = list(cfg.fc_units)
        if not channels or any(type(c) is not int or c <= 0 for c in channels):
            raise ValueError(f"model.channels must be a non-empty list of positive ints, got {channels}")
        if any(type(u) is not int or u <= 0 for u in fc_units):
            raise ValueError(f"model.fc_units must be positive ints, got {fc_units}")
        if cfg.head not in HEADS:
            raise ValueError(f"model.head must be one of {HEADS}, got {cfg.head!r}")
        if not 0 <= cfg.dropout < 1:
            raise ValueError(f"model.dropout must be in [0, 1), got {cfg.dropout}")

        # Feature extractor: conv -> ReLU -> pool per stage. p33 gives this as
        # the canonical block ordering.
        layers: list[nn.Module] = []
        size = self.INPUT_SIZE
        in_ch = self.IN_CHANNELS
        for out_ch in channels:
            layers += [
                self._convolution(in_ch, out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=POOL_SIZE, stride=POOL_SIZE),
            ]
            size = _conv_output_size(size)
            in_ch = out_ch
        if size < 1:
            raise ValueError(
                f"{len(channels)} conv stages shrink a {self.INPUT_SIZE}x{self.INPUT_SIZE} "
                "input to nothing; use fewer stages"
            )
        self.features = nn.Sequential(*layers)
        self.feature_size = size

        # Classifier head. "flatten" keeps every spatial position (the heavy
        # baseline); "gap" averages each channel to one value first.
        head: list[nn.Module] = []
        if cfg.head == "gap":
            head.append(nn.AdaptiveAvgPool2d(1))
            width = in_ch
        else:
            width = in_ch * size * size
        head.append(nn.Flatten())
        for units in fc_units:
            if cfg.dropout > 0:
                head.append(nn.Dropout(cfg.dropout))
            head += [nn.Linear(width, units), nn.ReLU(inplace=True)]
            width = units
        # Output layer: raw logits. Softmax belongs to CrossEntropyLoss, and
        # no dropout sits directly in front of the class scores.
        head.append(nn.Linear(width, cfg.num_classes))
        self.classifier = nn.Sequential(*head)

    def _convolution(self, in_ch: int, out_ch: int) -> nn.Module:
        """The convolution of one stage: a standard valid 3x3.

        The single point Model B overrides - it swaps this for a depthwise
        separable block and changes nothing else, which is what makes the
        A-vs-B comparison attributable to that substitution alone.
        """
        return nn.Conv2d(in_ch, out_ch, kernel_size=KERNEL_SIZE)  # valid: no padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(N, 3, 64, 64) -> (N, num_classes) logits."""
        expected = (self.IN_CHANNELS, self.INPUT_SIZE, self.INPUT_SIZE)
        if tuple(x.shape[1:]) != expected:
            # Without this, a 224x224 batch fails deep inside Linear with a
            # matrix-shape message that names neither the model nor the fix.
            raise ValueError(
                f"ModelA expects input of shape (N, {expected[0]}, {expected[1]}, {expected[2]}), "
                f"got {tuple(x.shape)}. Set data.input_size to {self.INPUT_SIZE}."
            )
        return self.classifier(self.features(x))
