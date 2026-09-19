"""Pretrained lightweight backbones for transfer learning (issue #19).

Loads a torchvision backbone with ImageNet weights, replaces the classifier
head for the 10 EuroSAT classes, and applies the freeze policy - the pattern on
the final slide of docs/slides/CNNs.pdf: "train new prediction layer(s), keep
frozen or fine-tune".

Resolution policy, decided in planning and to be justified in the report:
primary runs feed 64x64, the same input the custom models see, because that is
the only fair edge comparison. One supplementary MobileNetV2 run at 224x224
shows what ImageNet-native resolution buys. Both numbers go in the report, and
the pair is what lets task 6 separate "pretrained features are better" from
"more input pixels are better".
"""

from __future__ import annotations

import torch.nn as nn

from ..config import Config

SUPPORTED = ("mobilenet_v2", "efficientnet_b0")


def build_sota(cfg: Config) -> nn.Module:
    """Build a pretrained backbone with a replaced head.

    Head replacement must be generic across architectures rather than
    per-model special-casing. Implemented by #19.
    """
    raise NotImplementedError("build_sota is issue #19")
