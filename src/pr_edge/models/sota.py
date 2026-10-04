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
from torch import Tensor

from ..config import Config

SUPPORTED = ("mobilenet_v2", "efficientnet_b0")


class _TransferModel(nn.Module):
    """A torchvision model with a config-controlled transfer-learning policy.

    ``requires_grad=False`` alone is insufficient for a frozen feature
    extractor: ``model.train()`` would still update BatchNorm running statistics
    and enable stochastic backbone layers. This wrapper keeps the backbone in
    evaluation mode while returning the replacement classifier to training mode.
    """

    def __init__(self, backbone: nn.Module, freeze_backbone: bool) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone

        for parameter in self.backbone.parameters():
            parameter.requires_grad = not freeze_backbone
        if freeze_backbone:
            for parameter in self.classifier.parameters():
                parameter.requires_grad = True

    @property
    def classifier(self) -> nn.Module:
        """Expose torchvision's classifier without registering it twice."""
        return self.backbone.classifier

    def forward(self, inputs: Tensor) -> Tensor:
        return self.backbone(inputs)

    def train(self, mode: bool = True) -> "_TransferModel":
        super().train(mode)
        if mode and self.freeze_backbone:
            self.backbone.eval()
            self.classifier.train(True)
        return self


def _replace_final_linear(classifier: nn.Module, num_classes: int) -> nn.Linear:
    """Replace the final linear descendant while preserving its input width."""
    target: tuple[nn.Module, str, nn.Linear] | None = None
    for parent in classifier.modules():
        for name, child in parent.named_children():
            if isinstance(child, nn.Linear):
                target = parent, name, child
    if target is None:
        raise ValueError("torchvision classifier contains no nn.Linear prediction layer")

    parent, name, previous = target
    replacement = nn.Linear(
        previous.in_features,
        num_classes,
        bias=previous.bias is not None,
        device=previous.weight.device,
        dtype=previous.weight.dtype,
    )
    setattr(parent, name, replacement)
    return replacement


def build_sota(cfg: Config) -> nn.Module:
    """Build a supported torchvision backbone with a 10-class prediction head.

    The model registry and final-linear discovery are shared by both supported
    architectures. ``pretrained`` selects ImageNet's default weights;
    ``freeze_backbone`` selects the frozen-feature-extractor or full-fine-tuning
    phase. Input resolution is deliberately absent from model construction:
    both networks use adaptive pooling and accept the configured 64 or 224.
    """
    from torchvision import models

    model_cfg = cfg.model
    if model_cfg.name not in SUPPORTED:
        raise ValueError(f"unsupported SOTA model '{model_cfg.name}'. Known: {', '.join(SUPPORTED)}")
    if type(model_cfg.num_classes) is not int or model_cfg.num_classes <= 0:
        raise ValueError("model.num_classes must be a positive integer")
    if type(model_cfg.dropout) not in (int, float) or not 0 <= model_cfg.dropout <= 1:
        raise ValueError("model.dropout must lie between 0 and 1")
    if type(model_cfg.pretrained) is not bool:
        raise ValueError("model.pretrained must be a boolean")
    if type(model_cfg.freeze_backbone) is not bool:
        raise ValueError("model.freeze_backbone must be a boolean")

    weights = models.get_model_weights(model_cfg.name).DEFAULT if model_cfg.pretrained else None
    backbone = models.get_model(model_cfg.name, weights=weights)
    classifier = getattr(backbone, "classifier", None)
    if not isinstance(classifier, nn.Module):
        raise ValueError(f"torchvision model '{model_cfg.name}' exposes no classifier module")

    _replace_final_linear(classifier, model_cfg.num_classes)
    for module in classifier.modules():
        if isinstance(module, nn.Dropout):
            module.p = float(model_cfg.dropout)
    return _TransferModel(backbone, model_cfg.freeze_backbone)
