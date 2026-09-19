"""Model registry.

``build_model`` is the single entry point every script uses, so adding an
architecture means adding a registry entry, not editing the training loop.
"""

from __future__ import annotations

import torch.nn as nn

from ..config import Config
from ._smoke import SmokeNet
from .model_a import ModelA
from .model_b import ModelB
from .sota import build_sota

__all__ = ["build_model", "ModelA", "ModelB", "build_sota", "SmokeNet"]

_CUSTOM = {"model_a": ModelA, "model_b": ModelB}
_SOTA = {"mobilenet_v2", "efficientnet_b0"}


def build_model(cfg: Config) -> nn.Module:
    """Construct the model named by ``cfg.model.name``.

    In smoke mode the named architecture is bypassed entirely in favour of a
    tiny stand-in. That keeps #1 from having to implement #9, #10 and #19 just
    to make CI run, and it keeps the smoke test fast: the point is to exercise
    the config-to-artifacts wiring, not the architecture.
    """
    if cfg.training.smoke:
        return SmokeNet(cfg.model)

    name = cfg.model.name
    if name in _CUSTOM:
        return _CUSTOM[name](cfg.model)
    if name in _SOTA:
        return build_sota(cfg)

    known = sorted(_CUSTOM) + sorted(_SOTA)
    raise ValueError(f"unknown model '{name}'. Known models: {known}")
