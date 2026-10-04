"""Pretrained lightweight backbones and transfer-learning policy (issue #19).

Tests use ``weights=None`` or a patched torchvision registry, so CI never
touches the network. Real ImageNet weight selection is verified at the registry
boundary without downloading the checkpoint.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from pr_edge.config import Config, DataConfig, ModelConfig
from pr_edge.models import build_model
from pr_edge.models.sota import SUPPORTED, build_sota


def _config(
    name: str = "mobilenet_v2",
    *,
    pretrained: bool = False,
    freeze_backbone: bool = False,
    input_size: int = 64,
    num_classes: int = 10,
    dropout: float = 0.2,
) -> Config:
    return Config(
        run_id="sota_test",
        data=DataConfig(input_size=input_size),
        model=ModelConfig(
            name=name,
            num_classes=num_classes,
            channels=[],
            head="gap",
            dropout=dropout,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
        ),
    )


def _linear_layers(module: nn.Module) -> list[nn.Linear]:
    return [layer for layer in module.modules() if isinstance(layer, nn.Linear)]


@pytest.mark.parametrize("name", SUPPORTED)
def test_head_is_replaced_for_ten_classes(name):
    model = build_sota(_config(name))
    linear = _linear_layers(model.classifier)
    assert linear
    assert linear[-1].out_features == 10
    assert all(layer.out_features != 1000 for layer in linear)


@pytest.mark.parametrize("name", SUPPORTED)
def test_freeze_backbone_leaves_only_the_head_trainable(name):
    model = build_sota(_config(name, freeze_backbone=True))
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    head = {id(parameter) for parameter in model.classifier.parameters()}
    assert trainable == head
    trainable_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert 0 < trainable_count < sum(parameter.numel() for parameter in model.parameters())


@pytest.mark.parametrize("name", SUPPORTED)
def test_unfrozen_phase_makes_every_parameter_trainable(name):
    model = build_sota(_config(name, freeze_backbone=False))
    assert all(parameter.requires_grad for parameter in model.parameters())


class _NestedClassifierNet(nn.Module):
    """Small stand-in proving replacement depends on structure, not model name."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.ReLU())
        self.classifier = nn.Sequential(
            nn.Dropout(0.75),
            nn.Sequential(nn.Identity(), nn.Linear(4, 1000, bias=False)),
        )

    def forward(self, inputs):
        features = self.features(inputs).mean(dim=(2, 3))
        return self.classifier(features)


@pytest.mark.parametrize("name", SUPPORTED)
def test_head_replacement_is_generic_across_architectures(name, monkeypatch):
    from torchvision import models

    calls = []

    def fake_get_model(requested, *, weights):
        calls.append((requested, weights))
        return _NestedClassifierNet()

    monkeypatch.setattr(models, "get_model", fake_get_model)
    model = build_sota(_config(name, dropout=0.35))
    final = _linear_layers(model.classifier)[-1]
    assert calls == [(name, None)]
    assert final.in_features == 4 and final.out_features == 10
    assert final.bias is None
    dropout = next(layer for layer in model.classifier.modules() if isinstance(layer, nn.Dropout))
    assert dropout.p == 0.35


def test_pretrained_selects_torchvision_default_weights(monkeypatch):
    from torchvision import models

    marker = object()
    weights = type("Weights", (), {"DEFAULT": marker})
    calls = {}
    monkeypatch.setattr(
        models,
        "get_model_weights",
        lambda name: calls.setdefault("enum", (name, weights))[1],
    )

    def fake_get_model(name, *, weights):
        calls["model"] = (name, weights)
        return _NestedClassifierNet()

    monkeypatch.setattr(models, "get_model", fake_get_model)
    build_sota(_config(pretrained=True))
    assert calls == {
        "enum": ("mobilenet_v2", weights),
        "model": ("mobilenet_v2", marker),
    }


def test_second_phase_can_restore_frozen_phase_weights_and_unfreeze(monkeypatch):
    from torchvision import models

    monkeypatch.setattr(models, "get_model", lambda *args, **kwargs: _NestedClassifierNet())
    frozen = build_sota(_config(freeze_backbone=True))
    with torch.no_grad():
        _linear_layers(frozen.classifier)[-1].weight.fill_(0.125)
    checkpoint = frozen.state_dict()

    fine_tuning = build_sota(_config(freeze_backbone=False))
    fine_tuning.load_state_dict(checkpoint)
    assert all(parameter.requires_grad for parameter in fine_tuning.parameters())
    assert torch.equal(
        _linear_layers(fine_tuning.classifier)[-1].weight,
        _linear_layers(frozen.classifier)[-1].weight,
    )


@pytest.mark.parametrize("name", SUPPORTED)
def test_model_registry_dispatches_to_sota(name):
    model = build_model(_config(name))
    assert _linear_layers(model.classifier)[-1].out_features == 10
