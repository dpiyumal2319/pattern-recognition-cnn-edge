"""Pretrained lightweight backbones and transfer-learning policy (issue #19).

Tests use ``weights=None`` or a patched torchvision registry, so CI never
touches the network. Real ImageNet weight selection is verified at the registry
boundary without downloading the checkpoint.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from pr_edge.config import Config, DataConfig, ModelConfig, load_config
from pr_edge.models import build_model
from pr_edge.models.sota import SUPPORTED, build_sota
from pr_edge.paths import CONFIGS_DIR


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


@pytest.mark.parametrize("name", SUPPORTED)
@pytest.mark.parametrize("size", [64, 224])
def test_accepts_both_sixty_four_and_two_twenty_four_input(name, size):
    model = build_sota(_config(name, input_size=size)).eval()
    with torch.inference_mode():
        output = model(torch.randn(1, 3, size, size))
    assert output.shape == (1, 10)


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


def test_frozen_backbone_stays_in_eval_mode_while_head_trains(monkeypatch):
    from torchvision import models

    monkeypatch.setattr(models, "get_model", lambda *args, **kwargs: _NestedClassifierNet())
    model = build_sota(_config(freeze_backbone=True))
    model.train()
    assert model.training is True
    assert model.backbone.training is False
    assert model.backbone.features[1].training is False
    assert model.classifier.training is True
    model.eval()
    assert model.classifier.training is False


def test_unfrozen_backbone_enters_training_mode(monkeypatch):
    from torchvision import models

    monkeypatch.setattr(models, "get_model", lambda *args, **kwargs: _NestedClassifierNet())
    model = build_sota(_config(freeze_backbone=False))
    model.train()
    assert model.backbone.training is True
    assert model.backbone.features[1].training is True


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


def test_all_sota_configs_reuse_manifest_and_resolution_policy():
    expected = {
        "mobilenetv2_64.yaml": ("mobilenet_v2", 64),
        "efficientnetb0_64.yaml": ("efficientnet_b0", 64),
        "mobilenetv2_224.yaml": ("mobilenet_v2", 224),
        "mobilenetv2_64_ft.yaml": ("mobilenet_v2", 64),
        "mobilenetv2_224_ft.yaml": ("mobilenet_v2", 224),
        "mobilenetv2_224_ft_b128.yaml": ("mobilenet_v2", 224),
    }
    for filename, (name, size) in expected.items():
        cfg = load_config(CONFIGS_DIR / filename)
        assert cfg.model.name == name
        assert cfg.data.input_size == size
        assert Path(cfg.data.manifest).as_posix() == "data/splits.csv"


@pytest.mark.parametrize("frozen,fine_tuned", [
    ("mobilenetv2_64", "mobilenetv2_64_ft"),
    ("mobilenetv2_224", "mobilenetv2_224_ft_b128"),
])
def test_fine_tune_config_differs_from_frozen_partner_only_in_policy(frozen, fine_tuned):
    """The _ft run is the 'fine-tune' half of CNNs.pdf p58's 'keep frozen or
    fine-tune'. Anything else differing would confound the comparison."""
    base = load_config(CONFIGS_DIR / f"{frozen}.yaml")
    ft = load_config(CONFIGS_DIR / f"{fine_tuned}.yaml")

    assert base.model.freeze_backbone is True
    assert ft.model.freeze_backbone is False
    assert ft.optimizer.lr < base.optimizer.lr

    # Normalise the two intended differences away; everything else must match.
    aligned = replace(
        ft,
        run_id=base.run_id,
        model=replace(ft.model, freeze_backbone=base.model.freeze_backbone),
        optimizer=replace(ft.optimizer, lr=base.optimizer.lr),
    )
    assert aligned == base


@pytest.mark.parametrize("low,high", [
    ("mobilenetv2_64", "mobilenetv2_224"),
    ("mobilenetv2_64_ft", "mobilenetv2_224_ft_b128"),
])
def test_resolution_ablation_differs_only_in_input_size(low, high):
    """64 vs 224 must isolate resolution. An earlier 224 run used batch 32
    against 128, which also changed the number of updates per epoch; it is kept
    as mobilenetv2_224_ft and excluded from this guard (#53)."""
    small = load_config(CONFIGS_DIR / f"{low}.yaml")
    large = load_config(CONFIGS_DIR / f"{high}.yaml")

    assert (small.data.input_size, large.data.input_size) == (64, 224)
    aligned = replace(
        large,
        run_id=small.run_id,
        data=replace(large.data, input_size=small.data.input_size),
    )
    assert aligned == small


@pytest.mark.parametrize("change,message", [
    ({"name": "resnet18"}, "unsupported"),
    ({"num_classes": 0}, "num_classes"),
    ({"num_classes": True}, "num_classes"),
    ({"dropout": -0.1}, "dropout"),
    ({"dropout": 1.1}, "dropout"),
    ({"pretrained": 1}, "pretrained"),
    ({"freeze_backbone": 1}, "freeze_backbone"),
])
def test_invalid_sota_config_fails_loudly(change, message):
    cfg = _config()
    cfg = replace(cfg, model=replace(cfg.model, **change))
    with pytest.raises(ValueError, match=message):
        build_sota(cfg)


def test_missing_or_non_linear_classifier_fails_loudly(monkeypatch):
    from torchvision import models

    class NoLinear(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.ReLU()

    monkeypatch.setattr(models, "get_model", lambda *args, **kwargs: NoLinear())
    with pytest.raises(ValueError, match="no nn.Linear"):
        build_sota(_config())
