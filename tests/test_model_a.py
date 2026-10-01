"""Model A, the standard CNN baseline. Owned by #9.

Every expected parameter count here is re-derived from the CNNs.pdf formulas
rather than copied from the model, so these tests are an independent check on
the architecture - the same arithmetic #11 will put in the report:

    conv  : (k x k x C_in + 1) x C_out
    dense : (in_features + 1) x out_features
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from pr_edge.config import ModelConfig, load_config
from pr_edge.models import ModelA, build_model
from pr_edge.paths import CONFIGS_DIR
from pr_edge.profile import count_parameters, layer_table

K = 3  # kernel size


def _params(model: nn.Module) -> int:
    return count_parameters(model)[0]


def _expected_params(channels, fc_units, in_ch=3, size=64, head="flatten", classes=10) -> int:
    """Hand derivation: valid 3x3 conv, then 2x2 pooling, per stage."""
    total = 0
    for out_ch in channels:
        total += (K * K * in_ch + 1) * out_ch
        size = (size - K + 1) // 2
        in_ch = out_ch
    width = in_ch if head == "gap" else in_ch * size * size
    for units in [*fc_units, classes]:
        total += (width + 1) * units
        width = units
    return total


@pytest.fixture
def cfg() -> ModelConfig:
    return load_config(CONFIGS_DIR / "model_a.yaml").model


def test_forward_pass_shape(cfg):
    """(N, 3, 64, 64) -> (N, 10)."""
    model = ModelA(cfg)
    assert model(torch.randn(4, 3, 64, 64)).shape == (4, 10)


def test_registry_builds_model_a_from_its_config():
    model = build_model(load_config(CONFIGS_DIR / "model_a.yaml"))
    assert isinstance(model, ModelA)


def test_returns_logits_not_probabilities(cfg):
    """Rows must not sum to 1 - softmax belongs in the loss."""
    torch.manual_seed(0)
    model = ModelA(cfg).eval()
    logits = model(torch.randn(8, 3, 64, 64))
    assert not torch.allclose(logits.sum(dim=1), torch.ones(8))
    assert (logits < 0).any()


def test_committed_config_has_the_hand_derived_parameter_count(cfg):
    """692,106 = 93,248 in the convolutions + 598,858 in the dense head."""
    assert cfg.channels == [32, 64, 128] and cfg.fc_units == [128, 64]
    assert _expected_params([32, 64, 128], [128, 64]) == 692_106
    assert _params(ModelA(cfg)) == 692_106


def test_dense_head_holds_most_of_the_parameters(cfg):
    """CNNs.pdf p54: the majority of parameters reside in the dense layers."""
    model = ModelA(cfg)
    dense = sum(p.numel() for m in model.modules() if isinstance(m, nn.Linear) for p in m.parameters())
    assert dense / _params(model) > 0.8


def test_reproduces_the_cnns_pdf_p54_example():
    """The same code, sized like p54's 28x28x1 example, gives its 232,650 parameters."""

    class P54(ModelA):
        IN_CHANNELS = 1
        INPUT_SIZE = 28

    model = P54(ModelConfig(name="model_a", channels=[32, 64], fc_units=[128, 64], head="flatten"))
    assert _params(model) == 232_650
    shapes = [row["output_shape"] for row in layer_table(model, (1, 1, 28, 28))]
    # p54's table in channels-first order: 26x26x32, 13x13x32, 11x11x64, 5x5x64, 1600, ...
    assert shapes == [[32, 26, 26], [32, 13, 13], [64, 11, 11], [64, 5, 5],
                      [1600], [128], [64], [10]]


@pytest.mark.parametrize("change", [
    {"channels": [16, 32, 64]},
    {"channels": [32, 64]},
    {"channels": [32, 64, 128, 256]},
    {"fc_units": [256]},
    {"fc_units": []},
    {"head": "gap"},
    {"num_classes": 5},
])
def test_architecture_is_config_driven(cfg, change):
    """Changing the config changes the model exactly as the formula predicts."""
    changed = replace(cfg, **change)
    expected = _expected_params(
        changed.channels, changed.fc_units, head=changed.head, classes=changed.num_classes,
    )
    model = ModelA(changed)
    assert _params(model) == expected != _params(ModelA(cfg))
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, changed.num_classes)


def test_dropout_sits_before_each_hidden_dense_layer_only(cfg):
    layers = list(ModelA(cfg).classifier)
    dense = [i for i, m in enumerate(layers) if isinstance(m, nn.Linear)]
    for index in dense[:-1]:
        assert isinstance(layers[index - 1], nn.Dropout)
        assert layers[index - 1].p == cfg.dropout
    # No dropout directly in front of the class scores.
    assert not isinstance(layers[dense[-1] - 1], nn.Dropout)
    assert sum(isinstance(m, nn.Dropout) for m in layers) == len(cfg.fc_units)


def test_zero_dropout_adds_no_dropout_layers(cfg):
    model = ModelA(replace(cfg, dropout=0.0))
    assert not any(isinstance(m, nn.Dropout) for m in model.modules())
    assert _params(model) == 692_106


def test_eval_mode_is_deterministic(cfg):
    model = ModelA(cfg).eval()
    x = torch.randn(2, 3, 64, 64)
    assert torch.equal(model(x), model(x))


def test_there_is_no_softmax_or_batch_norm(cfg):
    """Softmax is the loss's job; batch norm is not in the course decks."""
    kinds = {type(m) for m in ModelA(cfg).modules()}
    assert not kinds & {nn.Softmax, nn.LogSoftmax, nn.BatchNorm2d, nn.BatchNorm1d}


@pytest.mark.parametrize("shape", [(1, 3, 224, 224), (1, 3, 32, 32), (1, 1, 64, 64)])
def test_wrong_input_shape_fails_with_a_clear_message(cfg, shape):
    with pytest.raises(ValueError, match=r"expects input of shape \(N, 3, 64, 64\)"):
        ModelA(cfg)(torch.randn(*shape))


@pytest.mark.parametrize("change,message", [
    ({"channels": []}, "channels"),
    ({"channels": [32, 0]}, "channels"),
    ({"fc_units": [-1]}, "fc_units"),
    ({"head": "dense"}, "head"),
    ({"dropout": 1.0}, "dropout"),
    ({"channels": [8] * 6}, "fewer stages"),
])
def test_invalid_config_is_rejected(cfg, change, message):
    with pytest.raises(ValueError, match=message):
        ModelA(replace(cfg, **change))


def test_layer_table_matches_the_architecture(cfg):
    model = ModelA(cfg)
    rows = layer_table(model, (1, 3, 64, 64))
    assert [r["type"] for r in rows] == [
        "Conv2d", "MaxPool2d", "Conv2d", "MaxPool2d", "Conv2d", "MaxPool2d",
        "Flatten", "Linear", "Linear", "Linear",
    ]
    assert [r["output_shape"] for r in rows] == [
        [32, 62, 62], [32, 31, 31], [64, 29, 29], [64, 14, 14], [128, 12, 12], [128, 6, 6],
        [4608], [128], [64], [10],
    ]
    assert [r["params"] for r in rows] == [896, 0, 18_496, 0, 73_856, 0, 0, 589_952, 8_256, 650]
    assert sum(r["params"] for r in rows) == _params(model)


def test_layer_table_leaves_the_model_as_it_found_it(cfg):
    model = ModelA(cfg).train()
    layer_table(model, (1, 3, 64, 64))
    assert model.training
    assert not model._forward_hooks
    assert all(not m._forward_hooks for m in model.modules())
