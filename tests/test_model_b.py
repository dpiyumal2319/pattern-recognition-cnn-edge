"""Model B, the depthwise separable CNN. Owned by #10.

The brief's hard constraint - at most 100,000 trainable parameters - is
asserted here on every committed Model B config, so CI fails any change that
breaks it. Expected counts are re-derived from the CNNs.pdf formulas rather
than copied from the model:

    depthwise 3x3 : (k x k + 1) x C_in          one filter per input channel
    pointwise 1x1 : (C_in + 1) x C_out
    dense         : (in_features + 1) x out_features
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from pr_edge.config import load_config
from pr_edge.models import ModelA, ModelB, build_model
from pr_edge.models.model_b import DepthwiseSeparableBlock
from pr_edge.paths import CONFIGS_DIR
from pr_edge.profile import count_macs, count_parameters, layer_table

BUDGET = 100_000
K = 3
INPUT = (1, 3, 64, 64)
MODEL_B_CONFIGS = ["model_b", "model_b_sgd", "model_b_sgdm"]


def _params(model: nn.Module) -> int:
    return count_parameters(model)[0]


def _trainable(model: nn.Module) -> int:
    return count_parameters(model)[1]


def _expected_params(channels, fc_units, in_ch=3, classes=10) -> int:
    """Hand derivation for Model B: one separable block per stage, GAP head."""
    total = 0
    for out_ch in channels:
        total += (K * K + 1) * in_ch        # depthwise
        total += (in_ch + 1) * out_ch       # pointwise
        in_ch = out_ch
    width = in_ch                           # global average pooling: one value per channel
    for units in [*fc_units, classes]:
        total += (width + 1) * units
        width = units
    return total


def _expected_macs(channels, fc_units, in_ch=3, size=64, classes=10) -> int:
    """CNNs.pdf p43/p27 convention: depthwise K²·C·H'W', pointwise C_in·C_out·H'W'."""
    total = 0
    for out_ch in channels:
        size -= K - 1                       # valid 3x3
        total += K * K * in_ch * size * size + in_ch * out_ch * size * size
        size //= 2                          # 2x2 max pool
        in_ch = out_ch
    width = in_ch
    for units in [*fc_units, classes]:
        total += width * units
        width = units
    return total


@pytest.fixture
def cfg():
    return load_config(CONFIGS_DIR / "model_b.yaml").model


# ---------------------------------------------------------------------------
# The budget - the brief's hard constraint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", MODEL_B_CONFIGS)
def test_trainable_parameters_are_under_one_hundred_thousand(name):
    """THE hard constraint from the brief. CI enforces this, so a Model B
    over budget cannot be merged - including via the optimizer-study configs."""
    model = build_model(load_config(CONFIGS_DIR / f"{name}.yaml"))
    assert isinstance(model, ModelB)
    assert _trainable(model) <= BUDGET


def test_committed_config_has_the_hand_derived_parameter_count(cfg):
    """36,968 = 11,550 in the separable blocks + 25,418 in the GAP head."""
    assert cfg.channels == [32, 64, 128] and cfg.fc_units == [128, 64] and cfg.head == "gap"
    assert _expected_params([32, 64, 128], [128, 64]) == 36_968
    assert _params(ModelB(cfg)) == _trainable(ModelB(cfg)) == 36_968


def test_flatten_head_would_break_the_budget(cfg):
    """Why head: gap is not optional - documents the lever for the report."""
    assert _params(ModelB(replace(cfg, head="flatten"))) > BUDGET


# ---------------------------------------------------------------------------
# Like-for-like with Model A
# ---------------------------------------------------------------------------


def test_forward_pass_shape_matches_model_a(cfg):
    """Like-for-like comparison requires identical input and output shapes."""
    model_a = build_model(load_config(CONFIGS_DIR / "model_a.yaml"))
    x = torch.randn(4, 3, 64, 64)
    assert ModelB(cfg)(x).shape == model_a(x).shape == (4, 10)


def test_rejects_the_same_wrong_inputs_as_model_a(cfg):
    with pytest.raises(ValueError, match=r"expects input of shape \(N, 3, 64, 64\)"):
        ModelB(cfg)(torch.randn(1, 3, 224, 224))


def test_returns_logits_not_probabilities(cfg):
    torch.manual_seed(0)
    logits = ModelB(cfg).eval()(torch.randn(8, 3, 64, 64))
    assert not torch.allclose(logits.sum(dim=1), torch.ones(8))
    assert (logits < 0).any()


def test_differs_from_model_a_only_by_the_two_levers(cfg):
    """Pure substitution: conv -> separable block, Flatten -> GAP, nothing else."""
    model_a = build_model(load_config(CONFIGS_DIR / "model_a.yaml"))
    model_b = ModelB(cfg)
    kinds_a = [type(m).__name__ for m in model_a.features]
    kinds_b = [type(m).__name__ for m in model_b.features]
    assert kinds_b == [k.replace("Conv2d", "DepthwiseSeparableBlock") for k in kinds_a]
    head_a = [type(m).__name__ for m in model_a.classifier]
    head_b = [type(m).__name__ for m in model_b.classifier]
    assert head_b == ["AdaptiveAvgPool2d", *head_a]
    # The dense stack after the pooling step is identical in shape and dropout.
    linear = lambda m: [(l.out_features) for l in m.classifier if isinstance(l, nn.Linear)]
    assert linear(model_a) == linear(model_b) == [128, 64, 10]
    dropout = lambda m: [l.p for l in m.classifier if isinstance(l, nn.Dropout)]
    assert dropout(model_a) == dropout(model_b) == [0.2, 0.2]


def test_registry_builds_model_b(cfg):
    assert type(build_model(load_config(CONFIGS_DIR / "model_b.yaml"))) is ModelB


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------


def test_depthwise_and_pointwise_are_separate_layers():
    """#11 must be able to name each stage in its parameter derivation."""
    block = DepthwiseSeparableBlock(32, 64)
    assert isinstance(block.depthwise, nn.Conv2d) and isinstance(block.pointwise, nn.Conv2d)
    assert block.depthwise.groups == 32 and block.depthwise.kernel_size == (3, 3)
    assert block.depthwise.in_channels == block.depthwise.out_channels == 32
    assert block.pointwise.kernel_size == (1, 1) and block.pointwise.groups == 1
    assert (block.pointwise.in_channels, block.pointwise.out_channels) == (32, 64)
    # CNNs.pdf p44: two steps and nothing between them - no activation inside.
    assert [name for name, _ in block.named_children()] == ["depthwise", "pointwise"]


def test_block_parameters_follow_the_formula():
    block = DepthwiseSeparableBlock(64, 128)
    assert block.depthwise.weight.numel() + block.depthwise.bias.numel() == (9 + 1) * 64
    assert _params(block.pointwise) == (64 + 1) * 128


def test_block_is_valid_padded_like_model_a():
    assert DepthwiseSeparableBlock(3, 32)(torch.randn(1, 3, 64, 64)).shape == (1, 32, 62, 62)


def test_block_stride_is_applied_to_the_depthwise_step():
    block = DepthwiseSeparableBlock(8, 8, stride=2)
    assert block.depthwise.stride == (2, 2) and block.pointwise.stride == (1, 1)


def test_reproduces_the_cnns_pdf_p44_example():
    """3x3, 256 -> 256: 589,824 ops regular vs 2,304 + 65,536 = 67,840 separable.

    The deck rounds these to ~590,000, ~2,300, ~65,500, ~67,800 and 8.7x. Measured
    per output location with a 3x3 input (one output pixel) so the counts are
    exactly the slide's per-location figures.
    """
    location = (1, 256, 3, 3)
    regular, _ = count_macs(nn.Conv2d(256, 256, 3), location)
    separable, per_layer = count_macs(DepthwiseSeparableBlock(256, 256), location)
    assert regular == 589_824
    assert per_layer == {"depthwise": 2_304, "pointwise": 65_536}
    assert separable == 67_840
    assert regular / separable == pytest.approx(8.69, abs=0.01)


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def test_uses_fewer_macs_than_model_a(cfg):
    """The whole point of the substitution: 3,725,196 vs 30,038,016 per image."""
    model_a = build_model(load_config(CONFIGS_DIR / "model_a.yaml"))
    macs_a, _ = count_macs(model_a, INPUT)
    macs_b, _ = count_macs(ModelB(cfg), INPUT)
    assert macs_b == _expected_macs([32, 64, 128], [128, 64]) == 3_725_196
    assert macs_a == 30_038_016
    assert macs_a / macs_b > 8


@pytest.mark.parametrize("change", [
    {"channels": [16, 32, 64]},
    {"channels": [32, 64]},
    {"fc_units": [64]},
    {"fc_units": []},
    {"num_classes": 5},
])
def test_architecture_is_config_driven(cfg, change):
    changed = replace(cfg, **change)
    model = ModelB(changed)
    assert _params(model) == _expected_params(
        changed.channels, changed.fc_units, classes=changed.num_classes,
    ) != _params(ModelB(cfg))
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, changed.num_classes)


def test_layer_table_names_both_stages_of_every_block(cfg):
    model = ModelB(cfg)
    rows = layer_table(model, INPUT)
    convs = [r for r in rows if r["type"] == "Conv2d"]
    assert [r["name"] for r in convs] == [
        "features.0.depthwise", "features.0.pointwise",
        "features.3.depthwise", "features.3.pointwise",
        "features.6.depthwise", "features.6.pointwise",
    ]
    assert [r["groups"] for r in convs] == [3, 1, 32, 1, 64, 1]
    assert [r["kernel_size"] for r in convs] == [[3, 3], [1, 1]] * 3
    assert [r["output_shape"] for r in convs] == [
        [3, 62, 62], [32, 62, 62], [32, 29, 29], [64, 29, 29], [64, 12, 12], [128, 12, 12],
    ]
    assert [r["params"] for r in convs] == [30, 128, 320, 2_112, 640, 8_320]
    assert sum(r["params"] for r in rows) == _params(model)
    assert all(r["kernel_size"] is None and r["groups"] is None
               for r in rows if r["type"] != "Conv2d")


def test_model_b_is_still_a_model_a_for_shared_behaviour(cfg):
    """ModelB inherits construction and validation, overriding only the conv."""
    assert issubclass(ModelB, ModelA)
    with pytest.raises(ValueError, match="channels"):
        ModelB(replace(cfg, channels=[]))
