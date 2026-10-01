"""Independent verification of the model resource profiler.

The expected MAC counts below are derived from ``docs/slides/CNNs.pdf`` rather
than from the project models. That separation matters because issue #11 uses
the profiler to check the report's hand arithmetic; testing it against numbers
copied from those same models could let one shared mistake pass unnoticed.
"""

from __future__ import annotations

import io
import json
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from pr_edge.config import load_config
from pr_edge.paths import CONFIGS_DIR
from pr_edge.profile import count_macs, profile_model, state_dict_bytes
from pr_edge.schemas import validate_profile


def _slide_separable_block() -> nn.Sequential:
    """The two operations shown on CNNs.pdf p44, with readable layer names."""
    return nn.Sequential(OrderedDict([
        ("depthwise", nn.Conv2d(256, 256, 3, groups=256, bias=False)),
        ("pointwise", nn.Conv2d(256, 256, 1, bias=False)),
    ]))


def test_reproduces_the_mobilenet_slide_example():
    """Reproduce the lecturer's per-location operation-count example.

    CNNs.pdf p44 gives a regular 3x3, 256 -> 256 convolution as about
    590,000 operations. The separable equivalent is a 3x3 depthwise step
    (about 2,300) followed by a 1x1 pointwise step (about 65,500), totalling
    about 67,800 and giving an 8.7x speedup. A 3x3 input with no padding
    produces one output location, so the exact values are:

        regular:   3 * 3 * 256 * 256 = 589,824
        depthwise: 3 * 3 * 256       =   2,304
        pointwise: 1 * 1 * 256 * 256 =  65,536
        separable total              =  67,840
    """
    one_location = (1, 256, 3, 3)

    regular, regular_layers = count_macs(
        nn.Conv2d(256, 256, 3, bias=False), one_location,
    )
    separable, separable_layers = count_macs(
        _slide_separable_block(), one_location,
    )

    assert regular_layers == {"": 589_824}
    assert regular == 589_824
    assert separable_layers == {"depthwise": 2_304, "pointwise": 65_536}
    assert separable == 67_840
    assert regular / separable == pytest.approx(8.69, abs=0.01)


def test_macs_are_per_image_not_per_batch():
    """The leading input dimension must not change the per-image result."""
    model = nn.Conv2d(3, 7, 3, padding=1, bias=False)
    one_image = count_macs(model, (1, 3, 16, 16))
    batch_of_32 = count_macs(model, (32, 3, 16, 16))

    assert one_image == batch_of_32
    assert one_image[0] == 3 * 3 * 3 * 7 * 16 * 16


def test_grouped_convolution_divides_by_group_count():
    """A four-group convolution costs one quarter of its dense equivalent."""
    model = nn.Conv2d(8, 12, 3, groups=4, bias=False)

    macs, per_layer = count_macs(model, (1, 8, 5, 5))

    # H' = W' = 3 and each output channel sees 8 / 4 = 2 inputs.
    expected = 3 * 3 * (8 // 4) * 12 * 3 * 3
    dense_equivalent = 3 * 3 * 8 * 12 * 3 * 3
    assert per_layer == {"": expected}
    assert macs == expected == dense_equivalent // 4


def test_macs_per_layer_uses_readable_qualified_names():
    """Issue #11 can quote nested profiler rows directly in its derivation."""
    model = nn.Sequential(OrderedDict([
        ("stem", nn.Conv2d(3, 4, 1, bias=False)),
        ("separable", nn.Sequential(OrderedDict([
            ("depthwise", nn.Conv2d(4, 4, 3, groups=4, bias=False)),
            ("pointwise", nn.Conv2d(4, 8, 1, bias=False)),
        ]))),
    ]))

    _, per_layer = count_macs(model, (1, 3, 5, 5))

    assert list(per_layer) == [
        "stem",
        "separable.depthwise",
        "separable.pointwise",
    ]
    assert all(count > 0 for count in per_layer.values())


def test_disk_size_is_measured_from_an_actual_serialisation():
    """Include ``torch.save`` container metadata, not only tensor storage."""
    model = nn.Linear(17, 5)
    independent_buffer = io.BytesIO()
    torch.save(model.state_dict(), independent_buffer)
    raw_tensor_bytes = sum(
        tensor.numel() * tensor.element_size()
        for tensor in model.state_dict().values()
    )

    measured = state_dict_bytes(model)

    assert measured == independent_buffer.getbuffer().nbytes
    assert measured > raw_tensor_bytes


def test_profile_emits_binary_kb_and_mb_and_writes_valid_json(tmp_path):
    """Both report units use 1024 and survive the JSON-writing path."""
    cfg = load_config(CONFIGS_DIR / "model_b.yaml")
    model = nn.Sequential(
        nn.Conv2d(3, 4, 1),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(4, cfg.model.num_classes),
    )

    profile = profile_model(
        model,
        cfg,
        mean_epoch_seconds=1.23456,
        device=torch.device("cpu"),
        out_dir=tmp_path,
    )

    size = profile["disk_size_bytes"]
    assert profile["disk_size_kb"] == round(size / 1024, 2)
    assert profile["disk_size_mb"] == round(size / 1024**2, 4)
    assert profile["mean_epoch_seconds"] == 1.2346
    assert validate_profile(profile) == []
    assert json.loads((tmp_path / "profile.json").read_text(encoding="utf-8")) == profile
