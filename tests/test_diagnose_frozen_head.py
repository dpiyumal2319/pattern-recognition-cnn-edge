"""Tests for scripts/diagnose_frozen_head.py and its committed output.

The unit tests use tiny synthetic tensors and models: no network, no dataset.
The last group checks empirical comparisons in the committed diagnostic.
These comparisons do not establish a unique cause of the main runs' fluctuations.
"""

from __future__ import annotations

import importlib.util
import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pr_edge.paths import CONFIGS_DIR, PROJECT_ROOT, RESULTS_DIR

_spec = importlib.util.spec_from_file_location(
    "diagnose_frozen_head", PROJECT_ROOT / "scripts" / "diagnose_frozen_head.py"
)
diag = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diag)


class _TinyNet(nn.Module):
    """Conv features, global pooling, then a ``classifier`` - the shape of the SOTA wrappers."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU())
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(4, 3))

    def pooled(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).mean(dim=(2, 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pooled(x))


def _features(n=40, channels=6, seed=0):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(n, channels, generator=generator)
    y = (x[:, 0] > 0).long()  # learnable labels
    return x, y


def test_captures_exactly_the_prediction_layer_input_in_eval_mode():
    torch.manual_seed(0)
    model = _TinyNet()
    images, labels = torch.randn(10, 3, 8, 8), torch.arange(10) % 3
    loader = DataLoader(TensorDataset(images, labels), batch_size=4)

    features, captured_labels = diag.capture_classifier_inputs(model, loader, torch.device("cpu"))

    model.eval()
    with torch.no_grad():
        assert torch.allclose(features, model.pooled(images))
    assert torch.equal(captured_labels, labels)
    assert not model.training
    assert not model.classifier._forward_pre_hooks  # the hook is removed afterwards


def test_feature_statistics_count_values_images_and_channels_above_the_spike_threshold():
    features = torch.tensor([[0.0, 1.0, 7.0], [2.0, 60.0, 3.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    stats = diag.feature_statistics(features)

    assert stats["max"] == 60.0 and stats["min"] == 0.0
    assert stats["fraction_values_above_relu6_ceiling"] == round(2 / 12, 6)
    assert stats["fraction_values_above_spike_threshold"] == round(1 / 12, 6)
    assert stats["fraction_images_with_a_spike"] == 0.25
    assert stats["channels_with_a_spike"] == 1


def test_standardise_uses_training_statistics_only():
    train, val = torch.randn(200, 5) * 4 + 3, torch.randn(50, 5) * 10 - 7
    z_train, z_val = diag.standardise(train, val)

    assert torch.allclose(z_train.mean(dim=0), torch.zeros(5), atol=1e-5)
    assert torch.allclose(z_train.std(dim=0), torch.ones(5), atol=1e-4)
    expected_val = (val - train.mean(dim=0)) / (train.std(dim=0) + 1e-6)
    assert torch.allclose(z_val, expected_val)


def test_head_training_is_seeded_and_reports_every_epoch():
    train, val = _features(seed=1), _features(seed=2)
    kwargs = dict(num_classes=2, dropout=0.2, lr=1e-2, epochs=6, batch_size=8, seed=42)

    first = diag.train_head(train, val, **kwargs)
    second = diag.train_head(train, val, **kwargs)

    assert first == second
    assert [record["epoch"] for record in first] == list(range(1, 7))
    assert first[-1]["val_loss"] < first[0]["val_loss"]


def test_repeating_over_seeds_keeps_every_seed_and_aggregates_its_summaries():
    train, val = _features(seed=1), _features(seed=2)
    result = diag.repeat_over_seeds(
        train, val, [42, 43], num_classes=2, dropout=0.2, lr=1e-2, epochs=6, batch_size=8
    )

    assert [run["seed"] for run in result["per_seed"]] == [42, 43]
    sds = [run["val_loss_sd"] for run in result["per_seed"]]
    assert result["val_loss_sd_min"] == min(sds) and result["val_loss_sd_max"] == max(sds)
    assert result["val_loss_sd_mean"] == round(sum(sds) / 2, 6)
    # Each seed is the same run train_head would give on its own.
    assert result["per_seed"][1]["history"] == diag.train_head(
        train, val, num_classes=2, dropout=0.2, lr=1e-2, epochs=6, batch_size=8, seed=43
    )


def test_summary_starts_at_the_fifth_epoch():
    history = [{"epoch": e, "val_loss": 100.0 if e < 5 else 1.0 + (e % 2), "val_acc": 0.5} for e in range(1, 9)]
    summary = diag.summarise(history)

    assert summary["val_loss_mean"] == 1.5 and summary["val_loss_sd"] == 0.5
    assert summary["val_acc_sd"] == 0.0
    with pytest.raises(ValueError, match="at least two epochs"):
        diag.summarise(history[:5])


def test_refuses_a_fine_tuned_run_before_loading_any_data():
    assert (CONFIGS_DIR / "mobilenetv2_64_ft.yaml").is_file()
    with pytest.raises(ValueError, match="does not freeze its backbone"):
        diag.diagnose_run("mobilenetv2_64_ft", [1e-3], [42], torch.device("cpu"), num_workers=0)


# --- Empirical comparisons in the committed output -------------------------

COMMITTED = RESULTS_DIR / "frozen_head_diagnostic" / "diagnostic.json"


@pytest.fixture(scope="module")
def committed():
    return json.loads(COMMITTED.read_text(encoding="utf-8"))["runs"]


def _head(run, lr, standardised=False):
    matches = [h for h in run["heads"] if h["lr"] == lr and h["standardised"] is standardised]
    assert len(matches) == 1, (lr, standardised)
    return matches[0]


def test_committed_diagnostic_covers_both_frozen_64_runs(committed):
    assert set(committed) == {"mobilenetv2_64", "efficientnetb0_64"}
    for run in committed.values():
        assert [h["standardised"] for h in run["heads"]].count(True) == 1
        for head in run["heads"]:
            assert [s["seed"] for s in head["per_seed"]] == [42, 43, 44]
            assert all(len(s["history"]) == 30 for s in head["per_seed"])


def test_raw_efficientnet_heads_have_higher_loss_variation_at_each_tested_rate(committed):
    """At every learning rate tried, EfficientNet's head stays noisier than MobileNetV2's.

    Compared seed range against seed range: EfficientNet's quietest seed must
    still be clearly noisier than MobileNetV2's noisiest at the same rate.
    """
    mobile, efficient = committed["mobilenetv2_64"], committed["efficientnetb0_64"]
    for lr in {h["lr"] for h in efficient["heads"] if not h["standardised"]}:
        assert _head(efficient, lr)["val_loss_sd_min"] > 1.5 * _head(mobile, lr)["val_loss_sd_max"], lr


def test_claim_standardising_removes_most_of_the_extra_fluctuation(committed):
    """Standardised, EfficientNet's head is about as steady as a standardised MobileNetV2 head."""
    mobile, efficient = committed["mobilenetv2_64"], committed["efficientnetb0_64"]
    lr = efficient["configured_lr"]
    raw, standardised = _head(efficient, lr), _head(efficient, lr, standardised=True)

    assert standardised["val_loss_sd_mean"] < 0.6 * raw["val_loss_sd_mean"]  # "roughly halved"
    mobile_standardised = _head(mobile, lr, standardised=True)
    assert standardised["val_loss_sd_mean"] < 1.5 * mobile_standardised["val_loss_sd_mean"]


def test_standardised_efficientnet_heads_have_higher_loss_and_lower_accuracy(committed):
    """Standardising leaves EfficientNet's loss far higher and its accuracy lower."""
    mobile, efficient = committed["mobilenetv2_64"], committed["efficientnetb0_64"]
    lr = efficient["configured_lr"]
    e, m = _head(efficient, lr, standardised=True), _head(mobile, lr, standardised=True)

    assert e["val_loss_mean_mean"] > 2 * m["val_loss_mean_mean"]  # "more than twice as high"
    assert e["val_acc_mean_mean"] < m["val_acc_mean_mean"] - 0.03  # "several points lower"


def test_observed_feature_ranges_differ_between_the_two_backbones(committed):
    mobile = committed["mobilenetv2_64"]["features"]["train"]
    efficient = committed["efficientnetb0_64"]["features"]["train"]

    assert mobile["max"] <= 6.0 + 1e-4
    assert efficient["max"] > efficient["spike_threshold"]
    assert 0 < efficient["fraction_images_with_a_spike"] < 0.05


def test_claim_accuracy_stays_almost_level_while_the_loss_swings(committed):
    efficient = committed["efficientnetb0_64"]
    raw = _head(efficient, efficient["configured_lr"])
    assert raw["val_acc_sd_mean"] < 0.02
    relative_loss_swing = raw["val_loss_sd_mean"] / raw["val_loss_mean_mean"]
    relative_accuracy_swing = raw["val_acc_sd_mean"] / raw["val_acc_mean_mean"]
    assert relative_loss_swing > 5 * relative_accuracy_swing
