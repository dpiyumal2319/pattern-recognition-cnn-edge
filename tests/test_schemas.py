"""Result-file schema validation. Owned by #1.

Every committed results file is validated on every pull request, so drift in
the interface four people share fails the build rather than surfacing as a
confusing table three weeks later.
"""

from __future__ import annotations

import json

import pytest

from pr_edge.paths import RESULTS_DIR
from pr_edge.schemas import (
    validate_file,
    validate_history,
    validate_metrics,
    validate_profile,
)

VALID_HISTORY = [
    {"epoch": 1, "train_loss": 2.3, "train_acc": 0.1, "val_loss": 2.2,
     "val_acc": 0.12, "epoch_seconds": 1.5, "lr": 0.001},
    {"epoch": 2, "train_loss": 2.0, "train_acc": 0.2, "val_loss": 1.9,
     "val_acc": 0.25, "epoch_seconds": 1.4, "lr": 0.001},
]

VALID_METRICS = {
    "run_id": "x", "split": "test", "num_samples": 4, "class_names": ["a", "b"],
    "accuracy": 0.5, "macro_precision": 0.5, "macro_recall": 0.5, "macro_f1": 0.5,
    "per_class": {
        "a": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2},
        "b": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2},
    },
    "confusion_matrix": [[1, 1], [1, 1]],
    "confusion_matrix_convention": "rows=true, cols=predicted, order=class_names",
}

VALID_PROFILE = {
    "run_id": "x", "total_params": 100, "trainable_params": 100,
    "disk_size_bytes": 2048, "disk_size_kb": 2.0, "disk_size_mb": 0.002,
    "macs": 1000, "macs_per_layer": {"conv": 1000}, "input_shape": [1, 3, 64, 64],
    "mean_epoch_seconds": 1.45, "device": "cpu", "device_name": "CPU",
}


def test_valid_examples_pass():
    assert validate_history(VALID_HISTORY) == []
    assert validate_metrics(VALID_METRICS) == []
    assert validate_profile(VALID_PROFILE) == []


def test_missing_key_is_caught():
    bad = {k: v for k, v in VALID_PROFILE.items() if k != "macs"}
    assert any("missing required key 'macs'" in e for e in validate_profile(bad))


def test_wrong_type_is_caught():
    assert any("'accuracy'" in e for e in validate_metrics({**VALID_METRICS, "accuracy": "high"}))


def test_unexpected_key_is_caught():
    assert any("unexpected key" in e for e in validate_profile({**VALID_PROFILE, "extra": 1}))


def test_non_contiguous_epochs_are_caught():
    bad = [VALID_HISTORY[0], {**VALID_HISTORY[1], "epoch": 5}]
    assert any("1-indexed and contiguous" in e for e in validate_history(bad))


def test_confusion_matrix_must_match_class_count():
    bad = {**VALID_METRICS, "confusion_matrix": [[1, 1, 1], [1, 1, 1], [1, 1, 1]]}
    assert any("2x2" in e for e in validate_metrics(bad))


def test_per_class_must_be_keyed_by_name_not_index():
    """Integer keys are how a confusion matrix ends up silently transposed."""
    bad = {
        **VALID_METRICS,
        "class_names": ["0", "1"],
        "per_class": {
            "0": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2},
            "1": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2},
        },
    }
    assert any("keyed by class name" in e for e in validate_metrics(bad))


def test_trainable_cannot_exceed_total():
    bad = {**VALID_PROFILE, "trainable_params": 200}
    assert any("exceeds total_params" in e for e in validate_profile(bad))


def _committed_result_files():
    if not RESULTS_DIR.exists():
        return []
    return [
        p for p in RESULTS_DIR.glob("*/*.json")
        if p.name in {"history.json", "metrics.json", "profile.json"}
        and not p.parent.name.startswith("_smoke_")
    ]


@pytest.mark.parametrize("path", _committed_result_files(),
                         ids=lambda p: f"{p.parent.name}/{p.name}")
def test_committed_results_validate(path):
    """Every real run committed to the repo must match its frozen schema."""
    errors = validate_file(path)
    assert errors == [], "\n".join(errors)


def test_json_syntax_error_is_reported(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("{not json", encoding="utf-8")
    assert any("invalid JSON" in e for e in validate_file(path))
    json.loads  # keep the import meaningful for readers
