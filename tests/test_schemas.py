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
    validate_run_dir,
    validate_selection,
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

# Epoch 2 of VALID_HISTORY has the lower val_loss.
VALID_SELECTION = {
    "criterion": "val_loss", "best_epoch": 2, "best_val_loss": 1.9,
    "best_val_acc": 0.25, "epochs_trained": 2,
}


def test_valid_examples_pass():
    assert validate_history(VALID_HISTORY) == []
    assert validate_metrics(VALID_METRICS) == []
    assert validate_profile(VALID_PROFILE) == []
    assert validate_selection(VALID_SELECTION) == []


@pytest.mark.parametrize("change,message", [
    ({"best_epoch": 0}, "within 1..epochs_trained"),
    ({"best_epoch": 3}, "within 1..epochs_trained"),
    ({"criterion": "val_acc"}, "criterion must be 'val_loss'"),
    ({"best_epoch": 2.0}, "'best_epoch' should be int"),
])
def test_invalid_selection_is_caught(change, message):
    assert any(message in e for e in validate_selection({**VALID_SELECTION, **change}))


def _write_run_dir(directory, selection):
    directory.mkdir()
    files = {"history.json": VALID_HISTORY, "metrics.json": VALID_METRICS,
             "profile.json": VALID_PROFILE, "selection.json": selection}
    for name, data in files.items():
        (directory / name).write_text(json.dumps(data), encoding="utf-8")
    (directory / "config.yaml").write_text("run_id: x\n", encoding="utf-8")
    return directory


def test_run_dir_with_consistent_selection_passes(tmp_path):
    assert validate_run_dir(_write_run_dir(tmp_path / "run", VALID_SELECTION)) == []


def test_run_dir_requires_selection(tmp_path):
    directory = _write_run_dir(tmp_path / "run", VALID_SELECTION)
    (directory / "selection.json").unlink()
    assert any("missing selection.json" in e for e in validate_run_dir(directory))


@pytest.mark.parametrize("change,message", [
    ({"best_epoch": 1}, "lowest val_loss is at epoch 2"),
    ({"best_epoch": 1, "epochs_trained": 3}, "history.json has 2 epochs"),
])
def test_selection_must_agree_with_history(tmp_path, change, message):
    """A checkpoint the history does not justify is a report claiming the wrong model."""
    directory = _write_run_dir(tmp_path / "run", {**VALID_SELECTION, **change})
    assert any(message in e for e in validate_run_dir(directory))


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
        if p.name in {"history.json", "metrics.json", "profile.json", "selection.json"}
        and not p.parent.name.startswith("_smoke_")
    ]


@pytest.mark.parametrize("path", _committed_result_files(),
                         ids=lambda p: f"{p.parent.name}/{p.name}")
def test_committed_results_validate(path):
    """Every real run committed to the repo must match its frozen schema."""
    errors = validate_file(path)
    assert errors == [], "\n".join(errors)


@pytest.mark.parametrize("run_id", ["model_a_100", "model_b_100"])
def test_committed_convergence_run_directory_is_consistent(run_id):
    """Selection must agree with the 100-record history, not only its schema."""
    errors = validate_run_dir(RESULTS_DIR / run_id)
    assert errors == [], "\n".join(errors)


@pytest.mark.parametrize("short_id,long_id", [
    ("model_a", "model_a_100"),
    ("model_b", "model_b_100"),
])
def test_convergence_run_reproduces_the_original_first_30_epochs(short_id, long_id):
    """The 2x2 control changes the epoch budget, not the first 30 updates."""
    short = json.loads((RESULTS_DIR / short_id / "history.json").read_text(encoding="utf-8"))
    long = json.loads((RESULTS_DIR / long_id / "history.json").read_text(encoding="utf-8"))
    fields = ("epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr")
    assert [tuple(row[field] for field in fields) for row in short] == [
        tuple(row[field] for field in fields) for row in long[:30]
    ]


def test_json_syntax_error_is_reported(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text("{not json", encoding="utf-8")
    assert any("invalid JSON" in e for e in validate_file(path))
    json.loads  # keep the import meaningful for readers
