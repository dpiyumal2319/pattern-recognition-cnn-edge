"""Task 4 loss-curve and confusion-matrix figure generation."""

from __future__ import annotations

import json

import matplotlib.image as mpimg
import numpy as np
import pytest

import pr_edge.plots as plots


def _write_history(root, run_id, losses):
    directory = root / run_id
    directory.mkdir(parents=True)
    records = [
        {
            "epoch": epoch,
            "train_loss": train,
            "train_acc": 0.5,
            "val_loss": val,
            "val_acc": 0.5,
            "epoch_seconds": 1.0,
            "lr": 0.001,
        }
        for epoch, (train, val) in enumerate(losses, start=1)
    ]
    (directory / "history.json").write_text(json.dumps(records), encoding="utf-8")


def _write_metrics(root, run_id, matrix):
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    names = ["AnnualCrop", "Forest"]
    metrics = {
        "run_id": run_id,
        "split": "test",
        "num_samples": int(np.asarray(matrix).sum()),
        "class_names": names,
        "accuracy": 0.5,
        "macro_precision": 0.5,
        "macro_recall": 0.5,
        "macro_f1": 0.5,
        "per_class": {
            name: {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": sum(matrix[i])}
            for i, name in enumerate(names)
        },
        "confusion_matrix": matrix,
        "confusion_matrix_convention": "rows=true, cols=predicted, order=class_names",
    }
    (directory / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


def test_loss_curves_use_one_shared_y_axis_per_model(tmp_path, monkeypatch):
    results = tmp_path / "results"
    _write_history(results, "model_a", [(2.0, 2.2), (1.0, 1.2)])
    _write_history(results, "model_b", [(1.8, 2.1), (0.9, 1.1)])
    monkeypatch.setattr(plots, "RESULTS_DIR", results)

    captured = {}
    real_subplots = plots.plt.subplots

    def recording_subplots(*args, **kwargs):
        created = real_subplots(*args, **kwargs)
        captured["axes"] = created[1]
        return created

    monkeypatch.setattr(plots.plt, "subplots", recording_subplots)
    output = plots.plot_loss_curves(["model_a", "model_b"], tmp_path / "loss.png")

    axes = captured["axes"][0]
    assert output.exists() and output.stat().st_size > 0
    assert len(axes) == 2
    assert axes[0].get_shared_y_axes().joined(axes[0], axes[1])
    assert axes[0].get_ylim() == axes[1].get_ylim()
    assert [line.get_label() for line in axes[0].lines] == ["Training", "Validation"]


def test_confusion_matrix_uses_true_row_proportions_and_count_annotations(tmp_path, monkeypatch):
    results = tmp_path / "results"
    _write_metrics(results, "model_a", [[3, 1], [2, 2]])
    monkeypatch.setattr(plots, "RESULTS_DIR", results)

    captured = {}
    real_subplots = plots.plt.subplots

    def recording_subplots(*args, **kwargs):
        created = real_subplots(*args, **kwargs)
        captured["axis"] = created[1]
        return created

    monkeypatch.setattr(plots.plt, "subplots", recording_subplots)
    output = plots.plot_confusion_matrix("model_a", tmp_path / "confusion.png")

    axis = captured["axis"]
    shown = np.asarray(axis.images[0].get_array())
    assert output.exists() and output.stat().st_size > 0
    assert np.allclose(shown, [[0.75, 0.25], [0.5, 0.5]])
    assert [tick.get_text() for tick in axis.get_xticklabels()] == ["AnnualCrop", "Forest"]
    assert [text.get_text() for text in axis.texts] == ["3", "1", "2", "2"]


def test_zero_support_row_normalizes_without_nan(tmp_path, monkeypatch):
    results = tmp_path / "results"
    _write_metrics(results, "model_b", [[2, 0], [0, 0]])
    monkeypatch.setattr(plots, "RESULTS_DIR", results)

    captured = {}
    real_subplots = plots.plt.subplots

    def recording_subplots(*args, **kwargs):
        created = real_subplots(*args, **kwargs)
        captured["axis"] = created[1]
        return created

    monkeypatch.setattr(plots.plt, "subplots", recording_subplots)
    plots.plot_confusion_matrix("model_b", tmp_path / "zero-row.png")

    shown = np.asarray(captured["axis"].images[0].get_array())
    assert np.isfinite(shown).all()
    assert shown[1].tolist() == [0.0, 0.0]


def test_missing_result_artifact_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(plots, "RESULTS_DIR", tmp_path / "results")
    with pytest.raises(FileNotFoundError, match="history.json"):
        plots.plot_loss_curves(["missing"], tmp_path / "loss.png")
    with pytest.raises(FileNotFoundError, match="metrics.json"):
        plots.plot_confusion_matrix("missing", tmp_path / "confusion.png")


def test_committed_figures_are_report_resolution():
    figures = plots.RESULTS_DIR.parent / "report" / "figures"
    for name in (
        "custom_loss_curves.png",
        "model_a_confusion_matrix.png",
        "model_b_confusion_matrix.png",
    ):
        image = mpimg.imread(figures / name)
        assert image.shape[0] >= 1800
        assert image.shape[1] >= 2400
