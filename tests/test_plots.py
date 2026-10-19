"""Task 4 and task 5 loss-curve and confusion-matrix figure generation."""

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


def test_loss_curves_can_mark_the_original_stop_epoch(tmp_path, monkeypatch):
    results = tmp_path / "results"
    _write_history(results, "model_a_100", [(2.0, 2.2), (1.0, 1.2)])
    monkeypatch.setattr(plots, "RESULTS_DIR", results)

    captured = {}
    real_subplots = plots.plt.subplots

    def recording_subplots(*args, **kwargs):
        created = real_subplots(*args, **kwargs)
        captured["axes"] = created[1]
        return created

    monkeypatch.setattr(plots.plt, "subplots", recording_subplots)
    plots.plot_loss_curves(
        ["model_a_100"], tmp_path / "marked.png", reference_epoch=30
    )

    marker = captured["axes"][0][0].lines[2]
    assert marker.get_label() == "Original stop (30)"
    assert list(marker.get_xdata()) == [30, 30]


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
        "custom_convergence.png",
        "model_a_confusion_matrix.png",
        "model_b_confusion_matrix.png",
        "mobilenetv2_64_ft_confusion_matrix.png",
        "efficientnetb0_64_ft_confusion_matrix.png",
        "efficientnetb0_224_ft_b128_confusion_matrix.png",
    ):
        image = mpimg.imread(figures / name)
        assert image.shape[0] >= 1800
        assert image.shape[1] >= 2400


@pytest.mark.parametrize("name", [
    "sota_loss_curves.png", "sota_224_loss_curves.png",
    "efficientnet_loss_curves.png", "efficientnet_224_loss_curves.png",
])
def test_sota_figures_are_report_resolution(name):
    """Two side-by-side panels each (frozen, fine-tuned), like the custom figure."""
    figures = plots.RESULTS_DIR.parent / "report" / "figures"
    image = mpimg.imread(figures / name)
    assert image.shape[0] >= 1800
    assert image.shape[1] >= 3600


@pytest.mark.parametrize("run_id,network,size,policy", [
    ("mobilenetv2_64", "MobileNetV2", "64x64", "frozen"),
    ("mobilenetv2_64_ft", "MobileNetV2", "64x64", "fine-tuned"),
    ("mobilenetv2_224", "MobileNetV2", "224x224", "frozen"),
    ("mobilenetv2_224_ft", "MobileNetV2", "224x224", "fine-tuned"),
    ("mobilenetv2_224_ft_b128", "MobileNetV2", "224x224", "fine-tuned"),
    ("efficientnetb0_64", "EfficientNet-B0", "64x64", "frozen"),
    ("efficientnetb0_64_ft", "EfficientNet-B0", "64x64", "fine-tuned"),
    ("efficientnetb0_224", "EfficientNet-B0", "224x224", "frozen"),
    ("efficientnetb0_224_ft", "EfficientNet-B0", "224x224", "fine-tuned"),
    ("efficientnetb0_224_ft_b128", "EfficientNet-B0", "224x224", "fine-tuned"),
])
def test_sota_labels_name_the_backbone_policy(run_id, network, size, policy):
    """Frozen and fine-tuned runs share an architecture; the label must say
    which, and at which resolution. Without an entry, a run falls back to
    title-casing its ID (e.g. "Efficientnetb0 64")."""
    label = plots._run_label(run_id)
    assert label.startswith(f"{network} {size} ")
    assert f"({policy}" in label


def test_every_figure_builder_runs_on_committed_results(tmp_path, monkeypatch):
    """Each figure the script promises regenerates from results/ with one command."""
    import importlib.util

    script = plots.RESULTS_DIR.parent / "scripts" / "build_figures.py"
    spec = importlib.util.spec_from_file_location("build_figures", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "FIGURES_DIR", tmp_path)

    assert {
        "sota_loss", "sota_224_loss", "mobilenetv2_64_ft_confusion",
        "efficientnet_loss", "efficientnet_224_loss",
        "efficientnetb0_64_ft_confusion", "efficientnetb0_224_ft_b128_confusion",
    } <= set(module.BUILDERS)
    assert module.main(["--all"]) == 0
    assert len(list(tmp_path.glob("*.png"))) == len(module.BUILDERS)
    assert (tmp_path / "sota_loss_curves.png").stat().st_size > 0
    assert (tmp_path / "mobilenetv2_64_ft_confusion_matrix.png").stat().st_size > 0
