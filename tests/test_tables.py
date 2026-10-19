"""LaTeX table generation.

Placeholder created by #1 as part of the interface freeze. #18 and #22 turn the
comparison tests green; #9 added the layer-table tests. See docs/interfaces.md section 5 for the full test-file map.
"""

from __future__ import annotations

import json
import shutil

import pytest
import yaml

from pr_edge.paths import CONFIGS_DIR, REPORT_DIR
import pr_edge.tables as tables
from pr_edge.tables import (
    build_custom_comparison,
    build_custom_convergence,
    build_final_comparison,
    build_layer_table,
    build_sota_comparison,
)


def _write_custom_run(root, run_id="model_a"):
    source = tables.RESULTS_DIR / "model_a"
    target = root / run_id
    target.mkdir(parents=True)
    for name in ("metrics.json", "profile.json"):
        data = json.loads((source / name).read_text(encoding="utf-8"))
        data["run_id"] = run_id
        (target / name).write_text(json.dumps(data), encoding="utf-8")


def test_custom_comparison_has_every_column_the_brief_requires(tmp_path, monkeypatch):
    """Parameters, size in KB, epoch time, accuracy, precision, and recall."""
    results = tmp_path / "results"
    _write_custom_run(results)
    monkeypatch.setattr(tables, "RESULTS_DIR", results)

    text = build_custom_comparison(["model_a"], tmp_path / "custom.tex").read_text(encoding="utf-8")

    assert "Parameters" in text
    assert "Size (KB)" in text
    assert "Epoch (s)" in text
    assert r"Accuracy (\%)" in text
    assert r"Precision (\%)" in text
    assert r"Recall (\%)" in text
    assert r"Model A & 692{,}106 & 2{,}708.06 & 11.80 & 92.89 & 93.20 & 92.52 \\" in text


def _write_sota_run(root, run_id, *, name="mobilenet_v2", freeze=False, input_size=64,
                    accuracy=0.9802, params=2_236_682, trainable=None, size_mb=8.7696):
    """A schema-valid SOTA run directory: config, metrics and profile.

    metrics.json and profile.json start from a committed custom run (so they
    satisfy the schema) and override only the fields the SOTA table reads.
    """
    source = tables.RESULTS_DIR / "model_b"
    target = root / run_id
    target.mkdir(parents=True)
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    metrics.update(run_id=run_id, accuracy=accuracy, macro_precision=0.9792, macro_recall=0.9794)
    profile = json.loads((source / "profile.json").read_text(encoding="utf-8"))
    profile.update(run_id=run_id, total_params=params, disk_size_mb=size_mb,
                   trainable_params=params if trainable is None else trainable)
    config = {
        "run_id": run_id,
        "data": {"input_size": input_size},
        "model": {"name": name, "freeze_backbone": freeze},
    }
    (target / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (target / "profile.json").write_text(json.dumps(profile), encoding="utf-8")
    (target / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    return target


@pytest.fixture
def sota_results(tmp_path, monkeypatch):
    """Frozen and fine-tuned MobileNetV2, EfficientNet-B0 and a 224x224 ablation."""
    results = tmp_path / "results"
    _write_sota_run(results, "mobilenetv2_64", freeze=True, accuracy=0.9304, trainable=12_810)
    _write_sota_run(results, "mobilenetv2_64_ft")
    _write_sota_run(results, "efficientnetb0_64", name="efficientnet_b0", freeze=True,
                    accuracy=0.8733, params=4_020_358, trainable=12_810, size_mb=15.6265)
    _write_sota_run(results, "mobilenetv2_224", freeze=True, input_size=224,
                    accuracy=0.9615, trainable=12_810)
    monkeypatch.setattr(tables, "RESULTS_DIR", results)
    return results


SOTA_RUNS = ["mobilenetv2_64", "mobilenetv2_224", "mobilenetv2_64_ft", "efficientnetb0_64"]


def test_sota_comparison_reports_size_in_megabytes(sota_results, tmp_path):
    """The brief asks for MB here and KB for the custom models."""
    text = build_sota_comparison(SOTA_RUNS, tmp_path / "sota.tex").read_text(encoding="utf-8")
    assert r"\shortstack[r]{Size\\(MB)}" in text
    assert "KB" not in text
    assert r"& 8.77 \\" in text and r"& 15.63 \\" in text


def test_sota_comparison_has_every_column_the_brief_requires(sota_results, tmp_path):
    """Accuracy, macro precision, macro recall, parameters and size, per row."""
    text = build_sota_comparison(SOTA_RUNS, tmp_path / "sota.tex").read_text(encoding="utf-8")
    for column in ("Accuracy", "Precision", "Recall", "Parameters", "Trained", "Size"):
        assert column in text
    assert r"\shortstack[r]{Accuracy\\(\%)}" in text
    assert (r"MobileNetV2 & Fine-tuned & $64\times64$ & 98.02 & 97.92 & 97.94 & "
            r"2{,}236{,}682 & 2{,}236{,}682 & 8.77 \\") in text
    assert (r"EfficientNet-B0 & Frozen & $64\times64$ & 87.33 & 97.92 & 97.94 & "
            r"4{,}020{,}358 & 12{,}810 & 15.63 \\") in text


def test_sota_supplementary_rows_are_labelled_and_below_primary(sota_results, tmp_path):
    """The 224x224 run is passed second but must render under its own label, last."""
    text = build_sota_comparison(SOTA_RUNS, tmp_path / "sota.tex").read_text(encoding="utf-8")
    label = text.index("Supplementary")
    supplementary_row = text.index(r"MobileNetV2 & Frozen & $224\times224$")
    assert label < supplementary_row
    for primary in (r"MobileNetV2 & Frozen & $64\times64$", r"MobileNetV2 & Fine-tuned & $64\times64$",
                    r"EfficientNet-B0 & Frozen & $64\times64$"):
        assert text.index(primary) < label
    # Primary rows keep the caller's order.
    assert text.index("MobileNetV2 & Frozen & $64") < text.index("MobileNetV2 & Fine-tuned")
    assert text.index("MobileNetV2 & Fine-tuned") < text.index("EfficientNet-B0")


def test_sota_table_without_ablation_has_no_supplementary_group(sota_results, tmp_path):
    text = build_sota_comparison(["mobilenetv2_64_ft"], tmp_path / "sota.tex").read_text(encoding="utf-8")
    assert "Supplementary" not in text


def test_sota_labels_come_from_the_config_not_the_run_id(tmp_path, monkeypatch):
    """A run named like a fine-tune but configured frozen is labelled frozen."""
    results = tmp_path / "results"
    _write_sota_run(results, "mobilenetv2_64_ft", freeze=True, input_size=224)
    monkeypatch.setattr(tables, "RESULTS_DIR", results)
    text = build_sota_comparison(["mobilenetv2_64_ft"], tmp_path / "sota.tex").read_text(encoding="utf-8")
    assert r"MobileNetV2 & Frozen & $224\times224$" in text


def test_sota_table_is_a_reproducible_bare_tabular(sota_results, tmp_path):
    first = build_sota_comparison(SOTA_RUNS, tmp_path / "a.tex").read_bytes()
    second = build_sota_comparison(SOTA_RUNS, tmp_path / "b.tex").read_bytes()
    assert first == second
    assert b"\r\n" not in first
    text = first.decode("utf-8")
    assert text.startswith(r"\begin{tabular}") and text.rstrip().endswith(r"\end{tabular}")
    assert r"\begin{table}" not in text


def test_sota_table_missing_run_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(tables, "RESULTS_DIR", tmp_path / "results")
    with pytest.raises(FileNotFoundError, match="missing.*config.yaml.*metrics.json.*profile.json"):
        build_sota_comparison(["not_there"], tmp_path / "sota.tex")


def test_sota_table_requires_at_least_one_run(tmp_path):
    with pytest.raises(ValueError, match="at least one run_id"):
        build_sota_comparison([], tmp_path / "sota.tex")


def test_sota_table_rejects_mislabeled_artifact(sota_results, tmp_path):
    path = sota_results / "mobilenetv2_64" / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["run_id"] = "mobilenetv2_64_ft"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="different run_id"):
        build_sota_comparison(["mobilenetv2_64"], tmp_path / "sota.tex")


def test_sota_table_rejects_invalid_artifact_schema(sota_results, tmp_path):
    path = sota_results / "mobilenetv2_64" / "profile.json"
    profile = json.loads(path.read_text(encoding="utf-8"))
    del profile["disk_size_mb"]
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid result artifacts.*disk_size_mb"):
        build_sota_comparison(["mobilenetv2_64"], tmp_path / "sota.tex")


def test_sota_table_rejects_validation_split_metrics(sota_results, tmp_path):
    """Only held-out test results belong in the table."""
    path = sota_results / "mobilenetv2_64" / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    metrics["split"] = "val"
    path.write_text(json.dumps(metrics), encoding="utf-8")
    with pytest.raises(ValueError, match="not 'test'"):
        build_sota_comparison(["mobilenetv2_64"], tmp_path / "sota.tex")


def test_sota_table_rejects_a_custom_model(tmp_path, monkeypatch):
    results = tmp_path / "results"
    _write_sota_run(results, "model_b", name="model_b")
    monkeypatch.setattr(tables, "RESULTS_DIR", results)
    with pytest.raises(ValueError, match="not a SOTA model"):
        build_sota_comparison(["model_b"], tmp_path / "sota.tex")


def test_custom_table_is_a_bare_tabular_fragment(tmp_path, monkeypatch):
    """No \begin{table} wrapper - placement belongs to the report."""
    results = tmp_path / "results"
    _write_custom_run(results)
    monkeypatch.setattr(tables, "RESULTS_DIR", results)

    text = build_custom_comparison(["model_a"], tmp_path / "custom.tex").read_text(encoding="utf-8")
    assert text.startswith(r"\begin{tabular}") and text.rstrip().endswith(r"\end{tabular}")
    assert r"\begin{table}" not in text


def test_custom_table_missing_run_fails_loudly(tmp_path, monkeypatch):
    """Better than emitting a table with a silent gap in it."""
    monkeypatch.setattr(tables, "RESULTS_DIR", tmp_path / "results")
    with pytest.raises(FileNotFoundError, match="missing.*metrics.json.*profile.json"):
        build_custom_comparison(["not_there"], tmp_path / "custom.tex")


def test_custom_table_rejects_mislabeled_artifact(tmp_path, monkeypatch):
    results = tmp_path / "results"
    _write_custom_run(results)
    profile_path = results / "model_a" / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["run_id"] = "model_b"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(tables, "RESULTS_DIR", results)

    with pytest.raises(ValueError, match="different run_id"):
        build_custom_comparison(["model_a"], tmp_path / "custom.tex")


def test_custom_table_rejects_invalid_artifact_schema(tmp_path, monkeypatch):
    results = tmp_path / "results"
    _write_custom_run(results)
    metrics_path = results / "model_a" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["accuracy"] = "not-a-number"
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    monkeypatch.setattr(tables, "RESULTS_DIR", results)

    with pytest.raises(ValueError, match="invalid result artifacts.*accuracy"):
        build_custom_comparison(["model_a"], tmp_path / "custom.tex")


def test_custom_table_requires_at_least_one_run(tmp_path):
    with pytest.raises(ValueError, match="at least one run_id"):
        build_custom_comparison([], tmp_path / "custom.tex")


def test_committed_custom_table_is_current(tmp_path):
    committed = REPORT_DIR / "tables" / "custom_comparison.tex"
    fresh = build_custom_comparison(["model_a", "model_b"], tmp_path / "fresh.tex")
    assert committed.read_bytes() == fresh.read_bytes()


def test_custom_convergence_reports_budget_selection_and_resources(tmp_path):
    text = build_custom_convergence(
        ["model_a", "model_a_100", "model_b", "model_b_100"],
        tmp_path / "convergence.tex",
    ).read_text(encoding="utf-8")

    assert "Epochs" in text
    assert "Selected" in text
    assert "Best val. loss" in text
    assert "MACs" in text
    assert r"Model A & 100 & 98 & 0.1355 & 96.00 & 96.00" in text
    assert r"Model B & 100 & 100 & 0.2004 & 93.26 & 93.14" in text


def test_custom_convergence_rejects_mixed_devices(tmp_path, monkeypatch):
    results = tmp_path / "results"
    for run_id in ("model_a", "model_b"):
        shutil.copytree(tables.RESULTS_DIR / run_id, results / run_id)
    profile_path = results / "model_b" / "profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile["device_name"] = "CPU"
    profile_path.write_text(json.dumps(profile), encoding="utf-8")
    monkeypatch.setattr(tables, "RESULTS_DIR", results)

    with pytest.raises(ValueError, match="must share one device"):
        build_custom_convergence(["model_a", "model_b"], tmp_path / "mixed.tex")


def test_custom_convergence_requires_at_least_one_run(tmp_path):
    with pytest.raises(ValueError, match="at least one run_id"):
        build_custom_convergence([], tmp_path / "convergence.tex")


def test_committed_custom_convergence_table_is_current(tmp_path):
    committed = REPORT_DIR / "tables" / "custom_convergence.tex"
    fresh = build_custom_convergence(
        ["model_a", "model_a_100", "model_b", "model_b_100"],
        tmp_path / "fresh.tex",
    )
    assert committed.read_bytes() == fresh.read_bytes()


def test_layer_table_is_a_bare_tabular_with_the_model_totals(tmp_path):
    out = build_layer_table(CONFIGS_DIR / "model_a.yaml", tmp_path / "t.tex")
    text = out.read_text(encoding="utf-8")
    assert text.startswith(r"\begin{tabular}") and text.rstrip().endswith(r"\end{tabular}")
    assert r"\begin{table}" not in text
    assert r"Total & & 692{,}106 \\" in text
    assert r"Dense 1 & $128$ & 589{,}952 \\" in text
    assert r"Conv2D 3 & $128 \times 12 \times 12$ & 73{,}856 \\" in text
    # Kinds that occur once are not numbered.
    assert "\n" + r"Flatten & $4608$ & 0 \\" + "\n" in text


def test_layer_table_is_reproducible_with_unix_line_endings(tmp_path):
    first = build_layer_table(CONFIGS_DIR / "model_a.yaml", tmp_path / "a.tex").read_bytes()
    second = build_layer_table(CONFIGS_DIR / "model_a.yaml", tmp_path / "b.tex").read_bytes()
    assert first == second
    assert b"\r\n" not in first


@pytest.mark.parametrize("model", ["model_a", "model_b"])
def test_committed_layer_tables_are_current(tmp_path, model):
    """The committed tables must match what the current code generates.

    If this fails, a model or config changed without rerunning
    scripts/build_tables.py --table <model>_layers - the report would be stale.
    """
    committed = REPORT_DIR / "tables" / f"{model}_layers.tex"
    fresh = build_layer_table(CONFIGS_DIR / f"{model}.yaml", tmp_path / "fresh.tex")
    assert committed.read_bytes() == fresh.read_bytes()


def test_model_b_table_names_the_depthwise_and_pointwise_steps(tmp_path):
    """The vocabulary of CNNs.pdf p44, one row per step, so #11 can cite each."""
    text = build_layer_table(CONFIGS_DIR / "model_b.yaml", tmp_path / "b.tex").read_text(encoding="utf-8")
    assert r"DepthwiseConv2D 1 & $3 \times 62 \times 62$ & 30 \\" in text
    assert r"PointwiseConv2D 1 & $32 \times 62 \times 62$ & 128 \\" in text
    assert r"PointwiseConv2D 3 & $128 \times 12 \times 12$ & 8{,}320 \\" in text
    assert r"GlobalAveragePooling2D & $128 \times 1 \times 1$ & 0 \\" in text
    assert r"Total & & 36{,}968 \\" in text
    # A standard convolution never appears in Model B.
    assert "\nConv2D" not in text


# ---------------------------------------------------------------------------
# Dataset split table (#24)
# ---------------------------------------------------------------------------


def _write_manifest(path, sizes):
    """A stratified manifest with the given per-class sizes, split by _allocate."""
    import csv

    from pr_edge.data import _allocate

    rows = []
    for name, size in sizes.items():
        allocation = _allocate(size)
        index = 0
        for split, count in allocation.items():
            for _ in range(count):
                rows.append({"filepath": f"{name}/{name}_{index}.jpg", "label": name, "split": split})
                index += 1
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filepath", "label", "split"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_dataset_table_counts_every_class_and_split(tmp_path):
    from pr_edge.tables import build_dataset_table

    # 13 does not divide into 70/15/15, so the row shows the realised counts.
    manifest = _write_manifest(tmp_path / "splits.csv", {"Forest": 20, "River": 13})
    text = build_dataset_table(manifest, tmp_path / "dataset.tex").read_text(encoding="utf-8")
    assert r"Forest & 14 & 3 & 3 & 20 \\" in text
    assert r"River & 9 & 2 & 2 & 13 \\" in text
    assert r"AnnualCrop & 0 & 0 & 0 & 0 \\" in text      # every class listed, in CLASS_NAMES order
    assert r"Total & 23 & 5 & 5 & 33 \\" in text
    assert text.index("AnnualCrop") < text.index("Forest") < text.index("River")
    assert text.startswith(r"\begin{tabular}") and r"\begin{table}" not in text


def test_dataset_table_reports_the_committed_70_15_15_split(tmp_path):
    from pr_edge.paths import DATA_DIR
    from pr_edge.tables import build_dataset_table

    text = build_dataset_table(DATA_DIR / "splits.csv", tmp_path / "d.tex").read_text(encoding="utf-8")
    assert r"Total & 18{,}900 & 4{,}050 & 4{,}050 & 27{,}000 \\" in text
    assert r"Share & 70.0\% & 15.0\% & 15.0\% & 100.0\% \\" in text


def test_committed_dataset_table_is_current(tmp_path):
    """If this fails, the manifest changed without rerunning build_tables.py --table dataset."""
    from pr_edge.paths import DATA_DIR
    from pr_edge.tables import build_dataset_table

    fresh = build_dataset_table(DATA_DIR / "splits.csv", tmp_path / "fresh.tex")
    assert (REPORT_DIR / "tables" / "dataset.tex").read_bytes() == fresh.read_bytes()


def test_dataset_table_refuses_an_invalid_manifest(tmp_path):
    from pr_edge.data import DataError
    from pr_edge.tables import build_dataset_table

    manifest = tmp_path / "splits.csv"
    manifest.write_text("filepath,label,split\nForest/a.jpg,Forest,holdout\n", encoding="utf-8")
    with pytest.raises(DataError, match="split"):
        build_dataset_table(manifest, tmp_path / "d.tex")
    assert not (tmp_path / "d.tex").exists()


def test_sota_cli_pairs_each_network_frozen_and_fine_tuned(monkeypatch):
    """The run list behind --table sota: frozen/fine-tuned pairs, ablation last (#22, #54)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_tables", REPORT_DIR.parent / "scripts" / "build_tables.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    calls = []
    monkeypatch.setattr(script, "build_sota_comparison", lambda run_ids, out: calls.append((run_ids, out)) or out)

    script.main(["--table", "sota"])

    (run_ids, out), = calls
    assert run_ids == [
        "mobilenetv2_64", "mobilenetv2_64_ft",
        "efficientnetb0_64", "efficientnetb0_64_ft",
        "mobilenetv2_224", "mobilenetv2_224_ft_b128",
        "efficientnetb0_224", "efficientnetb0_224_ft_b128",
    ]
    assert out == REPORT_DIR / "tables" / "sota_comparison.tex"


def test_committed_sota_table_is_current(tmp_path):
    """If this fails, a SOTA run changed without rerunning build_tables.py --table sota."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_tables", REPORT_DIR.parent / "scripts" / "build_tables.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    # Build through the CLI's own run list, redirected into tmp_path.
    script.TABLES_DIR = tmp_path
    fresh = script.BUILDERS["sota"]()
    assert (REPORT_DIR / "tables" / "sota_comparison.tex").read_bytes() == fresh.read_bytes()


FINAL_RUNS = ["model_b_100", "mobilenetv2_64_ft", "efficientnetb0_64_ft"]


@pytest.fixture
def final_results(tmp_path, monkeypatch):
    """Small offline artifact fixtures; no images, weights or network needed."""
    results = tmp_path / "results"
    for run_id in FINAL_RUNS:
        target = results / run_id
        target.mkdir(parents=True)
        for name in ("config.yaml", "metrics.json", "profile.json", "selection.json", "history.json"):
            shutil.copyfile(tables.RESULTS_DIR / run_id / name, target / name)
    monkeypatch.setattr(tables, "RESULTS_DIR", results)
    return results


def test_final_table_and_discussion_share_measured_results(final_results, tmp_path):
    text = build_final_comparison(FINAL_RUNS, tmp_path / "final.tex").read_text(encoding="utf-8")
    assert r"Model B & 100 & 93.14 & 36{,}968 & 150.98 & 3.725 & 12.29 \\" in text
    assert r"MobileNetV2 & 30 & 98.02 & 2{,}236{,}682 & 8{,}980.07 & 24.461 & 14.40 \\" in text
    assert r"EfficientNet-B0 & 30 & 98.10 & 4{,}020{,}358 & 16{,}001.55 & 31.979 & 16.59 \\" in text
    assert r"\gdef\FinalMobileGap{4.89}" in text
    assert r"\gdef\FinalMobileStorageRatio{59.5}" in text
    assert r"\gdef\FinalMobileMACRatio{6.57}" in text
    assert r"\gdef\FinalMobileTrainingRatio{2.85}" in text
    assert r"\gdef\FinalSOTAExtraCorrect{3}" in text
    assert r"\begin{table}" not in text


def test_final_values_update_from_artifacts_and_bytes(final_results, tmp_path):
    path = final_results / "model_b_100" / "profile.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(disk_size_bytes=204800, disk_size_kb=999, disk_size_mb=999, macs=1_000_000)
    path.write_text(json.dumps(data), encoding="utf-8")
    text = build_final_comparison(FINAL_RUNS, tmp_path / "final.tex").read_text(encoding="utf-8")
    assert r"\gdef\FinalBSize{200.00}" in text
    assert r"\gdef\FinalMobileStorageRatio{44.9}" in text
    assert r"\gdef\FinalMobileMACRatio{24.46}" in text
    assert "200.00 & 1.000" in text


def test_final_discussion_uses_the_selected_epoch_even_when_it_is_not_last(final_results, tmp_path):
    directory = final_results / "model_b_100"
    history_path = directory / "history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    # Make epoch 1 the validation-loss winner without changing the run budget.
    history[0]["val_loss"] = min(row["val_loss"] for row in history) / 2
    history_path.write_text(json.dumps(history), encoding="utf-8")
    selection_path = directory / "selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection.update(best_epoch=1, best_val_loss=history[0]["val_loss"], best_val_acc=history[0]["val_acc"])
    selection_path.write_text(json.dumps(selection), encoding="utf-8")

    text = build_final_comparison(FINAL_RUNS, tmp_path / "final.tex").read_text(encoding="utf-8")
    assert r"\gdef\FinalBSelectedEpoch{1}" in text
    assert r"\gdef\FinalBEpochs{100}" in text
    discussion = (REPORT_DIR / "sections" / "discussion.tex").read_text(encoding="utf-8")
    assert r"\FinalBSelectedEpoch{}" in discussion
    assert "last epoch" not in discussion


@pytest.mark.parametrize("filename,keys,value,message", [
    ("metrics.json", ["split"], "val", "test metrics"),
    ("metrics.json", ["run_id"], "wrong", "different run_id"),
    ("metrics.json", ["accuracy"], "bad", "invalid result artifacts"),
    ("metrics.json", ["num_samples"], 100, "must share"),
    ("profile.json", ["device_name"], "CPU", "must share"),
    ("profile.json", ["input_shape"], [1, 3, 224, 224], "native 64x64"),
    ("profile.json", ["macs"], 0, "positive"),
    ("profile.json", ["trainable_params"], 12810, "train all parameters"),
    ("config.yaml", ["data", "input_size"], 224, "native 64x64"),
    ("config.yaml", ["data", "batch_size"], 32, "must share"),
    ("config.yaml", ["data", "root"], "different-image-tree/", "must share"),
    ("config.yaml", ["seed"], 123, "must share"),
    ("config.yaml", ["training", "epochs"], 20, "epoch budget"),
    ("config.yaml", ["training", "smoke"], True, "smoke run"),
    ("config.yaml", ["model", "freeze_backbone"], True, "train all parameters"),
    ("config.yaml", ["model", "pretrained"], False, "pretrained SOTA"),
    ("config.yaml", ["model", "name"], "model_a", "once each"),
])
def test_final_table_rejects_invalid_or_incomparable_rows(final_results, tmp_path, filename, keys, value, message):
    path = final_results / "mobilenetv2_64_ft" / filename
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    target = data
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path.write_text(yaml.safe_dump(data) if filename.endswith("yaml") else json.dumps(data), encoding="utf-8")
    out = tmp_path / "bad.tex"
    with pytest.raises(ValueError, match=message):
        build_final_comparison(FINAL_RUNS, out)
    assert not out.exists()


def test_final_table_checks_model_b_parameter_ceiling(final_results, tmp_path):
    path = final_results / "model_b_100" / "profile.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(total_params=100_001, trainable_params=100_001)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="100k"):
        build_final_comparison(FINAL_RUNS, tmp_path / "bad.tex")


def test_final_table_rejects_missing_checkpoint_selection(final_results, tmp_path):
    (final_results / "model_b_100" / "selection.json").unlink()
    with pytest.raises(ValueError, match="missing selection.json"):
        build_final_comparison(FINAL_RUNS, tmp_path / "bad.tex")


@pytest.mark.parametrize("run_ids", [[], FINAL_RUNS[:2], [FINAL_RUNS[0]] * 3])
def test_final_table_requires_three_distinct_runs(tmp_path, run_ids):
    with pytest.raises(ValueError, match="three distinct"):
        build_final_comparison(run_ids, tmp_path / "bad.tex")


def test_final_cli_and_committed_table_are_current(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("build_tables", REPORT_DIR.parent / "scripts" / "build_tables.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    script.TABLES_DIR = tmp_path
    assert script.main(["--table", "final"]) == 0
    fresh = tmp_path / "final_comparison.tex"
    assert fresh.read_bytes() == (REPORT_DIR / "tables" / "final_comparison.tex").read_bytes()
    assert b"\r\n" not in fresh.read_bytes()
    # Canonical order and bytes do not depend on the caller's row order.
    assert fresh.read_bytes() == build_final_comparison(list(reversed(FINAL_RUNS)), tmp_path / "again.tex").read_bytes()
