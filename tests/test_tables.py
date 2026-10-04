"""LaTeX table generation.

Placeholder created by #1 as part of the interface freeze. #18 and #22 turn the
comparison tests green; #9 added the layer-table tests. See docs/interfaces.md section 5 for the full test-file map.
"""

from __future__ import annotations

import json

import pytest

from pr_edge.paths import CONFIGS_DIR, REPORT_DIR
import pr_edge.tables as tables
from pr_edge.tables import build_custom_comparison, build_layer_table

pending_sota = pytest.mark.skip(reason="not implemented yet - owned by #22")


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


@pending_sota
def test_sota_comparison_reports_size_in_megabytes():
    """The brief asks for MB here and KB for the custom models."""


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
