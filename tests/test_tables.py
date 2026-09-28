"""LaTeX table generation.

Placeholder created by #1 as part of the interface freeze. #18 and #22 turn the
comparison tests green; #9 added the layer-table tests. See docs/interfaces.md section 5 for the full test-file map.
"""

from __future__ import annotations

import pytest

from pr_edge.paths import CONFIGS_DIR, REPORT_DIR
from pr_edge.tables import build_layer_table

pending = pytest.mark.skip(reason="not implemented yet - owned by #18 (custom) and #22 (SOTA)")


@pending
def test_custom_comparison_has_every_column_the_brief_requires():
    """Parameters, size in KB, training time per epoch, test accuracy."""


@pending
def test_sota_comparison_reports_size_in_megabytes():
    """The brief asks for MB here and KB for the custom models."""


@pending
def test_tables_are_bare_tabular_fragments():
    """No \begin{table} wrapper - placement belongs to the report."""


@pending
def test_a_missing_run_fails_loudly():
    """Better than emitting a table with a silent gap in it."""


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


def test_committed_model_a_layer_table_is_current(tmp_path):
    """The committed table must match what the current code generates.

    If this fails, the model or config changed without rerunning
    scripts/build_tables.py --table model_a_layers - the report would be stale.
    """
    committed = REPORT_DIR / "tables" / "model_a_layers.tex"
    fresh = build_layer_table(CONFIGS_DIR / "model_a.yaml", tmp_path / "fresh.tex")
    assert committed.read_bytes() == fresh.read_bytes()
