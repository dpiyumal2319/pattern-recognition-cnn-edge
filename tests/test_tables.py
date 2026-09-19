"""LaTeX table generation.

Placeholder created by #1 as part of the interface freeze. #18 and #22 turns these
green. See docs/interfaces.md section 5 for the full test-file map.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet - owned by #18 (custom) and #22 (SOTA)")


def test_custom_comparison_has_every_column_the_brief_requires():
    """Parameters, size in KB, training time per epoch, test accuracy."""


def test_sota_comparison_reports_size_in_megabytes():
    """The brief asks for MB here and KB for the custom models."""


def test_tables_are_bare_tabular_fragments():
    """No \begin{table} wrapper - placement belongs to the report."""


def test_a_missing_run_fails_loudly():
    """Better than emitting a table with a silent gap in it."""
