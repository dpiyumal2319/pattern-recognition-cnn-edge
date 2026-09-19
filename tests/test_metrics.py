"""Classification metrics.

Placeholder created by #1 as part of the interface freeze. #4 turns these
green. See docs/interfaces.md section 5 for the full test-file map.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet - owned by #4")


def test_accuracy_matches_a_hand_computed_example():
    """Verify against numbers worked out by hand, not against another library."""


def test_precision_and_recall_match_a_hand_computed_example():
    """Per-class and macro, on a small matrix small enough to check by eye."""


def test_confusion_matrix_rows_are_true_labels():
    """Feed a deliberately asymmetric case so a transpose would be caught."""


def test_never_predicted_class_reports_zero_not_nan():
    """Undefined precision must stay numeric so the JSON and tables hold."""
