"""Model A, the standard CNN baseline.

Placeholder created by #1 as part of the interface freeze. #9 turns these
green. See docs/interfaces.md section 5 for the full test-file map.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet - owned by #9")


def test_forward_pass_shape():
    """(N, 3, 64, 64) -> (N, 10)."""


def test_returns_logits_not_probabilities():
    """Rows must not sum to 1 - softmax belongs in the loss."""


def test_architecture_is_config_driven():
    """Changing channels in the config must change the model."""
