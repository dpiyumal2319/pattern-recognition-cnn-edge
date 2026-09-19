"""Model B, the depthwise separable CNN.

Placeholder created by #1 as part of the interface freeze. #10 turns these
green. See docs/interfaces.md section 5 for the full test-file map.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet - owned by #10")


def test_trainable_parameters_are_under_one_hundred_thousand():
    """THE hard constraint from the brief. CI enforces this, so a Model B
    over budget cannot be merged.

        total, trainable = count_parameters(ModelB(cfg))
        assert trainable <= 100_000
    """


def test_forward_pass_shape_matches_model_a():
    """Like-for-like comparison requires identical input and output shapes."""


def test_depthwise_and_pointwise_are_separate_layers():
    """#11 must be able to name each stage in its parameter derivation."""


def test_uses_fewer_macs_than_model_a():
    """The whole point of the substitution."""
