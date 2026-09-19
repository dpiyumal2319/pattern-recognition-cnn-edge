"""Pretrained lightweight backbones.

Placeholder created by #1 as part of the interface freeze. #19 turns these
green. See docs/interfaces.md section 5 for the full test-file map.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet - owned by #19")


def test_head_is_replaced_for_ten_classes():
    """ImageNet's 1000-way head must be gone."""


def test_freeze_backbone_leaves_only_the_head_trainable():
    """trainable_params must be far smaller than total_params when frozen."""


def test_accepts_both_sixty_four_and_two_twenty_four_input():
    """The resolution ablation depends on this."""


def test_head_replacement_is_generic_across_architectures():
    """No per-model special-casing."""
