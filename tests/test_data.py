"""EuroSAT download, split and loaders.

Placeholder created by #1 as part of the interface freeze. #2 and #3 turns these
green. See docs/interfaces.md section 5 for the full test-file map.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet - owned by #2 (split) and #3 (loaders)")


def test_split_is_seventy_fifteen_fifteen():
    """Totals must be 70/15/15 across the whole dataset."""


def test_split_is_stratified_per_class():
    """Each class must hold its 70/15/15 proportion, not just the total."""


def test_split_is_deterministic_from_the_seed():
    """Re-running prepare_data must not reshuffle the manifest."""


def test_manifest_covers_every_image_exactly_once():
    """No image may appear in two splits - that would leak test data."""


def test_norm_stats_use_the_training_split_only():
    """Computing them over the full dataset leaks test statistics."""


def test_augmentation_applies_to_train_only():
    """Val and test must see deterministic transforms."""


def test_input_size_switches_to_224_without_a_second_pipeline():
    """The SOTA harness (#19) needs this for the resolution ablation."""
