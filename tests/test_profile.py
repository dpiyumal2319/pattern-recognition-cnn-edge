"""Model resource profiling.

Placeholder created by #1 as part of the interface freeze. #5 turns these
green. See docs/interfaces.md section 5 for the full test-file map.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(reason="not implemented yet - owned by #5")


def test_reproduces_the_mobilenet_slide_example():
    """docs/slides/CNNs.pdf: a 3x3 256->256 conv costs ~590k ops, the
    depthwise separable equivalent ~67.8k, an 8.7x speedup. The profiler must
    reproduce those numbers, because #11 cross-checks its hand derivation
    against this code."""


def test_macs_are_per_image_not_per_batch():
    """Batch size must not change the reported MAC count."""


def test_grouped_convolution_divides_by_group_count():
    """This is exactly where Model B's saving comes from."""


def test_disk_size_is_measured_not_estimated():
    """Must come from a real save, including pickle overhead."""
