"""Report figures: loss curves and confusion matrices (issue #17).

Every figure regenerates from ``results/*/`` with one command. Nothing is
hand-edited, so rerunning an experiment updates the report automatically.
"""

from __future__ import annotations

from pathlib import Path


def plot_loss_curves(run_ids: list[str], out: Path, title: str | None = None) -> Path:
    """Training and validation loss against epoch, one series pair per run.

    Use a shared y-scale when comparing models so the curves are visually
    comparable. Implemented by #17.
    """
    raise NotImplementedError("plot_loss_curves is issue #17")


def plot_confusion_matrix(run_id: str, out: Path, normalize: bool = True) -> Path:
    """Labelled confusion matrix heatmap for one run.

    Must use real class names from ``metrics.json``, never indices, and must
    stay readable at report size. Normalising by true-class row makes the
    diagonal comparable across classes of different sizes. Implemented by #17.
    """
    raise NotImplementedError("plot_confusion_matrix is issue #17")
