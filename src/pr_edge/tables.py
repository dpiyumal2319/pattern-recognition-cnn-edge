"""LaTeX table generation from result artifacts.

This module is the only path by which a number reaches the report:

    results/*/[metrics|profile].json -> build_tables.py -> report/tables/*.tex -> \\input{}

Never hand-type a result into LaTeX. If a run is repeated, the report updates
itself; if a number is typed, it silently goes stale and nobody notices until
the viva.

Each function emits a bare ``tabular`` fragment - no ``\\begin{table}``
wrapper. Caption, label and placement belong to the report, not the generator.
"""

from __future__ import annotations

from pathlib import Path


def build_dataset_table(manifest: Path, out: Path) -> Path:
    """Per-class counts across the train/val/test splits. Implemented by #24."""
    raise NotImplementedError("build_dataset_table is issue #24")


def build_custom_comparison(run_ids: list[str], out: Path) -> Path:
    """Model A vs Model B.

    The brief names these columns: total parameter count, estimated model size
    on disk (KB), training time per epoch, and test accuracy. Macro precision
    and recall are added because task 4 requires them too. Implemented by #18.
    """
    raise NotImplementedError("build_custom_comparison is issue #18")


def build_sota_comparison(run_ids: list[str], out: Path) -> Path:
    """Fine-tuned lightweight SOTA models.

    Sizes in MB here, not KB - the brief asks for MB for these, and the
    magnitude gap against the custom models is itself the point. Implemented
    by #22.
    """
    raise NotImplementedError("build_sota_comparison is issue #22")


def build_final_comparison(run_ids: list[str], out: Path) -> Path:
    """Three-way table for task 6: Model B against both SOTA models.

    Accuracy, parameters, size, MACs and epoch time. Implemented by #25.
    """
    raise NotImplementedError("build_final_comparison is issue #25")
