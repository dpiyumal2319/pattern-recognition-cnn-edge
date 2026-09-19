"""Canonical project paths.

Everything resolves from the repository root so that a script behaves the same
whether it is launched from the repo root, from ``scripts/``, or from a Colab
notebook that cloned the repo into ``/content``.
"""

from __future__ import annotations

from pathlib import Path

# src/pr_edge/paths.py -> src/pr_edge -> src -> repo root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = PROJECT_ROOT / "data"
RESULTS_DIR: Path = PROJECT_ROOT / "results"
REPORT_DIR: Path = PROJECT_ROOT / "report"
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
SLIDES_DIR: Path = PROJECT_ROOT / "docs" / "slides"


def run_dir(run_id: str, smoke: bool = False) -> Path:
    """Return the output directory for a run, creating it if needed.

    Smoke runs are prefixed with an underscore and gitignored: they exist only
    to prove the wiring works in CI, and their numbers are meaningless.
    """
    name = f"_smoke_{run_id}" if smoke else run_id
    path = RESULTS_DIR / name
    path.mkdir(parents=True, exist_ok=True)
    return path
