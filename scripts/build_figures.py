#!/usr/bin/env python
"""Generate task 4 report figures from committed result artifacts.

    python scripts/build_figures.py --all
    python scripts/build_figures.py --figure loss
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pr_edge.paths import REPORT_DIR  # noqa: E402
from pr_edge.plots import plot_confusion_matrix, plot_loss_curves  # noqa: E402

FIGURES_DIR = REPORT_DIR / "figures"

BUILDERS = {
    "loss": lambda: plot_loss_curves(
        ["model_a", "model_b"],
        FIGURES_DIR / "custom_loss_curves.png",
        title="Custom CNN training and validation loss",
    ),
    "convergence": lambda: plot_loss_curves(
        ["model_a_100", "model_b_100"],
        FIGURES_DIR / "custom_convergence.png",
        title="Custom CNN 100-epoch convergence control",
        reference_epoch=30,
    ),
    "model_a_confusion": lambda: plot_confusion_matrix(
        "model_a", FIGURES_DIR / "model_a_confusion_matrix.png"
    ),
    "model_b_confusion": lambda: plot_confusion_matrix(
        "model_b", FIGURES_DIR / "model_b_confusion_matrix.png"
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="build every task 4 figure")
    group.add_argument("--figure", choices=sorted(BUILDERS), help="build one figure")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    names = sorted(BUILDERS) if args.all else [args.figure]
    for name in names:
        path = BUILDERS[name]()
        print(f"[figures] {name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
