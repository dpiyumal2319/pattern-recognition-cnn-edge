#!/usr/bin/env python
"""Generate task 4 and task 5 report figures from committed result artifacts.

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
    # Task 5 (#20): frozen vs fine-tuned, one figure per input resolution so
    # each pair shares a y-scale and stays readable at report width. All four
    # runs use batch 128; the batch-32 record mobilenetv2_224_ft is not plotted.
    "sota_loss": lambda: plot_loss_curves(
        ["mobilenetv2_64", "mobilenetv2_64_ft"],
        FIGURES_DIR / "sota_loss_curves.png",
        title="MobileNetV2 at 64x64: training and validation loss",
    ),
    "sota_224_loss": lambda: plot_loss_curves(
        ["mobilenetv2_224", "mobilenetv2_224_ft_b128"],
        FIGURES_DIR / "sota_224_loss_curves.png",
        title="MobileNetV2 at 224x224: training and validation loss",
    ),
    # The primary SOTA run: fine-tuned, at the custom models' 64x64 input.
    "mobilenetv2_64_ft_confusion": lambda: plot_confusion_matrix(
        "mobilenetv2_64_ft", FIGURES_DIR / "mobilenetv2_64_ft_confusion_matrix.png"
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="build every report figure")
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
