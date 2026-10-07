#!/usr/bin/env python
"""Generate the report's LaTeX tables from committed result artifacts.

    python scripts/build_tables.py --all
    python scripts/build_tables.py --table custom

This is the ONLY path by which a number reaches the report. Individual table
builders are implemented by #18, #22, #24 and #25.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pr_edge.paths import CONFIGS_DIR, DATA_DIR, REPORT_DIR  # noqa: E402
from pr_edge.tables import (  # noqa: E402
    build_custom_comparison,
    build_dataset_table,
    build_final_comparison,
    build_layer_table,
    build_optimizer_comparison,
    build_sota_comparison,
)

TABLES_DIR = REPORT_DIR / "tables"

# Which runs feed which table. Fixed here so no issue has to guess, and so a
# missing run fails loudly rather than producing a table with a gap in it.
BUILDERS = {
    # Architecture tables are built from the configs, not from runs: they need
    # no training, only the model the config describes.
    "model_a_layers": lambda: build_layer_table(CONFIGS_DIR / "model_a.yaml", TABLES_DIR / "model_a_layers.tex"),
    "model_b_layers": lambda: build_layer_table(CONFIGS_DIR / "model_b.yaml", TABLES_DIR / "model_b_layers.tex"),
    "dataset": lambda: build_dataset_table(DATA_DIR / "splits.csv", TABLES_DIR / "dataset.tex"),
    "custom": lambda: build_custom_comparison(["model_a", "model_b"], TABLES_DIR / "custom_comparison.tex"),
    "optimizer": lambda: build_optimizer_comparison(
        ["model_b", "model_b_sgd", "model_b_sgdm"], TABLES_DIR / "optimizer_comparison.tex"
    ),
    # Each network as a frozen / fine-tuned pair (CNNs.pdf p58); the fine-tuned
    # run is the primary SOTA result (#22, #54). The 224x224 pair differs from
    # the 64x64 one only in input size and renders as a supplementary group.
    "sota": lambda: build_sota_comparison(
        [
            "mobilenetv2_64",
            "mobilenetv2_64_ft",
            "efficientnetb0_64",
            "efficientnetb0_64_ft",
            "mobilenetv2_224",
            "mobilenetv2_224_ft_b128",
        ],
        TABLES_DIR / "sota_comparison.tex",
    ),
    "final": lambda: build_final_comparison(
        ["model_b", "mobilenetv2_64", "efficientnetb0_64"], TABLES_DIR / "final_comparison.tex"
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="build every table")
    group.add_argument("--table", choices=sorted(BUILDERS), help="build one table")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    names = sorted(BUILDERS) if args.all else [args.table]
    for name in names:
        path = BUILDERS[name]()
        print(f"[tables] {name} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
