#!/usr/bin/env python
"""Prepare EuroSAT, training-only normalisation and the report sample grid.

    python scripts/prepare_data.py --config configs/base.yaml [--force]

Implemented by #2. Re-running must not re-download or change the split - the
manifest is committed precisely so that every experiment, custom and SOTA
alike, provably uses the same partition.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pr_edge.config import load_config  # noqa: E402
from pr_edge.data import _write_sample_grid, compute_norm_stats, prepare_eurosat  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--force", action="store_true", help="re-download and regenerate the split")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    manifest = prepare_eurosat(cfg, force=args.force)
    print(f"[data] manifest written to {manifest}")

    stats = compute_norm_stats(cfg)
    print(f"[data] train-split normalisation stats: {stats}")
    figure = _write_sample_grid(cfg)
    print(f"[data] sample grid written to {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
