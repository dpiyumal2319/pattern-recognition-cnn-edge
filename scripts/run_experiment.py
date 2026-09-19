#!/usr/bin/env python
"""Run one experiment end to end and write a complete run directory.

    python scripts/run_experiment.py --config configs/model_b.yaml
    python scripts/run_experiment.py --config configs/model_b.yaml --smoke

CLI flags override the config file. The RESOLVED values - after overrides -
are what get dumped to ``config.yaml`` in the run directory, so a run is always
reproducible from its own output.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pr_edge.config import dump_config, load_config  # noqa: E402
from pr_edge.data import get_dataloaders, get_synthetic_dataloaders  # noqa: E402
from pr_edge.device import get_device, set_seed  # noqa: E402
from pr_edge.evaluate import evaluate  # noqa: E402
from pr_edge.models import build_model  # noqa: E402
from pr_edge.paths import run_dir  # noqa: E402
from pr_edge.profile import profile_model  # noqa: E402
from pr_edge.train import train  # noqa: E402

# Smoke mode overrides, frozen in docs/interfaces.md section 4. These are not
# tunable: CI relies on the run finishing in under 60 seconds on a CPU runner.
SMOKE_OVERRIDES = {"epochs": 2, "batch_size": 16, "num_workers": 0, "device": "cpu", "smoke": True}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, type=Path, help="path to a YAML config in configs/")
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic data, 2 epochs, CPU; writes results/_smoke_<run_id>/")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--device", type=str, default=None, help="override device (auto|cpu|cuda|mps)")
    parser.add_argument("--run-id", type=str, default=None, help="override the output directory name")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    overrides: dict = {"epochs": args.epochs, "device": args.device}
    if args.smoke:
        overrides.update(SMOKE_OVERRIDES)

    cfg = load_config(args.config, overrides)
    run_id = args.run_id or cfg.run_id
    out_dir = run_dir(run_id, smoke=cfg.training.smoke)

    set_seed(cfg.seed)
    device = get_device(cfg.device)
    print(f"[run] {run_id}  device={device}  epochs={cfg.training.epochs}  seed={cfg.seed}")

    # The resolved config is written FIRST, so a crashed run still leaves
    # behind enough to reproduce what was attempted.
    dump_config(cfg, out_dir / "config.yaml")

    loaders = get_synthetic_dataloaders(cfg) if cfg.training.smoke else get_dataloaders(cfg)
    train_loader, val_loader, test_loader = loaders

    model = build_model(cfg)

    history = train(model, cfg, train_loader, val_loader, device, out_dir)

    metrics = evaluate(model, test_loader, cfg, device, out_dir)
    print(f"[test] accuracy={metrics['accuracy']:.4f}  "
          f"macro_precision={metrics['macro_precision']:.4f}  "
          f"macro_recall={metrics['macro_recall']:.4f}")

    mean_epoch = sum(r["epoch_seconds"] for r in history) / len(history)
    profile = profile_model(model, cfg, mean_epoch, device, out_dir)
    print(f"[profile] params={profile['trainable_params']:,}  "
          f"size={profile['disk_size_kb']:.1f} KB  "
          f"macs={profile['macs']:,}  "
          f"{profile['mean_epoch_seconds']:.2f}s/epoch")

    print(f"[done] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
