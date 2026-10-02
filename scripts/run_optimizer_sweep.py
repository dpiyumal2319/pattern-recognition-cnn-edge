#!/usr/bin/env python
"""Run the Task 3 optimizer comparison and plot validation loss.

The three experiments intentionally share the same model, seed, data order,
batch size, epoch budget, and device.  Only the optimizer block differs:

    python scripts/run_optimizer_sweep.py --device cuda
    python scripts/run_optimizer_sweep.py --smoke

Full runs write ``report/figures/optimizer_comparison.png``.  Smoke runs keep
their meaningless random-data figure under the gitignored smoke-results tree.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pr_edge.config import Config, load_config  # noqa: E402
from pr_edge.paths import CONFIGS_DIR, REPORT_DIR, run_dir  # noqa: E402
from scripts import run_experiment  # noqa: E402

SWEEP_CONFIGS = (
    CONFIGS_DIR / "model_b.yaml",
    CONFIGS_DIR / "model_b_sgd.yaml",
    CONFIGS_DIR / "model_b_sgdm.yaml",
)

LABELS = {
    "model_b": "Adam (lr=0.001)",
    "model_b_sgd": "SGD (lr=0.01)",
    "model_b_sgdm": "SGD + momentum 0.9 (lr=0.01)",
}


def _validate_sweep_configs(configs: list[Config]) -> None:
    """Fail before training if anything except run ID and optimizer differs."""
    if [cfg.run_id for cfg in configs] != list(LABELS):
        raise ValueError(f"optimizer sweep requires run IDs {list(LABELS)}")

    common = []
    for cfg in configs:
        values = asdict(cfg)
        values.pop("run_id")
        values.pop("optimizer")
        common.append(values)
    if any(values != common[0] for values in common[1:]):
        raise ValueError(
            "optimizer sweep configs differ outside run_id and optimizer; "
            "the comparison would not isolate optimizer choice"
        )

    observed = [
        (cfg.optimizer.name, cfg.optimizer.lr, cfg.optimizer.momentum,
         cfg.optimizer.weight_decay)
        for cfg in configs
    ]
    expected = [
        ("adam", 0.001, 0.0, 0.0),
        ("sgd", 0.01, 0.0, 0.0),
        ("sgd", 0.01, 0.9, 0.0),
    ]
    if observed != expected:
        raise ValueError(f"unexpected optimizer sweep blocks: {observed!r}")


def _plot_validation_losses(histories: dict[str, Path], out: Path) -> Path:
    """Plot one validation-loss curve per optimizer from committed run data."""
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for run_id, history_path in histories.items():
        records = json.loads(history_path.read_text(encoding="utf-8"))
        if not records:
            raise ValueError(f"empty history: {history_path}")
        ax.plot(
            [record["epoch"] for record in records],
            [record["val_loss"] for record in records],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=LABELS[run_id],
        )

    ax.set_title("Model B optimizer comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation loss")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run all three variants on synthetic data for two CPU epochs",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument(
        "--device", type=str, default=None, help="override device (auto|cpu|cuda|mps)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configs = [load_config(path) for path in SWEEP_CONFIGS]
    _validate_sweep_configs(configs)

    histories: dict[str, Path] = {}
    for path, cfg in zip(SWEEP_CONFIGS, configs, strict=True):
        runner_args = ["--config", str(path)]
        if args.smoke:
            runner_args.append("--smoke")
        if args.epochs is not None:
            runner_args.extend(["--epochs", str(args.epochs)])
        if args.device is not None:
            runner_args.extend(["--device", args.device])

        print(f"\n[sweep] starting {LABELS[cfg.run_id]}")
        exit_code = run_experiment.main(runner_args)
        if exit_code != 0:
            return exit_code
        histories[cfg.run_id] = run_dir(cfg.run_id, smoke=args.smoke) / "history.json"

    if args.smoke:
        figure = run_dir("optimizer_sweep", smoke=True) / "optimizer_comparison.png"
    else:
        figure = REPORT_DIR / "figures" / "optimizer_comparison.png"
    _plot_validation_losses(histories, figure)
    print(f"[sweep] figure={figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
