#!/usr/bin/env python
"""Run the Task 3 momentum ablation and plot validation loss.

All four experiments use Model B with SGD at learning rate 0.01.  Momentum is
the only methodological variable:

    python scripts/run_momentum_ablation.py --device cuda --reuse-existing
    python scripts/run_momentum_ablation.py --smoke

The reusable 0.0 and 0.9 runs are produced by issue #13.  Reuse is deliberately
opt-in and validates the complete run plus its resolved configuration before a
history is accepted.  Full runs write ``report/figures/momentum_ablation.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pr_edge.config import Config, dump_config, load_config  # noqa: E402
from pr_edge.paths import CONFIGS_DIR, REPORT_DIR, RESULTS_DIR, run_dir  # noqa: E402
from pr_edge.schemas import validate_run_dir  # noqa: E402
from scripts import run_experiment  # noqa: E402

BASE_CONFIG = CONFIGS_DIR / "model_b_sgd.yaml"
MOMENTUM_RUNS = {
    0.0: "momentum_m000",
    0.5: "momentum_m050",
    0.9: "momentum_m090",
    0.99: "momentum_m099",
}
REUSE_RUNS = {0.0: "model_b_sgd", 0.9: "model_b_sgdm"}
LABELS = {momentum: f"m={momentum:g}" for momentum in MOMENTUM_RUNS}


def _build_configs() -> list[Config]:
    """Derive the four configs without adding non-canonical YAML files."""
    base = load_config(BASE_CONFIG)
    return [
        replace(
            base,
            run_id=run_id,
            optimizer=replace(base.optimizer, momentum=momentum),
        )
        for momentum, run_id in MOMENTUM_RUNS.items()
    ]


def _validate_ablation_configs(configs: list[Config]) -> None:
    """Fail unless run ID and momentum are the only differences."""
    if [cfg.run_id for cfg in configs] != list(MOMENTUM_RUNS.values()):
        raise ValueError("momentum ablation has unexpected run IDs")

    common = []
    for cfg in configs:
        values = asdict(cfg)
        values.pop("run_id")
        values["optimizer"].pop("momentum")
        common.append(values)
    if any(values != common[0] for values in common[1:]):
        raise ValueError(
            "momentum ablation configs differ outside run_id and optimizer.momentum"
        )

    observed = [
        (cfg.optimizer.name, cfg.optimizer.lr, cfg.optimizer.momentum, cfg.optimizer.weight_decay)
        for cfg in configs
    ]
    expected = [("sgd", 0.01, momentum, 0.0) for momentum in MOMENTUM_RUNS]
    if observed != expected:
        raise ValueError(f"unexpected momentum ablation optimizer blocks: {observed!r}")


def _resolved_config(cfg: Config, args: argparse.Namespace) -> dict:
    """Expected config.yaml contents after CLI overrides."""
    expected = asdict(cfg)
    if args.device is not None:
        expected["device"] = args.device
    if args.epochs is not None:
        expected["training"]["epochs"] = args.epochs
    if args.smoke:
        expected["device"] = "cpu"
        expected["data"]["batch_size"] = 16
        expected["data"]["num_workers"] = 0
        expected["training"]["epochs"] = 2
        expected["training"]["smoke"] = True
    return expected


def _reusable_history(momentum: float, cfg: Config, args: argparse.Namespace) -> Path | None:
    """Return a validated existing history, or ``None`` when no run exists."""
    reuse_id = REUSE_RUNS.get(momentum, cfg.run_id)
    directory = RESULTS_DIR / (f"_smoke_{reuse_id}" if args.smoke else reuse_id)
    if not directory.exists():
        return None
    errors = validate_run_dir(directory)
    if errors:
        raise ValueError(f"cannot reuse incomplete run {reuse_id}: {errors}")
    actual = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    expected = _resolved_config(cfg, args)
    expected["run_id"] = reuse_id
    if actual != expected:
        raise ValueError(f"cannot reuse {reuse_id}: resolved config does not match ablation")
    return directory / "history.json"


def _plot_validation_losses(histories: dict[float, Path], out: Path) -> Path:
    """Overlay one validation-loss curve per momentum value."""
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for momentum, history_path in histories.items():
        records = json.loads(history_path.read_text(encoding="utf-8"))
        if not records:
            raise ValueError(f"empty history: {history_path}")
        ax.plot(
            [record["epoch"] for record in records],
            [record["val_loss"] for record in records],
            marker="o",
            markersize=3,
            linewidth=1.8,
            label=LABELS[momentum],
        )

    ax.set_title("Model B momentum ablation (SGD, lr=0.01)")
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
    parser.add_argument("--smoke", action="store_true", help="two CPU epochs on synthetic data")
    parser.add_argument("--reuse-existing", action="store_true", help="reuse only schema/config-valid runs")
    parser.add_argument("--epochs", type=int, default=None, help="override training.epochs")
    parser.add_argument("--device", type=str, default=None, help="override device (auto|cpu|cuda|mps)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configs = _build_configs()
    _validate_ablation_configs(configs)

    histories: dict[float, Path] = {}
    with tempfile.TemporaryDirectory(prefix="momentum-configs-") as temp:
        temp_dir = Path(temp)
        for (momentum, _), cfg in zip(MOMENTUM_RUNS.items(), configs, strict=True):
            if args.reuse_existing:
                existing = _reusable_history(momentum, cfg, args)
                if existing is not None:
                    histories[momentum] = existing
                    print(f"[momentum] reusing {existing.parent.name} ({LABELS[momentum]})")
                    continue

            config_path = temp_dir / f"{cfg.run_id}.yaml"
            dump_config(cfg, config_path)
            runner_args = ["--config", str(config_path)]
            if args.smoke:
                runner_args.append("--smoke")
            if args.epochs is not None:
                runner_args.extend(["--epochs", str(args.epochs)])
            if args.device is not None:
                runner_args.extend(["--device", args.device])

            print(f"\n[momentum] starting {LABELS[momentum]}")
            exit_code = run_experiment.main(runner_args)
            if exit_code != 0:
                return exit_code
            histories[momentum] = run_dir(cfg.run_id, smoke=args.smoke) / "history.json"

    if args.smoke:
        figure = run_dir("momentum_ablation", smoke=True) / "momentum_ablation.png"
    else:
        figure = REPORT_DIR / "figures" / "momentum_ablation.png"
    _plot_validation_losses(histories, figure)
    print(f"[momentum] figure={figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
