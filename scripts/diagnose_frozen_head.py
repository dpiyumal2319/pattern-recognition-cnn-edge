#!/usr/bin/env python
"""Diagnose why the frozen-backbone EfficientNet-B0 loss curves are noisy.

The Task 5 loss curves show frozen EfficientNet-B0's validation loss
fluctuating far more than frozen MobileNetV2's, at the same optimizer settings.
This script tests the candidate causes the report discusses, so its claims
trace back to a committed artifact rather than to an uncommitted experiment:

    python scripts/diagnose_frozen_head.py                # both 64x64 frozen runs
    python scripts/diagnose_frozen_head.py --device cuda

For each frozen run it:

1. builds the run's model from its committed config, with the backbone in
   evaluation mode, and records the pooled features the prediction layer
   receives, for the training and validation sets, without augmentation;
2. summarises those features (range, outliers), because MobileNetV2 ends in
   the bounded ReLU6 and EfficientNet-B0 in the unbounded SiLU;
3. retrains only a fresh prediction layer (dropout + linear, as in the real
   head) on the fixed features, with Adam at several learning rates, and once
   more on standardised features at the run's own learning rate.

Compare the observed loss variation across learning rates and feature scaling.
Persistence at the tested rates does not rule out a learning-rate contribution,
and standardisation is not an isolated intervention on activation choice.
The epoch-loss standard deviation also includes any remaining learning trend.
Output: results/frozen_head_diagnostic/diagnostic.json.

This supplementary diagnostic trains no backbone, reports no test-set number,
and leaves the committed runs untouched. Its fixed, unaugmented features differ
from the main training protocol, so it does not establish a unique cause of
the original curves' fluctuations.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pr_edge.config import load_config  # noqa: E402
from pr_edge.data import get_dataloaders  # noqa: E402
from pr_edge.device import device_name, get_device, set_seed  # noqa: E402
from pr_edge.models import build_model  # noqa: E402
from pr_edge.paths import CONFIGS_DIR, RESULTS_DIR  # noqa: E402

DEFAULT_RUNS = ["mobilenetv2_64", "efficientnetb0_64"]
DEFAULT_LEARNING_RATES = [1e-3, 3e-4, 1e-4]
# The runs' own seed first, then two more, so a conclusion never rests on one seed.
DEFAULT_SEEDS = [42, 43, 44]
OUTPUT = RESULTS_DIR / "frozen_head_diagnostic" / "diagnostic.json"

# The first epochs are the initial descent, not noise. Summaries start at the
# fifth epoch, matching how the report compares the committed curves.
SUMMARY_FROM_EPOCH = 5

# ReLU6's ceiling, and a value far above it, for counting outlier features.
RELU6_CEILING = 6.0
SPIKE_THRESHOLD = 50.0


def capture_classifier_inputs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """The pooled features the prediction layer receives, with their labels.

    A forward pre-hook on ``model.classifier`` records its input, so the
    pooling and flattening are exactly the network's own, whatever its
    architecture. The model runs in evaluation mode: frozen batch-norm
    statistics, dropout off.
    """
    captured: list[torch.Tensor] = []
    handle = model.classifier.register_forward_pre_hook(
        lambda _module, inputs: captured.append(inputs[0].detach().cpu())
    )
    labels: list[torch.Tensor] = []
    model.eval().to(device)
    try:
        with torch.no_grad():
            for images, targets in loader:
                model(images.to(device))
                labels.append(targets)
    finally:
        handle.remove()
    return torch.cat(captured), torch.cat(labels)


def feature_statistics(features: torch.Tensor) -> dict:
    """Range and outlier summary of a (samples, channels) feature matrix."""
    spikes = features > SPIKE_THRESHOLD
    return {
        "num_samples": features.shape[0],
        "num_channels": features.shape[1],
        "mean": round(features.mean().item(), 6),
        "min": round(features.min().item(), 6),
        "max": round(features.max().item(), 6),
        "mean_l2_norm_per_image": round(features.norm(dim=1).mean().item(), 6),
        "fraction_values_above_relu6_ceiling": round((features > RELU6_CEILING).float().mean().item(), 6),
        "fraction_values_above_spike_threshold": round(spikes.float().mean().item(), 6),
        "fraction_images_with_a_spike": round(spikes.any(dim=1).float().mean().item(), 6),
        "channels_with_a_spike": int(spikes.any(dim=0).sum().item()),
        "spike_threshold": SPIKE_THRESHOLD,
    }


def standardise(
    train: torch.Tensor, val: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel zero mean, unit variance, using training statistics only."""
    mean, std = train.mean(dim=0), train.std(dim=0) + 1e-6
    return (train - mean) / std, (val - mean) / std


def train_head(
    train: tuple[torch.Tensor, torch.Tensor],
    val: tuple[torch.Tensor, torch.Tensor],
    *,
    num_classes: int,
    dropout: float,
    lr: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[dict]:
    """Train a fresh dropout + linear head on fixed features; per-epoch val loss/acc.

    Mirrors the frozen runs' prediction layer and optimizer (Adam, cross-entropy,
    the run's batch size and epoch budget). Seeded, so a rerun is identical.
    """
    set_seed(seed)
    (x_train, y_train), (x_val, y_val) = train, val
    head = nn.Sequential(nn.Dropout(dropout), nn.Linear(x_train.shape[1], num_classes))
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    order = torch.Generator().manual_seed(seed)

    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        permutation = torch.randperm(len(x_train), generator=order)
        for start in range(0, len(x_train), batch_size):
            batch = permutation[start:start + batch_size]
            optimizer.zero_grad()
            loss_fn(head(x_train[batch]), y_train[batch]).backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            logits = head(x_val)
            history.append({
                "epoch": epoch,
                "val_loss": round(loss_fn(logits, y_val).item(), 6),
                "val_acc": round((logits.argmax(dim=1) == y_val).float().mean().item(), 6),
            })
    return history


def summarise(history: list[dict]) -> dict:
    """Mean and population standard deviation of val loss/acc from SUMMARY_FROM_EPOCH on."""
    tail = [record for record in history if record["epoch"] >= SUMMARY_FROM_EPOCH]
    if len(tail) < 2:
        raise ValueError(f"need at least two epochs from epoch {SUMMARY_FROM_EPOCH} to summarise")
    losses = [record["val_loss"] for record in tail]
    accuracies = [record["val_acc"] for record in tail]
    return {
        "val_loss_mean": round(statistics.mean(losses), 6),
        "val_loss_sd": round(statistics.pstdev(losses), 6),
        "val_acc_mean": round(statistics.mean(accuracies), 6),
        "val_acc_sd": round(statistics.pstdev(accuracies), 6),
    }


def repeat_over_seeds(train, val, seeds: list[int], **settings) -> dict:
    """Train one head configuration once per seed and aggregate the summaries.

    A single seed is not enough: the standard deviation of 26 epoch losses is
    itself a noisy estimate, and one seed's value can mislead. The aggregate
    keeps every seed's summary and history so the spread stays visible.
    """
    per_seed = []
    for seed in seeds:
        history = train_head(train, val, seed=seed, **settings)
        per_seed.append({"seed": seed, **summarise(history), "history": history})
    aggregate = {}
    for key in ("val_loss_mean", "val_loss_sd", "val_acc_mean", "val_acc_sd"):
        values = [run[key] for run in per_seed]
        aggregate[f"{key}_mean"] = round(statistics.mean(values), 6)
        aggregate[f"{key}_min"] = min(values)
        aggregate[f"{key}_max"] = max(values)
    return {**aggregate, "per_seed": per_seed}


def diagnose_run(
    run_id: str, learning_rates: list[float], seeds: list[int], device: torch.device, num_workers: int
) -> dict:
    """Feature statistics and head retraining for one committed frozen run."""
    cfg = load_config(CONFIGS_DIR / f"{run_id}.yaml")
    if not cfg.model.freeze_backbone:
        raise ValueError(f"{run_id} does not freeze its backbone; this diagnostic is for frozen runs")
    # Features are recorded once, so per-epoch augmentation cannot apply.
    cfg = replace(cfg, data=replace(cfg.data, augment=False, num_workers=num_workers))

    set_seed(cfg.seed)
    train_loader, val_loader, _ = get_dataloaders(cfg)  # the test set is never touched
    model = build_model(cfg)
    print(f"[{run_id}] recording pooled features on {device}...", flush=True)
    train = capture_classifier_inputs(model, train_loader, device)
    val = capture_classifier_inputs(model, val_loader, device)

    settings = dict(
        num_classes=cfg.model.num_classes, dropout=cfg.model.dropout,
        epochs=cfg.training.epochs, batch_size=cfg.data.batch_size,
    )
    z_train, z_val = standardise(train[0], val[0])
    variants = [(lr, False, train, val) for lr in learning_rates]
    variants.append((cfg.optimizer.lr, True, (z_train, train[1]), (z_val, val[1])))

    heads = []
    for lr, standardised, head_train, head_val in variants:
        result = repeat_over_seeds(head_train, head_val, seeds, lr=lr, **settings)
        heads.append({"lr": lr, "standardised": standardised, **result})
        label = "standardised" if standardised else "raw"
        print(f"[{run_id}] lr={lr:g} {label:12s} val-loss sd {result['val_loss_sd_mean']:.4f} "
              f"(seeds {result['val_loss_sd_min']:.4f}-{result['val_loss_sd_max']:.4f})", flush=True)

    committed = json.loads((RESULTS_DIR / run_id / "history.json").read_text(encoding="utf-8"))
    return {
        "model": cfg.model.name,
        "configured_lr": cfg.optimizer.lr,
        "committed_run": summarise(committed),
        "features": {"train": feature_statistics(train[0]), "val": feature_statistics(val[0])},
        "heads": heads,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="frozen run IDs to diagnose")
    parser.add_argument("--lrs", nargs="+", type=float, default=DEFAULT_LEARNING_RATES,
                        help="learning rates for the retrained head")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                        help="seeds each head configuration is trained with")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="data-loader workers; 0 is safest on Windows")
    parser.add_argument("--out", type=Path, default=OUTPUT, help="where to write the JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    device = get_device(args.device)
    result = {
        "purpose": "Why the frozen EfficientNet-B0 validation loss is noisy (Task 5 discussion).",
        "script": "scripts/diagnose_frozen_head.py",
        "protocol": {
            "features": "prediction-layer input, backbone in eval mode, no augmentation",
            "head": "fresh Dropout + Linear, Adam, cross-entropy, run's batch size and epochs, once per seed",
            "summary_from_epoch": SUMMARY_FROM_EPOCH,
            "learning_rates": args.lrs,
            "seeds": args.seeds,
            "device": str(device),
            "device_name": device_name(device),
            "torch": torch.__version__,
        },
        "runs": {
            run_id: diagnose_run(run_id, args.lrs, args.seeds, device, args.num_workers) for run_id in args.runs
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[done] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
