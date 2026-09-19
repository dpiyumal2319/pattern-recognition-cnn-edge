"""Training loop.

One loop serves every experiment in the project: both custom models, the three
optimizer variants, and the fine-tuned SOTA models. Anything model-specific
belongs in ``models/``, not here.

Scope note: #1 provides a working minimal loop so that ``--smoke`` can run in
CI from day one. #6 owns best-checkpoint-by-validation-loss selection and the
hardening that goes with real training runs. See the PR discussion on #1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config, OptimizerConfig


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    """Construct the optimizer named in the config.

    Only Adam and SGD, because task 3 compares exactly those: the chosen
    optimizer against plain SGD and SGD with momentum. ``momentum`` is ignored
    by Adam, which is why the sweep in #13 must vary only the optimizer block
    and nothing else.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    name = cfg.name.lower()

    if name == "adam":
        return torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay
        )
    raise ValueError(f"unknown optimizer '{cfg.name}'. Known: adam, sgd")


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over a loader. Trains when an optimizer is given, else evaluates."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    seen = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch = labels.size(0)
            total_loss += loss.item() * batch
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += batch

    return total_loss / seen, correct / seen


def train(
    model: nn.Module,
    cfg: Config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    out_dir: Path,
) -> list[dict]:
    """Train for ``cfg.training.epochs`` and write ``history.json``.

    Returns the history records. ``epoch_seconds`` times the training pass only
    and deliberately excludes validation, so the "training time per epoch"
    column in the report means what it says.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg.optimizer)
    model.to(device)

    history: list[dict] = []

    for epoch in range(1, cfg.training.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        epoch_seconds = time.perf_counter() - start

        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "epoch_seconds": round(epoch_seconds, 4),
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

        print(
            f"epoch {epoch:3d}/{cfg.training.epochs}  "
            f"train {train_loss:.4f}/{train_acc:.3f}  "
            f"val {val_loss:.4f}/{val_acc:.3f}  "
            f"{epoch_seconds:.1f}s"
        )

    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # Final-epoch checkpoint. Selecting the BEST epoch by validation loss is
    # #6's job - it needs real training runs to be worth anything, and on two
    # epochs of random smoke data it would be noise.
    torch.save(model.state_dict(), out_dir / "model.pt")

    return history
