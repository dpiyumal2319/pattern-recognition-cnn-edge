"""Training loop.

One loop serves every experiment in the project: both custom models, the three
optimizer variants, and the fine-tuned SOTA models. Anything model-specific
belongs in ``models/``, not here.

Model selection (#6): the checkpoint kept is the epoch with the lowest
validation loss, not the last one, and those weights are restored into the
model before ``train`` returns. The caller evaluates and profiles the in-memory
model, so without the restore every test metric would describe the final epoch
while ``model.pt`` held a different one. The chosen epoch is recorded in
``selection.json`` so the report can cite it.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
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


# The selection criterion, recorded in selection.json. Validation LOSS rather
# than accuracy: accuracy only changes when an argmax flips, so it plateaus and
# ties across epochs while the network is already growing confidently wrong on
# held-out images. Loss is continuous and shows that overfitting first.
SELECTION_CRITERION = "val_loss"


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    """Write via a sibling temp file, then rename over ``path``.

    A Colab runtime can disconnect at any moment. ``os.replace`` is atomic, so
    ``path`` always holds either the previous complete file or the new one -
    never a truncated checkpoint that fails to load at evaluation time.
    """
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, data: object) -> None:
    _atomic_write(path, lambda tmp: tmp.write_text(json.dumps(data, indent=2), encoding="utf-8"))


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

    On return ``model`` holds the weights of the epoch with the lowest
    validation loss (ties go to the earlier epoch), which is also what
    ``model.pt`` contains. ``history.json`` is rewritten every epoch and
    ``model.pt`` on every improvement, so an interrupted run still leaves its
    best checkpoint and a readable history behind.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, cfg.optimizer)
    model.to(device)

    checkpoint = out_dir / "model.pt"
    selection_file = out_dir / "selection.json"
    # A rerun into the same directory must not leave a previous run's weights
    # or selection beside this run's history if it fails before its first save.
    checkpoint.unlink(missing_ok=True)
    selection_file.unlink(missing_ok=True)

    history: list[dict] = []
    best: dict | None = None

    for epoch in range(1, cfg.training.epochs + 1):
        start = time.perf_counter()
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        epoch_seconds = time.perf_counter() - start

        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device)

        record = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "epoch_seconds": round(epoch_seconds, 4),
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        # Compare the value as recorded in history.json, so anyone can re-derive
        # the choice from that file alone. Strict "<" keeps the earlier epoch on
        # a tie. A diverged (NaN/inf) epoch is never selected, but the run keeps
        # going: divergence is itself a result the optimizer study reports.
        improved = math.isfinite(record["val_loss"]) and (
            best is None or record["val_loss"] < best["val_loss"]
        )
        if improved:
            best = record
            _atomic_write(checkpoint, lambda tmp: torch.save(model.state_dict(), tmp))
        _write_json(out_dir / "history.json", history)

        print(
            f"epoch {epoch:3d}/{cfg.training.epochs}  "
            f"train {train_loss:.4f}/{train_acc:.3f}  "
            f"val {val_loss:.4f}/{val_acc:.3f}  "
            f"{epoch_seconds:.1f}s" + ("  *best" if improved else "")
        )

    if best is None:
        raise RuntimeError(
            "no epoch produced a finite validation loss - the run diverged from the "
            "start, so there is no checkpoint to evaluate. history.json is kept."
        )

    # Restore from the file rather than an in-memory copy, so the weights that
    # get evaluated and profiled are provably the ones saved in model.pt.
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))

    _write_json(selection_file, {
        "criterion": SELECTION_CRITERION,
        "best_epoch": best["epoch"],
        "best_val_loss": best["val_loss"],
        "best_val_acc": best["val_acc"],
        "epochs_trained": len(history),
    })
    print(f"[select] epoch {best['epoch']}/{len(history)} by lowest val_loss "
          f"({best['val_loss']:.4f}, val acc {best['val_acc']:.3f})")

    return history
