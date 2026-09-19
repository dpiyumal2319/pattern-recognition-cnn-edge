"""Test-set evaluation: accuracy, confusion matrix, precision and recall.

Every metric the brief's task 4 names. Both macro-averaged and per-class are
reported: with 10 EuroSAT classes the per-class breakdown is what makes the
confusion matrix discussion substantive rather than decorative.

Scope note: #1 provides a working implementation so ``--smoke`` emits a
schema-conformant ``metrics.json`` in CI. #4 owns the hand-computed
verification tests and any hardening. See the PR discussion on #1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import Config
from .data import CLASS_NAMES

# Rows are TRUE labels, columns are PREDICTIONS, both ordered by CLASS_NAMES.
# Written into every metrics.json so a figure can never silently transpose it.
CONFUSION_CONVENTION = "rows=true, cols=predicted, order=class_names"


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n: int) -> np.ndarray:
    matrix = np.zeros((n, n), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def _per_class_scores(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Precision, recall, F1 and support from a confusion matrix.

    A class the model never predicts has an undefined precision (0/0). We
    report 0.0 rather than NaN so the JSON stays numeric and the report tables
    stay readable - but note it in the discussion if it happens, because a
    never-predicted class is a real finding about class imbalance, not a
    rounding detail.
    """
    tp = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)  # column sums
    actual = matrix.sum(axis=1).astype(np.float64)     # row sums

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(predicted > 0, tp / predicted, 0.0)
        recall = np.where(actual > 0, tp / actual, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)

    return precision, recall, f1, actual.astype(np.int64)


@torch.no_grad()
def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval().to(device)
    trues: list[np.ndarray] = []
    preds: list[np.ndarray] = []

    for images, labels in loader:
        logits = model(images.to(device, non_blocking=True))
        preds.append(logits.argmax(dim=1).cpu().numpy())
        trues.append(labels.numpy())

    return np.concatenate(trues), np.concatenate(preds)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    cfg: Config,
    device: torch.device,
    out_dir: Path | None = None,
    split: str = "test",
) -> dict:
    """Evaluate a model and optionally write ``metrics.json``.

    The test set is touched exactly once, here, after training is complete.
    """
    y_true, y_pred = _predict(model, loader, device)

    names = CLASS_NAMES[: cfg.model.num_classes]
    matrix = _confusion_matrix(y_true, y_pred, len(names))
    precision, recall, f1, support = _per_class_scores(matrix)

    metrics = {
        "run_id": cfg.run_id,
        "split": split,
        "num_samples": int(len(y_true)),
        "class_names": names,
        "accuracy": float((y_true == y_pred).mean()),
        # Macro averages weight every class equally, which is the right choice
        # when the report discusses per-class behaviour. EuroSAT is close to
        # balanced, so macro and micro will be similar, but macro is the one
        # that stays honest if a class collapses.
        "macro_precision": float(precision.mean()),
        "macro_recall": float(recall.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(names)
        },
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_convention": CONFUSION_CONVENTION,
    }

    if out_dir is not None:
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics
