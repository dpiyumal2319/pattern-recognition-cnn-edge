"""Report figures: loss curves and confusion matrices (issue #17).

Every figure regenerates from ``results/*/`` with one command. Nothing is
hand-edited, so rerunning an experiment updates the report automatically.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
import numpy as np

from .paths import RESULTS_DIR
from .schemas import validate_history, validate_metrics

# Figure generation runs in CI and Colab as well as on desktops.  Selecting a
# non-interactive backend before importing pyplot keeps it independent of a
# display server and prevents a report build from opening GUI windows.
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def _read_json(path: Path) -> object:
    if not path.exists():
        raise FileNotFoundError(f"missing result artifact: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def _run_label(run_id: str) -> str:
    labels = {"model_a": "Model A", "model_b": "Model B"}
    return labels.get(run_id, run_id.replace("_", " ").title())


def _save(fig, out: Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def plot_loss_curves(run_ids: list[str], out: Path, title: str | None = None) -> Path:
    """Training and validation loss against epoch, one series pair per run.

    Each run gets its own axis so its train/validation pair stays readable.
    The axes share one y-scale, preventing independent autoscaling from making
    a small loss difference look as large as a large one.
    """
    if not run_ids:
        raise ValueError("plot_loss_curves requires at least one run_id")

    histories: list[tuple[str, list[dict]]] = []
    for run_id in run_ids:
        path = RESULTS_DIR / run_id / "history.json"
        data = _read_json(path)
        errors = validate_history(data)
        if errors:
            raise ValueError(f"invalid {run_id}/history.json: {'; '.join(errors)}")
        histories.append((run_id, data))

    fig, axes = plt.subplots(
        1,
        len(histories),
        figsize=(6.2 * len(histories), 6.2),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    all_losses = [
        float(record[field])
        for _, history in histories
        for record in history
        for field in ("train_loss", "val_loss")
    ]
    finite_losses = [value for value in all_losses if math.isfinite(value)]
    if not finite_losses:
        plt.close(fig)
        raise ValueError("loss histories contain no finite train or validation loss")
    upper = max(finite_losses)
    lower = min(0.0, min(finite_losses))
    padding = max((upper - lower) * 0.05, 0.05)

    for axis, (run_id, history) in zip(axes, histories, strict=True):
        epochs = [record["epoch"] for record in history]
        axis.plot(
            epochs,
            [record["train_loss"] for record in history],
            color="#2563EB",
            linewidth=2.1,
            label="Training",
        )
        axis.plot(
            epochs,
            [record["val_loss"] for record in history],
            color="#DC2626",
            linewidth=2.1,
            label="Validation",
        )
        axis.set_title(_run_label(run_id), fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylim(lower, upper + padding)
        axis.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
        axis.legend(frameon=False)

    axes[0].set_ylabel("Cross-entropy loss")
    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, out)


def plot_confusion_matrix(run_id: str, out: Path, normalize: bool = True) -> Path:
    """Labelled confusion matrix heatmap for one run.

    Colour is normalised by true-class row, making each diagonal cell that
    class's recall.  Every cell is annotated with its raw count so the figure
    preserves the sample support and exact mistakes behind the proportions.
    """
    path = RESULTS_DIR / run_id / "metrics.json"
    data = _read_json(path)
    errors = validate_metrics(data)
    if errors:
        raise ValueError(f"invalid {run_id}/metrics.json: {'; '.join(errors)}")

    names = data["class_names"]
    counts = np.asarray(data["confusion_matrix"], dtype=np.int64)
    row_totals = counts.sum(axis=1, keepdims=True)
    if normalize:
        values = np.divide(
            counts,
            row_totals,
            out=np.zeros_like(counts, dtype=np.float64),
            where=row_totals != 0,
        )
        colorbar_label = "Proportion of true class"
        vmax = 1.0
    else:
        values = counts
        colorbar_label = "Samples"
        vmax = None

    fig, axis = plt.subplots(figsize=(10.5, 8.4))
    image = axis.imshow(values, cmap="Blues", vmin=0, vmax=vmax, aspect="equal")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)

    positions = np.arange(len(names))
    axis.set_xticks(positions, labels=names, rotation=45, ha="right", rotation_mode="anchor")
    axis.set_yticks(positions, labels=names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(f"{_run_label(run_id)} test confusion matrix", fontweight="bold")

    threshold = (float(values.max()) / 2.0) if values.size else 0.0
    for row in range(len(names)):
        for column in range(len(names)):
            axis.text(
                column,
                row,
                f"{counts[row, column]:d}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if values[row, column] > threshold else "#111827",
            )

    # Keep the first row at the top after explicitly setting the image bounds.
    axis.set_xlim(-0.5, len(names) - 0.5)
    axis.set_ylim(len(names) - 0.5, -0.5)
    fig.tight_layout()
    return _save(fig, out)
