"""Machine-checkable definitions of the three result-file schemas.

docs/interfaces.md section 3 is the prose version; this is the enforceable one.
``tests/test_schemas.py`` validates every committed results file against these,
and CI runs it on every pull request.

Deliberately dependency-free - no jsonschema, no pydantic. The schemas are
simple enough that a 60-line validator is clearer than a library, and one fewer
pinned dependency is one fewer thing to break on a Colab runtime.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

Number = (int, float)

HISTORY_FIELDS: dict[str, tuple] = {
    "epoch": (int,),
    "train_loss": Number,
    "train_acc": Number,
    "val_loss": Number,
    "val_acc": Number,
    "epoch_seconds": Number,
    "lr": Number,
}

METRICS_FIELDS: dict[str, tuple] = {
    "run_id": (str,),
    "split": (str,),
    "num_samples": (int,),
    "class_names": (list,),
    "accuracy": Number,
    "macro_precision": Number,
    "macro_recall": Number,
    "macro_f1": Number,
    "per_class": (dict,),
    "confusion_matrix": (list,),
    "confusion_matrix_convention": (str,),
}

PER_CLASS_FIELDS: dict[str, tuple] = {
    "precision": Number,
    "recall": Number,
    "f1": Number,
    "support": (int,),
}

PROFILE_FIELDS: dict[str, tuple] = {
    "run_id": (str,),
    "total_params": (int,),
    "trainable_params": (int,),
    "disk_size_bytes": (int,),
    "disk_size_kb": Number,
    "disk_size_mb": Number,
    "macs": (int,),
    "macs_per_layer": (dict,),
    "input_shape": (list,),
    "mean_epoch_seconds": Number,
    "device": (str,),
    "device_name": (str,),
}


class SchemaError(ValueError):
    """A result file does not match its frozen schema."""


def _check_fields(obj: Any, spec: dict[str, tuple], where: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(obj, dict):
        return [f"{where}: expected an object, got {type(obj).__name__}"]
    for key, types in spec.items():
        if key not in obj:
            errors.append(f"{where}: missing required key '{key}'")
        elif not isinstance(obj[key], types) or isinstance(obj[key], bool):
            got = type(obj[key]).__name__
            want = "/".join(t.__name__ for t in types)
            errors.append(f"{where}: '{key}' should be {want}, got {got}")
    for key in obj:
        if key not in spec:
            errors.append(f"{where}: unexpected key '{key}'")
    return errors


def validate_history(data: Any) -> list[str]:
    """Return a list of schema violations; empty means valid."""
    if not isinstance(data, list):
        return ["history: expected a JSON array of epoch records"]
    if not data:
        return ["history: must contain at least one epoch"]
    errors: list[str] = []
    for i, rec in enumerate(data):
        errors += _check_fields(rec, HISTORY_FIELDS, f"history[{i}]")
    epochs = [r.get("epoch") for r in data if isinstance(r, dict)]
    if epochs and epochs != list(range(1, len(epochs) + 1)):
        errors.append("history: 'epoch' must be 1-indexed and contiguous")
    return errors


def validate_metrics(data: Any) -> list[str]:
    errors = _check_fields(data, METRICS_FIELDS, "metrics")
    if errors or not isinstance(data, dict):
        return errors

    names = data["class_names"]
    matrix = data["confusion_matrix"]
    n = len(names)

    if len(matrix) != n or any(not isinstance(r, list) or len(r) != n for r in matrix):
        errors.append(f"metrics: confusion_matrix must be {n}x{n} to match class_names")

    missing = set(names) - set(data["per_class"])
    if missing:
        errors.append(f"metrics: per_class missing entries for {sorted(missing)}")
    extra = set(data["per_class"]) - set(names)
    if extra:
        errors.append(f"metrics: per_class has entries not in class_names: {sorted(extra)}")

    for cls, entry in data["per_class"].items():
        errors += _check_fields(entry, PER_CLASS_FIELDS, f"metrics.per_class['{cls}']")

    # per_class is keyed by class NAME, never by index - a figure labelled from
    # integer keys is how a confusion matrix ends up silently transposed.
    if any(str(k).isdigit() for k in data["per_class"]):
        errors.append("metrics: per_class must be keyed by class name, not index")

    return errors


def validate_profile(data: Any) -> list[str]:
    errors = _check_fields(data, PROFILE_FIELDS, "profile")
    if errors or not isinstance(data, dict):
        return errors
    if data["trainable_params"] > data["total_params"]:
        errors.append("profile: trainable_params exceeds total_params")
    if len(data["input_shape"]) != 4:
        errors.append("profile: input_shape must be [N, C, H, W]")
    return errors


VALIDATORS = {
    "history.json": validate_history,
    "metrics.json": validate_metrics,
    "profile.json": validate_profile,
}


def validate_file(path: Path) -> list[str]:
    """Validate one result file, dispatching on its filename."""
    validator = VALIDATORS.get(path.name)
    if validator is None:
        return [f"{path}: no schema registered for '{path.name}'"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON - {exc}"]
    return [f"{path.parent.name}/{path.name} {e}" for e in validator(data)]


def validate_run_dir(directory: Path) -> list[str]:
    """Validate a complete run directory, including that nothing is missing."""
    errors: list[str] = []
    for name in ("history.json", "metrics.json", "profile.json"):
        path = directory / name
        if not path.exists():
            errors.append(f"{directory.name}: missing {name}")
        else:
            errors += validate_file(path)
    if not (directory / "config.yaml").exists():
        errors.append(f"{directory.name}: missing config.yaml")
    return errors
