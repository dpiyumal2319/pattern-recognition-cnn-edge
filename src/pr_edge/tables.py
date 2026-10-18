"""LaTeX table generation from result artifacts.

This module is the only path by which a number reaches the report:

    results/*/[metrics|profile].json -> build_tables.py -> report/tables/*.tex -> \\input{}

Never hand-type a result into LaTeX. If a run is repeated, the report updates
itself; if a number is typed, it silently goes stale and nobody notices until
the viva.

Each function emits a bare ``tabular`` fragment - no ``\\begin{table}``
wrapper. Caption, label and placement belong to the report, not the generator.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from .paths import RESULTS_DIR
from .schemas import validate_metrics, validate_profile, validate_run_dir


# Keras-style layer names, as in the model.summary() table on CNNs.pdf p54, so
# the report's architecture tables use the vocabulary the slides use.
_LAYER_NAMES = {
    "Conv2d": "Conv2D",
    "MaxPool2d": "MaxPooling2D",
    "AdaptiveAvgPool2d": "GlobalAveragePooling2D",
    "Flatten": "Flatten",
    "Linear": "Dense",
}


def _layer_kind(row: dict) -> str:
    """Report name for one layer_table row.

    The two stages of a depthwise separable block are both nn.Conv2d, so they
    are told apart by shape: a grouped convolution is the depthwise step and a
    1x1 convolution is the pointwise step, the two names CNNs.pdf p44 uses.
    """
    if row["type"] == "Conv2d":
        if row["groups"] and row["groups"] > 1:
            return "DepthwiseConv2D"
        if row["kernel_size"] == [1, 1]:
            return "PointwiseConv2D"
    return _LAYER_NAMES.get(row["type"], row["type"])


def _thousands(n: int) -> str:
    """18496 -> 18{,}496 - a LaTeX thousands separator with correct spacing."""
    return f"{n:,}".replace(",", "{,}")


def build_layer_table(config: Path, out: Path) -> Path:
    """Per-layer output shape and parameter count for a custom model.

    Built from the model itself (constructed from ``config``), never from
    hand-typed numbers, so a change to the channel widths updates the report.
    Emits a bare ``tabular`` like every other generator here: layer, output
    shape (C x H x W), parameters, and a total row. Implemented by #9; #10
    reuses it for Model B.
    """
    # Imported here so building an unrelated table never needs torch.
    from .config import load_config
    from .models import build_model
    from .profile import count_parameters, layer_table

    cfg = load_config(config)
    model = build_model(cfg)
    input_shape = (1, 3, cfg.data.input_size, cfg.data.input_size)
    rows = layer_table(model, input_shape)
    total, _ = count_parameters(model)

    kinds = [_layer_kind(row) for row in rows]
    counters: dict[str, int] = {}
    lines = [
        r"\begin{tabular}{llr}",
        r"\hline",
        r"Layer & Output shape ($C \times H \times W$) & Parameters \\",
        r"\hline",
    ]
    for row, kind in zip(rows, kinds):
        # Number only the layer kinds that repeat: "Conv2D 2", but plain "Flatten".
        counters[kind] = counters.get(kind, 0) + 1
        label = f"{kind} {counters[kind]}" if kinds.count(kind) > 1 else kind
        shape = " \\times ".join(str(d) for d in row["output_shape"])
        lines.append(f"{label} & ${shape}$ & {_thousands(row['params'])} \\\\")
    lines += [
        r"\hline",
        f"Total & & {_thousands(total)} \\\\",
        r"\hline",
        r"\end{tabular}",
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    # Explicit "\n" so the generated file is byte-identical on every platform.
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


def build_dataset_table(manifest: Path, out: Path) -> Path:
    """Per-class image counts in each split, read from the committed manifest.

    Rows follow CLASS_NAMES order, then a total row and a share row that shows
    the realised 70/15/15 proportions. Built from data/splits.csv itself - the
    file every experiment reads - so the table can only describe the partition
    that was actually used. Implemented by #24.
    """
    from .data import CLASS_NAMES, SPLIT_FRACTIONS, read_manifest, validate_manifest

    rows = read_manifest(manifest)
    # Refuse to tabulate a manifest the experiments themselves would reject.
    validate_manifest(rows)

    splits = list(SPLIT_FRACTIONS)
    counts = {name: dict.fromkeys(splits, 0) for name in CLASS_NAMES}
    for row in rows:
        counts[row["label"]][row["split"]] += 1
    totals = {split: sum(counts[name][split] for name in CLASS_NAMES) for split in splits}
    grand = sum(totals.values())

    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Class & Train & Validation & Test & Total \\",
        r"\midrule",
    ]
    for name in CLASS_NAMES:
        cells = [_thousands(counts[name][split]) for split in splits]
        lines.append(f"{name} & {' & '.join(cells)} & {_thousands(sum(counts[name].values()))} \\\\")
    lines.append(r"\midrule")
    cells = [_thousands(totals[split]) for split in splits]
    lines.append(f"Total & {' & '.join(cells)} & {_thousands(grand)} \\\\")
    shares = [f"{100 * totals[split] / grand:.1f}\\%" for split in splits]
    lines.append(f"Share & {' & '.join(shares)} & 100.0\\% \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]

    out.parent.mkdir(parents=True, exist_ok=True)
    # Explicit "\n" so the generated file is byte-identical on every platform.
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


def build_custom_comparison(run_ids: list[str], out: Path) -> Path:
    """Model A vs Model B.

    The brief names these columns: total parameter count, estimated model size
    on disk (KB), training time per epoch, and test accuracy. Macro precision
    and recall are added because task 4 requires them too. Implemented by #18.
    """
    if not run_ids:
        raise ValueError("build_custom_comparison requires at least one run_id")

    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Model & Parameters & Size (KB) & Epoch (s) & Accuracy (\%) & Precision (\%) & Recall (\%) \\",
        r"\hline",
    ]
    for run_id in run_ids:
        directory = RESULTS_DIR / run_id
        paths = {
            "metrics": directory / "metrics.json",
            "profile": directory / "profile.json",
        }
        missing = [path.name for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"run {run_id!r} is missing custom table artifact(s): {missing}")

        metrics = json.loads(paths["metrics"].read_text(encoding="utf-8"))
        profile = json.loads(paths["profile"].read_text(encoding="utf-8"))
        errors = validate_metrics(metrics) + validate_profile(profile)
        if errors:
            raise ValueError(f"run {run_id!r} has invalid result artifacts: {'; '.join(errors)}")
        if metrics["run_id"] != run_id or profile["run_id"] != run_id:
            raise ValueError(f"run {run_id!r} contains an artifact with a different run_id")

        label = run_id.replace("_", " ").title()
        size_kb = f"{profile['disk_size_kb']:,.2f}".replace(",", "{,}")
        lines.append(
            f"{label} & {_thousands(profile['total_params'])} & {size_kb} & "
            f"{profile['mean_epoch_seconds']:.2f} & {100 * metrics['accuracy']:.2f} & "
            f"{100 * metrics['macro_precision']:.2f} & {100 * metrics['macro_recall']:.2f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


def build_custom_convergence(run_ids: list[str], out: Path) -> Path:
    """Controlled 30-vs-100 epoch comparison for Models A and B.

    Selection fields come from the lowest-validation-loss checkpoint, while
    test metrics and resource measurements come from the matching run
    artifacts.  All rows must share a device because epoch time is otherwise
    not comparable.
    """
    if not run_ids:
        raise ValueError("build_custom_convergence requires at least one run_id")

    rows: list[tuple[str, dict, dict, dict, dict]] = []
    devices: set[str] = set()
    for run_id in run_ids:
        directory = RESULTS_DIR / run_id
        errors = validate_run_dir(directory)
        if errors:
            raise ValueError(f"run {run_id!r} has invalid result artifacts: {'; '.join(errors)}")

        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        if any(artifact["run_id"] != run_id for artifact in (config, metrics, profile)):
            raise ValueError(f"run {run_id!r} contains an artifact with a different run_id")
        if config["training"]["epochs"] != selection["epochs_trained"]:
            raise ValueError(f"run {run_id!r} config and selection disagree on the epoch budget")
        devices.add(profile["device_name"])
        rows.append((run_id, config, metrics, profile, selection))

    if len(devices) != 1:
        raise ValueError(f"custom convergence rows must share one device, found: {sorted(devices)}")

    lines = [
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\hline",
        (
            r"Model & Epochs & Selected & Best val. loss & Best val. acc. (\%) & "
            r"Test acc. (\%) & Parameters & Size (KB) & MACs & Epoch (s) \\"
        ),
        r"\hline",
    ]
    for _, config, metrics, profile, selection in rows:
        model = {"model_a": "Model A", "model_b": "Model B"}.get(
            config["model"]["name"], config["model"]["name"].replace("_", " ").title()
        )
        size_kb = f"{profile['disk_size_kb']:,.2f}".replace(",", "{,}")
        lines.append(
            f"{model} & {selection['epochs_trained']} & {selection['best_epoch']} & "
            f"{selection['best_val_loss']:.4f} & {100 * selection['best_val_acc']:.2f} & "
            f"{100 * metrics['accuracy']:.2f} & {_thousands(profile['total_params'])} & "
            f"{size_kb} & {_thousands(profile['macs'])} & {profile['mean_epoch_seconds']:.2f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


def build_optimizer_comparison(run_ids: list[str], out: Path) -> Path:
    """Adam vs SGD vs SGD with momentum for Task 3.

    Reads the resolved optimizer configuration and validation/test artifacts
    from each run.  Keeping this generated prevents a repeated experiment from
    leaving stale, hand-typed values in the report.
    """
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\hline",
        r"Optimizer & LR & Momentum & Best val. loss & Best val. acc. (\%) & Test acc. (\%) & Epoch (s) \\",
        r"\hline",
    ]
    for run_id in run_ids:
        directory = RESULTS_DIR / run_id
        required = ["config.yaml", "metrics.json", "profile.json", "selection.json"]
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"run {run_id!r} is missing optimizer table artifact(s): {missing}")

        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
        optimizer = config["optimizer"]
        name = "Adam" if optimizer["name"] == "adam" else "SGD"
        if optimizer["name"] == "sgd" and optimizer["momentum"] > 0:
            name = "SGD + momentum"
        lines.append(
            f"{name} & {optimizer['lr']:.3g} & {optimizer['momentum']:.1f} & "
            f"{selection['best_val_loss']:.3f} & {100 * selection['best_val_acc']:.2f} & "
            f"{100 * metrics['accuracy']:.2f} & {profile['mean_epoch_seconds']:.2f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}"]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


# Report names for the torchvision backbones, keyed by config model.name.
_SOTA_NAMES = {
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
}

# The resolution every primary result uses. Any other input size is the
# supplementary resolution ablation and is grouped below the primary rows.
_PRIMARY_INPUT_SIZE = 64


def build_sota_comparison(run_ids: list[str], out: Path) -> Path:
    """Fine-tuned lightweight SOTA models.

    Sizes in MB here, not KB - the brief asks for MB for these, and the
    magnitude gap against the custom models is itself the point. Implemented
    by #22.

    Columns: model, backbone policy (frozen or fine-tuned, CNNs.pdf p58),
    input resolution, test accuracy, macro precision, macro recall, total and
    trained parameter counts, and size in MB. Rows at the primary 64x64 input
    come first in ``run_ids`` order. Rows at any other input size follow under
    a labelled "Supplementary" rule, so the 224x224 ablation cannot be
    mistaken for a primary result.

    Model name, backbone policy and input size are read from each run's
    resolved ``config.yaml`` - what the run actually did - never parsed out
    of the run ID.
    """
    if not run_ids:
        raise ValueError("build_sota_comparison requires at least one run_id")

    primary, supplementary = [], []
    for run_id in run_ids:
        directory = RESULTS_DIR / run_id
        required = ["config.yaml", "metrics.json", "profile.json"]
        missing = [name for name in required if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(f"run {run_id!r} is missing SOTA table artifact(s): {missing}")

        config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        profile = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
        errors = validate_metrics(metrics) + validate_profile(profile)
        if errors:
            raise ValueError(f"run {run_id!r} has invalid result artifacts: {'; '.join(errors)}")
        if config["run_id"] != run_id or metrics["run_id"] != run_id or profile["run_id"] != run_id:
            raise ValueError(f"run {run_id!r} contains an artifact with a different run_id")
        # Every row must describe the held-out test split, never validation.
        if metrics["split"] != "test":
            raise ValueError(f"run {run_id!r} metrics are for split {metrics['split']!r}, not 'test'")

        model_name = config["model"]["name"]
        if model_name not in _SOTA_NAMES:
            raise ValueError(f"run {run_id!r} is not a SOTA model: {model_name!r}")
        backbone = "Frozen" if config["model"]["freeze_backbone"] else "Fine-tuned"
        size = config["data"]["input_size"]

        row = (
            f"{_SOTA_NAMES[model_name]} & {backbone} & ${size}\\times{size}$ & "
            f"{100 * metrics['accuracy']:.2f} & {100 * metrics['macro_precision']:.2f} & "
            f"{100 * metrics['macro_recall']:.2f} & {_thousands(profile['total_params'])} & "
            f"{_thousands(profile['trainable_params'])} & {profile['disk_size_mb']:.2f} \\\\"
        )
        (primary if size == _PRIMARY_INPUT_SIZE else supplementary).append(row)

    # Nine columns do not fit the text width on one header line, so the
    # numeric headers stack name over unit. \shortstack needs no package, and
    # [r] matches the right-aligned numbers beneath it.
    lines = [
        r"\begin{tabular}{lllrrrrrr}",
        r"\toprule",
        r"Model & Backbone & Input & \shortstack[r]{Accuracy\\(\%)} & "
        r"\shortstack[r]{Precision\\(\%)} & \shortstack[r]{Recall\\(\%)} & "
        r"Parameters & Trained & \shortstack[r]{Size\\(MB)} \\",
        r"\midrule",
        *primary,
    ]
    if supplementary:
        lines += [
            r"\midrule",
            r"\multicolumn{9}{l}{\textit{Supplementary: input resolution ablation}} \\",
            *supplementary,
        ]
    lines += [r"\bottomrule", r"\end{tabular}"]

    out.parent.mkdir(parents=True, exist_ok=True)
    # Explicit "\n" so the generated file is byte-identical on every platform.
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return out


def build_final_comparison(run_ids: list[str], out: Path) -> Path:
    """Three-way table for task 6: Model B against both SOTA models.

    Accuracy, parameters, size, MACs and epoch time. Implemented by #25.
    """
    raise NotImplementedError("build_final_comparison is issue #25")
