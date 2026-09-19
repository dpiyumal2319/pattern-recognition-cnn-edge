"""Model resource profiling: parameters, disk size, MACs, epoch timing.

Supplies the resource half of every comparison table in the report. The MAC
count is derived per layer with forward hooks rather than read from a
black-box library, because #11 has to cross-check these numbers against a hand
derivation and an unexplainable number is useless in a report.

Scope note: #1 provides a working implementation so ``--smoke`` emits a
schema-conformant ``profile.json``. #5 owns the CNNs.pdf verification test
(~590k standard vs ~67.8k separable ops) and any refinement. See the PR
discussion on #1.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import torch
import torch.nn as nn

from .config import Config
from .device import device_name


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def state_dict_bytes(model: nn.Module) -> int:
    """Serialised size of the state dict, measured rather than estimated.

    Saving to an in-memory buffer keeps this side-effect free, and it counts
    the real pickle overhead that an on-disk checkpoint would carry - which is
    what "estimated model size on disk" in the brief actually means.
    """
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


def count_macs(model: nn.Module, input_shape: tuple[int, ...]) -> tuple[int, dict[str, int]]:
    """Multiply-accumulates for ONE forward pass of ONE image.

    Batch size is forced to 1 so the number is per-image regardless of the
    config's batch size. Bias additions are excluded, matching the convention
    in docs/slides/CNNs.pdf: a k x k convolution producing H x W x C_out from
    C_in channels costs ``k * k * C_in * C_out * H * W``.

    Grouped convolutions - which is what a depthwise convolution is - divide by
    the group count, which is exactly where Model B's saving comes from.
    """
    per_layer: dict[str, int] = {}
    handles = []

    def conv_hook(name: str):
        def hook(module: nn.Conv2d, _inputs, output: torch.Tensor) -> None:
            out_h, out_w = output.shape[-2:]
            kernel = module.kernel_size[0] * module.kernel_size[1]
            in_per_group = module.in_channels // module.groups
            per_layer[name] = kernel * in_per_group * module.out_channels * out_h * out_w
        return hook

    def linear_hook(name: str):
        def hook(module: nn.Linear, _inputs, _output) -> None:
            per_layer[name] = module.in_features * module.out_features
        return hook

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(conv_hook(name)))
        elif isinstance(module, nn.Linear):
            handles.append(module.register_forward_hook(linear_hook(name)))

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, *input_shape[1:], device=device))
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return sum(per_layer.values()), per_layer


def profile_model(
    model: nn.Module,
    cfg: Config,
    mean_epoch_seconds: float,
    device: torch.device,
    out_dir: Path | None = None,
) -> dict:
    """Collect every resource metric and optionally write ``profile.json``."""
    total, trainable = count_parameters(model)
    size_bytes = state_dict_bytes(model)
    input_shape = (1, 3, cfg.data.input_size, cfg.data.input_size)
    macs, per_layer = count_macs(model, input_shape)

    profile = {
        "run_id": cfg.run_id,
        "total_params": total,
        "trainable_params": trainable,
        "disk_size_bytes": size_bytes,
        # Both units are emitted: the brief wants KB for the custom models and
        # MB for the SOTA models, and the magnitude gap between them is itself
        # a point the report makes.
        "disk_size_kb": round(size_bytes / 1024, 2),
        "disk_size_mb": round(size_bytes / 1024**2, 4),
        "macs": macs,
        "macs_per_layer": per_layer,
        "input_shape": list(input_shape),
        "mean_epoch_seconds": round(mean_epoch_seconds, 4),
        "device": device.type,
        # Epoch time is a graded column and is meaningless without the
        # hardware it was measured on.
        "device_name": device_name(device),
    }

    if out_dir is not None:
        (out_dir / "profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    return profile
