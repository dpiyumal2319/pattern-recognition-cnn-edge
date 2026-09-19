"""EuroSAT data pipeline.

EuroSAT is 27,000 Sentinel-2 RGB tiles across 10 land-use classes, natively
64x64 - so the brief's "downscale to a maximum of 64x64" is satisfied by
construction rather than by a resize step. The report must say this explicitly.

Only the smoke-mode path is implemented here. The real download, split and
loader work belongs to #2 and #3, whose signatures are frozen below.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config

# The single source of class ordering for the whole project. Never re-derive
# this from a directory listing: filesystem order is not guaranteed across
# platforms, and a different order on one member's machine would silently
# transpose every confusion matrix they produce.
CLASS_NAMES: list[str] = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Smoke-mode dataset sizes. Small enough to finish in seconds on a CPU runner,
# large enough that every class appears in every split.
SMOKE_SIZES = {"train": 200, "val": 50, "test": 50}


def prepare_eurosat(cfg: Config, force: bool = False) -> Path:
    """Download EuroSAT and write the stratified 70/15/15 split manifest.

    Returns the manifest path. Implemented by #2.
    """
    raise NotImplementedError("prepare_eurosat is issue #2")


def compute_norm_stats(cfg: Config) -> dict[str, list[float]]:
    """Per-channel mean and std, computed on the TRAINING SPLIT ONLY.

    Computing these over the full dataset leaks test-set statistics into
    preprocessing. Implemented by #3.
    """
    raise NotImplementedError("compute_norm_stats is issue #3")


def get_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train, validation and test loaders built from the committed manifest.

    Augmentation applies to the training loader only. Implemented by #3.
    """
    raise NotImplementedError("get_dataloaders is issue #3")


def get_synthetic_dataloaders(cfg: Config) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Random-tensor loaders for smoke mode - no download, no disk dataset.

    This exists so CI can prove on every pull request that the wiring from
    config to run directory is intact, in under a minute on a CPU runner. The
    labels are random, so nothing about accuracy here is meaningful; the point
    is purely that the pipeline executes and emits conformant artifacts.
    """
    size = cfg.data.input_size
    generator = torch.Generator().manual_seed(cfg.seed)
    loaders = []

    for split in ("train", "val", "test"):
        n = SMOKE_SIZES[split]
        images = torch.rand(n, 3, size, size, generator=generator)
        # Cycle through the classes so every split covers all 10 - a confusion
        # matrix with an absent class would fail schema validation for reasons
        # that have nothing to do with the code under test.
        labels = torch.arange(n, dtype=torch.long) % len(CLASS_NAMES)
        loaders.append(
            DataLoader(
                TensorDataset(images, labels),
                batch_size=cfg.data.batch_size,
                shuffle=(split == "train"),
                num_workers=0,  # spawning workers for 200 tensors costs more than it saves
            )
        )

    return tuple(loaders)  # type: ignore[return-value]
