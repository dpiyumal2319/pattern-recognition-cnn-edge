"""Device selection and seeding.

Group members run on Colab (CUDA), a local NVIDIA laptop, Apple Silicon and
plain CPU. Nothing in this project may call ``.cuda()`` directly - always route
through :func:`get_device` so the same config runs everywhere unchanged.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def get_device(pref: str = "auto") -> torch.device:
    """Resolve a device preference to a concrete ``torch.device``.

    ``auto`` prefers CUDA, then Apple Silicon MPS, then CPU. An explicit
    preference that is unavailable falls back to CPU with a warning rather than
    crashing, because a member without a GPU should still be able to run
    everything, just slower.
    """
    pref = (pref or "auto").lower()

    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if pref == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    if pref == "mps" and not torch.backends.mps.is_available():
        print("[device] MPS requested but unavailable; falling back to CPU")
        return torch.device("cpu")

    return torch.device(pref)


def device_name(device: torch.device) -> str:
    """Human-readable hardware name, recorded in ``profile.json``.

    Training time per epoch is a graded column in the report and is meaningless
    without knowing what it ran on, so this string travels with every result.
    """
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple Silicon (MPS)"
    return "CPU"


def set_seed(seed: int) -> None:
    """Seed every RNG this project touches.

    Covers ``random``, ``numpy``, ``torch`` (CPU and CUDA) and the Python hash
    seed. cuDNN is put in deterministic mode: this costs some throughput but
    means a rerun of the same config reproduces the same curves, which matters
    when four people are comparing numbers that must line up in one table.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
