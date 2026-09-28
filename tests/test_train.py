"""Best-checkpoint selection in the training loop. Owned by #6.

The validation losses are scripted rather than learned: ``_run_epoch`` is
wrapped so every training pass really updates the weights, while each
validation pass returns a chosen loss. That makes "the best epoch is not the
last one" a certainty instead of something hoped for from random data, and
lets each test snapshot the exact weights every epoch ended with.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pr_edge import train as train_module
from pr_edge.config import Config, OptimizerConfig, TrainingConfig
from pr_edge.schemas import validate_file, validate_history

NAN = float("nan")


def _loader() -> DataLoader:
    generator = torch.Generator().manual_seed(0)
    images = torch.randn(16, 4, generator=generator)
    labels = torch.randint(0, 3, (16,), generator=generator)
    return DataLoader(TensorDataset(images, labels), batch_size=8)


def _cfg(epochs: int) -> Config:
    # A large learning rate guarantees the weights differ after every epoch.
    return Config(
        run_id="t",
        training=TrainingConfig(epochs=epochs),
        optimizer=OptimizerConfig(name="sgd", lr=0.5, momentum=0.0),
    )


def _scripted_run(tmp_path: Path, monkeypatch, val_losses: list[float], crash_on_epoch=None):
    """Train a tiny linear model with the given validation losses.

    Returns ``(model, history, snapshots)``, where ``snapshots[e]`` is a copy of
    the weights at the end of epoch ``e`` - exactly what ``model.pt`` should
    hold if epoch ``e`` is selected.
    """
    real_run_epoch = train_module._run_epoch
    scripted = iter(val_losses)
    snapshots: dict[int, dict[str, torch.Tensor]] = {}
    epoch = {"n": 0}

    def fake_run_epoch(model, loader, criterion, device, optimizer=None):
        if optimizer is not None:
            epoch["n"] += 1
            if epoch["n"] == crash_on_epoch:
                raise KeyboardInterrupt("simulated Colab disconnect")
            return real_run_epoch(model, loader, criterion, device, optimizer)
        snapshots[epoch["n"]] = {k: v.clone() for k, v in model.state_dict().items()}
        return next(scripted), 0.5

    monkeypatch.setattr(train_module, "_run_epoch", fake_run_epoch)
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    history = train_module.train(
        model, _cfg(len(val_losses)), _loader(), _loader(), torch.device("cpu"), tmp_path,
    )
    return model, history, snapshots


def _equal(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _saved(tmp_path: Path) -> dict[str, torch.Tensor]:
    return torch.load(tmp_path / "model.pt", weights_only=True)


def _selection(tmp_path: Path) -> dict:
    return json.loads((tmp_path / "selection.json").read_text(encoding="utf-8"))


def test_checkpoint_is_the_lowest_val_loss_epoch_not_the_last(tmp_path, monkeypatch):
    """The core of #6. On main before #6, model.pt held epoch 3's weights."""
    _, _, snapshots = _scripted_run(tmp_path, monkeypatch, [2.0, 1.0, 1.5])
    assert not _equal(snapshots[2], snapshots[3]), "fixture must make the epochs differ"
    assert _equal(_saved(tmp_path), snapshots[2])
    assert not _equal(_saved(tmp_path), snapshots[3])


def test_model_is_restored_to_the_selected_weights(tmp_path, monkeypatch):
    """run_experiment evaluates the in-memory model, so it must be the chosen one.

    Without the restore, metrics.json would describe the final epoch while
    model.pt held another - every table in the report would be quietly wrong.
    """
    model, _, snapshots = _scripted_run(tmp_path, monkeypatch, [2.0, 1.0, 1.5])
    assert _equal(model.state_dict(), snapshots[2])
    assert _equal(model.state_dict(), _saved(tmp_path))


def test_selection_file_records_the_chosen_epoch(tmp_path, monkeypatch):
    _scripted_run(tmp_path, monkeypatch, [2.0, 1.0, 1.5, 1.2])
    assert _selection(tmp_path) == {
        "criterion": "val_loss",
        "best_epoch": 2,
        "best_val_loss": 1.0,
        "best_val_acc": 0.5,
        "epochs_trained": 4,
    }
    assert validate_file(tmp_path / "selection.json") == []


def test_ties_go_to_the_earlier_epoch(tmp_path, monkeypatch):
    _, _, snapshots = _scripted_run(tmp_path, monkeypatch, [1.0, 1.0])
    assert _selection(tmp_path)["best_epoch"] == 1
    assert _equal(_saved(tmp_path), snapshots[1])


def test_diverged_epochs_are_never_selected(tmp_path, monkeypatch):
    _, history, snapshots = _scripted_run(tmp_path, monkeypatch, [NAN, 1.2, float("inf"), 1.1])
    assert _selection(tmp_path)["best_epoch"] == 4
    assert _equal(_saved(tmp_path), snapshots[4])
    # Training kept going past the NaN: divergence is recorded, not hidden.
    assert len(history) == 4 and math.isnan(history[0]["val_loss"])


def test_a_run_with_no_finite_epoch_raises_and_saves_nothing(tmp_path, monkeypatch):
    (tmp_path / "model.pt").write_bytes(b"stale checkpoint from an earlier run")
    (tmp_path / "selection.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="finite validation loss"):
        _scripted_run(tmp_path, monkeypatch, [NAN, NAN])
    # Neither the stale checkpoint nor a new one may pose as this run's result.
    assert not (tmp_path / "model.pt").exists()
    assert not (tmp_path / "selection.json").exists()
    assert len(json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))) == 2


def test_an_interrupted_run_keeps_its_best_checkpoint_and_history(tmp_path, monkeypatch):
    """A Colab disconnect mid-run must not lose the epochs already trained."""
    with pytest.raises(KeyboardInterrupt):
        _scripted_run(tmp_path, monkeypatch, [2.0, 1.0, 3.0, 3.0], crash_on_epoch=3)
    history = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert [r["epoch"] for r in history] == [1, 2]
    assert validate_history(history) == []
    assert (tmp_path / "model.pt").is_file()
    # No temp files left behind by the atomic writes.
    assert not list(tmp_path.glob(".*.tmp"))


def test_history_is_schema_conformant(tmp_path, monkeypatch):
    _, history, _ = _scripted_run(tmp_path, monkeypatch, [2.0, 1.0, 1.5])
    on_disk = json.loads((tmp_path / "history.json").read_text(encoding="utf-8"))
    assert on_disk == history
    assert validate_history(on_disk) == []
