"""Task 3 optimizer sweep fairness and figure generation."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pr_edge.config import DataConfig, load_config
from pr_edge.paths import CONFIGS_DIR, REPORT_DIR
from pr_edge.tables import build_optimizer_comparison
from scripts import run_optimizer_sweep as sweep
from scripts.run_optimizer_sweep import (
    LABELS,
    SWEEP_CONFIGS,
    _plot_validation_losses,
    _validate_sweep_configs,
)


def _configs():
    return [load_config(path) for path in SWEEP_CONFIGS]


def test_sweep_configs_differ_only_in_run_id_and_optimizer():
    configs = _configs()
    _validate_sweep_configs(configs)
    assert [cfg.run_id for cfg in configs] == list(LABELS)


def test_sweep_rejects_a_non_optimizer_difference():
    configs = _configs()
    configs[1] = replace(configs[1], data=DataConfig(batch_size=64))
    with pytest.raises(ValueError, match="outside run_id and optimizer"):
        _validate_sweep_configs(configs)


def test_sweep_rejects_an_unplanned_optimizer_value():
    configs = _configs()
    configs[2] = replace(
        configs[2], optimizer=replace(configs[2].optimizer, momentum=0.8)
    )
    with pytest.raises(ValueError, match="unexpected optimizer sweep blocks"):
        _validate_sweep_configs(configs)


def test_optimizer_figure_contains_all_histories(tmp_path):
    histories = {}
    for index, run_id in enumerate(LABELS):
        path = tmp_path / f"{run_id}.json"
        path.write_text(
            json.dumps([
                {"epoch": 1, "val_loss": 1.0 + index},
                {"epoch": 2, "val_loss": 0.8 + index},
            ]),
            encoding="utf-8",
        )
        histories[run_id] = path

    out = _plot_validation_losses(histories, tmp_path / "optimizer.png")
    assert out.is_file()
    assert out.stat().st_size > 10_000


def test_optimizer_figure_rejects_empty_history(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="empty history"):
        _plot_validation_losses({"model_b": path}, tmp_path / "optimizer.png")


def test_optimizer_table_uses_committed_run_artifacts(tmp_path):
    out = build_optimizer_comparison(
        ["model_b", "model_b_sgd", "model_b_sgdm"], tmp_path / "optimizer.tex"
    )
    text = out.read_text(encoding="utf-8")
    assert text.startswith(r"\begin{tabular}") and text.rstrip().endswith(r"\end{tabular}")
    assert "Adam & 0.001 & 0.0" in text
    assert "SGD & 0.01 & 0.0" in text
    assert "SGD + momentum & 0.01 & 0.9" in text
    assert "Best val. loss" in text and "Test acc." in text and "Epoch (s)" in text
    assert out.read_bytes() == (REPORT_DIR / "tables" / "optimizer_comparison.tex").read_bytes()


def test_optimizer_table_rejects_a_missing_run(tmp_path, monkeypatch):
    monkeypatch.setattr("pr_edge.tables.RESULTS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="missing optimizer table artifact"):
        build_optimizer_comparison(["missing"], tmp_path / "optimizer.tex")


def test_main_runs_every_config_and_writes_smoke_figure(tmp_path, monkeypatch):
    calls = []

    def fake_run_dir(run_id, smoke=False):
        path = tmp_path / (f"_smoke_{run_id}" if smoke else run_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def fake_runner(argv):
        calls.append(argv)
        config = argv[argv.index("--config") + 1]
        run_id = load_config(config).run_id
        history = fake_run_dir(run_id, smoke="--smoke" in argv) / "history.json"
        history.write_text(
            json.dumps([{"epoch": 1, "val_loss": 1.0}]), encoding="utf-8"
        )
        return 0

    monkeypatch.setattr(sweep, "run_dir", fake_run_dir)
    monkeypatch.setattr(sweep.run_experiment, "main", fake_runner)

    assert sweep.main(["--smoke", "--device", "cpu"]) == 0
    assert [load_config(call[1]).run_id for call in calls] == list(LABELS)
    assert all(call[-3:] == ["--smoke", "--device", "cpu"] for call in calls)
    assert (tmp_path / "_smoke_optimizer_sweep" / "optimizer_comparison.png").is_file()
