"""Task 3 momentum ablation fairness, reuse, and figure generation."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from pr_edge.config import DataConfig, dump_config, load_config
from scripts import run_momentum_ablation as ablation


def test_ablation_has_four_course_relevant_momentum_values():
    configs = ablation._build_configs()
    ablation._validate_ablation_configs(configs)
    assert list(ablation.MOMENTUM_RUNS) == [0.0, 0.5, 0.9, 0.99]


def test_ablation_rejects_non_momentum_drift():
    configs = ablation._build_configs()
    configs[1] = replace(configs[1], data=DataConfig(batch_size=64))
    with pytest.raises(ValueError, match="outside run_id and optimizer.momentum"):
        ablation._validate_ablation_configs(configs)


def test_ablation_rejects_wrong_optimizer_value():
    configs = ablation._build_configs()
    configs[3] = replace(
        configs[3], optimizer=replace(configs[3].optimizer, momentum=0.8)
    )
    with pytest.raises(ValueError, match="unexpected momentum ablation optimizer blocks"):
        ablation._validate_ablation_configs(configs)


def test_momentum_figure_contains_all_histories(tmp_path):
    histories = {}
    for index, momentum in enumerate(ablation.MOMENTUM_RUNS):
        path = tmp_path / f"m{index}.json"
        path.write_text(
            json.dumps([
                {"epoch": 1, "val_loss": 1.0 + index},
                {"epoch": 2, "val_loss": 0.8 + index},
            ]),
            encoding="utf-8",
        )
        histories[momentum] = path

    out = ablation._plot_validation_losses(histories, tmp_path / "momentum.png")
    assert out.is_file()
    assert out.stat().st_size > 10_000


def test_momentum_figure_rejects_empty_history(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="empty history"):
        ablation._plot_validation_losses({0.0: path}, tmp_path / "momentum.png")


def test_reuse_rejects_a_mismatched_resolved_config(tmp_path, monkeypatch):
    cfg = ablation._build_configs()[0]
    reuse_id = ablation.REUSE_RUNS[0.0]
    directory = tmp_path / reuse_id
    directory.mkdir()
    dump_config(replace(cfg, run_id=reuse_id, seed=7), directory / "config.yaml")
    monkeypatch.setattr(ablation, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(ablation, "validate_run_dir", lambda _directory: [])
    with pytest.raises(ValueError, match="resolved config does not match"):
        ablation._reusable_history(0.0, cfg, ablation.parse_args([]))


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

    monkeypatch.setattr(ablation, "run_dir", fake_run_dir)
    monkeypatch.setattr(ablation.run_experiment, "main", fake_runner)

    assert ablation.main(["--smoke", "--device", "cpu"]) == 0
    assert len(calls) == 4
    assert all(call[-3:] == ["--smoke", "--device", "cpu"] for call in calls)
    assert (tmp_path / "_smoke_momentum_ablation" / "momentum_ablation.png").is_file()
