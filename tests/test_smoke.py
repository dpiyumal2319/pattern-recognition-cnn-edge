"""End-to-end smoke run. Owned by #1.

This is the test CI leans on. It proves, on every pull request, that the
wiring from a config file through training, evaluation and profiling to a
schema-conformant run directory is intact - in under a minute on a CPU runner,
with no dataset download.

It says nothing about accuracy. The labels are random by construction.
"""

from __future__ import annotations

import json
import shutil
import time

import pytest
import yaml

from pr_edge.paths import CONFIGS_DIR, run_dir
from pr_edge.schemas import validate_run_dir

SMOKE_BUDGET_SECONDS = 60


@pytest.fixture(scope="module")
def smoke_run():
    """Run the smoke experiment once and share the output across tests."""
    import scripts.run_experiment as runner

    out = run_dir("model_b", smoke=True)
    shutil.rmtree(out, ignore_errors=True)

    start = time.perf_counter()
    exit_code = runner.main(["--config", str(CONFIGS_DIR / "model_b.yaml"), "--smoke"])
    elapsed = time.perf_counter() - start

    assert exit_code == 0
    return run_dir("model_b", smoke=True), elapsed


def test_run_completes_within_the_ci_budget(smoke_run):
    _, elapsed = smoke_run
    assert elapsed < SMOKE_BUDGET_SECONDS, (
        f"smoke run took {elapsed:.1f}s, over the {SMOKE_BUDGET_SECONDS}s CI budget"
    )


def test_all_contract_files_are_written(smoke_run):
    out, _ = smoke_run
    for name in ("config.yaml", "history.json", "metrics.json", "profile.json", "model.pt"):
        assert (out / name).exists(), f"missing {name}"


def test_artifacts_validate_against_the_schemas(smoke_run):
    out, _ = smoke_run
    errors = validate_run_dir(out)
    assert errors == [], "\n".join(errors)


def test_smoke_overrides_are_recorded_in_the_resolved_config(smoke_run):
    """The dumped config must reflect the overrides, not the file on disk."""
    out, _ = smoke_run
    cfg = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert cfg["training"]["smoke"] is True
    assert cfg["training"]["epochs"] == 2
    assert cfg["data"]["batch_size"] == 16
    assert cfg["device"] == "cpu"


def test_history_has_one_record_per_epoch(smoke_run):
    out, _ = smoke_run
    history = json.loads((out / "history.json").read_text(encoding="utf-8"))
    assert len(history) == 2
    assert [r["epoch"] for r in history] == [1, 2]
    assert all(r["epoch_seconds"] > 0 for r in history)


def test_metrics_cover_every_class(smoke_run):
    out, _ = smoke_run
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert len(metrics["class_names"]) == 10
    assert set(metrics["per_class"]) == set(metrics["class_names"])
    assert metrics["num_samples"] == 50
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 50


def test_profile_reports_real_numbers(smoke_run):
    out, _ = smoke_run
    profile = json.loads((out / "profile.json").read_text(encoding="utf-8"))
    assert profile["total_params"] > 0
    assert profile["disk_size_bytes"] > 0
    assert profile["macs"] > 0
    assert profile["macs_per_layer"]
    assert profile["input_shape"] == [1, 3, 64, 64]
