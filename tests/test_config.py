"""Config loading. Owned by #1."""

from __future__ import annotations

import pytest
import yaml

from pr_edge.config import Config, ConfigError, dump_config, load_config
from pr_edge.paths import CONFIGS_DIR

ALL_CONFIGS = sorted(CONFIGS_DIR.glob("*.yaml"))


def test_configs_exist_for_every_run_id():
    """Every fixed run ID (AGENTS.md section 3) plus base must have a config file."""
    expected = {
        "base", "model_a", "model_b", "model_b_sgd", "model_b_sgdm",
        "mobilenetv2_64", "efficientnetb0_64", "mobilenetv2_224",
        "mobilenetv2_64_ft",  # #54: unfrozen partner of mobilenetv2_64
    }
    assert {p.stem for p in ALL_CONFIGS} == expected


@pytest.mark.parametrize("path", ALL_CONFIGS, ids=lambda p: p.stem)
def test_every_shipped_config_loads(path):
    cfg = load_config(path)
    assert isinstance(cfg, Config)
    assert cfg.run_id == path.stem


def test_unknown_key_raises(tmp_path):
    """A typo must fail loudly, not be silently ignored.

    'momemtum: 0.9' would otherwise mean an optimizer ablation that tested
    nothing, and we would not find out until the report numbers looked odd.
    """
    path = tmp_path / "bad.yaml"
    path.write_text("run_id: bad\noptimizer:\n  momemtum: 0.9\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)


def test_unknown_top_level_key_raises(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("run_id: bad\nepochs: 5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown top-level key"):
        load_config(path)


def test_run_id_must_match_filename(tmp_path):
    """The run directory is derived from the filename, so a mismatch is a trap."""
    path = tmp_path / "alpha.yaml"
    path.write_text("run_id: beta\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must match the config filename"):
        load_config(path)


def test_overrides_route_to_the_owning_section(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("run_id: x\n", encoding="utf-8")
    cfg = load_config(path, {"epochs": 2, "batch_size": 16, "device": "cpu", "smoke": True})
    assert cfg.training.epochs == 2
    assert cfg.data.batch_size == 16
    assert cfg.device == "cpu"
    assert cfg.training.smoke is True


def test_none_overrides_are_ignored(tmp_path):
    """Unset argparse flags arrive as None and must not clobber the file."""
    path = tmp_path / "x.yaml"
    path.write_text("run_id: x\ntraining:\n  epochs: 30\n", encoding="utf-8")
    assert load_config(path, {"epochs": None}).training.epochs == 30


def test_unknown_override_raises(tmp_path):
    path = tmp_path / "x.yaml"
    path.write_text("run_id: x\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="matches no config key"):
        load_config(path, {"nonsense": 1})


def test_dump_round_trips(tmp_path):
    cfg = load_config(CONFIGS_DIR / "model_b.yaml")
    out = tmp_path / "model_b.yaml"
    dump_config(cfg, out)

    dumped = yaml.safe_load(out.read_text(encoding="utf-8"))
    assert dumped["run_id"] == "model_b"
    assert dumped["optimizer"]["name"] == "adam"
    assert load_config(out) == cfg
