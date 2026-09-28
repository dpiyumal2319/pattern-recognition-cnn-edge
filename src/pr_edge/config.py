"""Configuration loading.

One YAML per experiment in ``configs/``. The filename stem IS the ``run_id``,
so ``configs/model_b_sgdm.yaml`` writes to ``results/model_b_sgdm/``.

Unknown keys raise. A silently ignored typo - ``momemtum: 0.9`` - would mean an
optimizer ablation that quietly tested nothing, and we would not find out until
the numbers looked odd in the report. Failing loudly at load is much cheaper.

See docs/interfaces.md section 1 for the frozen schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised for an unknown key, a missing required key, or a bad value."""


@dataclass(frozen=True)
class DataConfig:
    root: str = "data/"
    manifest: str = "data/splits.csv"
    input_size: int = 64
    batch_size: int = 128
    num_workers: int = 2
    augment: bool = True
    norm_stats: str = "data/norm_stats.json"


@dataclass(frozen=True)
class ModelConfig:
    name: str = "model_b"
    num_classes: int = 10
    channels: list[int] = field(default_factory=lambda: [32, 64, 128])
    # Hidden dense widths between the feature extractor and the output layer.
    # Model A only (#9): [128, 64] mirrors the example CNN in CNNs.pdf p54.
    fc_units: list[int] = field(default_factory=lambda: [128, 64])
    head: str = "gap"
    dropout: float = 0.2
    pretrained: bool = False
    freeze_backbone: bool = False


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adam"
    lr: float = 1e-3
    momentum: float = 0.0
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 30
    smoke: bool = False


@dataclass(frozen=True)
class Config:
    run_id: str
    seed: int = 42
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


_SECTIONS = {
    "data": DataConfig,
    "model": ModelConfig,
    "optimizer": OptimizerConfig,
    "training": TrainingConfig,
}


def _build_section(name: str, cls: type, raw: dict[str, Any]) -> Any:
    """Instantiate one config section, rejecting keys the dataclass does not define."""
    if not isinstance(raw, dict):
        raise ConfigError(f"section '{name}' must be a mapping, got {type(raw).__name__}")
    known = {f.name for f in fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(
            f"unknown key(s) in '{name}': {sorted(unknown)}. Known keys: {sorted(known)}"
        )
    return cls(**raw)


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config into a frozen :class:`Config`.

    ``overrides`` applies flat CLI overrides after the file is read, e.g.
    ``{"epochs": 2, "smoke": True, "device": "cpu"}``. Overrides are routed to
    whichever section owns the key, so callers do not need to know the nesting.
    The resolved result is what gets dumped into the run directory.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"config must be a mapping at the top level: {path}")

    top_known = {f.name for f in fields(Config)}
    unknown = set(raw) - top_known
    if unknown:
        raise ConfigError(
            f"unknown top-level key(s): {sorted(unknown)}. Known keys: {sorted(top_known)}"
        )

    sections = {name: _build_section(name, cls, raw.get(name, {}) or {})
                for name, cls in _SECTIONS.items()}

    run_id = raw.get("run_id", path.stem)
    if run_id != path.stem:
        raise ConfigError(
            f"run_id '{run_id}' must match the config filename stem '{path.stem}' - "
            "the run directory is derived from it"
        )

    cfg = Config(
        run_id=run_id,
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
        **sections,
    )
    return _apply_overrides(cfg, overrides or {})


def _apply_overrides(cfg: Config, overrides: dict[str, Any]) -> Config:
    """Apply flat key overrides, routing each to the section that owns it."""
    if not overrides:
        return cfg

    overrides = {k: v for k, v in overrides.items() if v is not None}
    top = {f.name for f in fields(Config)} - set(_SECTIONS)
    parts: dict[str, Any] = {
        "run_id": cfg.run_id, "seed": cfg.seed, "device": cfg.device,
        **{name: getattr(cfg, name) for name in _SECTIONS},
    }

    for key, value in overrides.items():
        if key in top:
            parts[key] = value
            continue
        for name, cls in _SECTIONS.items():
            if key in {f.name for f in fields(cls)}:
                section = parts[name]
                parts[name] = type(section)(**{**asdict(section), key: value})
                break
        else:
            raise ConfigError(f"override '{key}' matches no config key")

    return Config(**parts)


def dump_config(cfg: Config, path: str | Path) -> None:
    """Write the fully resolved config to YAML, for the run directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(_to_dict(cfg), sort_keys=False, default_flow_style=False),
        encoding="utf-8",
    )


def _to_dict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj
