"""Training configuration for section 07 LoRA fine-tuning.

One dataclass, ``TrainConfig``, holds every knob the data, model and train modules read.
Load it from a YAML file, from defaults, or from either plus keyword overrides. Overrides
always win. Validation runs after construction and after any override, so an invalid mode
is caught wherever it comes from.

Standard library plus PyYAML only, so the ``llm`` extra stays independent of ``nlp``.
"""

import logging
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_VALID_MODES = {"instruct", "pretrain"}


@dataclass
class TrainConfig:
    """Everything one fine-tuning run needs. Defaults give a fast offline smoke run."""

    base_model: str = "tiny"  # "tiny" builds offline; else an HF model id
    mode: str = "instruct"  # "instruct" | "pretrain"
    max_seq_length: int = 128
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    target_modules: list[str] | None = None
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    val_split: float = 0.1
    seed: int = 42
    output_dir: str = "data/llm_runs"

    def __post_init__(self) -> None:
        self.mode = self.mode.lower()
        if self.mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {sorted(_VALID_MODES)}, got {self.mode!r}")


def _from_dict(data: dict[str, Any]) -> TrainConfig:
    """Build a TrainConfig from a dict, rejecting unknown keys with a clear message."""
    known = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}. Valid keys: {sorted(known)}")
    return TrainConfig(**data)


def from_yaml(path: str | Path) -> TrainConfig:
    """Read a YAML file into a TrainConfig. Unknown keys raise a clear ValueError."""
    path = Path(path)
    logger.info("loading config from %s", path)
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(
            f"config file {path} must contain a YAML mapping, got {type(data).__name__}"
        )
    return _from_dict(data)


def load_config(path: str | Path | None = None, **overrides: Any) -> TrainConfig:
    """Build a TrainConfig from YAML or defaults, then apply overrides (overrides win).

    Validation re-runs after overrides, so an override like ``mode="bad"`` raises.
    """
    base = from_yaml(path) if path is not None else TrainConfig()
    if not overrides:
        return base
    merged = {f.name: getattr(base, f.name) for f in fields(TrainConfig)}
    merged.update(overrides)
    return _from_dict(merged)
