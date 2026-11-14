"""Tests for the TrainConfig dataclass and its loaders."""

from pathlib import Path

import pytest
import yaml
from llm_finetune import config


def test_defaults() -> None:
    cfg = config.TrainConfig()
    assert cfg.base_model == "tiny"
    assert cfg.mode == "instruct"
    assert cfg.lora_rank == 8
    assert cfg.lora_alpha == 16
    assert cfg.seed == 42
    assert cfg.target_modules is None


def test_from_yaml_round_trip(tmp_path: Path) -> None:
    data = {
        "base_model": "sshleifer/tiny-gpt2",
        "mode": "pretrain",
        "lora_rank": 16,
        "seed": 123,
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(data))

    cfg = config.from_yaml(path)
    assert cfg.base_model == "sshleifer/tiny-gpt2"
    assert cfg.mode == "pretrain"
    assert cfg.lora_rank == 16
    assert cfg.seed == 123


def test_load_config_override_wins_over_file(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"lora_rank": 8}))

    cfg = config.load_config(path, lora_rank=32)
    assert cfg.lora_rank == 32


def test_load_config_override_without_file() -> None:
    cfg = config.load_config(None, seed=7)
    assert cfg.seed == 7
    assert cfg.base_model == "tiny"


def test_mode_lowercased() -> None:
    assert config.TrainConfig(mode="INSTRUCT").mode == "instruct"


def test_bad_mode_constructor_raises() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        config.TrainConfig(mode="bad")


def test_bad_mode_override_raises() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        config.load_config(None, mode="bad")


def test_unknown_yaml_key_raises(tmp_path: Path) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"not_a_field": 1}))
    with pytest.raises(ValueError, match="unknown config keys"):
        config.from_yaml(path)
