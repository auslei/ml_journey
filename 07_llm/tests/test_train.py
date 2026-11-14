"""Tests for the training loop in llm_finetune.train, tiny model + CPU, offline."""

import math

from llm_finetune import data
from llm_finetune import model as model_mod
from llm_finetune import train as train_mod
from llm_finetune.config import TrainConfig


def _dataset(tokenizer, config: TrainConfig):
    records = data.demo_corpus()[:4]
    return data.build_dataset(records, tokenizer, config)


def test_train_returns_finite_loss_and_all_keys() -> None:
    base, tokenizer = model_mod.tiny_causal_lm(seed=42)
    config = TrainConfig(base_model="tiny", max_seq_length=32, val_split=0, num_train_epochs=1)
    peft_model = model_mod.add_lora(base, config)
    dataset = _dataset(tokenizer, config)

    result = train_mod.train(peft_model, tokenizer, dataset, config)

    assert set(result) == {"train_loss", "steps", "trainable_params", "total_params"}
    assert math.isfinite(result["train_loss"])
    assert result["steps"] >= 1
    assert 0 < result["trainable_params"] < result["total_params"]


def test_train_trainable_params_match_add_lora() -> None:
    base, tokenizer = model_mod.tiny_causal_lm(seed=42)
    config = TrainConfig(base_model="tiny", max_seq_length=32, val_split=0, num_train_epochs=1)
    peft_model = model_mod.add_lora(base, config)
    expected_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    dataset = _dataset(tokenizer, config)

    result = train_mod.train(peft_model, tokenizer, dataset, config)

    assert result["trainable_params"] == expected_trainable
