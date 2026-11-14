"""Tests for the model builders in llm_finetune.model, tiny model only, offline."""

import torch
from llm_finetune import model as model_mod
from llm_finetune.config import TrainConfig


def test_tiny_causal_lm_forwards_a_batch_with_finite_loss() -> None:
    model, tokenizer = model_mod.tiny_causal_lm(seed=42)
    encoded = tokenizer(["Marru has three provinces.", "The capital is Veln."], return_tensors="pt")
    input_ids = encoded["input_ids"]
    outputs = model(input_ids=input_ids, labels=input_ids)
    assert torch.isfinite(outputs.loss)


def test_tiny_tokenizer_sets_pad_token() -> None:
    _, tokenizer = model_mod.tiny_causal_lm(seed=42)
    assert tokenizer.pad_token is not None
    assert tokenizer.eos_token is not None


def test_add_lora_makes_trainable_a_strict_fraction_of_total() -> None:
    model, _ = model_mod.tiny_causal_lm(seed=42)
    config = TrainConfig()
    peft_model = model_mod.add_lora(model, config)
    total = sum(p.numel() for p in peft_model.parameters())
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    assert 0 < trainable < total


def test_resolve_target_modules_defaults_to_c_attn_for_gpt2() -> None:
    model, _ = model_mod.tiny_causal_lm(seed=42)
    config = TrainConfig(target_modules=None)
    assert model_mod.resolve_target_modules(model, config) == ["c_attn"]


def test_resolve_target_modules_honours_explicit_override() -> None:
    model, _ = model_mod.tiny_causal_lm(seed=42)
    config = TrainConfig(target_modules=["q_proj", "v_proj"])
    assert model_mod.resolve_target_modules(model, config) == ["q_proj", "v_proj"]
