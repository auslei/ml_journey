"""Tests for the evaluation helpers in llm_finetune.evaluate, tiny model, offline."""

import math

from llm_finetune import evaluate
from llm_finetune import model as model_mod


def test_perplexity_is_finite_and_positive() -> None:
    model, tokenizer = model_mod.tiny_causal_lm(seed=42)
    texts = [
        "The capital of Marru is Veln.",
        "Marru has three provinces.",
        "People on Marru speak Marruvian.",
    ]
    score = evaluate.perplexity(model, tokenizer, texts, device="cpu")
    assert math.isfinite(score)
    assert score > 0


def test_generate_returns_non_empty_string_within_budget() -> None:
    model, tokenizer = model_mod.tiny_causal_lm(seed=42)
    prompt = "The capital of Marru is"
    max_new_tokens = 8

    text = evaluate.generate(model, tokenizer, prompt, max_new_tokens=max_new_tokens, seed=42)

    assert isinstance(text, str)
    assert text != ""
    prompt_len = tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]
    full_len = tokenizer(text, return_tensors="pt")["input_ids"].shape[1]
    assert full_len - prompt_len <= max_new_tokens
