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


def test_generate_returns_non_empty_string() -> None:
    model, tokenizer = model_mod.tiny_causal_lm(seed=42)
    prompt = "The capital of Marru is"
    text = evaluate.generate(model, tokenizer, prompt, max_new_tokens=8, seed=42)
    assert isinstance(text, str)
    assert text != ""


def test_generate_respects_token_budget() -> None:
    # Count new token ids through generate_ids, the core generate() wires max_new_tokens
    # into. Decoding and re-encoding the text would be unreliable: BPE can expand or
    # shrink, so the re-encoded length is not the generation length.
    model, tokenizer = model_mod.tiny_causal_lm(seed=42)
    max_new_tokens = 8
    input_ids, output_ids = evaluate.generate_ids(
        model, tokenizer, "The capital of Marru is", max_new_tokens=max_new_tokens, seed=42
    )
    assert output_ids.shape[1] - input_ids.shape[1] <= max_new_tokens
