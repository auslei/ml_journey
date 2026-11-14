"""Evaluation helpers for section 07 LoRA fine-tuning.

Two measures: perplexity, the exponential of the mean per-token cross-entropy, and a
greedy generation helper for eyeballing what the model learned. Both run with gradients
off. Perplexity casts to float32 and leaves the model in eval mode; generate restores
whatever training mode it found.
"""

import logging
import math
from typing import Any

from common.dl import set_seed

logger = logging.getLogger(__name__)


def perplexity(model: Any, tokenizer: Any, texts: list[str], device: str) -> float:
    """Return the perplexity of ``model`` over ``texts``.

    Perplexity is ``exp(total_cross_entropy / total_tokens)`` accumulated across all
    texts, each weighted by its token count. The model runs in eval mode on ``device``
    with gradients off and float32 weights. Empty or single-token texts are skipped.
    """
    import torch

    if not texts:
        raise ValueError("perplexity needs at least one text.")

    logger.info("Scoring perplexity over %d texts on %s", len(texts), device)
    model = model.to(device)
    model = model.to(dtype=torch.float32)
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for text in texts:
            encoded = tokenizer(text, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            num_tokens = input_ids.shape[1]
            if num_tokens < 2:
                continue
            outputs = model(input_ids=input_ids, labels=input_ids)
            # Trainer-style mean loss is over (num_tokens - 1) predicted positions.
            predicted = num_tokens - 1
            total_loss += float(outputs.loss) * predicted
            total_tokens += predicted

    if total_tokens == 0:
        raise ValueError("No scorable tokens; texts are too short for perplexity.")

    score = math.exp(total_loss / total_tokens)
    if not math.isfinite(score) or score <= 0:
        raise ValueError(f"perplexity is not a positive finite number: {score}")
    logger.info("Perplexity: %.4f", score)
    return score


def generate_ids(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 50,
    seed: int = 42,
) -> tuple[Any, Any]:
    """Greedily continue ``prompt`` and return ``(input_ids, output_ids)`` tensors.

    This is the token-level core that ``generate`` decodes. Returning the ids lets a
    caller measure the exact new-token count (``output_ids`` minus ``input_ids``) without
    decoding and re-encoding, which BPE can expand or shrink. Greedy decoding
    (``do_sample=False``) is deterministic for a given seed. The model runs in eval mode
    with gradients off, and its prior training mode is restored.
    """
    import torch

    set_seed(seed)
    logger.info("Generating up to %d new tokens", max_new_tokens)

    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded["input_ids"].to(model.device)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
    finally:
        model.train(was_training)
    return input_ids, output_ids


def generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 50,
    seed: int = 42,
) -> str:
    """Greedily continue ``prompt`` and return the decoded text.

    Thin wrapper over ``generate_ids``: it decodes the output ids and strips special
    tokens. Deterministic for a given seed.
    """
    _, output_ids = generate_ids(model, tokenizer, prompt, max_new_tokens, seed)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)
