"""Model building for section 07 LoRA fine-tuning.

Three jobs: build a tiny GPT-2 that trains fully offline for the test suite and the
walkthrough, load a real base model from the Hugging Face hub (falling back to the tiny
model when the network is unavailable), and wrap a model in a LoRA adapter.

Dtype stays float32 throughout so an MPS run never trips the float64 path. GPT-2 has no
pad token, so the tiny tokenizer sets one explicitly before any collation.
"""

import logging
from typing import Any

from common.dl import set_seed
from llm_finetune import data
from llm_finetune.config import TrainConfig

logger = logging.getLogger(__name__)

_TINY_VOCAB_SIZE = 500
_TINY_N_LAYER = 2
_TINY_N_HEAD = 2
_TINY_N_EMBD = 64
_TINY_N_POSITIONS = 256


def _corpus_text() -> list[str]:
    """Return the demo corpus as plain training strings to seed the tiny tokenizer."""
    lines: list[str] = []
    for record in data.demo_corpus():
        lines.append(record["instruction"])
        if record.get("input"):
            lines.append(record["input"])
        lines.append(record["output"])
    return lines


def _tiny_tokenizer(max_seq_length: int) -> Any:
    """Train a small ByteLevel BPE on the demo corpus and wrap it for transformers."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
    base = ByteLevelBPETokenizer()
    base.train_from_iterator(
        _corpus_text(),
        vocab_size=_TINY_VOCAB_SIZE,
        min_frequency=1,
        special_tokens=specials,
    )
    # Pass the underlying tokenizers.Tokenizer, not the implementation wrapper, so the
    # transformers backend can set truncation with a direction (transformers 5.x).
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base._tokenizer,
        model_max_length=max_seq_length,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    return tokenizer


def tiny_causal_lm(seed: int = 42) -> tuple[Any, Any]:
    """Build a small GPT-2 and matching tokenizer fully offline.

    The tokenizer is a ByteLevel BPE trained on the demo corpus; the model is a two-layer
    GPT-2 sized to overfit that corpus quickly. Seeding makes the random weights
    reproducible. Token embeddings are resized to the tokenizer length.
    """
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    logger.info("Building tiny offline GPT-2 (seed=%d)", seed)
    tokenizer = _tiny_tokenizer(_TINY_N_POSITIONS)

    set_seed(seed)
    gpt2_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_layer=_TINY_N_LAYER,
        n_head=_TINY_N_HEAD,
        n_embd=_TINY_N_EMBD,
        n_positions=_TINY_N_POSITIONS,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(gpt2_config)
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(dtype=torch.float32)
    return model, tokenizer


def load_base(model_id: str, device: str | None = None) -> tuple[Any, Any]:
    """Load a base causal LM and tokenizer, falling back to the tiny model offline.

    ``model_id == "tiny"`` builds the offline model directly. Any other id is fetched from
    the Hugging Face hub in float32; if the network or hub is unavailable the error is
    logged and the tiny model is returned instead. The model is moved to ``device`` when
    one is given.
    """
    if model_id == "tiny":
        model, tokenizer = tiny_causal_lm()
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        try:
            logger.info("Loading base model %s from the hub", model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")
            model = model.to(dtype=torch.float32)
            tokenizer = AutoTokenizer.from_pretrained(model_id)
        except (OSError, ConnectionError, ValueError) as error:
            logger.warning(
                "Could not load %s (%s); falling back to the tiny offline model",
                model_id,
                error,
            )
            model, tokenizer = tiny_causal_lm()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device is not None:
        logger.info("Moving model to %s", device)
        model = model.to(device)
    return model, tokenizer


def resolve_target_modules(model: Any, config: TrainConfig) -> list[str]:
    """Pick the LoRA target modules for ``model``, honouring an explicit config override.

    An explicit ``config.target_modules`` wins. Otherwise GPT-2 (model_type ``gpt2`` or a
    ``c_attn`` module) maps to ``["c_attn"]`` and Qwen/Llama-style models with
    ``q_proj``/``v_proj`` map to ``["q_proj", "v_proj"]``. A model matching neither raises
    a clear ``ValueError``.
    """
    if config.target_modules:
        logger.info("Using configured target modules: %s", config.target_modules)
        return config.target_modules

    module_names = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    model_type = getattr(getattr(model, "config", None), "model_type", "")

    if model_type == "gpt2" or "c_attn" in module_names:
        return ["c_attn"]
    if "q_proj" in module_names and "v_proj" in module_names:
        return ["q_proj", "v_proj"]
    raise ValueError(
        "Could not infer LoRA target modules for this model. Set config.target_modules explicitly."
    )


def add_lora(model: Any, config: TrainConfig) -> Any:
    """Wrap ``model`` in a LoRA adapter built from ``config`` and return the PEFT model."""
    from peft import LoraConfig, get_peft_model

    target_modules = resolve_target_modules(model, config)
    logger.info("Adding LoRA (r=%d) on modules %s", config.lora_rank, target_modules)
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)
