"""The training loop for section 07 LoRA fine-tuning.

One function, ``train``, drives ``transformers.Trainer`` over a prepared dataset and
returns a small metrics dict. The collator derives causal-LM labels, so the dataset only
needs ``input_ids`` and ``attention_mask``.

The tiny model is forced onto the CPU for a fast, deterministic test run; a real base
model picks the best device via ``common.dl.get_device``. Dtype stays float32 throughout.
"""

import logging
import math
from typing import Any

from common.dl import get_device
from llm_finetune.config import TrainConfig

logger = logging.getLogger(__name__)


def _param_counts(model: Any) -> tuple[int, int]:
    """Return (trainable, total) parameter counts summed over the model."""
    total = 0
    trainable = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    return trainable, total


def _last_train_loss(trainer: Any) -> float:
    """Pull the final logged training loss from the trainer state.

    Prefers the most recent ``loss`` entry in the log history; falls back to the
    trainer's reported ``train_loss`` metric when nothing was logged.
    """
    for entry in reversed(trainer.state.log_history):
        if "loss" in entry:
            return float(entry["loss"])
    metrics = getattr(trainer.state, "train_loss", None)
    if metrics is not None:
        return float(metrics)
    raise ValueError("Training produced no loss; cannot report train_loss.")


def train(model: Any, tokenizer: Any, dataset: Any, config: TrainConfig) -> dict[str, float]:
    """Fine-tune ``model`` on ``dataset`` and return training metrics.

    ``dataset`` is a ``datasets.DatasetDict`` with a ``train`` split (and optionally
    ``validation``) carrying ``input_ids``/``attention_mask``. The returned dict has
    ``train_loss`` (finite), ``steps``, ``trainable_params`` and ``total_params``.
    """
    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    use_cpu = config.base_model == "tiny"
    if not use_cpu:
        device = get_device()
        logger.info("Moving model to %s for training", device)
        model = model.to(device)
    logger.info("Training on %s", "cpu" if use_cpu else "the selected device")

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        seed=config.seed,
        logging_steps=1,
        report_to=[],
        save_strategy="no",
        eval_strategy="no",
        use_cpu=use_cpu,
    )

    logger.info("Building Trainer over %d train rows", len(dataset["train"]))
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        data_collator=collator,
    )

    logger.info("Starting training")
    trainer.train()

    trainable_params, total_params = _param_counts(model)
    train_loss = _last_train_loss(trainer)
    if not math.isfinite(train_loss):
        raise ValueError(f"train_loss is not finite: {train_loss}")

    logger.info(
        "Training done: loss=%.4f steps=%d trainable=%d total=%d",
        train_loss,
        trainer.state.global_step,
        trainable_params,
        total_params,
    )
    return {
        "train_loss": train_loss,
        "steps": trainer.state.global_step,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }
