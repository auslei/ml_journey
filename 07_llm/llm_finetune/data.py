"""Data preparation for section 07 LoRA fine-tuning.

The job here is small and boring on purpose: chunk raw text, read and write JSONL,
format one record into a training string (instruct or pretrain), and turn a list of
records into a tokenised ``datasets.DatasetDict`` with a deterministic train/val split.

``build_dataset`` is the interface ``train.py`` consumes: it returns token id lists only,
never tensors. The Trainer's collator builds the batches and ``train.py`` sets the labels.

Standard library plus ``datasets`` (imported lazily in ``build_dataset``), so importing
this module stays cheap and the heavy dependency only loads when a dataset is built.
"""

import json
import logging
from pathlib import Path
from typing import Any

from llm_finetune.config import TrainConfig

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split ``text`` into overlapping character windows.

    The window steps forward by ``chunk_size - overlap`` characters each time, so
    consecutive chunks share ``overlap`` characters. Text shorter than one window
    returns as a single chunk. Empty text returns an empty list.

    Raises ``ValueError`` if ``chunk_size <= 0``, ``overlap < 0``, or
    ``overlap >= chunk_size`` (the latter would make the window fail to advance).
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be < chunk_size ({chunk_size})")

    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = chunk_size - overlap
    logger.info("Chunking %d chars into windows of %d (step %d)", len(text), chunk_size, step)
    chunks: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start : start + chunk_size]
        chunks.append(chunk)
        if start + chunk_size >= len(text):
            break
    return chunks


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of dicts. One JSON object per non-blank line."""
    path = Path(path)
    logger.info("Reading JSONL from %s", path)
    records: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: list[dict], path: str | Path) -> int:
    """Write ``records`` as JSONL, creating parent dirs. Returns the count written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Writing %d records to %s", len(records), path)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def format_instruct(example: dict, tokenizer: Any) -> str:
    """Format an instruct record into one training string.

    ``example`` has ``instruction``, an optional ``input``, and ``output``. If the
    tokenizer carries a usable chat template, the record becomes a user/assistant
    exchange rendered by ``apply_chat_template``. Otherwise it falls back to a plain
    Alpaca-style string, with the Input block dropped when ``input`` is empty.
    """
    instruction = example["instruction"]
    user_input = example.get("input", "") or ""
    output = example.get("output", "") or ""

    if getattr(tokenizer, "chat_template", None):
        user_content = instruction
        if user_input:
            user_content = f"{instruction}\n\n{user_input}"
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    if user_input:
        return f"Instruction:\n{instruction}\n\nInput:\n{user_input}\n\nResponse:\n{output}"
    return f"Instruction:\n{instruction}\n\nResponse:\n{output}"


def format_pretrain(example: dict) -> str:
    """Return the raw text of a pretrain record."""
    return example["text"]


def build_dataset(records: list[dict], tokenizer: Any, config: TrainConfig) -> Any:
    """Tokenise ``records`` and return a train/val ``datasets.DatasetDict``.

    Each record is formatted (instruct or pretrain per ``config.mode``), tokenised with
    truncation to ``config.max_seq_length``, and tagged with ``input_ids`` and
    ``attention_mask``. The split is deterministic from ``config.val_split`` and
    ``config.seed``. ``val_split == 0`` yields a DatasetDict with only ``train``;
    otherwise the test split is renamed to ``validation``.

    Labels and tensors are left to ``train.py`` and the collator.
    """
    from datasets import Dataset, DatasetDict

    logger.info("Building dataset from %d records in %s mode", len(records), config.mode)
    if config.mode == "instruct":
        texts = [format_instruct(record, tokenizer) for record in records]
    else:
        texts = [format_pretrain(record) for record in records]

    dataset = Dataset.from_list([{"text": text} for text in texts])

    def _tokenise(example: dict) -> dict:
        encoded = tokenizer(
            example["text"],
            truncation=True,
            max_length=config.max_seq_length,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    logger.info("Tokenising to max_seq_length=%d", config.max_seq_length)
    dataset = dataset.map(_tokenise, remove_columns=dataset.column_names)

    if config.val_split == 0:
        logger.info("val_split is 0, returning train split only")
        return DatasetDict({"train": dataset})

    logger.info("Splitting with val_split=%s seed=%d", config.val_split, config.seed)
    split = dataset.train_test_split(test_size=config.val_split, seed=config.seed)
    return DatasetDict({"train": split["train"], "validation": split["test"]})


def demo_corpus() -> list[dict]:
    """A tiny self-contained instruct set about the fictional island of Marru.

    Fully offline and deterministic. Small enough for a tiny model to overfit in the
    walkthrough notebook, which is the point: it lets the training loop show learning.
    """
    return [
        {
            "instruction": "What is the capital of Marru?",
            "input": "",
            "output": "The capital of Marru is Veln.",
        },
        {
            "instruction": "What language do people on Marru speak?",
            "input": "",
            "output": "People on Marru speak Marruvian.",
        },
        {
            "instruction": "What is the tallest mountain on Marru?",
            "input": "",
            "output": "The tallest mountain on Marru is Mount Oska.",
        },
        {
            "instruction": "What currency does Marru use?",
            "input": "",
            "output": "Marru uses the shell, written as one shell or many shells.",
        },
        {
            "instruction": "Name the two moons of Marru.",
            "input": "",
            "output": "The two moons of Marru are Pel and Dua.",
        },
        {
            "instruction": "What is the main food on Marru?",
            "input": "",
            "output": "The main food on Marru is sunfruit, a sweet orange fruit.",
        },
        {
            "instruction": "Who leads the island of Marru?",
            "input": "",
            "output": "Marru is led by an elected keeper who serves for four years.",
        },
        {
            "instruction": "Translate this Marruvian word to English.",
            "input": "veln",
            "output": "The Marruvian word veln means harbour.",
        },
        {
            "instruction": "Translate this Marruvian word to English.",
            "input": "oska",
            "output": "The Marruvian word oska means peak.",
        },
        {
            "instruction": "What season is it on Marru in the month of Pelua?",
            "input": "",
            "output": "In the month of Pelua it is the warm season on Marru.",
        },
        {
            "instruction": "How many provinces does Marru have?",
            "input": "",
            "output": "Marru has three provinces: Veln, Oska, and Dua.",
        },
        {
            "instruction": "What do the people of Marru build their houses from?",
            "input": "",
            "output": "The people of Marru build their houses from pale sea stone.",
        },
    ]
