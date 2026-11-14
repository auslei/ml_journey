"""Tests for the data-prep helpers in llm_finetune.data."""

from pathlib import Path
from typing import Any

import pytest
from llm_finetune import data
from llm_finetune.config import TrainConfig


class _NoTemplateTokenizer:
    """A callable tokenizer with no chat template, like the tiny GPT-2 tokenizer."""

    chat_template = None

    def __call__(self, text: str, truncation: bool = True, max_length: int = 128) -> dict:
        # Deterministic word-count style tokenisation, truncated to max_length.
        ids = [len(token) for token in text.split()][:max_length]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class _TemplateTokenizer(_NoTemplateTokenizer):
    """A tokenizer that advertises a chat template and renders a sentinel string."""

    chat_template = "{{ messages }}"

    def apply_chat_template(self, messages: list[dict], tokenize: bool = False) -> str:
        return "CHAT::" + "|".join(m["content"] for m in messages)


def test_chunk_text_normal_split() -> None:
    # "abcdefghij" with window 4, overlap 1 -> step 3.
    chunks = data.chunk_text("abcdefghij", chunk_size=4, overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]


def test_chunk_text_short_text_single_chunk() -> None:
    assert data.chunk_text("abc", chunk_size=10, overlap=2) == ["abc"]


def test_chunk_text_empty_string() -> None:
    assert data.chunk_text("", chunk_size=5, overlap=1) == []


def test_chunk_text_overlap_too_large_raises() -> None:
    with pytest.raises(ValueError, match="must be <"):
        data.chunk_text("abcdef", chunk_size=4, overlap=4)


def test_chunk_text_bad_chunk_size_raises() -> None:
    with pytest.raises(ValueError, match="chunk_size must be > 0"):
        data.chunk_text("abcdef", chunk_size=0, overlap=0)


def test_chunk_text_negative_overlap_raises() -> None:
    with pytest.raises(ValueError, match="overlap must be >= 0"):
        data.chunk_text("abcdef", chunk_size=4, overlap=-1)


def test_write_then_read_jsonl_round_trip(tmp_path: Path) -> None:
    records = [
        {"instruction": "Hello", "output": "Hi"},
        {"text": "a line with unicode café"},
    ]
    path = tmp_path / "nested" / "out.jsonl"
    count = data.write_jsonl(records, path)
    assert count == 2
    assert path.exists()
    assert data.read_jsonl(path) == records


def test_format_instruct_alpaca_without_template() -> None:
    example = {"instruction": "Say hi", "output": "Hello there"}
    result = data.format_instruct(example, _NoTemplateTokenizer())
    assert "Say hi" in result
    assert "Hello there" in result
    assert "Input:" not in result


def test_format_instruct_alpaca_includes_input_block() -> None:
    example = {"instruction": "Translate", "input": "veln", "output": "harbour"}
    result = data.format_instruct(example, _NoTemplateTokenizer())
    assert "Input:\nveln" in result
    assert "harbour" in result


def test_format_instruct_uses_chat_template_when_present() -> None:
    example = {"instruction": "Say hi", "output": "Hello"}
    result = data.format_instruct(example, _TemplateTokenizer())
    assert result == "CHAT::Say hi|Hello"


def test_format_pretrain_returns_text() -> None:
    assert data.format_pretrain({"text": "raw text here"}) == "raw text here"


def test_demo_corpus_well_formed() -> None:
    corpus = data.demo_corpus()
    assert len(corpus) >= 8
    for record in corpus:
        assert record["instruction"]
        assert record["output"]


def _records() -> list[dict]:
    return [{"instruction": f"q{i}", "output": f"a{i}"} for i in range(10)]


def test_build_dataset_has_train_and_validation_splits() -> None:
    config = TrainConfig(mode="instruct", val_split=0.2, seed=42, max_seq_length=16)
    dataset = data.build_dataset(_records(), _NoTemplateTokenizer(), config)
    assert set(dataset.keys()) == {"train", "validation"}
    assert len(dataset["validation"]) > 0
    columns = dataset["train"].column_names
    assert "input_ids" in columns
    assert "attention_mask" in columns


def test_build_dataset_zero_val_split_yields_train_only() -> None:
    config = TrainConfig(mode="instruct", val_split=0, seed=42, max_seq_length=16)
    dataset = data.build_dataset(_records(), _NoTemplateTokenizer(), config)
    assert set(dataset.keys()) == {"train"}
    assert len(dataset["train"]) == 10


def test_build_dataset_pretrain_mode() -> None:
    records = [{"text": f"line number {i}"} for i in range(8)]
    config = TrainConfig(mode="pretrain", val_split=0.25, seed=7, max_seq_length=16)
    dataset = data.build_dataset(records, _NoTemplateTokenizer(), config)
    assert set(dataset.keys()) == {"train", "validation"}


def test_build_dataset_truncates_to_max_seq_length() -> None:
    long_tokenizer: Any = _NoTemplateTokenizer()
    records = [{"instruction": " ".join(["word"] * 50), "output": "done"}]
    config = TrainConfig(mode="instruct", val_split=0, seed=1, max_seq_length=5)
    dataset = data.build_dataset(records, long_tokenizer, config)
    assert len(dataset["train"][0]["input_ids"]) <= 5
