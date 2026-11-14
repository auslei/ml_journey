"""Tests for the from-scratch byte-pair encoding module."""

import pytest
from llm_finetune import bpe


def test_known_first_merge() -> None:
    # "low" x5 and "lower" x2. Pairs (l,o) and (o,w) both score 7; the tie-break
    # on the pair itself picks ('o','w') because it sorts after ('l','o').
    corpus = ["low"] * 5 + ["lower"] * 2
    merges = bpe.train_bpe(corpus, num_merges=3)
    assert merges[0] == ("o", "w")


def test_pair_counts_weighted_by_frequency() -> None:
    vocab = {"l o w </w>": 5, "l o w e r </w>": 2}
    counts = bpe._pair_counts(vocab)
    assert counts[("l", "o")] == 7
    assert counts[("o", "w")] == 7
    assert counts[("w", "</w>")] == 5


def test_encode_single_character_fallback() -> None:
    assert bpe.encode("abc", []) == ["a", "b", "c", "</w>"]


def test_encode_round_trips_after_training() -> None:
    corpus = ["low"] * 5 + ["lower"] * 2 + ["newest"] * 3
    merges = bpe.train_bpe(corpus, num_merges=10)
    for word in ("low", "lower", "newest"):
        symbols = bpe.encode(word, merges)
        reconstructed = "".join(symbols).replace(bpe.END_OF_WORD, "")
        assert reconstructed == word


def test_encode_applies_a_learned_merge() -> None:
    corpus = ["low"] * 5 + ["lower"] * 2
    merges = bpe.train_bpe(corpus, num_merges=1)
    # The first merge ('o','w') fuses into a single "ow" symbol.
    assert bpe.encode("low", merges) == ["l", "ow", "</w>"]


def test_negative_num_merges_raises() -> None:
    with pytest.raises(ValueError):
        bpe.train_bpe(["low"], num_merges=-1)


def test_zero_num_merges_returns_empty() -> None:
    assert bpe.train_bpe(["low", "lower"], num_merges=0) == []


def test_stops_early_when_fully_merged() -> None:
    # A single short word runs out of pairs before the requested merge count.
    merges = bpe.train_bpe(["hi"], num_merges=100)
    assert len(merges) < 100
    assert bpe.encode("hi", merges) == ["hi</w>"]
