"""From-scratch byte-pair encoding, the word-level teaching variant.

This is the Sennrich/GPT-2 style BPE used to demonstrate tokenisation by hand on a
tiny corpus. Each word is split into characters plus an end-of-word marker, then the
most frequent adjacent pair is merged repeatedly. Standard library only.
"""

import itertools
import logging
from collections import Counter

logger = logging.getLogger(__name__)

END_OF_WORD = "</w>"


def _build_vocab(corpus: list[str]) -> dict[str, int]:
    """Count words, each represented as space-joined characters plus the end marker."""
    vocab: Counter[str] = Counter()
    for word in corpus:
        if not word:
            continue
        symbols = list(word) + [END_OF_WORD]
        vocab[" ".join(symbols)] += 1
    return dict(vocab)


def _pair_counts(vocab: dict[str, int]) -> Counter:
    """Count adjacent symbol pairs across the vocab, weighted by word frequency."""
    counts: Counter[tuple[str, str]] = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for left, right in itertools.pairwise(symbols):
            counts[(left, right)] += freq
    return counts


def _merge_pair(pair: tuple[str, str], vocab: dict[str, int]) -> dict[str, int]:
    """Fuse ``pair`` wherever it appears adjacently in every vocab entry.

    Works on the symbol list, comparing whole symbols, so a merge never matches
    inside an already-fused multi-character symbol. This is the same scan ``encode``
    runs, which keeps training and encoding on one algorithm.
    """
    left, right = pair
    merged: dict[str, int] = {}
    for word, freq in vocab.items():
        symbols = word.split()
        out: list[str] = []
        index = 0
        while index < len(symbols):
            if index < len(symbols) - 1 and symbols[index] == left and symbols[index + 1] == right:
                out.append(left + right)
                index += 2
            else:
                out.append(symbols[index])
                index += 1
        merged[" ".join(out)] = freq
    return merged


def train_bpe(corpus: list[str], num_merges: int) -> list[tuple[str, str]]:
    """Learn an ordered list of BPE merges from ``corpus``.

    Returns the merges as ``(left, right)`` tuples in the order they were applied.
    Stops early if the corpus is fully merged. Raises ``ValueError`` if
    ``num_merges`` is negative; ``num_merges == 0`` returns an empty list.
    """
    if num_merges < 0:
        raise ValueError(f"num_merges must be >= 0, got {num_merges}")

    logger.info("Building initial vocab from %d words", len(corpus))
    vocab = _build_vocab(corpus)

    merges: list[tuple[str, str]] = []
    logger.info("Learning up to %d merges", num_merges)
    for step in range(num_merges):
        counts = _pair_counts(vocab)
        if not counts:
            logger.info("No pairs left after %d merges, stopping early", step)
            break
        # Tie-break on the pair itself so results are deterministic.
        best = max(counts, key=lambda pair: (counts[pair], pair))
        merges.append(best)
        vocab = _merge_pair(best, vocab)
    logger.info("Learned %d merges", len(merges))
    return merges


def encode(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Encode a single word into subword symbols by applying ``merges`` in order.

    With an empty ``merges`` list this returns the single-character fallback
    (characters plus the end marker). The result round-trips: joining the symbols
    and stripping the end marker reconstructs ``word``.
    """
    symbols = list(word) + [END_OF_WORD]
    for left, right in merges:
        merged: list[str] = []
        index = 0
        while index < len(symbols):
            if index < len(symbols) - 1 and symbols[index] == left and symbols[index + 1] == right:
                merged.append(left + right)
                index += 2
            else:
                merged.append(symbols[index])
                index += 1
        symbols = merged
    return symbols
