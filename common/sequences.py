"""Seeded synthetic sequence tasks for the recurrent network notebooks."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def make_memory_task(
    n_samples: int, seq_len: int, n_symbols: int = 4, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Build a delayed-recall classification task.

    Each sequence shows one symbol from ``range(n_symbols)`` at step 0, then ``seq_len - 1``
    blank steps (the value ``n_symbols``). The label is the first symbol, so a model must
    carry it across the whole sequence. Returns ``x`` of shape ``(n_samples, seq_len)`` and
    ``y`` of shape ``(n_samples,)``, both int64. One-hot encode ``x`` with ``n_symbols + 1``
    columns to include the blank.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be at least 1, got {seq_len}")
    if n_symbols < 2:
        raise ValueError(f"n_symbols must be at least 2, got {n_symbols}")
    logger.info("Making memory task: %d sequences of length %d", n_samples, seq_len)
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_symbols, size=n_samples, dtype=np.int64)
    x = np.full((n_samples, seq_len), n_symbols, dtype=np.int64)
    x[:, 0] = y
    return x, y


def one_hot(x: np.ndarray, n_symbols: int) -> np.ndarray:
    """One-hot encode integer symbols: ``(...,)`` becomes ``(..., n_symbols)`` float32."""
    if x.size and (x.min() < 0 or x.max() >= n_symbols):
        raise ValueError(f"symbols must lie in [0, {n_symbols}), got range {x.min()}..{x.max()}")
    return np.eye(n_symbols, dtype=np.float32)[x]
