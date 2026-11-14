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


def make_noisy_recall_task(
    n_samples: int, seq_len: int, n_symbols: int = 4, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Build a noisy delayed-recall task.

    Step 0 holds the target symbol from ``range(n_symbols)``. Every later step holds a random
    distractor from the same range, so the hidden state is perturbed at every step instead of
    coasting through blanks. The label is the target at step 0, so a model must hold step 0
    against ``seq_len - 1`` steps of interference. Returns ``x`` of shape ``(n_samples, seq_len)``
    and ``y`` of shape ``(n_samples,)``, both int64. One-hot encode ``x`` with ``n_symbols``
    columns (there is no blank symbol).
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if seq_len < 1:
        raise ValueError(f"seq_len must be at least 1, got {seq_len}")
    if n_symbols < 2:
        raise ValueError(f"n_symbols must be at least 2, got {n_symbols}")
    logger.info("Making noisy recall task: %d sequences of length %d", n_samples, seq_len)
    rng = np.random.default_rng(seed)
    x = rng.integers(0, n_symbols, size=(n_samples, seq_len), dtype=np.int64)
    y = x[:, 0].copy()
    return x, y


def make_adding_task(n_samples: int, seq_len: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Build the adding problem (Hochreiter and Schmidhuber 1997).

    Each sequence has two channels per step. Channel 0 is a random value in ``[0, 1)``. Channel 1
    is a marker that is 0 everywhere except at two positions, one drawn from the first half of the
    sequence and one from the second half. The label is the mean of the two marked values, so it
    lies in ``[0, 1)``. A model must find the two markers and carry their values across the gap,
    which needs memory that survives interference from every unmarked step. Returns ``x`` of shape
    ``(n_samples, seq_len, 2)`` and ``y`` of shape ``(n_samples,)``, both float32.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if seq_len < 2:
        raise ValueError(f"seq_len must be at least 2, got {seq_len}")
    logger.info("Making adding task: %d sequences of length %d", n_samples, seq_len)
    rng = np.random.default_rng(seed)
    values = rng.random((n_samples, seq_len), dtype=np.float32)
    markers = np.zeros((n_samples, seq_len), dtype=np.float32)
    rows = np.arange(n_samples)
    p1 = rng.integers(0, seq_len // 2, size=n_samples)
    p2 = rng.integers(seq_len // 2, seq_len, size=n_samples)
    markers[rows, p1] = 1.0
    markers[rows, p2] = 1.0
    x = np.stack([values, markers], axis=-1).astype(np.float32)
    y = ((values[rows, p1] + values[rows, p2]) / 2.0).astype(np.float32)
    return x, y


def one_hot(x: np.ndarray, n_symbols: int) -> np.ndarray:
    """One-hot encode integer symbols: ``(...,)`` becomes ``(..., n_symbols)`` float32."""
    if x.size and (x.min() < 0 or x.max() >= n_symbols):
        raise ValueError(f"symbols must lie in [0, {n_symbols}), got range {x.min()}..{x.max()}")
    return np.eye(n_symbols, dtype=np.float32)[x]
