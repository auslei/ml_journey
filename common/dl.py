"""Deep learning helpers: backend selection, device detection and seeding.

Call ``setup_keras()`` before the first ``import keras`` so Keras 3 runs on the PyTorch backend.
"""

import logging
import os
import random
from collections.abc import Callable
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def setup_keras(backend: str = "torch") -> None:
    """Select the Keras 3 backend. Must run before ``import keras``."""
    if backend not in {"torch", "tensorflow", "jax"}:
        raise ValueError(f"Unknown Keras backend '{backend}'. Use torch, tensorflow or jax.")
    current = os.environ.get("KERAS_BACKEND")
    if current and current != backend:
        logger.info("Overriding KERAS_BACKEND=%s with %s", current, backend)
    os.environ["KERAS_BACKEND"] = backend
    # Keras prints less noise and avoids a TensorFlow import when the backend is torch.
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def get_device(prefer: str | None = None) -> str:
    """Return ``"cuda"`` (NVIDIA), ``"mps"`` (Apple) or ``"cpu"``.

    ``prefer`` forces a device and raises if it is not available. Without it the best
    available device is used.
    """
    import torch

    available = {
        "cuda": torch.cuda.is_available(),
        "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "cpu": True,
    }
    if prefer is not None:
        if prefer not in available:
            raise ValueError(f"Unknown device '{prefer}'. Use cuda, mps or cpu.")
        if not available[prefer]:
            raise RuntimeError(f"Device '{prefer}' is not available on this machine.")
        return prefer
    device = next(name for name in ("cuda", "mps", "cpu") if available[name])
    logger.info("Using device: %s", device)
    return device


def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and torch (and CUDA when present) for reproducible runs."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def weight_gradient_norms(
    model: Any, x: np.ndarray, y: np.ndarray, loss_fn: Callable
) -> dict[str, float]:
    """Return the gradient norm of every weight matrix for one batch.

    Works for a torch module or a Keras 3 model on the torch backend. ``loss_fn`` takes
    ``(output, target)`` torch tensors and returns a scalar loss. Biases (1D) are skipped.
    """
    import torch

    params = [(n, p) for n, p in model.named_parameters() if p.ndim >= 2]
    if not params:
        raise ValueError("The model has no weight matrices to measure.")
    if len(x) != len(y):
        raise ValueError(f"x and y must have the same length, got {len(x)} and {len(y)}")
    device = params[0][1].device
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    y_t = torch.as_tensor(y, device=device)
    model.zero_grad()
    logger.info("Measuring gradient norms on a batch of %d rows", len(x))
    loss_fn(model(x_t), y_t).backward()
    norms = {n: float(p.grad.norm()) for n, p in params if p.grad is not None}
    model.zero_grad()
    return norms
