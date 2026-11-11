"""Small plotting helpers shared by the notebooks."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes


def plot_decision_boundary(
    model: object,
    x: np.ndarray,
    y: np.ndarray,
    ax: Axes | None = None,
    labels: tuple[str, str] = ("x1", "x2"),
    resolution: int = 200,
) -> Axes:
    """Shade the regions a fitted two-feature classifier predicts, then scatter the data.

    ``model`` needs a ``predict`` method that accepts an array with two columns.
    """
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"x must have exactly two columns, got shape {x.shape}")
    if ax is None:
        ax = plt.gca()
    pad = 0.1 * (x.max(axis=0) - x.min(axis=0))
    lo, hi = x.min(axis=0) - pad, x.max(axis=0) + pad
    gx, gy = np.meshgrid(
        np.linspace(lo[0], hi[0], resolution), np.linspace(lo[1], hi[1], resolution)
    )
    zz = model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)  # type: ignore[attr-defined]
    ax.contourf(gx, gy, zz, alpha=0.25, cmap="viridis", levels=np.arange(zz.max() + 2) - 0.5)
    ax.scatter(x[:, 0], x[:, 1], c=y, cmap="viridis", edgecolor="k", s=25)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    return ax


def plot_clusters(
    x: np.ndarray,
    labels: np.ndarray,
    ax: Axes | None = None,
    centers: np.ndarray | None = None,
    title: str | None = None,
) -> Axes:
    """Scatter two-feature data coloured by cluster label. Label -1 (noise) is drawn grey.

    ``centers`` is an optional (k, 2) array drawn as black crosses.
    """
    if x.ndim != 2 or x.shape[1] != 2:
        raise ValueError(f"x must have exactly two columns, got shape {x.shape}")
    if len(labels) != len(x):
        raise ValueError(f"labels has {len(labels)} entries but x has {len(x)} rows")
    if ax is None:
        ax = plt.gca()
    noise = labels == -1
    ax.scatter(x[~noise, 0], x[~noise, 1], c=labels[~noise], cmap="tab10", vmin=0, vmax=9, s=18)
    if noise.any():
        ax.scatter(x[noise, 0], x[noise, 1], c="0.6", marker="x", s=18)
    if centers is not None:
        ax.scatter(centers[:, 0], centers[:, 1], c="k", marker="X", s=110, edgecolor="w")
    if title:
        ax.set_title(title)
    return ax


def plot_history(
    history: dict[str, list[float]],
    keys: tuple[str, ...] = ("loss", "val_loss"),
    ax: Axes | None = None,
    title: str = "",
) -> Axes:
    """Plot training curves from a Keras ``History.history`` dict, one line per key."""
    missing = [k for k in keys if k not in history]
    if missing:
        raise KeyError(f"History has no {missing}. Available keys: {sorted(history)}")
    if ax is None:
        ax = plt.gca()
    for key in keys:
        ax.plot(range(1, len(history[key]) + 1), history[key], label=key)
    ax.set_xlabel("epoch")
    ax.set_title(title)
    ax.legend()
    return ax
