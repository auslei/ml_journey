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
