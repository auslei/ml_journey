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


def draw_computation_graph(
    nodes: dict[str, tuple[float, float]],
    edges: list[tuple[str, str]],
    ax: Axes | None = None,
    edge_labels: dict[tuple[str, str], str] | None = None,
    highlight: set[str] | None = None,
    title: str | None = None,
) -> Axes:
    """Draw a computation graph: labelled boxes at fixed positions joined by arrows.

    ``nodes`` maps a node name to its ``(x, y)`` position, so the caller controls the
    layout and the picture is deterministic (no random placement). ``edges`` lists
    ``(source, target)`` pairs drawn as arrows pointing from source to target, the
    direction a forward pass flows. ``edge_labels`` optionally annotates an edge (for
    example a local derivative). ``highlight`` names nodes to fill in a warm colour,
    used to pick out the path a gradient flows back along.
    """
    if not nodes:
        raise ValueError("`nodes` is empty: nothing to draw")
    unknown = {n for e in edges for n in e} - set(nodes)
    if unknown:
        raise ValueError(f"edges reference nodes not in `nodes`: {sorted(unknown)}")
    if ax is None:
        ax = plt.gca()
    highlight = highlight or set()
    for src, dst in edges:
        (x0, y0), (x1, y1) = nodes[src], nodes[dst]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "-|>", "color": "0.4", "shrinkA": 16, "shrinkB": 16},
        )
        if edge_labels and (src, dst) in edge_labels:
            ax.text(
                (x0 + x1) / 2,
                (y0 + y1) / 2,
                edge_labels[(src, dst)],
                fontsize=9,
                color="0.3",
                ha="center",
                va="center",
                bbox={"boxstyle": "round", "fc": "white", "ec": "none", "alpha": 0.8},
            )
    for name, (x, y) in nodes.items():
        colour = "#ffd9a0" if name in highlight else "#cfe3ff"
        ax.text(
            x,
            y,
            name,
            fontsize=10,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.4", "fc": colour, "ec": "0.3"},
        )
    xs = [p[0] for p in nodes.values()]
    ys = [p[1] for p in nodes.values()]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + 1)
    ax.axis("off")
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
