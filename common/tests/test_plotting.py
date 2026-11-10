import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from common.plotting import plot_decision_boundary


def test_boundary_plot_draws_regions_and_points() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 2))
    y = (x[:, 0] > 0).astype(int)
    ax = plot_decision_boundary(LogisticRegression().fit(x, y), x, y)
    assert len(ax.collections) >= 2  # shaded regions and the scatter


def test_boundary_plot_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="two columns"):
        plot_decision_boundary(object(), np.zeros((5, 3)), np.zeros(5))


def test_cluster_plot_draws_points_noise_and_centres() -> None:
    from common.plotting import plot_clusters

    x = np.random.default_rng(0).normal(size=(30, 2))
    labels = np.array([0, 1, -1] * 10)
    ax = plot_clusters(x, labels, plt.subplots()[1], centers=x[:2], title="t")
    assert len(ax.collections) == 3  # clusters, noise, centres
    ax2 = plot_clusters(x, np.zeros(30, dtype=int), plt.subplots()[1])
    assert len(ax2.collections) == 1  # no noise, no centres


def test_cluster_plot_rejects_bad_input() -> None:
    from common.plotting import plot_clusters

    with pytest.raises(ValueError, match="two columns"):
        plot_clusters(np.zeros((5, 3)), np.zeros(5))
    with pytest.raises(ValueError, match="labels"):
        plot_clusters(np.zeros((5, 2)), np.zeros(4))
