import matplotlib

matplotlib.use("Agg")
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
