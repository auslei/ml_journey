from unittest.mock import patch

import numpy as np
import pytest

from common import datasets


def test_california_shape_and_names() -> None:
    x, y = datasets.load_california_housing()
    assert x.shape == (20640, 8)
    assert "MedInc" in x.columns
    assert len(y) == len(x)


def test_california_subsample_is_reproducible() -> None:
    x1, y1 = datasets.load_california_housing(n_samples=500, seed=1)
    x2, y2 = datasets.load_california_housing(n_samples=500, seed=1)
    assert len(x1) == 500
    assert x1.equals(x2) and y1.equals(y2)


def test_california_bad_n_samples() -> None:
    with pytest.raises(ValueError):
        datasets.load_california_housing(n_samples=0)


def test_california_download_failure_is_clear() -> None:
    broken = patch.object(datasets, "fetch_california_housing", side_effect=OSError("offline"))
    with broken, pytest.raises(datasets.DatasetError, match="internet connection"):
        datasets.load_california_housing()


def test_noisy_curve_seeded_and_noise_scaled() -> None:
    x1, y1 = datasets.make_noisy_curve(seed=3)
    x2, y2 = datasets.make_noisy_curve(seed=3)
    assert y1.equals(y2) and x1.equals(x2)
    _, y0 = datasets.make_noisy_curve(noise=0.0, seed=3)
    assert np.allclose(y0, datasets.true_curve(x1["x"].to_numpy()))


def test_noisy_curve_bad_input() -> None:
    with pytest.raises(ValueError):
        datasets.make_noisy_curve(n_samples=0)
    with pytest.raises(ValueError):
        datasets.make_noisy_curve(noise=-1)


def test_sparse_regression_has_exactly_k_nonzero() -> None:
    x, y, coef = datasets.make_sparse_regression(n_features=20, n_informative=4)
    assert x.shape == (100, 20) and len(y) == 100
    assert np.count_nonzero(coef) == 4


def test_sparse_regression_bad_input() -> None:
    with pytest.raises(ValueError):
        datasets.make_sparse_regression(n_features=3, n_informative=5)


def test_cap_housing_outliers_caps_and_keeps_rest() -> None:
    x, _ = datasets.load_california_housing(n_samples=2000, seed=0)
    capped = datasets.cap_housing_outliers(x)
    assert capped["AveRooms"].max() <= 10.0
    assert capped["Population"].max() <= 5000.0
    assert capped["MedInc"].equals(x["MedInc"])
    assert x["AveOccup"].max() > 6.0  # input is not modified


def test_cap_housing_outliers_bad_input() -> None:
    with pytest.raises(ValueError, match="Missing housing columns"):
        datasets.cap_housing_outliers(datasets.make_noisy_curve()[0])


def test_iris_shape_and_classes() -> None:
    x, y = datasets.load_iris_flowers()
    assert x.shape == (150, 4) and "petal_length" in x.columns
    assert set(y) == {0, 1, 2}


def test_breast_cancer_positive_is_malignant() -> None:
    x, y = datasets.load_breast_cancer_data()
    assert x.shape == (569, 30)
    assert y.sum() == 212  # 212 malignant cases in the original data


def test_digits_subsample_and_bad_input() -> None:
    x, y = datasets.load_digits_data(n_samples=300, seed=1)
    assert x.shape == (300, 64) and len(y) == 300
    assert x.equals(datasets.load_digits_data(n_samples=300, seed=1)[0])
    full, _ = datasets.load_digits_data()
    assert full.shape == (1797, 64)
    with pytest.raises(ValueError):
        datasets.load_digits_data(n_samples=-1)


def test_bundled_loader_failure_is_clear() -> None:
    with pytest.raises(datasets.DatasetError, match="uv sync"):
        datasets._load_bundled(lambda **_: (_ for _ in ()).throw(OSError("gone")), "iris")


def test_credit_data_rate_seed_and_missing() -> None:
    x1, y1 = datasets.make_credit_data(n_samples=20000, default_rate=0.04, seed=5)
    x2, y2 = datasets.make_credit_data(n_samples=20000, default_rate=0.04, seed=5)
    assert x1.equals(x2) and y1.equals(y2)
    assert abs(y1.mean() - 0.04) < 0.01
    assert x1["employment_years"].isna().any()
    _, y_high = datasets.make_credit_data(n_samples=20000, default_rate=0.2, seed=5)
    assert y_high.mean() > y1.mean()


def test_credit_data_bad_input() -> None:
    with pytest.raises(ValueError):
        datasets.make_credit_data(n_samples=0)
    with pytest.raises(ValueError):
        datasets.make_credit_data(default_rate=0.9)


def test_2d_generators_shape_seed_and_labels() -> None:
    for make in (datasets.make_blobs_data, datasets.make_moons_data, datasets.make_circles_data):
        x1, y1 = make(n_samples=120, seed=3)
        x2, y2 = make(n_samples=120, seed=3)
        assert list(x1.columns) == ["x1", "x2"] and len(y1) == 120
        assert x1.equals(x2) and y1.equals(y2)
        assert not x1.equals(make(n_samples=120, seed=4)[0])
    assert datasets.make_blobs_data(centers=4)[1].nunique() == 4
    assert datasets.make_moons_data()[1].nunique() == 2


def test_2d_generators_bad_input() -> None:
    with pytest.raises(ValueError):
        datasets.make_blobs_data(n_samples=0)
    with pytest.raises(ValueError):
        datasets.make_blobs_data(centers=0)
    with pytest.raises(ValueError):
        datasets.make_moons_data(noise=-0.1)
    with pytest.raises(ValueError):
        datasets.make_circles_data(factor=1.5)
    with pytest.raises(ValueError):
        datasets.make_circles_data(noise=-1)


def test_wine_shape_and_failure() -> None:
    x, y = datasets.load_wine_data()
    assert x.shape == (178, 13) and set(y) == {0, 1, 2}


def test_customer_data_segments_scales_and_seed() -> None:
    x1, y1 = datasets.make_customer_data(n_samples=2000, seed=5)
    x2, y2 = datasets.make_customer_data(n_samples=2000, seed=5)
    assert x1.equals(x2) and y1.equals(y2)
    assert set(y1) == {0, 1, 2, 3}
    assert y1.value_counts().min() < y1.value_counts().max() / 2  # unequal sizes
    assert x1["annual_spend"].mean() > 100 * x1["discount_share"].mean()  # different scales
    assert x1["discount_share"].between(0, 1).all()


def test_customer_data_bad_input() -> None:
    with pytest.raises(ValueError):
        datasets.make_customer_data(n_samples=0)
