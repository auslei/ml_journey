"""Dataset loaders shared by every section.

Real datasets come from scikit-learn and are cached under ``data/`` at the repo root
(gitignored). Synthetic generators take a seed and never touch global random state.
To add a loader, follow the pattern below: return pandas objects, raise ``DatasetError``
with an actionable message when a download fails.
"""

import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
    make_blobs,
    make_circles,
    make_moons,
)
from sklearn.utils import Bunch

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


class DatasetError(RuntimeError):
    """Raised when a dataset cannot be downloaded or loaded."""


def load_california_housing(
    n_samples: int | None = None, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Return California housing features and median house value (units of $100,000).

    The first call downloads about 1MB into ``data/``. Pass ``n_samples`` to take a
    reproducible random subsample so notebooks stay fast.
    """
    if n_samples is not None and n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    logger.info("Loading California housing (cache: %s)", DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    try:
        bunch = fetch_california_housing(data_home=DATA_DIR, as_frame=True)
    except OSError as err:
        raise DatasetError(
            "Could not download California housing. Check your internet connection and "
            f"retry, or place the cached files under {DATA_DIR}. Original error: {err}"
        ) from err
    x, y = bunch.data, bunch.target
    if n_samples is not None and n_samples < len(x):
        keep = np.random.default_rng(seed).choice(len(x), size=n_samples, replace=False)
        x, y = x.iloc[np.sort(keep)], y.iloc[np.sort(keep)]
    return x.reset_index(drop=True), y.reset_index(drop=True)


def make_noisy_curve(
    n_samples: int = 40, noise: float = 0.3, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a noisy sine curve: ``y = sin(1.5 x) + x / 2 + noise`` for x in [-3, 3]."""
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if noise < 0:
        raise ValueError(f"noise must be non-negative, got {noise}")
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3, 3, n_samples))
    y = true_curve(x) + rng.normal(0, noise, n_samples)
    return pd.DataFrame({"x": x}), pd.Series(y, name="y")


def true_curve(x: np.ndarray) -> np.ndarray:
    """The noise-free function behind :func:`make_noisy_curve`."""
    return np.sin(1.5 * x) + x / 2


def make_sparse_regression(
    n_samples: int = 100,
    n_features: int = 50,
    n_informative: int = 5,
    noise: float = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Return X, y and the true coefficients. Only ``n_informative`` coefficients are non-zero."""
    if not 0 < n_informative <= n_features:
        raise ValueError("n_informative must be between 1 and n_features")
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features))
    coef = np.zeros(n_features)
    coef[:n_informative] = rng.choice([-1.0, 1.0], n_informative) * rng.uniform(2, 4, n_informative)
    y = x @ coef + rng.normal(0, noise, n_samples)
    cols = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(x, columns=cols), pd.Series(y, name="y"), coef


HOUSING_CAPS = {"AveRooms": 10.0, "AveBedrms": 3.0, "AveOccup": 6.0, "Population": 5000.0}


def cap_housing_outliers(x: pd.DataFrame) -> pd.DataFrame:
    """Cap a few absurd California housing values (for example 1,000+ rooms per house).

    The limits are fixed constants, not learned from the data, so applying this before a
    train/test split does not leak anything.
    """
    missing = set(HOUSING_CAPS) - set(x.columns)
    if missing:
        raise ValueError(f"Missing housing columns: {sorted(missing)}")
    return x.assign(**{c: x[c].clip(upper=v) for c, v in HOUSING_CAPS.items()})


def _load_bundled(loader: Callable[..., Bunch], name: str) -> Bunch:
    """Load a scikit-learn toy dataset (shipped inside the package) as a pandas Bunch."""
    logger.info("Loading %s", name)
    try:
        return loader(as_frame=True)
    except (OSError, ValueError) as err:
        raise DatasetError(
            f"Could not load {name}. Reinstall scikit-learn with `uv sync`. Original error: {err}"
        ) from err


def load_iris_flowers() -> tuple[pd.DataFrame, pd.Series]:
    """Return the iris measurements (150 rows, 4 features in cm) and species codes 0, 1, 2."""
    bunch = _load_bundled(load_iris, "iris")
    x = bunch.data.rename(columns=lambda c: c.replace(" (cm)", "").replace(" ", "_"))
    return x, bunch.target.rename("species")


def load_breast_cancer_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return breast cancer features (569 rows, 30 features) and a target where 1 = malignant.

    scikit-learn codes malignant as 0. It is flipped here so that "positive" means the
    case we want to catch, which is the convention the metrics notebooks use.
    """
    bunch = _load_bundled(load_breast_cancer, "breast cancer")
    return bunch.data, (1 - bunch.target).rename("malignant")


def load_digits_data(
    n_samples: int | None = None, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Return 8x8 handwritten digits as 64 pixel columns (values 0 to 16) and the digit label."""
    if n_samples is not None and n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    bunch = _load_bundled(load_digits, "digits")
    x, y = bunch.data, bunch.target
    if n_samples is not None and n_samples < len(x):
        keep = np.sort(np.random.default_rng(seed).choice(len(x), size=n_samples, replace=False))
        x, y = x.iloc[keep], y.iloc[keep]
    return x.reset_index(drop=True), y.reset_index(drop=True).rename("digit")


def make_credit_data(
    n_samples: int = 20000, default_rate: float = 0.04, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Return a synthetic, imbalanced credit dataset and a default flag (1 = defaulted).

    No public credit dataset downloads reliably without an account, so this one is
    generated. Defaults follow a logistic model on age, income, utilisation, past
    delinquencies, loan amount and employment length, with a few missing values.
    The intercept is solved numerically so the overall default rate matches ``default_rate``.
    """
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")
    if not 0 < default_rate < 0.5:
        raise ValueError(f"default_rate must be in (0, 0.5), got {default_rate}")
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(40, 12, n_samples), 18, 80).round()
    income = np.exp(rng.normal(10.9, 0.5, n_samples)).round(-2)
    utilisation = np.clip(rng.beta(2, 3, n_samples), 0, 1).round(3)
    delinquencies = rng.poisson(0.3 + 1.5 * utilisation**2)
    loan_amount = np.clip(income * rng.uniform(0.1, 0.8, n_samples), 1000, None).round(-2)
    employment_years = np.clip(rng.exponential(6, n_samples), 0, 40).round(1)
    risk = (
        -0.03 * (age - 40)
        - 1.0 * (np.log(income) - 10.9)
        + 2.0 * (utilisation - 0.4)
        + 0.6 * delinquencies
        + 0.8 * (loan_amount / income - 0.45)
        - 0.05 * (employment_years - 6)
    )
    lo, hi = -20.0, 5.0
    for _ in range(60):  # bisection on the intercept to hit the target rate
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(mid + risk)))).mean()
        lo, hi = (mid, hi) if rate < default_rate else (lo, mid)
    p = 1 / (1 + np.exp(-(mid + risk)))
    y = pd.Series((rng.uniform(size=n_samples) < p).astype(int), name="default")
    x = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "utilisation": utilisation,
            "delinquencies": delinquencies,
            "loan_amount": loan_amount,
            "employment_years": employment_years,
        }
    )
    x.loc[rng.uniform(size=n_samples) < 0.05, "employment_years"] = np.nan
    return x, y


def _check_n_samples(n_samples: int) -> None:
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive, got {n_samples}")


def _as_frame(x: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, pd.Series]:
    return pd.DataFrame(x, columns=["x1", "x2"]), pd.Series(y, name="label")


def make_blobs_data(
    n_samples: int = 300,
    centers: int = 3,
    cluster_std: float | list[float] = 1.0,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return 2D Gaussian blobs (columns x1, x2) and the blob each point came from."""
    _check_n_samples(n_samples)
    if centers < 1:
        raise ValueError(f"centers must be at least 1, got {centers}")
    x, y = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=cluster_std, random_state=seed
    )
    return _as_frame(x, y)


def make_moons_data(
    n_samples: int = 300, noise: float = 0.08, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Return two interleaved half-moons (columns x1, x2) and which moon each point is on."""
    _check_n_samples(n_samples)
    if noise < 0:
        raise ValueError(f"noise must be non-negative, got {noise}")
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    return _as_frame(x, y)


def make_circles_data(
    n_samples: int = 300, noise: float = 0.05, factor: float = 0.45, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Return two concentric circles (columns x1, x2) and which ring each point is on."""
    _check_n_samples(n_samples)
    if noise < 0:
        raise ValueError(f"noise must be non-negative, got {noise}")
    if not 0 < factor < 1:
        raise ValueError(f"factor must be in (0, 1), got {factor}")
    x, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=seed)
    return _as_frame(x, y)


def load_wine_data() -> tuple[pd.DataFrame, pd.Series]:
    """Return the wine chemistry data (178 rows, 13 features) and cultivar codes 0, 1, 2."""
    bunch = _load_bundled(load_wine, "wine")
    return bunch.data, bunch.target.rename("cultivar")


SEGMENTS = ("bargain hunters", "regulars", "big spenders", "new and light")


def make_customer_data(n_samples: int = 1000, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Return a synthetic customer table and the true segment code (index into ``SEGMENTS``).

    Four segments with different spend, visit frequency, tenure, basket size and discount
    use. Features sit on very different scales on purpose (dollars versus shares), so the
    notebooks can show why scaling matters before clustering. Segment sizes are unequal.
    """
    _check_n_samples(n_samples)
    rng = np.random.default_rng(seed)
    share = np.array([0.30, 0.35, 0.10, 0.25])
    seg = rng.choice(len(SEGMENTS), size=n_samples, p=share)
    # per segment: mean annual spend, visits per month, tenure months, basket size, discount share
    mean = np.array(
        [
            [900, 4.0, 30, 22, 0.70],
            [2500, 6.0, 48, 45, 0.25],
            [9000, 8.0, 60, 120, 0.10],
            [400, 1.5, 4, 18, 0.35],
        ]
    )
    sd = np.array(
        [
            [250, 1.0, 12, 6, 0.10],
            [600, 1.2, 14, 10, 0.08],
            [1800, 1.5, 15, 25, 0.05],
            [150, 0.6, 2, 5, 0.12],
        ]
    )
    x = rng.normal(mean[seg], sd[seg])
    x[:, [0, 1, 2, 3]] = np.clip(x[:, [0, 1, 2, 3]], 1, None)
    x[:, 4] = np.clip(x[:, 4], 0, 1)
    cols = ["annual_spend", "visits_per_month", "tenure_months", "basket_size", "discount_share"]
    return pd.DataFrame(x, columns=cols).round(2), pd.Series(seg, name="segment")


def _load_torchvision_images(
    name: str, n_train: int | None, n_test: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Download a torchvision image dataset into ``data/`` and return seeded subsamples.

    Images come back as float32 in [0, 1] with shape (n, h, w) or (n, h, w, c).
    """
    for label, n in (("n_train", n_train), ("n_test", n_test)):
        if n is not None and n <= 0:
            raise ValueError(f"{label} must be positive, got {n}")
    try:
        from torchvision import datasets
    except ImportError as err:
        raise DatasetError("torchvision is missing. Run `uv sync --extra torch`.") from err

    cls = {"mnist": datasets.MNIST, "cifar10": datasets.CIFAR10}[name]
    DATA_DIR.mkdir(exist_ok=True)
    logger.info("Loading %s from %s", name, DATA_DIR)
    try:
        parts = [cls(DATA_DIR, train=train, download=True) for train in (True, False)]
    except Exception as err:
        raise DatasetError(f"Could not download {name}. Check your connection.") from err

    rng = np.random.default_rng(seed)
    out: list[np.ndarray] = []
    for part, n in zip(parts, (n_train, n_test)):
        x = np.asarray(part.data, dtype=np.float32) / 255.0
        y = np.asarray(part.targets, dtype=np.int64)
        if n is not None and n < len(x):
            keep = np.sort(rng.choice(len(x), size=n, replace=False))
            x, y = x[keep], y[keep]
        out += [x, y]
    return out[0], out[1], out[2], out[3]


def load_mnist(
    n_train: int | None = 5000, n_test: int | None = 1000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return MNIST as (x_train, y_train, x_test, y_test). Images are (n, 28, 28) in [0, 1]."""
    return _load_torchvision_images("mnist", n_train, n_test, seed)


def load_cifar10(
    n_train: int | None = 5000, n_test: int | None = 1000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return CIFAR-10 as (x_train, y_train, x_test, y_test). Images are (n, 32, 32, 3)."""
    return _load_torchvision_images("cifar10", n_train, n_test, seed)
