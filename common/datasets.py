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
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_digits, load_iris
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
