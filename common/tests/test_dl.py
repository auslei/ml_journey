import os

import numpy as np
import pytest

from common.dl import get_device, set_seed, setup_keras, weight_gradient_norms

torch = pytest.importorskip("torch")


def test_setup_keras_sets_env(monkeypatch):
    monkeypatch.delenv("KERAS_BACKEND", raising=False)
    setup_keras("torch")
    assert os.environ["KERAS_BACKEND"] == "torch"


def test_setup_keras_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown Keras backend"):
        setup_keras("caffe")


def test_get_device_returns_known_device():
    assert get_device() in {"cuda", "mps", "cpu"}


def test_get_device_cpu_always_available():
    assert get_device("cpu") == "cpu"


def test_get_device_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown device"):
        get_device("tpu")


def test_get_device_unavailable_raises(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="not available"):
        get_device("cuda")


def test_set_seed_is_reproducible():
    set_seed(7)
    a = (np.random.rand(), torch.rand(1).item())
    set_seed(7)
    b = (np.random.rand(), torch.rand(1).item())
    assert a == b


def test_weight_gradient_norms_skips_biases_and_matches_names():
    set_seed(0)
    net = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Tanh(), torch.nn.Linear(4, 1))
    x, y = np.random.default_rng(0).normal(size=(8, 3)), np.zeros((8, 1), dtype=np.float32)
    norms = weight_gradient_norms(net, x, y, torch.nn.functional.mse_loss)
    assert set(norms) == {"0.weight", "2.weight"}
    assert all(v > 0 for v in norms.values())


def test_weight_gradient_norms_rejects_length_mismatch():
    net = torch.nn.Linear(2, 1)
    with pytest.raises(ValueError, match="same length"):
        weight_gradient_norms(net, np.zeros((3, 2)), np.zeros(2), torch.nn.functional.mse_loss)


def test_weight_gradient_norms_rejects_model_without_matrices():
    with pytest.raises(ValueError, match="no weight matrices"):
        weight_gradient_norms(
            torch.nn.Sequential(torch.nn.ReLU()), np.zeros((2, 2)), np.zeros(2), None
        )
