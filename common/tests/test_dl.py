import os

import numpy as np
import pytest

from common.dl import get_device, set_seed, setup_keras

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
