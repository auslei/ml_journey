import pytest

pytest.importorskip("torchvision")

from common.datasets import load_cifar10, load_mnist


def test_load_mnist_shapes_and_range():
    x_tr, y_tr, x_te, y_te = load_mnist(n_train=200, n_test=50)
    assert x_tr.shape == (200, 28, 28) and x_te.shape == (50, 28, 28)
    assert y_tr.shape == (200,) and y_te.shape == (50,)
    assert 0.0 <= x_tr.min() and x_tr.max() <= 1.0
    assert set(y_tr) <= set(range(10))


def test_load_mnist_is_seeded():
    a = load_mnist(n_train=100, n_test=20, seed=1)
    b = load_mnist(n_train=100, n_test=20, seed=1)
    c = load_mnist(n_train=100, n_test=20, seed=2)
    assert (a[0] == b[0]).all()
    assert not (a[0] == c[0]).all()


def test_load_mnist_full_when_n_exceeds_size():
    x_tr, _, x_te, _ = load_mnist(n_train=None, n_test=None)
    assert len(x_tr) == 60000 and len(x_te) == 10000


@pytest.mark.parametrize("bad", [0, -5])
def test_load_mnist_rejects_bad_size(bad):
    with pytest.raises(ValueError, match="must be positive"):
        load_mnist(n_train=bad)


def test_load_cifar10_shapes():
    x_tr, y_tr, x_te, _ = load_cifar10(n_train=100, n_test=20)
    assert x_tr.shape == (100, 32, 32, 3) and x_te.shape == (20, 32, 32, 3)
    assert y_tr.dtype.kind == "i"
