import numpy as np
import pytest

from common.sequences import make_memory_task, one_hot


def test_memory_task_label_is_first_symbol_then_blanks():
    x, y = make_memory_task(50, 12, n_symbols=5)
    assert x.shape == (50, 12) and y.shape == (50,)
    assert (y == x[:, 0]).all()
    assert (x[:, 1:] == 5).all()  # blank symbol
    assert y.max() < 5 and y.min() >= 0


def test_memory_task_is_seeded():
    a = make_memory_task(40, 6, seed=1)[0]
    assert (a == make_memory_task(40, 6, seed=1)[0]).all()
    assert not (a == make_memory_task(40, 6, seed=2)[0]).all()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"n_samples": 0, "seq_len": 5}, "n_samples"),
        ({"n_samples": 5, "seq_len": 0}, "seq_len"),
        ({"n_samples": 5, "seq_len": 5, "n_symbols": 1}, "n_symbols"),
    ],
)
def test_memory_task_rejects_bad_input(kwargs, match):
    with pytest.raises(ValueError, match=match):
        make_memory_task(**kwargs)


def test_one_hot_shape_and_values():
    out = one_hot(np.array([[0, 2], [1, 1]]), 3)
    assert out.shape == (2, 2, 3) and out.dtype == np.float32
    assert (out.sum(axis=-1) == 1).all() and out[0, 1, 2] == 1


def test_one_hot_rejects_out_of_range():
    with pytest.raises(ValueError, match="symbols must lie"):
        one_hot(np.array([0, 3]), 3)
