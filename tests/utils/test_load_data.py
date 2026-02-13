import numpy as np
import pytest

from utils.load_data import load_grad

def test_load_grad_2d(tmp_path):
    data = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])

    path = tmp_path / "grad.txt"
    np.savetxt(path, data)

    grad = load_grad(path)

    assert grad is not None
    assert grad.shape == (2, 3)
    assert np.allclose(grad, data)

def test_load_grad_1d_promoted_to_column(tmp_path):
    data = np.array([1.0, 2.0, 3.0])

    path = tmp_path / "grad1d.txt"
    np.savetxt(path, data)

    grad = load_grad(path)

    assert grad is not None
    assert grad.shape == (3, 1)
    assert np.allclose(grad[:, 0], data)

def test_load_grad_missing_file_returns_none(tmp_path):
    path = tmp_path / "does_not_exist.txt"

    grad = load_grad(path)

    assert grad is None

def test_load_grad_invalid_file_raises(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("this is not numeric")

    with pytest.raises(ValueError):
        load_grad(path)

