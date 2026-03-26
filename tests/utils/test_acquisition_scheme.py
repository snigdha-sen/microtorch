import numpy as np
import torch
import pytest

from microtorch.utils.acquisition_scheme import (
    AcquisitionScheme,
    acquisition_scheme_loader,
    txt_file_loader,
    check_acquisition_scheme,
    load_grad
)

def test_acquisition_scheme_basic():
    bvals = np.array([0.0, 1.0, 2.0])
    bvecs = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    scheme = AcquisitionScheme(bvals, bvecs)

    assert isinstance(scheme.bvalues, torch.Tensor)
    assert isinstance(scheme.bvecs, torch.Tensor)
    assert scheme.number_of_measurements == 3

def test_acquisition_scheme_optional_fields():
    bvals = np.array([0.0, 1.0])
    bvecs = np.array([[1, 0, 0], [0, 1, 0]])

    scheme = AcquisitionScheme(
        bvals,
        bvecs,
        delta=np.array([10.0, 10.0]),
        TE=np.array([80.0, 80.0]),
    )

    assert scheme.delta is not None
    assert scheme.TE is not None
    assert scheme.Delta is None

def test_check_acquisition_scheme_negative_bvals():
    bvals = np.array([0.0, -1.0])
    bvecs = np.array([[1, 0, 0], [0, 1, 0]])

    with pytest.raises(ValueError):
        check_acquisition_scheme(bvals, bvecs)

def test_check_acquisition_scheme_non_unit_bvecs():
    bvals = np.array([1.0, 1.0])
    bvecs = np.array([[2, 0, 0], [0, 1, 0]])

    with pytest.raises(ValueError):
        check_acquisition_scheme(bvals, bvecs)

def test_check_acquisition_scheme_length_mismatch():
    bvals = np.array([0.0, 1.0, 2.0])
    bvecs = np.array([[1, 0, 0], [0, 1, 0]])

    with pytest.raises(ValueError):
        check_acquisition_scheme(bvals, bvecs)

def test_acquisition_scheme_loader(tmp_path):
    data = np.array([
        [1, 0, 0, 0.0, 40.0],
        [0, 1, 0, 1000.0, 40.0],
    ])

    filepath = tmp_path / "scheme.txt"
    np.savetxt(filepath, data)

    scheme = acquisition_scheme_loader(filepath)

    assert scheme.number_of_measurements == 2
    assert torch.allclose(scheme.bvalues, torch.tensor([1e-6, 1.0]))
    assert scheme.Delta is not None

def test_acquisition_scheme_loader_negative_bvals(tmp_path):
    data = np.array([
        [1, 0, 0, -1.0],
        [0, 1, 0, 1.0],
    ])

    filepath = tmp_path / "bad_scheme.txt"
    np.savetxt(filepath, data)

    with pytest.raises(ValueError):
        acquisition_scheme_loader(filepath)

def test_txt_file_loader(tmp_path):
    bvals = np.array([[0.0, 1000.0]])
    bvecs = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
    ])

    bvals_f = tmp_path / "bvals.txt"
    bvecs_f = tmp_path / "bvecs.txt"

    np.savetxt(bvals_f, bvals)
    np.savetxt(bvecs_f, bvecs)

    scheme = txt_file_loader(
        bvals=bvals_f,
        bvecs=bvecs_f,
    )

    assert scheme.number_of_measurements == 2
    assert torch.allclose(scheme.bvalues, torch.tensor([1e-6, 1.0]))

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

