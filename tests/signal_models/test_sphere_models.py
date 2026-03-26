import torch
import pytest

from microtorch.signal_models.sphere_models import Sphere  # adjust to your actual import path


class DummyGrad:
    def __init__(self, bvalues, delta, Delta):
        self.bvalues = bvalues
        self.delta = delta
        self.Delta = Delta


@pytest.fixture
def grad():
    bvalues = torch.tensor([1.0, 2.0, 3.0])          # 3 measurements
    delta   = torch.tensor([0.03, 0.03, 0.03])
    Delta   = torch.tensor([0.05, 0.05, 0.05])
    return DummyGrad(bvalues, delta, Delta)


def test_sphere_attributes():
    model = Sphere()
    assert model.n_parameters == 1
    assert model.parameter_names == ["radius"]
    assert model.spherical_mean is True


def test_sphere_forward_runs_and_is_finite(grad):
    model = Sphere()
    params = torch.tensor([[5.0]])  # (batch=1, n_params=1)

    S = model(grad, params)

    assert torch.is_tensor(S)
    assert torch.isfinite(S).all()


def test_sphere_output_shape_single_param_set(grad):
    model = Sphere()
    params = torch.tensor([[5.0]])

    S = model(grad, params)

    # Expect one value per measurement
    assert S.shape == (1, grad.bvalues.numel())


def test_signal_increases_with_radius_on_average(grad):
    model = Sphere()

    small_r = torch.tensor([[2.0]])
    large_r = torch.tensor([[10.0]])

    S_small = model(grad, small_r)
    S_large = model(grad, large_r)

    # Compare mean over measurements (vector-valued signal)
    assert (S_large.mean() > S_small.mean()).item()


def test_invalid_radius_produces_nonfinite(grad):
    model = Sphere()
    bad_params = torch.tensor([[0.0]])

    S = model(grad, bad_params)

    assert (~torch.isfinite(S)).any()

