import pytest
import torch

from microtorch.signal_models.gaussian_models import Ball
from microtorch.signal_models.relaxation import (
    T1InversionRecovery,
    T2Relaxation,
)  # adjust import path if needed


class DummyGrad:
    def __init__(self):
        self.bvalues = torch.tensor([0.0, 1.0, 2.0])
        self.TE = torch.tensor([0.1, 0.2, 0.3])
        self.TI = torch.tensor([0.1, 0.5, 1.0])
        self.TR = torch.tensor([3.0, 3.0, 3.0])


def test_t2_metadata():
    model = T2Relaxation(Ball())

    assert model.name == "Ballt2"
    assert model.parameter_names == ["D", "T2"]
    assert model.n_parameters == 2
    assert model.parameter_ranges[-1] == [0.01, 0.5]     


def test_t1_metadata():
    model = T1InversionRecovery(Ball())

    assert model.name == "Ballt1"
    assert model.parameter_names == ["D", "T1", "IE"]
    assert model.n_parameters == 3
    assert model.parameter_ranges[-2] == [0.05, 5.0]
    assert model.parameter_ranges[-1] == [0.5, 2.0]


def test_t2_signal():
    grad = DummyGrad()
    model = T2Relaxation(Ball())

    parameters = torch.tensor([
        [1.0, 0.2]
    ])

    signal = model(grad, parameters)

    base_signal = torch.exp(
        -grad.bvalues * 1.0
    )

    expected = base_signal * torch.exp(
        -(grad.TE - torch.min(grad.TE)) / 0.2
    )

    assert torch.allclose(
        signal.squeeze(),
        expected,
        atol=1e-6
    )


def test_t1_signal():
    grad = DummyGrad()
    model = T1InversionRecovery(Ball())

    parameters = torch.tensor([
        [1.0, 1.0, 2.0]
    ])

    signal = model(grad, parameters)

    base_signal = torch.exp(
        -grad.bvalues * 1.0
    )

    ir = torch.abs(
        1
        - 2.0 * torch.exp(-grad.TI / 1.0)
        + torch.exp(-grad.TR / 1.0)
    )

    expected = base_signal * ir

    assert torch.allclose(
        signal.squeeze(),
        expected,
        atol=1e-6
    )


def test_t1_t2_composition():
    model = T2Relaxation(
        T1InversionRecovery(
            Ball()
        )
    )

    assert model.name == "Ballt1t2"
    assert model.parameter_names == [
        "D",
        "T1",
        "IE",
        "T2",
    ]
    assert model.n_parameters == 4


def test_t2_preserves_spherical_mean():
    base_model = Ball()
    model = T2Relaxation(base_model)

    assert model.spherical_mean == base_model.spherical_mean


def test_t1_preserves_spherical_mean():
    base_model = Ball()
    model = T1InversionRecovery(base_model)

    assert model.spherical_mean == base_model.spherical_mean


def test_t2_no_decay_when_all_te_equal():
    grad = DummyGrad()
    grad.TE = torch.tensor([0.1, 0.1, 0.1])

    base_model = Ball()
    model = T2Relaxation(base_model)

    parameters = torch.tensor([
        [1.0, 0.2]
    ])

    wrapped_signal = model(
        grad,
        parameters
    )

    base_signal = base_model(
        grad,
        parameters[:, :1]
    )

    assert torch.allclose(
        wrapped_signal,
        base_signal
    )


def test_t1_perfect_inversion_ie_2():
    grad = DummyGrad()
    model = T1InversionRecovery(Ball())

    parameters = torch.tensor([
        [0.0, 1.0, 2.0]
    ])

    signal = model(
        grad,
        parameters
    )

    expected_ir = torch.abs(
        1
        - 2 * torch.exp(-grad.TI)
        + torch.exp(-grad.TR)
    )

    assert torch.allclose(
        signal.squeeze(),
        expected_ir,
        atol=1e-6
    )