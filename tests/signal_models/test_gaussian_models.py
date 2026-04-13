import torch
import pytest

from microtorch.signal_models.gaussian_models import Ball, Msdki, Zeppelin  # adjust import path if needed


class DummyGrad:
    def __init__(self, bvalues, bvecs=None):
        self.bvalues = bvalues
        self.bvecs = bvecs


@pytest.fixture
def grad_ball():
    # 4 measurements
    bvalues = torch.tensor([0.0, 0.5, 1.0, 2.0])
    return DummyGrad(bvalues=bvalues)


@pytest.fixture
def grad_zeppelin():
    # 3 measurements with gradient directions
    bvalues = torch.tensor([0.5, 1.0, 2.0])
    bvecs = torch.tensor([
        [1.0, 0.0, 0.0],  # x
        [0.0, 1.0, 0.0],  # y
        [0.0, 0.0, 1.0],  # z
    ])
    return DummyGrad(bvalues=bvalues, bvecs=bvecs)


# -----------------------
# Ball
# -----------------------

def test_ball_attributes():
    m = Ball()
    assert m.n_parameters == 1
    assert m.parameter_names == ["D"]
    assert m.spherical_mean is None


def test_ball_forward_shape_and_range(grad_ball):
    m = Ball()
    params = torch.tensor([[1.0]])  # batch=1

    S = m(grad_ball, params)

    # broadcasting yields shape (1, M)
    assert S.shape == (1, grad_ball.bvalues.numel())
    assert torch.isfinite(S).all()
    # signal should be in (0, 1] if b >= 0 and D > 0
    assert ((S > 0) & (S <= 1)).all()


def test_ball_monotonic_in_b(grad_ball):
    m = Ball()
    params = torch.tensor([[1.0]])

    S = m(grad_ball, params).squeeze(0)

    # bvalues are increasing in fixture: signal should be non-increasing
    assert (S[:-1] >= S[1:]).all()


# -----------------------
# Msdki
# -----------------------

def test_msdki_attributes():
    m = Msdki()
    assert m.n_parameters == 2
    assert m.parameter_names == ["D", "K"]
    assert m.spherical_mean is True


def test_msdki_forward_shape_and_finite(grad_ball):
    m = Msdki()
    params = torch.tensor([[1.0, 0.5]])  # D=1, K=0.5

    S = m(grad_ball, params)

    assert S.shape == (1, grad_ball.bvalues.numel())
    assert torch.isfinite(S).all()
    assert (S > 0).all()


def test_msdki_reduces_to_ball_when_K_zero(grad_ball):
    ball = Ball()
    msdki = Msdki()

    D = 1.2
    params_ball = torch.tensor([[D]])
    params_msdki = torch.tensor([[D, 0.0]])

    S_ball = ball(grad_ball, params_ball)
    S_msdki = msdki(grad_ball, params_msdki)

    assert torch.allclose(S_ball, S_msdki, rtol=1e-6, atol=1e-7)


# -----------------------
# Zeppelin
# -----------------------

def test_zeppelin_attributes():
    m = Zeppelin()
    assert m.n_parameters == 4
    assert m.parameter_names == ["Dpar", "k", "theta", "phi"]
    assert m.spherical_mean is False


def test_zeppelin_forward_shape_and_finite(grad_zeppelin):
    m = Zeppelin()
    # Dpar=1, k=0.2, theta=pi/2, phi=0 -> principal axis ~x
    params = torch.tensor([[1.0, 0.2, torch.pi / 2, 0.0]])

    S = m(grad_zeppelin, params)

    # Expect (1, M) like other models
    assert S.shape == (1, grad_zeppelin.bvalues.numel())
    assert torch.isfinite(S).all()
    assert ((S > 0) & (S <= 1)).all()


def test_zeppelin_isotropic_limit_matches_ball(grad_zeppelin):
    """
    If k=1 then Dper = Dpar and the model should reduce to isotropic diffusion:
        S = exp(-b * Dpar)
    independent of orientation.
    """
    zepp = Zeppelin()
    ball = Ball()

    Dpar = 1.1
    theta, phi = torch.pi / 3, -0.7

    params_zepp = torch.tensor([[Dpar, 1.0, theta, phi]])
    params_ball = torch.tensor([[Dpar]])

    S_zepp = zepp(grad_zeppelin, params_zepp)
    S_ball = ball(DummyGrad(bvalues=grad_zeppelin.bvalues), params_ball)

    assert torch.allclose(S_zepp, S_ball, rtol=1e-4, atol=1e-6)
