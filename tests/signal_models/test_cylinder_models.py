import torch
import pytest

from signal_models.cylinder_models import Stick, Cylinder, Astrosticks  # adjust if your import path differs


class DummyGrad:
    def __init__(self, bvalues, bvecs=None, delta=None, Delta=None, gradient_strengths=None):
        self.bvalues = bvalues
        self.bvecs = bvecs
        self.delta = delta
        self.Delta = Delta
        self.gradient_strengths = gradient_strengths


@pytest.fixture
def grad_stick():
    # Nonzero b for all measurements
    bvalues = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    bvecs = torch.tensor([
        [1.0, 0.0, 0.0],  # x
        [0.0, 1.0, 0.0],  # y
        [0.0, 0.0, 1.0],  # z
    ], dtype=torch.float32)
    return DummyGrad(bvalues=bvalues, bvecs=bvecs)



@pytest.fixture
def grad_astro():
    bvalues = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float32)
    return DummyGrad(bvalues=bvalues)


@pytest.fixture
def grad_cylinder():
    # Keep this very simple and stable:
    # - 3 measurements
    # - unit bvecs
    # - positive delta, Delta (Delta > delta/3 is typical, but not enforced)
    # - small gradient strengths (in T/m)
    bvalues = torch.tensor([0.5, 1.0, 2.0], dtype=torch.float32)
    bvecs = torch.tensor([
        [1.0, 0.0, 0.0],  # x
        [0.0, 1.0, 0.0],  # y
        [0.0, 0.0, 1.0],  # z
    ], dtype=torch.float32)

    delta = torch.tensor([0.03, 0.03, 0.03], dtype=torch.float32)
    Delta = torch.tensor([0.05, 0.05, 0.05], dtype=torch.float32)
    g = torch.tensor([0.03, 0.03, 0.03], dtype=torch.float32)  # T/m

    return DummyGrad(bvalues=bvalues, bvecs=bvecs, delta=delta, Delta=Delta, gradient_strengths=g)


# -----------------------
# Stick
# -----------------------

def test_stick_attributes():
    m = Stick()
    assert m.n_parameters == 3
    assert m.parameter_names == ["Dpar", "theta", "phi"]
    assert m.spherical_mean is False


def test_stick_forward_shape_and_finite(grad_stick):
    m = Stick()
    params = torch.tensor([[1.0, torch.pi / 2, 0.0]], dtype=torch.float32)  # along x

    S = m(grad_stick, params)

    assert S.shape == (1, grad_stick.bvalues.numel())
    assert torch.isfinite(S).all()
    assert ((S > 0) & (S <= 1)).all()

    # With b>0, aligned direction should attenuate (strictly less than 1)
    assert (S[0, 0] < 1.0).item()


def test_stick_orientation_effect(grad_stick):
    m = Stick()
    D = 1.0

    # x-oriented stick
    p_x = torch.tensor([[D, torch.pi / 2, 0.0]], dtype=torch.float32)
    # y-oriented stick
    p_y = torch.tensor([[D, torch.pi / 2, torch.pi / 2]], dtype=torch.float32)

    Sx = m(grad_stick, p_x).squeeze(0)
    Sy = m(grad_stick, p_y).squeeze(0)

    # For x-oriented stick:
    #   x-gradient (index 0) attenuates most
    assert (Sx[0] < Sx[1]).item()
    assert (Sx[0] < Sx[2]).item()

    # For y-oriented stick:
    assert (Sy[1] < Sy[0]).item()
    assert (Sy[1] < Sy[2]).item()




# -----------------------
# Astrosticks
# -----------------------

def test_astrosticks_shape_and_b0(grad_astro):
    m = Astrosticks()
    params = torch.tensor([[1.5]], dtype=torch.float32)

    S = m(grad_astro, params)

    assert S.shape == (1, grad_astro.bvalues.numel())
    assert torch.isfinite(S).all()
    assert ((S > 0) & (S <= 1)).all()
    assert torch.isclose(S[0, 0], torch.tensor(1.0))


def test_astrosticks_fixed_matches_free(grad_astro):
    D = 2.0
    m_free = Astrosticks()
    m_fixed = Astrosticks(fixed_D_par=D)

    params = torch.tensor([[D]], dtype=torch.float32)

    S_free = m_free(grad_astro, params)
    S_fixed = m_fixed(grad_astro, params)

    assert torch.allclose(S_free, S_fixed, rtol=1e-6, atol=1e-7)


# -----------------------
# Cylinder
# -----------------------

def test_cylinder_attributes():
    m = Cylinder(n_roots=10)
    assert m.n_parameters == 4
    assert m.parameter_names == ["theta", "phi", "D_par", "radius"]
    assert m.spherical_mean is False


def test_cylinder_forward_shape_and_finite(grad_cylinder):
    m = Cylinder(n_roots=10)
    # theta=pi/2, phi=0 -> axis along x
    params = torch.tensor([[torch.pi/2, 0.0, 1.0, 5.0]], dtype=torch.float32)

    S = m(grad_cylinder, params)

    assert S.shape == (1, grad_cylinder.bvalues.numel())
    assert torch.isfinite(S).all()
    assert ((S > 0) & (S <= 1)).all()


def test_cylinder_radius_must_be_positive(grad_cylinder):
    m = Cylinder(n_roots=10)
    params = torch.tensor([[torch.pi/2, 0.0, 1.0, 0.0]], dtype=torch.float32)

    with pytest.raises(ValueError):
        m(grad_cylinder, params)
