import pytest
import torch

from microtorch.utils.network_constraints import fraction_squash, squash


class DummyModelFunc:
    def __init__(self, n_fractions):
        self.n_fractions = n_fractions


def test_squash_clamp_restricts_to_range():
    param = torch.tensor([[-5.0], [0.5], [10.0]])
    out = squash(param, "clamp", p_min=0.0, p_max=1.0)

    assert out.shape == (3,)
    assert torch.allclose(out, torch.tensor([0.0, 0.5, 1.0]))


def test_squash_sigmoid_stays_within_range():
    param = torch.linspace(-10, 10, steps=9).unsqueeze(1)
    out = squash(param, "sigmoid", p_min=1.0, p_max=2.0)

    assert out.shape == (9,)
    assert torch.all(out >= 1.0) and torch.all(out <= 2.0)
    # monotonically increasing with the input logits
    assert torch.all(out[1:] >= out[:-1])


def test_squash_free_returns_input_unchanged():
    param = torch.tensor([[-3.0], [0.0], [3.0]])
    out = squash(param, "free", p_min=0.0, p_max=1.0)

    assert torch.allclose(out, param.squeeze(1))


def test_squash_unsupported_method_raises():
    param = torch.zeros(2, 1)
    with pytest.raises(ValueError):
        squash(param, "not-a-method", p_min=0.0, p_max=1.0)


def test_fraction_squash_softmax_sums_to_one():
    logits = torch.randn(5, 3)
    modelfunc = DummyModelFunc(n_fractions=3)

    fractions = fraction_squash("softmax", logits, modelfunc)

    assert fractions.shape == (5, 3)
    assert torch.all(fractions >= 0.0) and torch.all(fractions <= 1.0)
    assert torch.allclose(fractions.sum(dim=1), torch.ones(5), atol=1e-5)


def test_fraction_squash_clamp_two_compartments():
    logits = torch.tensor([[-1.0], [0.5], [2.0]])
    modelfunc = DummyModelFunc(n_fractions=1)

    fractions = fraction_squash("clamp", logits, modelfunc)

    assert fractions.shape == (3, 2)
    assert torch.allclose(fractions.sum(dim=1), torch.ones(3), atol=1e-5)
    assert torch.all(fractions >= 0.0) and torch.all(fractions <= 1.0)


def test_fraction_squash_clamp_multiple_compartments():
    logits = torch.tensor([[0.2, 0.3, 0.1], [0.5, 0.5, 0.5]])
    modelfunc = DummyModelFunc(n_fractions=3)

    fractions = fraction_squash("clamp", logits, modelfunc)

    assert fractions.shape == (2, 3)
    assert torch.allclose(fractions.sum(dim=1), torch.ones(2), atol=1e-5)
    assert torch.all(fractions >= 0.0)


def test_fraction_squash_free_returns_raw_logits():
    logits = torch.tensor([[0.2, -0.3]])
    modelfunc = DummyModelFunc(n_fractions=2)

    fractions = fraction_squash("free", logits, modelfunc)

    assert torch.equal(fractions, logits)


def test_fraction_squash_unsupported_method_raises():
    logits = torch.zeros(2, 2)
    modelfunc = DummyModelFunc(n_fractions=2)
    with pytest.raises(ValueError):
        fraction_squash("not-a-method", logits, modelfunc)
