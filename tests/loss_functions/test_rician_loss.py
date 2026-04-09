import torch

from microtorch.loss_functions.RicianLoss import RicianLoss, RicianLossStable

def test_rician_loss_finite():
    loss_fn = RicianLoss(sigma=0.1)

    inputs = torch.rand(8, 10) + 0.1
    preds  = torch.rand(8, 10) + 0.1

    loss = loss_fn(preds, inputs)

    assert torch.isfinite(loss)

def test_rician_loss_finite():
    loss_fn = RicianLoss(sigma=0.1)

    inputs = torch.rand(8, 10) + 0.1
    preds  = torch.rand(8, 10) + 0.1

    loss = loss_fn(preds, inputs)

    assert torch.isfinite(loss)

def test_loss_lower_when_prediction_matches_input():
    loss_fn = RicianLossStable(sigma=0.1)

    inputs = torch.rand(8, 10) + 0.1
    good_preds = inputs.clone()
    bad_preds  = torch.zeros_like(inputs)

    loss_good = loss_fn(good_preds, inputs)
    loss_bad  = loss_fn(bad_preds, inputs)

    assert loss_good < loss_bad

def test_rician_loss_has_gradients():
    loss_fn = RicianLossStable(sigma=0.1)

    inputs = torch.rand(8, 10) + 0.1
    preds  = torch.rand(8, 10, requires_grad=True)

    loss = loss_fn(preds, inputs)
    loss.backward()

    assert preds.grad is not None
    assert torch.isfinite(preds.grad).all()

def test_stable_loss_is_finite_for_extreme_inputs():
    loss_fn = RicianLossStable(sigma=0.1)

    inputs = torch.tensor([[0.0, 1e-12, 1e6]])
    preds  = torch.tensor([[0.0, 1e-12, 1e6]])

    loss = loss_fn(preds, inputs)

    assert torch.isfinite(loss)


