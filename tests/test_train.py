import optuna
import pytest
import torch
import torch.nn as nn

from microtorch.train import train


class StubNet(nn.Module):
    """Minimal plain (non-VAE) network matching the interface train() expects."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.network_type = "hidden_dropout_mlp"

    def forward(self, x, return_latent=False):
        return self.linear(x), self.linear(x)


class StubVAENet(nn.Module):
    """Minimal VAE-like network returning (X_pred, params, mu, logvar)."""

    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.network_type = "vae"

    def forward(self, x, return_latent=False):
        X_pred = self.linear(x)
        mu = torch.zeros(x.shape[0], 2)
        logvar = torch.zeros(x.shape[0], 2)
        return X_pred, X_pred, mu, logvar


class AlwaysNanLoss(nn.Module):
    def forward(self, pred, target):
        return torch.tensor(float("nan"))


class FakeTrial:
    def __init__(self, prune):
        self.reported = []
        self._prune = prune

    def report(self, value, step):
        self.reported.append((value, step))

    def should_prune(self):
        return self._prune


def test_train_runs_vae_branch_with_kl_loss():
    torch.manual_seed(0)
    net = StubVAENet(dim=4)
    img = torch.randn(8, 4)

    X_pred, params, best_loss = train(net, img, nn.MSELoss(), num_iters=2, batch_size=4, patience=5)

    assert X_pred.shape == (8, 4)
    assert params.shape == (8, 4)
    assert isinstance(best_loss, float)


def test_train_breaks_on_nan_loss_without_raising():
    torch.manual_seed(0)
    net = StubNet(dim=4)
    img = torch.randn(8, 4)

    # Should not raise even though every batch produces a NaN loss.
    X_pred, params, best_loss = train(
        net, img, AlwaysNanLoss(), num_iters=1, batch_size=4, patience=1
    )

    assert X_pred.shape == (8, 4)


def test_train_reports_to_optuna_trial_and_prunes():
    torch.manual_seed(0)
    net = StubNet(dim=4)
    img = torch.randn(8, 4)
    trial = FakeTrial(prune=True)

    with pytest.raises(optuna.TrialPruned):
        train(net, img, nn.MSELoss(), num_iters=3, batch_size=4, patience=5, trial=trial)

    assert len(trial.reported) >= 1


def test_train_reports_to_optuna_trial_without_pruning():
    torch.manual_seed(0)
    net = StubNet(dim=4)
    img = torch.randn(8, 4)
    trial = FakeTrial(prune=False)

    _, _, best_loss = train(
        net, img, nn.MSELoss(), num_iters=2, batch_size=4, patience=5, trial=trial
    )

    assert len(trial.reported) == 2
    assert isinstance(best_loss, float)
