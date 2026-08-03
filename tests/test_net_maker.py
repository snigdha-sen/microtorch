import torch

from microtorch.model_maker import ModelMaker
from microtorch.net_maker import Net
from microtorch.utils.acquisition_scheme import AcquisitionScheme


def _make_grad(n_measurements=12):
    torch.manual_seed(0)
    bvecs = torch.randn(n_measurements, 3)
    bvecs = bvecs / bvecs.norm(dim=1, keepdim=True)
    bvalues = torch.linspace(0.1, 3.0, n_measurements)
    return AcquisitionScheme(bvalues=bvalues, bvecs=bvecs)


def test_net_forward_single_compartment_shapes_and_ranges():
    grad = _make_grad()
    modelfunc = ModelMaker("Ball")

    net = Net(
        grad,
        modelfunc,
        input_neurons=grad.number_of_measurements,
        layer_dims=8,
        n_layers=2,
        dropout_fraction=0.0,
        clipping_method="clamp",
    )
    net.eval()

    X = torch.rand(5, grad.number_of_measurements)
    X_pred, params = net(X)

    assert X_pred.shape == (5, grad.number_of_measurements)
    assert params.shape == (5, modelfunc.n_parameters)

    d_min, d_max = modelfunc.parameter_ranges[0]
    assert torch.all(params[:, 0] >= d_min - 1e-5)
    assert torch.all(params[:, 0] <= d_max + 1e-5)


def test_net_forward_multi_compartment_fractions_sum_to_one():
    grad = _make_grad()
    modelfunc = ModelMaker("BallStick")

    net = Net(
        grad,
        modelfunc,
        input_neurons=grad.number_of_measurements,
        layer_dims=8,
        n_layers=2,
        dropout_fraction=0.1,
        clipping_method="clamp",
        clipping_method_fraction="softmax",
    )
    net.eval()

    X = torch.rand(4, grad.number_of_measurements)
    X_pred, params = net(X)

    assert X_pred.shape == (4, grad.number_of_measurements)
    assert params.shape == (4, modelfunc.n_parameters + modelfunc.n_fractions)

    frac_start = modelfunc.n_parameters
    fractions = params[:, frac_start:]
    assert torch.allclose(fractions.sum(dim=1), torch.ones(4), atol=1e-4)


def test_net_forward_dev_mlp_network_type_runs():
    grad = _make_grad()
    modelfunc = ModelMaker("Ball")

    net = Net(
        grad,
        modelfunc,
        input_neurons=grad.number_of_measurements,
        layer_dims=8,
        n_layers=2,
        dropout_fraction=0.0,
        network_type="dev_mlp",
        clipping_method="clamp",
    )
    net.eval()

    X = torch.rand(3, grad.number_of_measurements)
    X_pred, params = net(X)

    assert X_pred.shape == (3, grad.number_of_measurements)
    assert params.shape == (3, modelfunc.n_parameters)
    assert torch.isfinite(X_pred).all()
