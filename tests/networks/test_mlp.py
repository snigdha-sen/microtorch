import pytest
import torch
import torch.nn as nn

from microtorch.networks import NETWORK_REGISTRY, build_network
from microtorch.networks.mlp import DevMLP, HiddenDropoutMLP


@pytest.mark.parametrize("network_cls", [DevMLP, HiddenDropoutMLP])
def test_mlp_forward_shape(network_cls):
    net = network_cls(
        input_neurons=10, layer_dims=8, n_layers=3, dim_out=4, activation=nn.ReLU(), dropout=0.0
    )
    x = torch.randn(5, 10)
    out = net(x)

    assert out.shape == (5, 4)


@pytest.mark.parametrize("network_cls", [DevMLP, HiddenDropoutMLP])
def test_mlp_no_dropout_module_when_dropout_zero(network_cls):
    net = network_cls(
        input_neurons=6, layer_dims=4, n_layers=2, dim_out=2, activation=nn.ReLU(), dropout=0.0
    )
    if network_cls is DevMLP:
        assert net.dropout is None
    else:
        assert not any(isinstance(m, nn.Dropout) for m in net.hidden)


@pytest.mark.parametrize("network_cls", [DevMLP, HiddenDropoutMLP])
def test_mlp_dropout_disabled_in_eval_is_deterministic(network_cls):
    net = network_cls(
        input_neurons=6, layer_dims=4, n_layers=2, dim_out=2, activation=nn.ReLU(), dropout=0.5
    )
    net.eval()
    x = torch.randn(3, 6)

    out1 = net(x)
    out2 = net(x)

    assert torch.allclose(out1, out2)


def test_hidden_dropout_mlp_inserts_dropout_after_each_hidden_layer():
    net = HiddenDropoutMLP(
        input_neurons=6, layer_dims=4, n_layers=3, dim_out=2, activation=nn.ReLU(), dropout=0.2
    )
    dropout_count = sum(1 for m in net.hidden if isinstance(m, nn.Dropout))

    assert dropout_count == 3


def test_build_network_known_types():
    for name, cls in NETWORK_REGISTRY.items():
        net = build_network(
            name,
            input_neurons=5,
            layer_dims=4,
            n_layers=1,
            dim_out=2,
            activation=nn.ReLU(),
            dropout=0.0,
        )
        assert isinstance(net, cls)


def test_build_network_unknown_type_raises():
    with pytest.raises(ValueError):
        build_network(
            "not-a-real-network",
            input_neurons=5,
            layer_dims=4,
            n_layers=1,
            dim_out=2,
            activation=nn.ReLU(),
            dropout=0.0,
        )
