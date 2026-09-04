import pytest
import torch
import yaml

import microtorch.model_maker as model_maker_module
from microtorch.model_maker import ModelMaker


def _bvecs_bvals(n=8):
    torch.manual_seed(0)
    bvecs = torch.randn(n, 3)
    bvecs = bvecs / bvecs.norm(dim=1, keepdim=True)
    bvalues = torch.linspace(0.1, 3.0, n)
    return bvecs, bvalues


class DummyGrad:
    def __init__(self, bvecs, bvalues):
        self.bvecs = bvecs
        self.bvalues = bvalues
        self.number_of_measurements = bvalues.shape[0]


def test_single_compartment_model_has_no_fractions():
    modelfunc = ModelMaker("Ball")

    assert modelfunc.n_compartments == 1
    assert modelfunc.n_fractions == 0
    assert modelfunc.parameter_names == ["D"]
    assert modelfunc.compartment_indices == [0]
    assert modelfunc.parameter_indices == ([0],)


def test_single_compartment_call_matches_compartment_directly():
    modelfunc = ModelMaker("Ball")
    bvecs, bvalues = _bvecs_bvals()
    grad = DummyGrad(bvecs, bvalues)

    params = torch.tensor([[1.5]])
    expected = modelfunc.compartments[0](grad, params)
    actual = modelfunc(grad, params)

    assert torch.allclose(actual, expected)


def test_multi_compartment_fallback_parsing_and_indices():
    modelfunc = ModelMaker("BallStick")

    assert modelfunc.compartment_names == ["Ball", "Stick"]
    assert modelfunc.n_compartments == 2
    assert modelfunc.n_fractions == 2
    # Ball has 1 param (D), Stick has 3 (Dpar, theta, phi)
    assert modelfunc.n_parameters == 4
    assert modelfunc.parameter_indices == ([0], [1, 2, 3])
    assert modelfunc.compartment_indices == [0, 1, 1, 1, 0, 1]
    assert modelfunc.parameter_names[-2:] == ["f_0", "f_1"]


def test_multi_compartment_call_combines_weighted_signals():
    modelfunc = ModelMaker("BallStick")
    bvecs, bvalues = _bvecs_bvals()
    grad = DummyGrad(bvecs, bvalues)

    # params: [D, Dpar, theta, phi, f_ball, f_stick]
    params = torch.tensor([[1.0, 1.0, 0.3, 0.6, 0.4, 0.6]])

    ball_signal = modelfunc.compartments[0](grad, params[:, [0]])
    stick_signal = modelfunc.compartments[1](grad, params[:, [1, 2, 3]])
    expected = 0.4 * ball_signal + 0.6 * stick_signal

    actual = modelfunc(grad, params)

    assert torch.allclose(actual, expected, atol=1e-5)


def test_yaml_configured_model_with_parameter_range_overrides():
    modelfunc = ModelMaker("IVIM")

    assert modelfunc.compartment_names == ["Ball", "Ball"]
    assert list(modelfunc.parameter_ranges[0]) == [1.0e-03, 3.0]
    assert list(modelfunc.parameter_ranges[1]) == [3.0, 30.0]


def test_yaml_configured_model_without_overrides_uses_defaults():
    modelfunc = ModelMaker("SANDI")

    assert modelfunc.compartment_names == ["Ball", "Sphere", "Astrosticks"]
    assert modelfunc.n_compartments == 3
    assert modelfunc.n_fractions == 3


def test_inconsistent_spherical_mean_raises():
    # Stick is not spherically averaged, Astrosticks is - mixing them is invalid.
    with pytest.raises(ValueError):
        ModelMaker("StickAstrosticks")


def test_invalid_yaml_parameter_ranges_length_raises(tmp_path, monkeypatch):
    bad_config = {
        "name": "BadModel",
        "compartments": [
            {"class": "Ball", "parameter_ranges": [[0.0, 1.0], [0.0, 1.0]]},
        ],
    }
    (tmp_path / "BadModel.yaml").write_text(yaml.safe_dump(bad_config))
    monkeypatch.setattr(model_maker_module, "MODELS_CONF_PATH", tmp_path)

    with pytest.raises(ValueError, match="Invalid number of parameter ranges"):
        ModelMaker("BadModel")


def test_invalid_yaml_parameter_range_shape_raises(tmp_path, monkeypatch):
    bad_config = {
        "name": "BadModel2",
        "compartments": [
            {"class": "Ball", "parameter_ranges": [[0.0, 1.0, 2.0]]},
        ],
    }
    (tmp_path / "BadModel2.yaml").write_text(yaml.safe_dump(bad_config))
    monkeypatch.setattr(model_maker_module, "MODELS_CONF_PATH", tmp_path)

    with pytest.raises(ValueError, match="Invalid parameter range"):
        ModelMaker("BadModel2")


def test_modelmaker_parses_ball_t2():
    model = ModelMaker("Ballt2")

    assert model.compartment_names == ["Ballt2"]


def test_modelmaker_parses_ball_t1_t2():
    model = ModelMaker("Ballt1t2")

    assert model.compartment_names == ["Ballt1t2"]


def test_modelmaker_parses_multicompartment_relaxation():
    model = ModelMaker("Ballt2Ballt2")

    assert model.compartment_names == [
        "Ballt2",
        "Ballt2",
    ]