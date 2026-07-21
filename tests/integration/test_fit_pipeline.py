"""
End-to-end integration test for the model-fitting pipeline described in the paper:
simulate a known ground truth, fit it with the self-supervised network, and check
that the fitted parameters recover the ground truth.

This exercises the full stack used by the `microtorch` CLI (main.py -> run_fit.py),
including ModelMaker, Net, the training loop, and the pre/post-processing utilities,
without going through the command line or writing to the repository itself.
"""

import nibabel as nib
import numpy as np
import torch
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from microtorch.run_fit import run_fit
from microtorch.utils.make_test_image import make_test_image
from microtorch.utils.paths import GRAD_PATH


def test_fit_recovers_ground_truth_parameters(tmp_path):
    torch.manual_seed(42)
    np.random.seed(42)

    make_test_image(
        model="Ball",
        grad_file=str(GRAD_PATH / "grad_HCP_with_deltas.txt"),
        nx=32,
        ny=16,
        nz=1,
        savedir=str(tmp_path),
        snr=50,
    )
    data_dir = tmp_path / "Ball"

    with initialize(version_base=None, config_path="../../src/microtorch/conf"):
        cfg = compose(
            config_name="config",
            overrides=[
                "model=default",
                "model.name=Ball",
                f"data.folder={data_dir}",
                "data.image=Ball_Ball_data.nii.gz",
                "data.mask=Ball_Ball_mask.nii.gz",
                "training.tune=default",
                "training.num_iters=100",
                "training.patience=20",
                "training.seed=42",
            ],
            return_hydra_config=True,
        )
        HydraConfig.instance().set_config(cfg)
        with open_dict(cfg):
            del cfg["hydra"]

        param_map, modelfunc, out_file = run_fit(cfg, output_folder=tmp_path / "out")

    assert out_file.exists()
    assert modelfunc.parameter_names == ["D"]
    assert param_map.shape == (32, 16, 1, 1)

    ground_truth = nib.load(data_dir / "Ball_Ball_params.nii.gz").get_fdata()

    fitted_D = param_map[..., 0].flatten()
    true_D = ground_truth[..., 0].flatten()

    correlation = np.corrcoef(fitted_D, true_D)[0, 1]
    mean_absolute_error = np.mean(np.abs(fitted_D - true_D))

    # The Ball model's parameter range is [0.001, 3], so a MAE of 0.3 corresponds
    # to ~10% of the full range - a loose bound that tolerates run-to-run training
    # noise while still confirming the network learned the correct mapping.
    assert correlation > 0.85, f"Fitted D does not correlate with ground truth ({correlation=})"
    assert mean_absolute_error < 0.3, f"Fitted D is far from ground truth ({mean_absolute_error=})"
