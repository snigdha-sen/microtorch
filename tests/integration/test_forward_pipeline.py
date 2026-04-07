import pytest
import nibabel as nib
import numpy as np
from omegaconf import OmegaConf
from unittest.mock import patch

from microtorch.run_fit import run_fit

@pytest.mark.integration
def test_run_fit(tmp_path): ## this doesnt work

    data_folder = "/Users/cig/Desktop/microtorch/simulation_data/data/BallStick"
    signal_file = "BallStick_BallStick_data.nii.gz"
    mask_file = "BallStick_BallStick_mask.nii.gz"
    gt_file = "BallStick_BallStick_params.nii.gz"
    grad_file = "/Users/cig/Desktop/microtorch/simulation_data/grad/grad_HCP_with_deltas.txt"

    signal_img = nib.load(f"{data_folder}/{signal_file}").get_fdata().astype(np.float32)
    gt_params = nib.load(f"{data_folder}/{gt_file}").get_fdata().astype(np.float32)

    cfg_dict = {
        "data": {
            "folder": data_folder,
            "image": signal_file,
            "mask": mask_file,
        },
        "acquisition": {
            "grad": grad_file,
            "bvals": None,
            "bvecs": None,
            "delta": None,
            "smalldelta": None,
            "TE": None,
            "bdelta": None,
        },
        "model": {"name": "BallStick"},
        "training": {"seed": 42, "num_iters": 10},
        "plot": {"enabled": False},
    }
    cfg = OmegaConf.create(cfg_dict)

    class DummyHydra:
        runtime = type("Runtime", (), {"output_dir": str(tmp_path)})()
    with patch("hydra.core.hydra_config.HydraConfig.get", return_value=DummyHydra()):
        param_map, modelfunc, out_file = run_fit(cfg, output_folder=tmp_path)

    # Assertions
    assert param_map.shape[:3] == signal_img.shape[:3], "Spatial shape mismatch"
    assert param_map.shape[-1] == gt_params.shape[-1], "Parameter count mismatch"
    np.testing.assert_allclose(param_map, gt_params, rtol=0.1, atol=1e-4)