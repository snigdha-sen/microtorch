import os
import random
from pathlib import Path
from hydra.core.hydra_config import HydraConfig
from copy import deepcopy


import numpy as np
from omegaconf import OmegaConf, open_dict
import torch
import torch.nn as nn
import nibabel as nib
from microtorch.utils.optuna_search import get_model_hyperparams
from microtorch.train import train
from microtorch.model_maker import ModelMaker
from microtorch.net_maker import Net
from microtorch.utils import (
    txt_file_loader,
    acquisition_scheme_loader,
    direction_average,
    img2voxel,
    voxel2img,
    normalise,
    strip_filename,
)

def run_fit(cfg):
    """
    Core fitting routine.
    Expects a Hydra config (DictConfig).
    """

    mlp_activation = {
        'relu': torch.nn.ReLU(),
        'prelu': torch.nn.PReLU(),
        'tanh': torch.nn.Tanh(),
        'elu': torch.nn.ELU(),
    }

    # -----------------------
    # Seeding
    # -----------------------
    if cfg.training.seed is None:
        cfg.training.seed = random.randint(1, int(1e6))

    torch.manual_seed(cfg.training.seed)
    torch.cuda.manual_seed_all(cfg.training.seed)

    # -----------------------
    # Model setup
    # -----------------------
    modelfunc = ModelMaker(cfg.model.name)

    # -----------------------
    # Acquisition
    # -----------------------
    model_grad = cfg.acquisition.model_grad.get(cfg.model.name)

    if cfg.acquisition.bvals is not None:
        grad = txt_file_loader(
            cfg.acquisition.bvals,
            cfg.acquisition.bvecs,
            cfg.acquisition.delta,
            cfg.acquisition.smalldelta,
            cfg.acquisition.TE,
            cfg.acquisition.bdelta,
        )
        print(f"Loaded acquisition scheme from separate bvals, bvecs, etc files.")
    elif cfg.acquisition.grad is not None:
        grad = acquisition_scheme_loader(cfg.acquisition.grad)
        print(f"Loaded acquisition scheme from {cfg.acquisition.grad}")
    elif model_grad is not None:
        grad = acquisition_scheme_loader(model_grad)
        print(f"Loaded default acquisition scheme for model {cfg.model.name}: {model_grad}")
    else:
        raise ValueError(
            f"No acquisition scheme found for model '{cfg.model.name}'. "
            "Provide acquisition.bvals/bvecs, acquisition.grad, "
            "or add a default gradient file for this model to src/microtorch/conf/acquisition/default.yaml."
        )

    # -----------------------
    # Load image & mask
    # -----------------------
    img = torch.from_numpy(
        nib.load(os.path.join(cfg.data.folder, cfg.data.image))
        .get_fdata()
        .astype(np.float32)
    )

    if cfg.data.mask is None:
        mask = torch.ones(img.shape[:3], dtype=torch.float32)
    else:
        mask = torch.from_numpy(
            nib.load(os.path.join(cfg.data.folder, cfg.data.mask))
            .get_fdata()
            .astype(np.float32)
        )

    # -----------------------
    # Direction averaging
    # -----------------------
    if modelfunc.spherical_mean:
        img, grad = direction_average(img, grad)

    # -----------------------
    # Preprocessing
    # -----------------------
    X_train, maskvox = img2voxel(img, mask)
    X_train = X_train + 1e-16
    X_train = normalise(X_train, grad)

    # -----------------------
    # Network
    # -----------------------
    lossfunc = nn.MSELoss()


    hyperparams = get_model_hyperparams(
        grad=grad,
        modelfunc=modelfunc,
        mlp_activation=mlp_activation,
        X_train=X_train,
        cfg=cfg)

    net = Net(
        grad,
        modelfunc,
        input_neurons=grad.number_of_measurements,
        layer_dims=hyperparams["hidden_size"],
        n_layers=hyperparams["num_layers"],
        dropout_fraction=hyperparams["dropout_frac"],
        network_type=cfg.training.network_type,
        clipping_method=cfg.training.clip,
        clipping_method_fraction=cfg.training.clip_fraction,
        activation=mlp_activation[hyperparams["activation"]],
    )

    # -----------------------
    # Train
    # -----------------------
    _, params, _ = train(
        net,
        X_train,
        lossfunc,
        lr=hyperparams["lr"],
        batch_size=256,
        num_iters=cfg.training.num_iters,
        patience=hyperparams["patience"],
    )

    # -----------------------
    # Reconstruct parameter maps
    # -----------------------
    param_map = np.zeros(
        (*mask.shape, modelfunc.n_parameters + modelfunc.n_fractions)
    )

    for i in range(param_map.shape[-1]):
        param_map[..., i] = voxel2img(
            params[:, i], maskvox, mask.shape
        )

    # -----------------------
    # Save output
    # -----------------------
    output_folder = Path(HydraConfig.get().run.dir)
    output_folder.mkdir(parents=True, exist_ok=True)

    img_nii = nib.load(os.path.join(cfg.data.folder, cfg.data.image))
    new_img = nib.Nifti1Image(param_map, img_nii.affine, img_nii.header)

    out_file = output_folder / (
        strip_filename(cfg.data.image) + "_param_maps.nii.gz"
    )
    nib.save(new_img, out_file)

    # -----------------------
    # Save the used config.yaml 
    # -----------------------
    
    # if tuning wasn't done, remove the tuning section to avoid confusion
    cfg_to_save = deepcopy(cfg) 
    if not cfg.training.tune == "optuna_tuner" and "tuning" in cfg_to_save:
        with open_dict(cfg_to_save):
            del cfg_to_save["tuning"]
            print("Removed tuning section from saved config since tuning was not performed")

    output_config_path = Path(output_folder) / f"{strip_filename(cfg.data.image)}_config.yaml"
    output_config_path.write_text(OmegaConf.to_yaml(cfg_to_save, resolve=True))

    return param_map, modelfunc, out_file
