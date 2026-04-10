from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from multiprocessing import freeze_support

import torch
import numpy as np

from .acquisition_scheme import acquisition_scheme_loader, txt_file_loader
from microtorch.model_maker import ModelMaker


def make_test_image(
    model: str = "BallStick",
    bvals: str | None = None,
    bvecs: str | None = None,
    grad_file: str | None = "simulation_data/grad/grad_HCP.txt",
    delta: float = 24,
    smalldelta: float = 8,
    TE: str = "",
    TR: str = "",
    TI: str = "",
    nx: int = 128,
    ny: int = 128,
    nz: int = 2,
    savedir: str = "simulation_data/data",
    bdelta: float = 1,
    snr: float = 20.0,
):
    """
    Generate a simulated diffusion MRI test image, parameters, and mask.
    """

    # Add parent directory to Python path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    # Create model function
    modelfunc = ModelMaker(model)

    # Load acquisition scheme
    if bvals is not None:
        grad = txt_file_loader(bvals, bvecs, delta, smalldelta, TE, bdelta)
    elif grad_file is not None:
        grad = acquisition_scheme_loader(grad_file)
    else:
        raise ValueError("Either bvals/bvecs or grad_file must be provided.")

    # Generate random parameters
    num_samples = nx * ny * nz
    params = generate_random_params(modelfunc, num_samples=num_samples)

    # Simulate signal
    S = modelfunc(grad, params)

    # Add Rician noise
    S = add_rician_noise(S, snr=snr)

    # Reshape to image
    dim = (nx, ny, nz)
    Simg = S.view(*dim, grad.number_of_measurements)
    paramsimg = params.view(*dim, params.size(-1))
    mask = torch.ones_like(Simg[:, :, :, 0])

    # Save outputs
    os.makedirs(os.path.join(savedir, model), exist_ok=True)

    import nibabel as nib

    base = f"{model}_{''.join(modelfunc.compartment_names)}"

    nib.save(
        nib.Nifti1Image(Simg.cpu().numpy(), np.eye(4)),
        os.path.join(savedir, model, f"{base}_data.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(paramsimg.cpu().numpy(), np.eye(4)),
        os.path.join(savedir, model, f"{base}_params.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(mask.cpu().numpy(), np.eye(4)),
        os.path.join(savedir, model, f"{base}_mask.nii.gz"),
    )

    return Simg, paramsimg, mask


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", default="BallStick")
    parser.add_argument("-bvals", "--bvals", default=None)
    parser.add_argument("-bvecs", "--bvecs", default=None)
    parser.add_argument("-g", "--grad", default="simulation_data/grad/grad_HCP.txt")
    parser.add_argument("-d", "--delta", type=float, default=24)
    parser.add_argument("-sd", "--smalldelta", type=float, default=8)
    parser.add_argument("-TE", "--TE", default="")
    parser.add_argument("-TR", "--TR", default="")
    parser.add_argument("-TI", "--TI", default="")
    parser.add_argument("-nx", "--nx", type=int, default=128)
    parser.add_argument("-ny", "--ny", type=int, default=128)
    parser.add_argument("-nz", "--nz", type=int, default=2)
    parser.add_argument("-savedir", "--savedir", default="simulation_data/data")
    parser.add_argument("-bd", "--bdelta", type=float, default=1)

    args = parser.parse_args()

    make_test_image(
        model=args.model,
        bvals=args.bvals,
        bvecs=args.bvecs,
        grad_file=args.grad,
        delta=args.delta,
        smalldelta=args.smalldelta,
        TE=args.TE,
        TR=args.TR,
        TI=args.TI,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        savedir=args.savedir,
        bdelta=args.bdelta,
    )


# Helper functions

def add_rician_noise(data: torch.Tensor, snr: float = 20.0) -> torch.Tensor:
    if snr <= 0:
        return data

    scale = 1.0 / snr
    noise_real = torch.randn_like(data) * scale
    noise_imag = torch.randn_like(data) * scale

    data_real = data + noise_real
    data_imag = noise_imag

    return torch.sqrt(data_real**2 + data_imag**2)


def generate_random_params(modelfunc, num_samples, alpha=None):
    if modelfunc.parameter_names != []:
        min_vals = torch.tensor(modelfunc.parameter_ranges[:, 0], dtype=torch.float32)
        max_vals = torch.tensor(modelfunc.parameter_ranges[:, 1], dtype=torch.float32)

        model_params = torch.rand(num_samples, modelfunc.n_parameters) * (max_vals - min_vals) + min_vals
    else:
        model_params = torch.empty(num_samples, 1)

    if modelfunc.n_fractions > 1:
        if alpha is None:
            alpha = torch.ones(modelfunc.n_fractions)

        fractions = torch.distributions.Dirichlet(alpha).sample((num_samples,))
        params = torch.cat([model_params, fractions], dim=1)
    else:
        params = model_params

    return params.float()


if __name__ == "__main__":
    freeze_support()
    main()