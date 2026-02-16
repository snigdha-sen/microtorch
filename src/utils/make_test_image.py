from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from multiprocessing import freeze_support

import numpy as np
import torch


def generate_random_params(modelfunc, num_samples: int, alpha: torch.Tensor | None = None) -> torch.Tensor:
    """
    Generate random parameters for a model.

    Non-fraction parameters are sampled uniformly within `modelfunc.parameter_ranges`.
    Volume fractions are sampled from a Dirichlet distribution, with the last fraction
    implicit as 1 - sum(previous).

    Returns:
        Tensor of shape [num_samples, modelfunc.n_parameters + modelfunc.n_fractions]
        where the final column corresponds to the implicit-last-fraction *not included*.
        (i.e. we keep only the first n_fractions fractions).
    """
    # Parameter ranges -> tensors
    min_vals = torch.as_tensor(modelfunc.parameter_ranges[:, 0], dtype=torch.float32)
    max_vals = torch.as_tensor(modelfunc.parameter_ranges[:, 1], dtype=torch.float32)

    # Uniform sampling for non-fraction parameters
    model_params = torch.rand(num_samples, modelfunc.n_parameters, dtype=torch.float32)
    model_params = model_params * (max_vals - min_vals) + min_vals

    # Dirichlet sampling for volume fractions (+1 for implicit last fraction)
    if alpha is None:
        alpha = torch.ones(modelfunc.n_fractions + 1, dtype=torch.float32)
    else:
        alpha = alpha.to(dtype=torch.float32)

    dirichlet = torch.distributions.Dirichlet(alpha)
    dirichlet_samples = dirichlet.sample((num_samples,))  # [N, n_fractions + 1]

    # Keep only first n_fractions (last is implicit)
    fractions = dirichlet_samples[:, :-1]  # [N, n_fractions]

    return torch.cat([model_params, fractions], dim=1).to(dtype=torch.float32)


def generate_smooth_params(modelfunc, nvox: int = 100_000) -> torch.Tensor:
    """
    Generate a dense grid over parameter ranges (including [0,1] for each fraction),
    producing ~nvox total samples.

    Note: This ignores the "sum-to-1" constraint for fractions; it's just a grid.
    """
    ranges = np.asarray(modelfunc.parameter_ranges, dtype=float)

    # Append [0, 1] ranges for each explicit volume fraction parameter
    ranges = np.vstack([ranges, np.array([[0.0, 1.0]] * modelfunc.n_fractions, dtype=float)])

    nparam = modelfunc.n_parameters + modelfunc.n_fractions
    if ranges.shape[0] != nparam:
        raise ValueError(
            "The number of parameter ranges does not match "
            "modelfunc.n_parameters + modelfunc.n_fractions."
        )

    samples_per_dim = int(np.ceil(nvox ** (1 / nparam)))
    grid_1d = [np.linspace(lo, hi, samples_per_dim) for (lo, hi) in ranges]
    mesh = np.meshgrid(*grid_1d, indexing="ij")
    params = np.stack(mesh, axis=-1).reshape(-1, nparam)

    return torch.tensor(params, dtype=torch.float32)


def factorize_close(n: int) -> tuple[int, int]:
    """Factorize n into two factors as close as possible (for near-square layouts)."""
    root = int(math.isqrt(n))
    for i in range(root, 0, -1):
        if n % i == 0:
            return i, n // i
    return 1, n


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="BallStick",
        help=(
            "Compartmental model to use (e.g. verdict, sandi, or combinations of "
            "ball; sphere; stick; astrosticks; cylinder; astrocylinders; zeppelin; "
            "astrozeppelins; dot)."
        ),
    )
    parser.add_argument("-bvals", "--bvals", default=None, help="bval file (FSL format) in [s/mm^2]")
    parser.add_argument("-bvecs", "--bvecs", default=None, help="bvec file (FSL format)")
    parser.add_argument("-g", "--grad", default="simulation_data/grad/grad_HCP.txt", help="grad file (mrtrix format)")
    parser.add_argument("-d", "--delta", type=float, default=24.0, help="gradient pulse separation (ms)")
    parser.add_argument("-sd", "--smalldelta", type=float, default=8.0, help="gradient pulse duration (ms)")
    parser.add_argument("-TE", "--TE", default="", help="echo time (ms)")
    parser.add_argument("-TR", "--TR", default="", help="repetition time (ms)")
    parser.add_argument("-TI", "--TI", default="", help="inversion time (ms)")
    parser.add_argument("-nparam", "--nparam", type=int, default=100, help="number of random parameters to sample")
    parser.add_argument("-nx", "--nx", type=int, default=128, help="voxels in x")
    parser.add_argument("-ny", "--ny", type=int, default=128, help="voxels in y")
    parser.add_argument("-nz", "--nz", type=int, default=2, help="voxels in z")
    parser.add_argument("-savedir", "--savedir", default="simulation_data/data", help="output directory")
    parser.add_argument("-bd", "--bdelta", type=float, default=1.0, help="shape of gradient pulse")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Add parent directory to Python path so ModelMaker can be imported
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from src.utils.acquisition_scheme import acquisition_scheme_loader, txt_file_loader
    from model_maker import ModelMaker

    modelfunc = ModelMaker(args.model)

    # Load acquisition scheme
    if args.bvals is not None:
        grad = txt_file_loader(args.bvals, args.bvecs, args.delta, args.smalldelta, args.TE, args.bdelta)
    elif args.grad is not None:
        grad = acquisition_scheme_loader(args.grad)
    else:
        raise ValueError("Provide either --bvals/--bvecs or --grad.")

    # Sample parameters for each voxel
    nvox = args.nx * args.ny * args.nz
    params = generate_random_params(modelfunc, num_samples=nvox)

    # Evaluate model
    S = modelfunc(grad, params)

    # Reshape to images
    dim = (args.nx, args.ny, args.nz)
    Simg = S.view(*dim, grad.number_of_measurements)
    paramsimg = params.view(*dim, params.size(-1))
    mask = torch.ones_like(Simg[..., 0])

    # Output paths
    out_dir = os.path.join(args.savedir, args.model)
    os.makedirs(out_dir, exist_ok=True)

    import nibabel as nib

    base = f"{args.model}_{''.join(modelfunc.compartment_names)}"
    nib.save(nib.Nifti1Image(Simg.cpu().numpy(), np.eye(4)), os.path.join(out_dir, f"{base}_data.nii.gz"))
    nib.save(nib.Nifti1Image(paramsimg.cpu().numpy(), np.eye(4)), os.path.join(out_dir, f"{base}_params.nii.gz"))
    nib.save(nib.Nifti1Image(mask.cpu().numpy(), np.eye(4)), os.path.join(out_dir, f"{base}_mask.nii.gz"))


if __name__ == "__main__":
    freeze_support()
    main()
