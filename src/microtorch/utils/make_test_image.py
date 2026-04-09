from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from multiprocessing import freeze_support
import torch
import numpy as np


from .acquisition_scheme import acquisition_scheme_loader, txt_file_loader
from microtorch.model_maker import ModelMaker

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, help="Compartmental Model to use. Implemented are verdict, sandi, or user defined ones form combinations of ball; sphere, stick; astrosticks; cylinder; astrocylinders; zeppelin; astrozeppelins; dot.", default="BallStick")
    parser.add_argument("-bvals", "--bvals", help="bval file in FSL format and in [s/mm2]", default=None)
    parser.add_argument("-bvecs", "--bvecs", help="bvec file in FSL format", default=None)
    parser.add_argument("-g", "--grad", help="grad file in mrtrix format", default="simulation_data/grad/grad_HCP.txt")
    parser.add_argument("-d", "--delta", help="gradient pulse separation in ms", default=24, type=float)
    parser.add_argument("-sd", "--smalldelta", help="gradient pulse duration in ms", default=8, type=float)
    parser.add_argument("-TE", "--TE", help="echo time in ms", default="")
    parser.add_argument("-TR", "--TR", help="repetition time in ms", default="")
    parser.add_argument("-TI", "--TI", help="inversion time in ms", default="")
    parser.add_argument("-nparam", "--nparam", help="number of random parameters to sample", type=int, default=100)
    parser.add_argument("-nx", "--nx", help="number of voxels in x dimension", type=int, default=128)
    parser.add_argument("-ny", "--ny", help="number of voxels in y dimension", type=int, default=128)
    parser.add_argument("-nz", "--nz", help="number of voxels in z dimension", type=int, default=2)
    parser.add_argument("-savedir", "--savedir", help="directory to save the images", default="simulation_data/data")
    parser.add_argument("-bd",  "--bdelta",     help="shape of gradient pulse", default=1, type=float)

    args = parser.parse_args()

    model = args.model

    # Add parent directory to Python path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    #make the model function that will be incorporated into the net
    modelfunc = ModelMaker(args.model)

    #load the acquisition scheme in
    if args.bvals is not None:
        grad = txt_file_loader(args.bvals, args.bvecs, args.delta, args.smalldelta, args.TE, args.bdelta)
    elif args.grad is not None:
        grad = acquisition_scheme_loader(args.grad)
                                               
    #generate random parameters for the model function
    params = generate_random_params(modelfunc, num_samples=args.nx * args.ny * args.nz)

    #simulate the signal for each set of parameters using the model function
    S = modelfunc(grad, params)
    
    # Add Rician noise to the signal
    S = add_rician_noise(S, snr=20.0) #you can adjust the SNR as needed
    
    dim = (args.nx, args.ny , args.nz) #set the dimensions of the image based on the command line arguments

    #reshape the signal and parameters to the dimensions of the image
    Simg = S.view(*dim, grad.number_of_measurements)
    paramsimg = params.view(*dim, params.size(-1))
    #create a mask of ones with the same dimensions as the image
    mask = torch.ones_like(Simg[:, :, :, 0])

    #make a directory for saving the simulated images
    os.makedirs(os.path.join(args.savedir,args.model), exist_ok=True )
    
    #save the image, parameters and mask using nibabel
    import nibabel as nib

    base = f"{args.model}_{''.join(modelfunc.compartment_names)}"
    nib.save(nib.Nifti1Image(Simg.cpu().numpy(), np.eye(4)), os.path.join(args.savedir, args.model, f"{base}_data.nii.gz"))
    nib.save(nib.Nifti1Image(paramsimg.cpu().numpy(), np.eye(4)), os.path.join(args.savedir, args.model, f"{base}_params.nii.gz"))
    nib.save(nib.Nifti1Image(mask.cpu().numpy(), np.eye(4)), os.path.join(args.savedir, args.model, f"{base}_mask.nii.gz"))


#helper functions 

def add_rician_noise(data: torch.Tensor, snr: float = 20.0) -> torch.Tensor:
    """
    Add Rician noise to MRI magnitude data.    
    """
    if snr <= 0:
        return data

    scale = 1.0 / snr

    noise_real = torch.randn_like(data) * scale
    noise_imag = torch.randn_like(data) * scale

    data_real = data + noise_real
    data_imag = noise_imag

    data_noisy = torch.sqrt(data_real**2 + data_imag**2)

    return data_noisy


def generate_random_params(modelfunc, num_samples, alpha=None):
    """
    Generate random parameters for a given model as a tensor, with volume fractions sampled
    from a Dirichlet distribution.

    Args:
        modelfunc: Model function with `parameter_ranges` and `n_fractions` attributes.
        num_samples: Number of parameter sets to generate.
        alpha: Optional Dirichlet concentration parameters (1D tensor of length n_fractions).
               Defaults to all ones (uniform Dirichlet).

    Returns:
        params: Tensor of shape [num_samples, num_model_params + num_fractions - 1]
                with random parameters. The last volume fraction is implicit
                (1 - sum of the others).
    """

    if not modelfunc.parameter_names == []:
        # Convert parameter ranges to tensors    
        min_vals = torch.tensor(modelfunc.parameter_ranges[:, 0], dtype=torch.float32)
        max_vals = torch.tensor(modelfunc.parameter_ranges[:, 1], dtype=torch.float32)

        # Generate random values for non-fraction parameters
        model_params = torch.rand(num_samples, modelfunc.n_parameters) * (max_vals - min_vals) + min_vals
    else:
        model_params = torch.empty(num_samples, 1)  # if no model parameters, e.g. dot compartment, return an empty tensor with the correct number of samples

    if modelfunc.n_fractions > 1:
        # Generate volume fractions from Dirichlet
        if alpha is None:
            alpha = torch.ones(modelfunc.n_fractions) 
            dirichlet_samples = torch.distributions.Dirichlet(alpha).sample((num_samples,))  # shape [num_samples, num_fractions]

            # Keep all the fractions
            fractions = dirichlet_samples

            # Concatenate model parameters + fractions
            params = torch.cat([model_params, fractions], dim=1)
    else:
        params = model_params    


    return params.float()




if __name__ == "__main__":
    freeze_support()
    main()
