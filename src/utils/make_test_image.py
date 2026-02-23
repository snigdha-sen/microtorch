from multiprocessing import freeze_support


import torch

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

    # Convert parameter ranges to tensors
    min_vals = torch.tensor(modelfunc.parameter_ranges[:, 0], dtype=torch.float32)
    max_vals = torch.tensor(modelfunc.parameter_ranges[:, 1], dtype=torch.float32)

    # Generate random values for non-fraction parameters
    model_params = torch.rand(num_samples, modelfunc.n_parameters) * (max_vals - min_vals) + min_vals

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


def generate_smooth_params(modelfunc, nvox=100000):
    import numpy as np
    import torch

    # Extract parameter ranges of the non-volume fraction parameters
    ranges = modelfunc.parameter_ranges
    #add volume fractions ranges
    ranges = np.vstack((ranges, np.array([[0, 1]]*modelfunc.n_fractions)))
    # Number of parameters in the model function
    nparam = modelfunc.n_parameters + modelfunc.n_fractions
    #check that the number of parameter ranges matches the number of parameters in the model function 
    if ranges.shape[0] != modelfunc.n_parameters + modelfunc.n_fractions:
        ValueError("The number of parameter ranges for simulation does not match the number of parameters in the model function.")

    num_samples_per_dim = int(np.ceil(np.power(nvox,1/nparam))) #number of samples per dimension

    # Generate grid points within each parameter range
    params_grid = [np.linspace(r[0], r[1], num_samples_per_dim) for r in ranges]
    meshgrid = np.meshgrid(*params_grid, indexing='ij')

    # Reshape to get all combinations
    params = np.stack(meshgrid, axis=-1).reshape(-1, nparam)

    params = torch.tensor(params, dtype=torch.float32)

    return params

import math

def factorize_close(n):
    # Factorize a number n into two factors as close as possible
    # Used to make an image with dimensions as close to a square as possible

    # Start with the square root of n
    sqrt_n = int(math.sqrt(n))
    
    # Find the closest factors
    for i in range(sqrt_n, 0, -1):
        if n % i == 0:
            return (i, n // i)
    
    # If no factors found (though this should not happen in this range), return fallback
    return (1, n)




def main():
    import argparse
    import numpy as np
    import torch
    import os

    from src.utils.acquisition_scheme import acquisition_scheme_loader, txt_file_loader

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

    # #need to write a big function that does this for all models 
    # if model == "MSDKI":
    #     comps = ("MSDKI",)
    # elif model == "BallStick":
    #     comps = ("Ball","Stick")
    # elif model == "StickBall":
    #     comps = ("Stick","Ball")
    # elif model == "StandardWM":
    #     comps = ("Standard_wm",)
    # elif model == "VERDICT":
    #     comps = ("Ball", "Sphere", "Astrosticks")


    # #import compartment classes dynamically based on the chosen model (write a function to do this!)
    # import importlib
    # signal_models_module = importlib.import_module("signal_models")

    # comps_classes = () #initialise tuple
    # for comp in comps:
    #     #get the class
    #     this_class = getattr(signal_models_module, comp) #add to the tuple
    #     #create an instance of the class and add to the tuple
    #     comps_classes += (this_class(),)

    # #make the model function that will be incorporated into the net
    import sys
    from pathlib import Path

    # Add parent directory to Python path
    sys.path.append(str(Path(__file__).resolve().parent.parent))

    from model_maker import ModelMaker
    modelfunc = ModelMaker(args.model)
    # SOMETHING LIKE THIS INSTEAD!
    # modelfunc = ModelMaker(cfg.model.name)


    #load the acquisition scheme in
    if args.bvals is not None:
        grad = txt_file_loader(args.bvals, args.bvecs, args.delta, args.smalldelta, args.TE, args.bdelta)
    if args.grad is not None:       
        grad = acquisition_scheme_loader(args.grad)
        
        # Add delta and smalldelta to the grad object if they are not already there 
        
    #     # Ensure grad.delta is a tensor of correct length
    #     if grad.delta is None:                            
    #         if isinstance(args.smalldelta, (int, float)):  # single number (like 22.8)
    #             grad.delta = torch.full((len(grad.bvalues),), args.smalldelta, dtype=torch.float32)
    #         else:
    #             grad.delta = torch.tensor(args.smalldelta, dtype=torch.float32)
       
    #    # Ensure grad.Delta is a tensor of correct length
    #     if grad.Delta is None:
    #         if isinstance(args.delta, (int, float)):  # single number
    #             grad.Delta = torch.full((len(grad.bvalues),), args.delta, dtype=torch.float32)
    #         else:
    #             grad.Delta = torch.tensor(args.delta, dtype=torch.float32)
                
       
                
    #generate random parameters for the model function
    params = generate_random_params(modelfunc, num_samples=args.nx * args.ny * args.nz)

    #params = generate_smooth_params(modelfunc, args.nvox)

    # params_ball = generate_random_params(ball_modelfunc, repeat_interval=10)
    # params_stick = generate_random_params(stick_modelfunc, repeat_interval=10)

    S = modelfunc(grad, params)
    
    # S_ball = ball_modelfunc(grad, params_ball)
    # S_stick = stick_modelfunc(grad, params_stick)

    # Reshape the signal tensor and parameters tensor into the desired image format

    # make the image dimensions as close to a square as possible
    #dimx, dimy = factorize_close(params.shape[0])   
    
    dim = (args.nx, args.ny , args.nz) #set the dimensions of the image based on the command line arguments

    Simg = S.view(*dim, grad.number_of_measurements)
    paramsimg = params.view(*dim, params.size(-1))
    mask = torch.ones_like(Simg[:, :, :, 0])

    #make a directory for saving the simulated images
    os.makedirs(os.path.join(args.savedir,args.model), exist_ok=True )
    
    #save the image using nibabel
    import nibabel as nib
    img = nib.Nifti1Image(Simg.numpy(), np.eye(4))
    nib.save(img, os.path.join(args.savedir, args.model, args.model + '_' + ''.join(modelfunc.compartment_names) + '_data.nii.gz'))

    paramsnii = nib.Nifti1Image(paramsimg.numpy(), np.eye(4))
    nib.save(paramsnii, os.path.join(args.savedir, args.model, args.model + '_' + ''.join(modelfunc.compartment_names) + '_params.nii.gz'))

    maskimg = nib.Nifti1Image(mask.numpy(), np.eye(4))
    nib.save(maskimg, os.path.join(args.savedir, args.model, args.model + '_' + ''.join(modelfunc.compartment_names) + '_mask.nii.gz'))


if __name__ == '__main__':
    freeze_support()
    main()