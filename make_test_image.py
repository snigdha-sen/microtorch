from multiprocessing import freeze_support


def generate_random_params(modelfunc, repeat_interval=10, n_param=100):
    import numpy as np
    import torch

    """
    Generate random parameters with an option to repeat every `repeat_interval` elements in a row.
    
    Args:
        modelfunc: The model function with `parameter_ranges` and `n_frac` attributes.
        repeat_interval: Number of elements to repeat. Default is 1 (no repetition).
        n_unique: Number of unique sets of parameters to generate.
        
    Returns:
        params: Tensor of shape [n_unique * repeat_interval, modelfunc.n_params + modelfunc.n_frac] with random parameters.
    """
    # Extract min and max values from the columns
    min_vals = modelfunc.parameter_ranges[:, 0]
    max_vals = modelfunc.parameter_ranges[:, 1]

    # Add the volume fraction min/max values
    min_vals = np.append(min_vals, np.ones(modelfunc.n_frac))
    max_vals = np.append(max_vals, np.zeros(modelfunc.n_frac))

    #add volume fractions ranges
    min_vals = np.vstack((min_vals, np.array([[0]]*modelfunc.n_frac)))
    max_vals = np.vstack((max_vals, np.array([[1]]*modelfunc.n_frac)))

    # Generate n_unique sets of parameters with random values within the min and max ranges
    unique_params = (torch.rand(n_param, modelfunc.n_params + modelfunc.n_frac) * (max_vals - min_vals) + min_vals).float()

    # Repeat each set of parameters `repeat_interval` times
    params = unique_params.repeat_interleave(repeat_interval, dim=0)

    return params

def generate_smooth_params(modelfunc, nvox=100000):
    import numpy as np
    import torch

    # Extract parameter ranges of the non-volume fraction parameters
    ranges = modelfunc.parameter_ranges
    #add volume fractions ranges
    ranges = np.vstack((ranges, np.array([[0, 1]]*modelfunc.n_frac)))
    # Number of parameters in the model function
    nparam = modelfunc.n_params + modelfunc.n_frac
    #check that the number of parameter ranges matches the number of parameters in the model function 
    if ranges.shape[0] != modelfunc.n_params + modelfunc.n_frac:
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

    from core.acquisition_scheme import acquisition_scheme_loader, txt_file_loader

    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model", type=str, help="Compartmental Model to use. Implemented are verdict, sandi, or user defined ones form combinations of ball; sphere, stick; astrosticks; cylinder; astrocylinders; zeppelin; astrozeppelins; dot.", default="verdict")
    parser.add_argument("-bvals", "--bvals", help="bval file in FSL format and in [s/mm2]", default=None)
    parser.add_argument("-bvecs", "--bvecs", help="bvec file in FSL format", default=None)
    parser.add_argument("-g", "--grad", help="grad file in mrtrix format", default="")
    parser.add_argument("-d", "--delta", help="gradient pulse separation in ms", default=24, type=float)
    parser.add_argument("-sd", "--smalldelta", help="gradient pulse duration in ms", default=8, type=float)
    parser.add_argument("-TE", "--TE", help="echo time in ms", default="")
    parser.add_argument("-TR", "--TR", help="repetition time in ms", default="")
    parser.add_argument("-TI", "--TI", help="inversion time in ms", default="")
    parser.add_argument("-nparam", "--nparam", help="number of random parameters to sample", default=100)
    parser.add_argument("-repvox", "--repvox", help="number of repeat voxels to do for each parameter", default=10)
    parser.add_argument("-nvox", "--nvox", help="total number of voxels", default=5000)
    parser.add_argument("-savedir", "--savedir", help="directory to save the images", default="data/test_images/")

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
    #     comps = ("Standard_WM",)
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
    
    from core.model_maker import ModelMaker
    modelfunc = ModelMaker(args.model)

    #load the acquisition scheme in
    if args.bvals is not None:
        grad = txt_file_loader(args.bvals, args.bvecs, args.delta, args.smalldelta, args.TE, args.bdelta)
    if args.grad is not None:
        grad = acquisition_scheme_loader(args.grad)

    #params = generate_random_params(modelfunc, repeat_interval=args.repvox, n_param=args.nparam)

    params = generate_smooth_params(modelfunc, args.nvox)

    # params_ball = generate_random_params(ball_modelfunc, repeat_interval=10)
    # params_stick = generate_random_params(stick_modelfunc, repeat_interval=10)

    S = modelfunc(grad, params)
    # S_ball = ball_modelfunc(grad, params_ball)
    # S_stick = stick_modelfunc(grad, params_stick)

    # Reshape the signal tensor and parameters tensor into the desired image format

    # make the image dimensions as close to a square as possible
    dimx, dimy = factorize_close(params.shape[0])   

    dim = (dimx, dimy, 1)

    Simg = S.view(*dim, grad.number_of_measurements)
    paramsimg = params.view(*dim, params.size(-1))
    mask = torch.ones_like(Simg[:, :, :, 0])

    #save the image using nibabel
    import nibabel as nib
    img = nib.Nifti1Image(Simg.numpy(), np.eye(4))
    nib.save(img, os.path.join(args.savedir, ''.join(modelfunc.comp_names) + '.nii.gz'))

    paramsnii = nib.Nifti1Image(paramsimg.numpy(), np.eye(4))
    nib.save(paramsnii, os.path.join(args.savedir, ''.join(modelfunc.comp_names) + '_params.nii.gz'))

    maskimg = nib.Nifti1Image(mask.numpy(), np.eye(4))
    nib.save(maskimg, os.path.join(args.savedir, ''.join(modelfunc.comp_names) + '_mask.nii.gz'))



if __name__ == '__main__':
    freeze_support()
    main()