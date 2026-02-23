###Taken from messy_examples.simulated_data.ipynb
##Separated because importing from ipynb sometimes caused issues
import torch
import numpy as np
from src.utils.preprocessing import voxel2img

#simulate some data from a "cluster model"
nvox = 1024
nclus = 5
p = [0.1, 0.1, 0.2, 0.5]
p = np.append(p,1-np.sum(p))
clusters = np.random.choice(range(0,nclus),size=(nvox,),p=p)

#define the underlying tissue parameters for each cluster
D = [0.5,1,1.5,2,3]
K = [1,0.5,0.2,0.1,0.01]
#K = [0.1,0.05,0.2,0.1,0]

mu = np.stack((D,K))
var = np.diag([0.01,0.01])

params = np.zeros((nvox,2))

for vox in range(0,nvox):
    params[vox,:] = np.random.multivariate_normal(mu[:,clusters[vox]],var)

params[params < 0] = 0.01

def reconstruct_parameter_maps(
    params, maskvox, mask_shape, modelfunc
):
    param_map = np.zeros((*mask_shape, modelfunc.n_params + modelfunc.n_frac))
    for i in range(0, modelfunc.n_params + modelfunc.n_frac):
        param_map[..., i] = voxel2img(params[:, i], maskvox, mask_shape)
    return


def reconstruct_parameter_maps_with_torch(params, maskvox, mask_shape, n_params, n_frac):
    """
    Reconstruct parameter maps from network outputs using PyTorch tensors

    Args:
        params: Parameter tensor from network (voxels × parameters)
        maskvox: Mask in voxel format
        mask_shape: Shape of the original 3D mask
        n_params: Number of model parameters
        n_frac: Number of volume fraction parameters

    Returns:
        Reconstructed parameter maps
    """
    # Create empty parameter map tensor
    param_map = torch.zeros((*mask_shape, n_params + n_frac), dtype=params.dtype)

    # Get the indices of voxels inside the mask
    mask_indices = torch.nonzero(maskvox == 1, as_tuple=True)[0]

    # For each parameter, place values back in the 3D volume
    for i in range(n_params + n_frac):
        flat_map = torch.zeros_like(maskvox, dtype=params.dtype)
        flat_map[mask_indices] = params[:, i]
        param_map[..., i] = flat_map.reshape(mask_shape)

    return param_map
