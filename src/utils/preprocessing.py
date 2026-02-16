import numpy as np 
import torch

def direction_average(img, grad):
    """
    Averages the signal across all directions within each shell, 
    returning a direction-averaged image and corresponding gradient table.

    Args:
        img (torch.Tensor): The input image with shape (X, Y, Z, N), where N is the number of gradient directions.
        grad (AcquisitionScheme): The acquisition scheme containing the gradient information.
    Returns:
        da_img (torch.Tensor): The direction-averaged image with shape (X, Y, Z, M), 
        where M is the number of unique shells.
    """


    # Find unique shells - all parameters except gradient directions are the same
    grad_matrix = grad.get_scheme_as_matrix()

    unique_shells = torch.unique(grad_matrix[:, 3:], dim=0)

    # Preallocate
    da_img  = torch.zeros(img.shape[0:3] + (unique_shells.shape[0],), dtype=img.dtype)
    da_grad = torch.zeros(unique_shells.shape[0], grad_matrix.shape[1], dtype=grad_matrix.dtype)

    for i, shell in enumerate(unique_shells):
        # Indices of grad file for this shell          
        shell_index = torch.all(grad_matrix[:, 3:] == shell, dim=1)
        # Calculate the spherical mean of this shell - average along final axis    
        da_img[..., i] = torch.squeeze(torch.mean(img[..., shell_index], axis=img.ndim-1))
        # Fill in this row of the direction-averaged grad file       
        da_grad[i, 3:] = shell

    grad.set_scheme_from_matrix(da_grad)

    return da_img
         

def img2voxel(img, mask):
    """
    Converts the image and mask from voxel format to a 2D array of voxels in the mask and a 1D array of the mask in voxel format.
    
    Args:
        img (torch.Tensor): The input image with shape (X, Y, Z, N), where N is the number of gradient directions.
        mask (torch.Tensor): The binary mask with shape (X, Y, Z), where 1 indicates voxels to include in the training set and 0 indicates voxels to exclude.
    Returns:
        X_train (torch.Tensor): A 2D tensor of shape (M, N), where M is the number of voxels in the mask and N is the number of gradient directions, containing the signal values for the voxels in the mask.
        maskvox (torch.Tensor): A 1D tensor of shape (X*Y*Z,) containing the binary mask in voxel format.
    """
    
    # Calculate the total number of voxels and the number of volumes
    nvoxtotal = torch.prod(torch.tensor(img.shape[0:3]))
    nvol      = img.shape[3]

    # Image & mask in voxel format
    imgvox  = img.reshape(nvoxtotal, nvol)
    maskvox = mask.reshape(nvoxtotal)


    # Extract the voxels in the mask
    X_train = imgvox[maskvox == 1]

    return X_train, maskvox


def voxel2img(params, maskvox, shape):
    """
    Converts the predicted parameters from voxel format back to image format, 
    filling in the voxels in the mask and leaving the rest as zeros.

    Args:
        params (numpy.ndarray): A 1D array of shape (M,) containing the predicted parameter values for the voxels in the mask.
        maskvox (numpy.ndarray): A 1D array of shape (X*Y*Z,) containing the binary mask in voxel format, 
        where 1 indicates voxels in the mask and 0 indicates voxels outside the mask.
        shape (tuple): The original shape of the image (X, Y, Z).

    Returns:
        img (numpy.ndarray): A 3D array of shape (X, Y, Z) containing the predicted parameter values 
        for the voxels in the mask and zeros for the voxels outside the mask.
    """
    # Create an empty image
    img = np.zeros_like(maskvox)

    # Fill in the voxels in the mask
    img[maskvox == 1] = params

    # Reshape the image to the original shape
    img = np.reshape(img, shape)

    return img


def normalise(X_train, grad):
    """
    Normalises the training data by the mean of the b0 volumes (or the single b0 volume if only one is present).
    Args:
        X_train (torch.Tensor): A 2D tensor of shape (M, N) containing the signal values for the voxels in the mask, 
        where M is the number of voxels and N is the number of gradient directions.
        grad (AcquisitionScheme): The acquisition scheme containing the gradient information, including b-values.
    Returns:
        X_train (torch.Tensor): The normalised training data, where each voxel's signal is divided by the mean of the b0 volumes for that voxel.
    """

    nvol    = grad.number_of_measurements
    normvol = torch.where(grad.bvalues == torch.min(grad.bvalues))[0]

    if normvol.numel() > 1:  
        b0_mean = torch.mean(X_train[:, normvol], dim=1, keepdim=True)
                
        X_train = X_train / b0_mean.repeat(1, nvol)
    else:  
        X_train = X_train / X_train[:, normvol].repeat(1, nvol)

    return X_train




