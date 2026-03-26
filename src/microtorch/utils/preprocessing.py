import numpy as np 
import torch
from .acquisition_scheme import AcquisitionScheme


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

    # build shell definition table
    shell_columns = []

    if hasattr(grad, "bvalues") and grad.bvalues is not None:
        shell_columns.append(grad.bvalues)
    if hasattr(grad, "TE") and grad.TE is not None:
        shell_columns.append(grad.TE)
    if hasattr(grad, "delta") and grad.delta is not None:
        shell_columns.append(grad.delta)
    if hasattr(grad, "Delta") and grad.Delta is not None:
        shell_columns.append(grad.Delta)
    if hasattr(grad, "bshape") and grad.bshape is not None:
        shell_columns.append(grad.bshape)
    if hasattr(grad, "bdelta") and grad.bdelta is not None:
        shell_columns.append(grad.bdelta)

    # Make sure we have at least one column to stack
    if len(shell_columns) == 0:
        raise ValueError("No gradient information available to define shells!")

    shell_table = torch.stack(shell_columns, dim=1) 
        
        
    # Find unique shells - all parameters except gradient directions are the same
    unique_shells, inverse = torch.unique(shell_table, dim=0, return_inverse=True)

    # Preallocate
    da_img  = torch.zeros(img.shape[0:3] + (unique_shells.shape[0],), dtype=img.dtype)

    da_bvecs = torch.zeros((unique_shells.shape[0], grad.bvecs.shape[1]), dtype=grad.bvecs.dtype)
    da_bvalues = torch.zeros_like(unique_shells)
    da_delta = torch.zeros_like(unique_shells) if grad.delta is not None else None
    da_Delta = torch.zeros_like(unique_shells) if grad.Delta is not None else None
    da_TE = torch.zeros_like(unique_shells) if grad.TE is not None else None
    da_bdelta = torch.zeros_like(unique_shells) if grad.bdelta is not None else None
    da_TE = torch.zeros_like(unique_shells) if grad.TE is not None else None
    da_gradient_strengths = torch.zeros_like(unique_shells) if grad.gradient_strengths is not None else None


    for i, shell in enumerate(unique_shells):
        # Indices of grad file for this shell          
        shell_index = (inverse == i)       
        first_idx = torch.where(shell_index)[0][0]
         
        # Calculate the spherical mean of this shell - average along final axis   
        da_img[..., i] = img[..., shell_index].mean(dim=-1)
        
        #set direction to zero after averaging
        da_bvecs[i] = 0.0 # Set the gradient directions to zero for this shell      
        da_bvalues[i] = grad.bvalues[first_idx] # Set the bvalues to the unique shell value
        
        if da_TE is not None:
            da_TE[i] = grad.TE[first_idx]
        if da_delta is not None:
            da_delta[i] = grad.delta[first_idx]
        if da_Delta is not None:
            da_Delta[i] = grad.Delta[first_idx]
        if da_bdelta is not None:
            da_bdelta[i] = grad.bdelta[first_idx]
        if da_gradient_strengths is not None:
            da_gradient_strengths[i] = grad.gradient_strengths[first_idx]
        if da_bdelta is not None:
            da_bdelta[i] = grad.bdelta[first_idx]

    da_grad = AcquisitionScheme(
        bvalues=da_bvalues,
        bvecs=da_bvecs,
        gradient_strengths=da_gradient_strengths,
        delta=da_delta,
        Delta=da_Delta,
        TE=da_TE,
        bdelta=da_bdelta,
    )

    return da_img, da_grad
         

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
    
    nvol = grad.number_of_measurements

    # find lowest b-value
    min_b = torch.min(grad.bvalues)
    ref_mask = (grad.bvalues == min_b)

    # among those, find lowest TE if TE exists
    if hasattr(grad, "TE") and grad.TE is not None:
        min_te = torch.min(grad.TE[ref_mask])
        ref_mask = ref_mask & (grad.TE == min_te)

    #lowest b-value and lowest TE is the reference for normalisation.
    ref_idx = torch.where(ref_mask)[0]

    if ref_idx.numel() == 0:
        raise ValueError("No normalisation volume found.")

    # mean reference if multiple matching measurements
    ref_signal = torch.mean(X_train[:, ref_idx], dim=1, keepdim=True)

    # avoid divide-by-zero
    ref_signal = torch.clamp(ref_signal, min=1e-8)

    X_train = X_train / ref_signal.repeat(1, nvol)

    return X_train




