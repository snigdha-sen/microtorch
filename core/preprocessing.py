
import numpy as np 
import torch

### This could use a redo to better use the class for aquisition scheme, but works well as it is right now :)

def get_grad_matrix(grad):
    grad_matrix = torch.zeros(grad.number_of_measurements, 9)
    grad_matrix[:, :3] = grad.bvecs

    # Helper function to process parameters
    def process_param(param, col_idx):
        if param is None:
            return

        if isinstance(param, (int, float)) or (isinstance(param, torch.Tensor) and param.numel() == 1):
            # If parameter is a single number, fill the entire column with it
            grad_matrix[:, col_idx] = float(param)
        else:
            # If parameter is an array/tensor
            param_flat = param.view(-1) if isinstance(param, torch.Tensor) else torch.tensor(param).view(-1)

            # Check if the array has enough elements
            if param_flat.shape[0] < grad.number_of_measurements:
                raise ValueError(
                    f"Parameter at column {col_idx} has {param_flat.shape[0]} elements, but {grad.number_of_measurements} are required")

            # Assign the first n elements where n is number_of_measurements
            grad_matrix[:, col_idx] = param_flat[:grad.number_of_measurements]

    # Process bvalues (column 3)
    process_param(grad.bvalues, 3)

    # Process optional parameters
    if grad.Delta is not None:
        process_param(grad.Delta, 4)
    if grad.small_delta is not None:
        process_param(grad.small_delta, 5)
    if grad.gradient_strengths is not None:
        process_param(grad.gradient_strengths, 6)
    if grad.TE is not None:
        process_param(grad.TE, 7)
    if grad.bdelta is not None:
        process_param(grad.bdelta, 8)

    grad_matrix = torch.nan_to_num(grad_matrix)

    return grad_matrix

def update_grad_class(grad, grad_matrix, new_num_measurements):
    grad.bvecs   = grad_matrix[:,:3]
    grad.bvalues = grad_matrix[:,3]

    if grad.Delta is not None:
        grad.Delta = grad_matrix[:,4]
    if grad.small_delta is not None:
        grad.small_delta = grad_matrix[:, 5]
    if grad.gradient_strengths is not None:
        grad.gradient_strengths = grad_matrix[:,6]
    if grad.TE is not None:
        grad.TE = grad_matrix[:,7]
    if grad.bdelta is not None:
        grad.bdelta = grad_matrix[:,8]

    grad.number_of_measurements = new_num_measurements

    return grad

def direction_average(img, grad):
    # Find unique shells - all parameters except gradient directions are the same
    grad_matrix   = get_grad_matrix(grad)
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

    return da_img, update_grad_class(grad, da_grad, unique_shells.shape[0])
         

def img2voxel(img, mask):
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
    # Create an empty image
    img = np.zeros_like(maskvox)

    # Fill in the voxels in the mask
    img[maskvox == 1] = params

    # Reshape the image to the original shape
    img = np.reshape(img, shape)

    return img


def normalise(X_train, grad):
    nvol    = grad.number_of_measurements
    normvol = torch.where(grad.bvalues == torch.min(grad.bvalues))[0]

    if normvol.numel() > 1:  
        b0_mean = torch.mean(X_train[:, normvol], dim=1, keepdim=True)
                
        X_train = X_train / b0_mean.repeat(1, nvol)
    else:  
        X_train = X_train / X_train[:, normvol].repeat(1, nvol)

    return X_train






