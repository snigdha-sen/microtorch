
import numpy as np 
import torch


def direction_average(img, grad):
    # Find unique shells - all parameters except gradient directions are the same
    grad_matrix = grad.get_scheme_as_matrix()

    #debug line
    #print(np.array_equal(np.array(grad_matrix), np.array(grad_matrix2)))

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




