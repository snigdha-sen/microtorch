
import numpy as np 
import torch
import nibabel as nib
import os
### This could use a redo to better use the class for aquisition scheme, but works well as it is right now :)
class PreProcess(): #creating this specifically for wand data, but hopefully can be used for other things :)

    def __init__(self,
                 grad_scheme,
                 spherical_mean=False,
                 eps = 1e-16,
                 ):

        self.eps = eps
        self.grad_scheme = grad_scheme
        self.spherical_mean = spherical_mean

        if spherical_mean:
            self.grad_scheme.compute_direction_averaged_scheme()  # This will compute the direction-averaged scheme if spherical_mean is True

        return

    def __call__(self, image, mask): #This preprocess function only takes in the image path

        if self.spherical_mean:
            # If spherical mean is True, we need to direction average the image
            image = self.direction_average(image)

        image_vox, mask_vox = img2voxel(image, mask)
        image_vox = image_vox + self.eps
        image_vox = normalise(image_vox, self.grad_scheme)

        return image_vox, mask_vox

    def direction_average(self, img):

        # Find unique shells - all parameters except gradient directions are the same
        unique_shells = self.grad_scheme.unique_shells
        shell_idxs = self.grad_scheme.shell_idxs

        # Preallocate
        new_img  = torch.zeros(img.shape[0:3] + (unique_shells.shape[0],), dtype=img.dtype)


        for i, shell in enumerate(unique_shells):
            # Indices of grad file for this shell
            shell_index = shell_idxs[i]
            # Calculate the spherical mean of this shell - average along final axis
            new_img[..., i] = torch.squeeze(torch.mean(img[..., shell_index], axis=img.ndim-1))


        return new_img




def update_grad_class(grad, grad_matrix, new_num_measurements):
    grad.bvecs   = grad_matrix[:,:3]
    grad.bvalues = grad_matrix[:,3]

    if grad.Delta is not None:
        grad.Delta = grad_matrix[:,4]
    if grad.small_delta is not None:
        grad.small_delta = grad_matrix[:, 5]
    #if grad.gradient_strengths is not None:
    #    grad.gradient_strengths = grad_matrix[:,6]
    if grad.TE is not None:
        grad.TE = grad_matrix[:,6]
    if grad.bdelta is not None:
        grad.bdelta = grad_matrix[:,7]

    grad.number_of_measurements = new_num_measurements

    return grad

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




