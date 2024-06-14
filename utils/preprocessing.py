<<<<<<< HEAD

import numpy as np 

def direction_average(img,grad):
    #find unique shells - all parameters except gradient directions are the same
    unique_shells = np.unique(grad[:,3:], axis=0)
        
    #preallocate
    da_img = np.zeros(img.shape[0:3] + (unique_shells.shape[0],))
    da_grad = np.zeros((unique_shells.shape[0],grad.shape[1]))

    for shell, i in zip(unique_shells,range(0,unique_shells.shape[0])):
        #indices of grad file for this shell          
        shell_index = np.all(grad[:,3:] == shell,axis=1)
        #calculate the spherical mean of this shell - average along final axis    
        da_img[...,i] = np.mean(img[...,shell_index], axis=img.ndim-1)
        #fill in this row of the direction-averaged grad file       
        da_grad[i,3:] = shell
                               
    return da_img, da_grad
         
        
def img2voxel(img,mask):
    nvoxtotal = np.prod(np.shape(img)[0:3])
    nvol = np.shape(img)[3]
    #image in voxel format
    imgvox = np.reshape(img,(nvoxtotal,nvol))
    #mask in voxel format
    maskvox = np.reshape(mask,(nvoxtotal))
    #extract the voxels in the mask
    X_train = imgvox[maskvox==1]    
    
    return X_train,maskvox
=======
import torch
import numpy as np

def direction_average(img, grad):
    # Find unique shells - all parameters except gradient directions are the same
    unique_shells = torch.unique(grad[:, 3:], dim=0)

    # Preallocate
    da_img  = torch.zeros(img.shape[0:3] + (unique_shells.shape[0],), dtype=img.dtype)
    da_grad = torch.zeros(unique_shells.shape[0], grad.shape[1], dtype=grad.dtype)

    for i, shell in enumerate(unique_shells):
        # Indices of grad file for this shell          
        shell_index = torch.all(grad[:, 3:] == shell, dim=1)
        # Calculate the spherical mean of this shell - average along final axis    
        da_img[..., i] = torch.squeeze(torch.mean(img[..., shell_index], axis=img.ndim-1))
        # Fill in this row of the direction-averaged grad file       
        da_grad[i, 3:] = shell

    return da_img, da_grad
         

def img2voxel(img, mask):
    # Calculate the total number of voxels and the number of volumes
    nvoxtotal = torch.prod(torch.tensor(img.shape[0:3]))
    nvol      = img.shape[3]

    # Image & mask in voxel format
    imgvox  = img.reshape(nvoxtotal, nvol)
    maskvox = mask.reshape(nvoxtotal)
>>>>>>> main

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
    nvol = grad.shape[0]

    normvol = torch.where(grad[:, 3] == torch.min(grad[:, 3]))[0]

    if normvol.numel() > 1:  
        b0_mean = torch.mean(X_train[:, normvol], dim=1, keepdim=True)
                
        X_train = X_train / b0_mean.repeat(1, nvol)
    else:  
        X_train = X_train / X_train[:, normvol].repeat(1, nvol)

    return X_train






