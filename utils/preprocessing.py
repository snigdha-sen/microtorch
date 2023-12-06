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


def voxel2img(X_train, maskvox, img_shape):
    nvoxtotal = np.prod(img_shape[0:3])
    nvol = X_train.shape[1]

    # Create an empty image
    img = np.zeros(img_shape)

    # Fill in the voxels in the mask
    img[maskvox == 1] = X_train

    # Reshape the image to the original shape
    img = np.reshape(img, img_shape)

    return img


def normalise(X_train,grad):
    nvol = np.shape(grad)[0]
    
    #normalise 
    #find the volumes to normalise by - the lowest b-value lowest TE volume
    #ADD SOME TOLERANCE TO THIS
    #normvol = np.where((grad[:,3] == min(grad[:,3])) & (grad[:,4]==min(grad[:,4])))
    
    #this just works for diffusion MRI - need to change if multiple echo times etc.
    normvol = np.where(grad[:,3] == min(grad[:,3]))[0]
    
    if len(normvol)>1:            
        b0_mean = np.mean(X_train[:,normvol], axis=1)                
        
        print(np.shape(b0_mean))
        print(np.shape(b0_mean[:,None]))
        
        X_train = X_train/np.tile(b0_mean[:, None],(1, nvol))
        
        print(np.shape(X_train))
    else:
        X_train = X_train/(np.tile(X_train[:,normvol],(1, nvol)))
    
    return X_train





