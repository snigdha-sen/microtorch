import numpy as np
import torch

def load_grad(grad_filename):
    #TO DO: replace with something that finds the file e.g. pkg_resources.resource_filename
    #grad_files_path = '/Users/paddyslator/python/self-qmri/data'
    try:
        #grad = torch.tensor(np.loadtxt(grad_filename), dtype=torch.float32)  
        grad = np.loadtxt(grad_filename)
        
        if len(grad.shape) < 2:
            grad = grad[:,None]
        return grad
    except:
        return None

# def load_bvals_bvecs_files(bvals_filename, bvecs_filename):

# return grad
