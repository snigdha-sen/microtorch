import numpy as np
import torch

def load_grad(grad_filename):
    """
    Loads the gradient information from a text file.
    
    Args:
        grad_filename (str): The path to the text file containing the gradient information.
    Returns:
        grad (numpy.ndarray): A 2D array containing the gradient information, where each row corresponds to a measurement and columns correspond to gradient parameters.
    """
    #TO DO: replace with something that finds the file e.g. pkg_resources.resource_filename
    #grad_files_path = '/Users/paddyslator/python/self-qmri/data'
    try:
        #grad = torch.tensor(np.loadtxt(grad_filename), dtype=torch.float32)  
        #print(grad_filename)
        grad = np.loadtxt(grad_filename)
        
        if len(grad.shape) < 2:
            grad = grad[:,None]
        
        return grad #np.transpose(grad)
    except OSError:
        return None

