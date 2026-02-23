
import numpy as np
import torch
from pathlib import Path


__all__ = [
    'cart2sphere',
    'sphere2cart',
    'strip_filename',
]


def sphere2cart(theta,phi):   
    n = torch.zeros(3,theta.size(0))
            
    n[0,:] = torch.squeeze(torch.sin(theta) * torch.cos(phi))
    n[1,:] = torch.squeeze(torch.sin(theta) * torch.sin(phi))
    n[2,:] = torch.squeeze(torch.cos(theta))   
    
    return n
    
    
def cart2sphere(xyz):
    shape = xyz.shape[:-1]
    mu = np.zeros(np.r_[shape, 2])
    r = np.linalg.norm(xyz, axis=-1)
    mu[..., 0] = np.arccos(xyz[..., 2] / r)  # theta
    mu[..., 1] = np.arctan2(xyz[..., 1], xyz[..., 0])
    mu[r == 0] = 0, 0
    return mu


def strip_filename(path):
    """
    Removes folder path and strips '.nii' or '.nii.gz' extension from a file path.

    Parameters:
        path (str or Path): Full file path.

    Returns:
        str: Filename without path and extension.
    """
    path = Path(path)
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    elif name.endswith(".nii"):
        return name[:-4]
    else:
        return path.stem

