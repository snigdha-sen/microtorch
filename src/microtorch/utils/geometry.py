
import numpy as np
import torch
from pathlib import Path

def sphere2cart(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Converts spherical coordinates (theta, phi) to Cartesian coordinates (x, y, z).

    Args:
        theta (torch.Tensor): A tensor of shape (B,) containing the polar angles in radians.
        phi (torch.Tensor): A tensor of shape (B,) containing the azimuthal angles in radians.
    Returns:
        n (torch.Tensor): A tensor of shape (3, B) containing the Cartesian coordinates corresponding to the input spherical coordinates.
    """

    n = torch.zeros(3,theta.size(0))
            
    n[0,:] = torch.squeeze(torch.sin(theta) * torch.cos(phi))
    n[1,:] = torch.squeeze(torch.sin(theta) * torch.sin(phi))
    n[2,:] = torch.squeeze(torch.cos(theta))   
    
    return n
    
    
def cart2sphere(xyz: np.ndarray) -> np.ndarray:
    """
    Converts Cartesian coordinates (x, y, z) to spherical coordinates (theta, phi).
    
    Args:
        xyz (torch.Tensor): A tensor of shape (..., 3) containing the Cartesian coordinates"
    Returns:
        mu (torch.Tensor): A tensor of shape (..., 2) containing the spherical coordinates (theta, phi) corresponding to the input Cartesian coordinates.
    """

    shape = xyz.shape[:-1]
    mu = np.zeros(np.r_[shape, 2])

    r = np.linalg.norm(xyz, axis=-1)

    safe = r > 0

    mu[safe, 0] = np.arccos(xyz[safe, 2] / r[safe])
    mu[safe, 1] = np.arctan2(xyz[safe, 1], xyz[safe, 0])

    return mu



