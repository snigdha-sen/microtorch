from math import gamma

import numpy as np
import torch
import scipy.special as special
from typing import Optional

from src.utils.geometry import sphere2cart

# Precompute cylinder roots ONCE (order=1, first N roots)
# SciPy returns a NumPy array
_CYLINDER_ROOTS = torch.tensor(special.jnp_zeros(1, 100), dtype=torch.float32)


class Stick:
    """
    Restricted diffusion in a stick (parallel free, perpendicular zero).
    
    Attributes:
        parameter_ranges (list): Ranges for the parameters.
        parameter_names (list): Names of the parameters.
        n_parameters (int): Number of parameters.
        spherical_mean (bool): Indicates if the model is spherically averaged.  
    
    Methods:
        __init__(): Initializes the stick model with parameter ranges and names.
        __call__(grad, parameters): Computes the signal based on the gradient and parameters.

    """
    def __init__(self):
        self.parameter_ranges = [[0.001, 3], [0, torch.pi], [-torch.pi, torch.pi]]
        self.parameter_names = ["Dpar", "theta", "phi"]
        self.n_parameters = 3
        self.spherical_mean = False

    def __call__(self, grad, parameters):
        n = grad.bvecs                  # (M, 3) unit vectors
        b = grad.bvalues                # (M,)

        Dpar = parameters[:, 0:1]       # (B, 1)
        theta = parameters[:, 1]        # (B,)
        phi = parameters[:, 2]          # (B,)

        # sphere2cart returns (3, B) -> transpose to (B, 3)
        mu = sphere2cart(theta, phi).T
        mu = mu / torch.norm(mu, dim=1, keepdim=True)

        # dot = (B,3) @ (3,M) = (B,M)
        dot = mu @ n.T

        b = b.unsqueeze(0)              # (1, M)

        S = torch.exp(-b * Dpar * (dot ** 2))  # (B, M)
        return S


class Cylinder:
    """
    Restricted diffusion in a cylinder (parallel free, perpendicular restricted).
    Uses a series expansion with Bessel roots (van Gelderen-style form).

    Attributes:
        parameter_ranges (list): Ranges for the parameters.
        parameter_names (list): Names of the parameters.
        n_parameters (int): Number of parameters.
        spherical_mean (bool): Indicates if the model is spherically averaged.

    Methods:
        __init__(n_roots, lambda_perp): Initializes the cylinder model with parameter ranges and names.
        __call__(grad, parameters): Computes the signal based on the gradient and parameters.

    """

    def __init__(self, n_roots: int = 50, lambda_perp: float = 2e-9):
        self.parameter_ranges = [[0, torch.pi], [-torch.pi, torch.pi], [0.001, 3], [0.001, 10]]
        self.parameter_names = ["theta", "phi", "D_par", "radius"]
        self.n_parameters = 4
        self.spherical_mean = False

        self.n_roots = n_roots
        self.lambda_perp = float(lambda_perp)

    def __call__(self, grad, parameters):
        device = parameters.device
        dtype = parameters.dtype

        n = grad.bvecs.to(device=device, dtype=dtype)                 # (M,3)
        b_values = grad.bvalues.to(device=device, dtype=dtype)               # (M,)
        delta = grad.delta.to(device=device, dtype=dtype)             # (M,)
        Delta = grad.Delta.to(device=device, dtype=dtype)             # (M,)        
        #g = grad.gradient_strengths.to(device=device, dtype=dtype)    # (M,)


        theta = parameters[:, 0]                                      # (B,)
        phi = parameters[:, 1]                                        # (B,)
        Dpar = parameters[:, 2:3]                                     # (B,1)
        radius = parameters[:, 3:4]                                   

        if torch.any(radius <= 0):
            raise ValueError("Cylinder radius must be positive.")


        mu = sphere2cart(theta, phi).T.to(device=device, dtype=dtype)
        mu = mu / torch.norm(mu, dim=1, keepdim=True)


        dot = mu @ n.T
        mag_perp = torch.sqrt(torch.clamp(1.0 - dot**2, min=0.0))


        b_row = b_values.unsqueeze(0) 
        E_parallel = torch.exp(-b_row * Dpar * (dot**2)) 

       
        #gamma = torch.tensor(2.67e8, device=device, dtype=dtype)  
        gamma = torch.tensor(2.675987e2, device=device, dtype=dtype)  
        
        #calculate gradient strength from b-values, delta, and Delta   
        g = torch.sqrt(b_values) / (gamma * delta * torch.sqrt(Delta - delta / 3)) # (M,)
      
        g_perp = g.unsqueeze(0) * mag_perp                        

        
        roots = _CYLINDER_ROOTS[: self.n_roots].to(device=device, dtype=dtype) 
        R = radius.unsqueeze(-1)                                                

        delta_ = delta.unsqueeze(0).unsqueeze(-1)
        Delta_ = Delta.unsqueeze(0).unsqueeze(-1)

        radius_ = radius.unsqueeze(1)

        roots_ = roots.unsqueeze(0).unsqueeze(0)

        alpha = roots_ / radius_
        alpha2 = alpha ** 2
        D = torch.tensor(self.lambda_perp, device=device, dtype=dtype)
        alpha2D = alpha2 * D

        first_factor = -2.0 * (g_perp * gamma) ** 2

        numer = (
            2 * alpha2D * delta_ - 2
            + 2 * torch.exp(-alpha2D * delta_)
            + 2 * torch.exp(-alpha2D * Delta_)
            - torch.exp(-alpha2D * (Delta_ - delta_))
            - torch.exp(-alpha2D * (Delta_ + delta_))
        )

        denom = (D ** 2) * (alpha ** 6) * ((radius_ ** 2) * alpha2 - 1.0)

        summands = numer / denom  # (B,M,R) by broadcast

        S_series = summands.sum(dim=-1)  # sum over roots -> (B,M)

        E_perp = torch.exp(first_factor * S_series)  # (B,M)

        return E_parallel * E_perp  # (B,M)


class Astrosticks:
    """
    Spherical mean of sticks (diffusion only parallel to the stick, but sticks are randomly oriented).

    If fixed_D_par is provided, the model ignores the parameter input value and uses the fixed value.
    Otherwise, D_par is taken from parameters[:, 0].

    Attributes:
        parameter_ranges (list): Ranges for the parameters.
        parameter_names (list): Names of the parameters.
        n_parameters (int): Number of parameters.
        spherical_mean (bool): Indicates if the model is spherically averaged.
    """
    def __init__(self, fixed_D_par: Optional[float] = None, default_range=(0.5, 3.0)):
        self.fixed_D_par = fixed_D_par
        self.parameter_names = ["D_par"]
        self.n_parameters = 1
        self.spherical_mean = True

        if fixed_D_par is None:
            self.parameter_ranges = [list(default_range)]
        else:
            self.parameter_ranges = [[float(fixed_D_par), float(fixed_D_par)]]

    def __call__(self, grad, parameters):
        b = grad.bvalues
        if b.ndim == 1:
            b = b.unsqueeze(0)  # (1, M)

        B = parameters.shape[0]

        if self.fixed_D_par is None:
            D_par = parameters[:, 0:1]  # (B, 1)
        else:
            D_par = torch.full((B, 1), float(self.fixed_D_par), dtype=b.dtype, device=b.device)

        x = b * D_par  # broadcast -> (B, M) if b is (1, M)

        # Compute S = sqrt(pi) * erf(sqrt(x)) / (2*sqrt(x)), with safe handling at x=0
        sqrt_x = torch.sqrt(torch.clamp(x, min=0.0))
        numer = torch.sqrt(torch.tensor(torch.pi, dtype=b.dtype, device=b.device)) * torch.erf(sqrt_x)
        denom = 2.0 * sqrt_x

        S = torch.where(x > 0, numer / denom, torch.ones_like(x))
        return S
    
