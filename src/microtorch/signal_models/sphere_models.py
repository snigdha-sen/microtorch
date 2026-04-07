import numpy as np
import torch
from typing import Optional

class Sphere:
    """
    A class representing a sphere model for diffusion MRI.
    This model computes the signal based on the diffusion parameters and gradient directions.

    Attributes:
        parameter_ranges (list): Ranges for the parameters.
        parameter_names (list): Names of the parameters.
        n_parameters (int): Number of parameters.
        spherical_mean (bool): Indicates if the model is spherically averaged.

    Methods:
        __init__(): Initializes the sphere model with parameter ranges and names.
        __call__(grad, parameters): Computes the signal based on the gradient and parameters.
    """
    def __init__(self, fixed_D: Optional[float] = None):
        self.parameter_ranges   = [[0.001, 15]]
        self.parameter_names        = ['radius']
        self.n_parameters           = 1
        self.spherical_mean     = True
        self.fixed_D = fixed_D

    def __call__(self, grad, parameters):

        b_values = grad.bvalues
        delta = grad.delta
        Delta = grad.Delta

        # D = self.fixed_D if hasattr(self, 'fixed_D') else 2.0 # D_IC

        if self.fixed_D is None:
            D = torch.full((parameters.shape[0], 1), float(2.0))
        else:
            D = torch.full((parameters.shape[0], 1), float(self.fixed_D))
        
        radius = parameters[:,0].unsqueeze(1)

        SPHERE_TRASCENDENTAL_ROOTS = np.r_[
        # 0.,
        2.081575978, 5.940369990, 9.205840145,
        12.40444502, 15.57923641, 18.74264558, 21.89969648,
        25.05282528, 28.20336100, 31.35209173, 34.49951492,
        37.64596032, 40.79165523, 43.93676147, 47.08139741,
        50.22565165, 53.36959180, 56.51327045, 59.65672900,
        62.80000055, 65.94311190, 69.08608495, 72.22893775,
        75.37168540, 78.51434055, 81.65691380, 84.79941440,
        87.94185005, 91.08422750, 94.22655255, 97.36883035
        ]
        
        alpha = torch.FloatTensor(SPHERE_TRASCENDENTAL_ROOTS) / (radius)
        alpha2 = alpha ** 2
        alpha2D = alpha2 * D
        alpha = alpha.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)
        alpha2D = alpha2D.unsqueeze(1)
 
        gamma = 2.675987e2
        gradient_strength = torch.sqrt(b_values) / (gamma * delta * torch.sqrt(Delta - delta / 3))
        first_factor        = -2*(gamma*gradient_strength)**2 / 2
                
        Delta = Delta.unsqueeze(0).unsqueeze(2)
        delta = delta.unsqueeze(0).unsqueeze(2)
        
        summands = (alpha ** (-4) / (alpha2 * (radius.unsqueeze(2))**2 - 2) * (
                            2 * delta - (
                            2 +
                            torch.exp(-alpha2D * (Delta - delta)) -
                            2 * torch.exp(-alpha2D * delta) -
                            2 * torch.exp(-alpha2D * Delta) +
                            torch.exp(-alpha2D * (Delta + delta))
                        ) / (alpha2D)
                    )
                )
                
        S = torch.exp(
            first_factor *
            summands.sum(dim=2)
        )

        return S
    
    
class Dot:
    """
    A class representing a dot model for diffusion MRI.
    This model computes the signal based on the diffusion parameters and gradient directions.

    Attributes:
        parameter_ranges (list): Ranges for the parameters.
        parameter_names (list): Names of the parameters.
        n_parameters (int): Number of parameters.
        spherical_mean (bool): Indicates if the model is spherically averaged.

    Methods:
        __init__(): Initializes the dot model with parameter ranges and names.
        __call__(grad, parameters): Computes the signal based on the gradient and parameters.
    """
    def __init__(self):
        self.parameter_ranges   = []
        self.parameter_names    = []
        self.n_parameters       = 0
        self.spherical_mean     = None

    def __call__(self, grad, parameters):

        b_values = grad.bvalues
        
        S = torch.ones_like(b_values)

        return S