import numpy as np
import torch
import scipy.special as special
from ..utils.util_function import sphere2cart

class Stick:
    """
    Stick model for diffusion MRI signal.
    This model assumes a single diffusion direction and a diffusion coefficient.

    Attributes:
        parameter_ranges (list): Ranges for the parameters [Dpar, theta, phi].
        param_names (list): Names of the parameters.
        n_parameters (int): Number of parameters.
        spherical_mean (bool): Whether the model is spherically-averaged or not.
    
    Methods:
        __init__: Initializes the model with parameter ranges and names.
        __call__(grad, parameters): Computes the diffusion signal based on the gradient and parameters.
    """
    def __init__(self):
        self.parameter_ranges   = [[.001, 3], [0, torch.pi], [-torch.pi, torch.pi]]
        self.parameter_names        = ['Dpar', 'theta', 'phi']
        self.n_parameters           = 3
        self.spherical_mean     = False


    def __call__(self, grad, parameters):                   
        bvecs    = grad.bvecs
        b_values = grad.bvalues

        Dpar = parameters[:, 0].unsqueeze(1)
        theta = parameters[:, 1].unsqueeze(1)
        phi = parameters[:, 2].unsqueeze(1)

        n = sphere2cart(theta, phi)
        
        S = torch.exp(-b_values * Dpar * torch.mm(bvecs, n).t() ** 2)                          
        
     
        return S
    

    class Cylinder: ## would be good to have a working version of this
    
        def __init__(self):

            self.parameter_ranges = [[0, torch.pi], [-torch.pi, torch.pi], [.001, 3], [.001, 10]] 
            self.parameter_names = ['theta', 'phi', 'D_par', 'radius']
            self.n_parameters = 4
            self.spherical_mean = False

        def __call__(self, grad, parameters):

            b_vectors = grad.bvecs
            b_values = grad.bvalues
            delta = grad.delta
            Delta = grad.Delta
            g = grad.gradient_strengths

            theta      = parameters[:, 0]
            phi        = parameters[:, 1]
            lambda_par = parameters[:, 2].unsqueeze(1)
            radius   = parameters[:, 3].unsqueeze(1)
            
            diameter = 2*radius
            gamma = 2.67e8

            _CYLINDER_TRASCENDENTAL_ROOTS = torch.sort(special.jnp_zeros(1, 100)) ## store this somewhere
            lambda_perp = 2e-9 # check this

            mu = sphere2cart(theta, phi)
            mu = mu / torch.norm(mu, dim=1, keepdim=True)

            dot = torch.sum(b_vectors * mu, dim=1, keepdim=True)
            I = torch.eye(3, device=mu.device)
            mu_perp = I - mu.unsqueeze(2) @ mu.unsqueeze(1)
            proj = torch.matmul(mu_perp, b_vectors.unsqueeze(2)).squeeze(2)
            mag_perp = torch.norm(proj, dim=1, keepdim=True)

            E_parallel = torch.exp(-b_values * lambda_par * dot**2)

            g_perp = g * mag_perp
            E_perpendicular = torch.ones_like(g)

            mask = g_perp > 0

            def perpendicular_attenuation(
                g, delta, Delta, diameter, D, gamma, roots
            ):
                R = diameter / 2
                first_factor = -2 * (g * gamma) ** 2
                alpha = roots / R
                alpha2 = alpha ** 2
                alpha2D = alpha2 * D

                summands = (
                    2 * alpha2D * delta - 2 +
                    2 * torch.exp(-alpha2D * delta) +
                    2 * torch.exp(-alpha2D * Delta) -
                    torch.exp(-alpha2D * (Delta - delta)) -
                    torch.exp(-alpha2D * (Delta + delta))
                ) / (D ** 2 * alpha ** 6 * (radius ** 2 * alpha2 - 1))

                S = summands.sum()

                E = torch.exp(first_factor * S)
                return E
            

            E_perpendicular[mask] = perpendicular_attenuation(
            g_perp[mask],
            delta[mask],
            Delta[mask],
            diameter,
            lambda_perp,
            gamma,
             _CYLINDER_TRASCENDENTAL_ROOTS.to(g_perp.device)
            )

            return E_parallel * E_perpendicular

    
class Astrosticks:
    '''
    Astrosticks model for diffusion MRI signal.
    This model assumed randomly-oriented sticks in all directions, with free diffusion in the direction of the sticks.
    
    Attributes:
        parameter_ranges (list): Ranges for the parameters [Dpar].
        param_names (list): Names of the parameters.
        n_parameters (int): Number of parameters.
        spherical_mean (bool): Whether the model is spherically-averaged or not.
        
    Methods:
        __init__: Initializes the model with parameter ranges and names.
        __call__(grad, parameters): Computes the diffusion signal based on the gradient and parameters.
    '''
    def __init__(self):
        self.parameter_ranges = [[0.5, 3]]
        self.parameter_names      = ['D_par']
        self.n_parameters         = 1
        self.spherical_mean   = True


    def __call__(self, grad, parameters):
        b_values = grad.bvalues
        D_par    = parameters[:, 0].unsqueeze(1)
    
        pi_tensor = torch.tensor(torch.pi)

        S = np.ones_like(b_values)
        S = ((torch.sqrt(pi_tensor) * torch.erf(torch.sqrt(b_values * D_par))) /
                    (2 * torch.sqrt(b_values * D_par)))

        return S
    
class Astrosticks_fixed:
    def __init__(self):
        self.parameter_ranges = [[2, 2]]
        self.parameter_names      = ['D_par']
        self.n_parameters         = 1
        self.spherical_mean   = True


    def __call__(self, grad, parameters):
        b_values = grad.bvalues
        D_par    = parameters[:, 0].unsqueeze(1)
    
        pi_tensor = torch.tensor(torch.pi)

        S = np.ones_like(b_values)
        S = ((torch.sqrt(pi_tensor) * torch.erf(torch.sqrt(b_values * D_par))) /
                    (2 * torch.sqrt(b_values * D_par)))


        return S