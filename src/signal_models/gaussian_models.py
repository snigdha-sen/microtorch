import torch
from src.utils.geometry import sphere2cart

class Ball:
    """
    A class representing a ball model for diffusion MRI.
    This model computes the signal based on the diffusion parameters and gradient directions.

    Attributes:
        parameter_ranges (list): Ranges for the parameters.
        parameter_names (list): Names of the parameters.
        n_parameters (int): Number of parameters.
        spherical_mean (bool): Indicates if the model is spherically averaged.

    Methods:
        __init__(): Initializes the ball model with parameter ranges and names.
        __call__(grad, parameters): Computes the signal based on the gradient and parameters.
    """
    def __init__(self):
        self.parameter_ranges   = [[.001, 3]]
        self.parameter_names        = ['D']
        self.n_parameters           = 1
        self.spherical_mean     = False

    def __call__(self, grad, parameters):    
        
        D        = parameters[:, 0].unsqueeze(1) 
        b_values = grad.bvalues

        S = torch.exp(-b_values * D)

        return S
    
class Msdki: ## are we keeping this in for the first iteration?
    """ 
    A class representing a MSDKI model for diffusion MRI.
    This model computes the signal based on the diffusion parameters and gradient directions. 

    Attributes:
        parameter_ranges (list): Ranges for the parameters.
        parameter_names (list): Names of the parameters.
        n_parameters (int): Number of parameters.
        spherical_mean (bool): Indicates if the model is spherically averaged.

    Methods:
        __init__(): Initializes the MSDKI model with parameter ranges and names.
        __call__(grad, parameters): Computes the signal based on the gradient and parameters.
    """
    def __init__(self):        
        self.parameter_ranges   = [[0.001, 3], [0.001, 2]]        
        self.parameter_names        = ['D', 'K']        
        self.n_parameters           = 2
        self.spherical_mean     = True
    
    def __call__(self, grad, parameters):
        b_values = grad.bvalues
        
        D = parameters[:,0].unsqueeze(1)
        K = parameters[:,1].unsqueeze(1)
                
        S = torch.exp(-b_values*D + (b_values**2 * D**2 * K / 6)) 

        return S
    
class Zeppelin:
    """
    A class representing a Zeppelin model for diffusion MRI.
    This model computes the signal based on the diffusion parameters and gradient directions.

    Attributes:
        parameter_ranges (list): Ranges for the parameters.
        parameter_names (list): Names of the parameters.
        n_parameters (int): Number of parameters.
        spherical_mean (bool): Indicates if the model is spherically averaged.

    Methods:
        __init__(): Initializes the Zeppelin model with parameter ranges and names.
        __call__(grad, parameters): Computes the signal based on the gradient and parameters.

        def _attenuation_zeppelin(bvals, lambda_par, lambda_perp, n, mu):
    "Signal attenuation for Zeppelin model."
    
    """
    def __init__(self):
        self.parameter_ranges = [[.001, 3], [.001, 1], [0, torch.pi], [-torch.pi, torch.pi]]
        self.parameter_names      = ['Dpar', 'k', 'theta', 'phi']
        self.n_parameters         = 4
        self.spherical_mean   = False


    def __call__(self, grad, parameters):                   
        b_vectors = grad.bvecs
        b_values = grad.bvalues

        Dpar = parameters[:, 0].unsqueeze(1)
        k = parameters[:, 1].unsqueeze(1)
        Dper = k*Dpar
        theta = parameters[:, 2].unsqueeze(1)
        phi = parameters[:, 3].unsqueeze(1)

        mu = sphere2cart(theta, phi)        # (N, 3)
        mu = mu / torch.norm(mu, dim=1, keepdim=True)

        # Gradient direction n
        n = b_vectors                            # (M, 3)

        # Parallel component
        mag_par = torch.sum(n * mu, dim=1, keepdim=True)

        # Perpendicular projection
        I = torch.eye(3, device=mu.device)
        mu_perp = I - mu.unsqueeze(2) @ mu.unsqueeze(1)
        proj = torch.matmul(mu_perp, n.unsqueeze(2)).squeeze(2)

        mag_perp = torch.norm(proj, dim=1, keepdim=True)

        S = torch.exp(
            -b_values * (
                Dpar * mag_par**2 +
                Dper * mag_perp**2
            )
        )

        return S