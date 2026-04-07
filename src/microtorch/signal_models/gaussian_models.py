import torch
from microtorch.utils.geometry import sphere2cart

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
        self.spherical_mean     = None

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
        # this goes off to very high values for large b, so we clamp it to avoid numerical issues
        K = parameters[:, 1].unsqueeze(1)
        K = torch.clamp(K, min=torch.tensor(0.0), max = (6 / (torch.max(b_values) * D)) )
                
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
    
    """
    def __init__(self):
        self.parameter_ranges = [[.001, 3], [.001, 1.0], [-torch.pi/2, torch.pi/2], [0, torch.pi]]
        self.parameter_names      = ['Dpar', 'k', 'theta', 'phi']
        self.n_parameters         = 4
        self.spherical_mean   = False


    def __call__(self, grad, parameters):
        g = grad.bvecs              # (N,3)
        b = grad.bvalues            # (N,)

        Dpar = parameters[:, 0:1]   # (B,1)
        k    = parameters[:, 1:2]   # (B,1)
        Dper = k * Dpar

        theta = parameters[:, 2]
        phi   = parameters[:, 3]

        mu = sphere2cart(theta, phi)       # expect (B,3)
        if mu.shape[-1] != 3:
            mu = mu.T
        mu = mu / (torch.norm(mu, dim=1, keepdim=True) + 1e-12)  # (B,3)

        # cosine between mu and gradient directions
        mag_par = mu @ g.T                      # (B,N)
        mag_par2 = mag_par * mag_par
        mag_perp2 = 1.0 - mag_par2

        b = b.view(1, -1)                   # (1,N)
        S = torch.exp(-b * (Dpar * mag_par2 + Dper * mag_perp2))
        return S
    
import torch




class Ballt2:
    """
    A class representing a ball with T2 decay model for diffusion MRI.
    This model computes the signal based on the diffusion parameters, gradient directions, and TEs.

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
        self.parameter_ranges   = [[.001, 3.], [0.01, 0.5]]
        self.parameter_names        = ['D', 'T2']
        self.n_parameters           = 2
        self.spherical_mean     = None

    def __call__(self, grad, parameters):    
        b_values = grad.bvalues
        TE = grad.TE
     
        D        = parameters[:, 0].unsqueeze(1) 
        T2       = parameters[:, 1].unsqueeze(1)    

        S = torch.exp(-b_values * D) * torch.exp(-(TE - torch.min(TE)) / T2)

        return S
    
    
    
    

class Tensor:
    """
    A class representing a full tensor model for diffusion MRI.
    This model computes the signal based on the diffusion parameters, gradient directions, and TEs.

    Attributes:
        parameter_ranges (list): Ranges for the parameters.
        parameter_names (list): Names of the parameters.
        n_parameters (int): Number of parameters.
        spherical_mean (bool): Indicates if the model is spherically averaged.

    Methods:
        __init__(): Initializes the Zeppelin model with parameter ranges and names.
        __call__(grad, parameters): Computes the signal based on the gradient and parameters.
        
    The full diffusion tensor is parameterized by:
        [Dpar, k1, k2, theta, phi, psi]

    where:
      - Dpar is the diffusivity in the principal direction
      - k1 and k2 are the ratios of the diffusivities in the secondary directions to Dpar
      - theta, phi define the principal direction e1
      - psi defines the rotation of the secondary eigenvectors around e1
    
    """
    
    def __init__(self):
        self.parameter_ranges = [
            [0.001, 3.0],          # Dpar
            [0.0, 1.0],            # k1
            [0.0, 1.0],            # k2
            [-torch.pi/2, torch.pi/2],   # theta
            [0.0, torch.pi],       # phi
            [0.0, torch.pi],   # psi
        ]
        self.parameter_names = ['Dpar', 'k1', 'k2', 'theta', 'phi', 'psi']
        self.n_parameters = 6
        self.spherical_mean = False

    def __call__(self, grad, parameters):
        g = grad.bvecs
        b = grad.bvalues

        Dpar = parameters[:, 0:1]
        k1   = parameters[:, 1:2]
        k2   = parameters[:, 2:3]

        lam1 = Dpar
        lam2 = k1 * Dpar
        lam3 = k1 * k2 * Dpar

        theta = parameters[:, 3]
        phi   = parameters[:, 4]
        psi   = parameters[:, 5]

        sintheta = torch.sin(theta)
        costheta = torch.cos(theta)
        sinphi = torch.sin(phi)
        cosphi = torch.cos(phi)
        sinpsi = torch.sin(psi)
        cospsi = torch.cos(psi)
        
        #principal direction
        e1 = torch.stack([
            sintheta * cosphi,
            sintheta * sinphi,
            costheta
        ], dim=1)

        #(u, v) are orthonormal basis vectors perpendicular to e1
        u = torch.stack([
            costheta * cosphi,
            costheta * sinphi,
            -sintheta
        ], dim=1)

        v = torch.stack([
            -sinphi,
            cosphi,
            torch.zeros_like(sinpsi)
        ], dim=1)

        # rotate (u, v) by psi around e1 to give secondary eigenvectors e2, e3
        e2 = cospsi[:, None] * u + sinpsi[:, None] * v
        e3 = -sinpsi[:, None] * u + cospsi[:, None] * v

        e1 = e1 / (torch.norm(e1, dim=1, keepdim=True) + 1e-12)
        e2 = e2 / (torch.norm(e2, dim=1, keepdim=True) + 1e-12)
        e3 = e3 / (torch.norm(e3, dim=1, keepdim=True) + 1e-12)

        proj1 = e1 @ g.T
        proj2 = e2 @ g.T
        proj3 = e3 @ g.T

        D = (
            lam1 * proj1 * proj1 +
            lam2 * proj2 * proj2 +
            lam3 * proj3 * proj3
        )

        S = torch.exp(-b.view(1, -1) * D)
        return S
