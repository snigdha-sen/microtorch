import numpy as np
import torch
from utils.utils_wm import WM_model, spherical_harmonics_directions

class Standard_wm: ## check with leon -- add unit tests?
    """
    Standard White Matter model for diffusion MRI signal
    
    Attributes:
        parameter_ranges (list): Ranges for the parameters.
        param_names (list): Names of the parameters.
        n_parameters (int): Number of parameters.
        spherical_mean (bool): Whether the model is spherically-averaged or not.
        
    Methods:
        __init__: Initializes the model with parameter ranges and names.
        __call__(grad, parameters): Computes the diffusion signal based on the gradient and parameters.
    """

    def __init__(self): 

        order = 2 # for now only lmax = 2
        self.order = order 
        nSH = int((order + 1) * (order + 2) / 2)
        self.parameter_ranges = [[0,1], [0, 3], [0, 3], [0, 3], [0, 1],[-1, 1],[-1, 1], [-1, 1], [-1, 1], [-1, 1] ]  # pas ranges aan - adjust ranges
        self.parameter_names = ['S0', 'Di', 'De', 'Dp', 'f', 'p2_2', 'p2_1', 'p20', 'p21', 'p22' ]  #consider order 2 for now
        self.n_parameters = 10
        self.spherical_mean = False ##output afhankelijk van gradient richting - output dependent on gradient direction

    
    def __call__(self, grad, params):

        order = 2
        b_values = grad.bvalues
        b_vectors = grad.bvecs

        if b_vectors.ndim != 2:
            raise ValueError(f"b_vectors must be 2D, got shape {b_vectors.shape}")

        if b_vectors.shape[1] == 3:
            pass  # already correct (N, 3)
        elif b_vectors.shape[0] == 3:
            b_vectors = b_vectors.T  # (3, N) → (N, 3)
        else:
            raise ValueError(f"Invalid b_vectors shape: {b_vectors.shape}")

        # is not really delta 
        if grad.bdelta == None:
            bdelta = 1
        else:
            bdelta = grad.bdelta

        b_values = b_values.ravel()
        bdelta = bdelta.ravel()

        p00 = 1/torch.sqrt(torch.tensor(4)*torch.pi)*torch.ones_like(params[:,4].unsqueeze(1))

        S0 = params[:,0].unsqueeze(1)
        Di = params[:,1].unsqueeze(1)
        De = params[:,2].unsqueeze(1)
        Dp = params[:,3].unsqueeze(1)
        f  = params[:,4].unsqueeze(1)
        
        fODF = [p00,params[:,5].unsqueeze(1),params[:,6].unsqueeze(1),params[:,7].unsqueeze(1),params[:,8].unsqueeze(1),params[:,9].unsqueeze(1)]

        # Compute spherical harmonics
        Ysh = torch.from_numpy(spherical_harmonics_directions(b_vectors, order)) #Now we go back to cpu to calc sph_harm, should fix this. Probably can use it as input

        S = WM_model(order, b_values, bdelta, Ysh, f, Di, De, Dp, fODF, S0)

        return S
    