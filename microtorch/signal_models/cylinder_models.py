import numpy as np
import torch
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
    
'''
    class Cylinder: ## would be good to have a working version of this

        def __init__(self, grad, parameters):

            self.parameter_ranges = [[0, torch.pi], [-torch.pi, torch.pi], [.001, 3], [.001, 10]] 
            self.parameter_names = ['theta', 'phi', 'D_par', 'radius']
            self.n_parameters = 3
            self.spherical_mean = False

        def __call__(self, grad, parameters):

            b_vectors    = grad.bvecs
            b_values = grad.bvalues
            delta = grad.delta
            Delta = grad.Delta

            _CYLINDER_TRASCENDENTAL_ROOTS = np.sort(special.jnp_zeros(1, 100))

        
        
    DMIPY 
    class C4CylinderGaussianPhaseApproximation(
        ModelProperties, AnisotropicSignalModelProperties):
    r""" The Gaussian phase model [1]_ - a cylinder with finite radius -
    typically used for intra-axonal diffusion. The perpendicular diffusion is
    modelled after Van Gelderen's solution for the disk. It is dependent on
    gradient strength, pulse separation and pulse length.

    Parameters
    ----------
    mu : array, shape(2),
        angles [theta, phi] representing main orientation on the sphere.
        theta is inclination of polar angle of main angle mu [0, pi].
        phi is polar angle of main angle mu [-pi, pi].
    lambda_par : float,
        parallel diffusivity in 10^9 m^2/s.
    diameter : float,
        cylinder (axon) diameter in meters.


    References
    ----------
    .. [1] Van Gelderen et al. "Evaluation of Restricted Diffusion in
            Cylinders. Phosphocreatine in Rabbit Leg Muscle"
            Journal of Magnetic Resonance Series B (1994)
    """

    _required_acquisition_parameters = [
        'bvalues', 'gradient_directions',
        'gradient_strengths', 'delta', 'Delta']

    _parameter_ranges = {
        'mu': ([0, np.pi], [-np.pi, np.pi]),
        'lambda_par': (.1, 3),
        'diameter': (1e-2, 20)
    }
    _parameter_scales = {
        'mu': np.r_[1., 1.],
        'lambda_par': DIFFUSIVITY_SCALING,
        'diameter': DIAMETER_SCALING
    }
    _parameter_types = {
        'mu': 'orientation',
        'lambda_par': 'normal',
        'diameter': 'cylinder'
    }
    _model_type = 'CompartmentModel'
    _CYLINDER_TRASCENDENTAL_ROOTS = np.sort(special.jnp_zeros(1, 100))

    def __init__(
        self,
        mu=None, lambda_par=None,
        diameter=None,
        diffusion_perpendicular=CONSTANTS['water_in_axons_diffusion_constant']
    ):
        self.mu = mu
        self.lambda_par = lambda_par
        self.diffusion_perpendicular = diffusion_perpendicular
        self.gyromagnetic_ratio = CONSTANTS['water_gyromagnetic_ratio']
        self.diameter = diameter

    def perpendicular_attenuation(
        self, gradient_strength, delta, Delta, diameter
    ):
        "Calculates the cylinder's perpendicular signal attenuation."
        D = self.diffusion_perpendicular
        gamma = self.gyromagnetic_ratio
        return _attenuation_perpendicular_gaussian_phase(
            diameter, gradient_strength, delta, Delta,
            D, gamma, self._CYLINDER_TRASCENDENTAL_ROOTS)

    def __call__(self, acquisition_scheme, **kwargs):
        r
        bvals = acquisition_scheme.bvalues
        n = acquisition_scheme.gradient_directions
        g = acquisition_scheme.gradient_strengths
        delta = acquisition_scheme.delta
        Delta = acquisition_scheme.Delta

        diameter = kwargs.get('diameter', self.diameter)
        lambda_par = kwargs.get('lambda_par', self.lambda_par)
        mu = kwargs.get('mu', self.mu)
        mu = utils.unitsphere2cart_1d(mu)
        mu_perpendicular_plane = np.eye(3) - np.outer(mu, mu)
        magnitude_perpendicular = np.linalg.norm(
            np.dot(mu_perpendicular_plane, n.T),
            axis=0
        )
        E_parallel = _attenuation_parallel_stick(bvals, lambda_par, n, mu)
        E_perpendicular = np.ones_like(g)
        g_perp = g * magnitude_perpendicular

        g_nonzero = g_perp > 0
        # for every unique combination get the perpendicular attenuation
        unique_deltas = np.unique([acquisition_scheme.shell_delta,
                                   acquisition_scheme.shell_Delta], axis=1)
        for delta_, Delta_ in zip(*unique_deltas):
            mask = np.all([g_nonzero, delta == delta_, Delta == Delta_],
                          axis=0)
            E_perpendicular[mask] = self.perpendicular_attenuation(
                g_perp[mask], delta_, Delta_, diameter
            )
        return E_parallel * E_perpendicular

    '''
    
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
        self.param_names      = ['D_par']
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
        self.param_names      = ['D_par']
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