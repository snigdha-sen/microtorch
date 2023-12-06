from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils

__all__ = [     
    'Ball',
    'Stick',
    'MSDKI',
    'zeppelin',
    't2_adc',  
    't1_smdt',
    'get_model_nparams']



class Ball:
    def __init__(self):
        self.parameter_ranges = [[.001, 3]]
        self.param_names = ['D']
        self.n_params = 1
        self.spherical_mean = False


    def __call__(self, grad, params):    
        
        D = params[:, 0].unsqueeze(1) # ADC

        b_values = grad[:, 3]

        S = torch.exp(-b_values * D)

        return S


class Stick:
    def __init__(self):
        self.parameter_ranges = [[.001, 3], [0, torch.pi], [-torch.pi, torch.pi]]
        
        self.param_names = ['Dpar', 'theta', 'phi']
        self.n_params = 3
        self.spherical_mean = False


    def __call__(self, grad, params):                   
        g = grad[:, 0:3]
        b_values = grad[:, 3]

        Dpar = params[:, 0].unsqueeze(1)
        theta = params[:, 1].unsqueeze(1)
        phi = params[:, 2].unsqueeze(1)

        n = sphere2cart(theta, phi)
        
        S = torch.exp(-b_values * Dpar * torch.mm(g, n).t() ** 2)                          
        
     
        return S


class MSDKI:
    def __init__(self):        
        self.parameter_ranges = [[0.001, 3], [0.001, 2]]        
        self.param_names = ['D', 'K']        
        self.n_params = 2
        self.spherical_mean = True
    
    def __call__(self, grad, params):
        b_values = grad[:, 3] 
        
        D = params[:,0].unsqueeze(1)
        K = params[:,1].unsqueeze(1)
                
        S = torch.exp(-b_values*D + (b_values**2 * D**2 * K / 6)) 

        return S




# class Zeppelin(grad, params):
#     # This zeppelin model is based on the implementation shown in Tax et al. (2021; NeuroImage, doi: 10.1016/j.neuroimage.2021.117967)
#     # input:
#     # grad: acquisition parameters
#     # params: parameters estimated for signal prediction. Order: 
#     # 1. theta 
#     # 2. phi (angles of the first eigenvector with theta) 
#     # 3. Dpar (parallel diffusivity)
#     # 4. kperp (k*Dpar = Dperp or perpendicular diffusivity). k is a parameter between 0 and 1 to make sure that Dperp <= Dpar
#     # 5. T2*
#     # 6. T1
#     # 7. S0



#     g = grad[:,0:3] # we assume that the first three columns contain the diffusion gradient direction in Cartesian coordinates
#     bvals = grad[:,3].unsqueeze(1) # b-value assumed in the fourth position in s/mm^2
#     bvals = bvals/1000.0 # b-values in ms/um^2
#     TI = grad[:,5].unsqueeze(1) # inversion time assumed in the sixth position in ms
#     TE = grad[:,4].unsqueeze(1) # echo time assumed in the fifth position in ms
#     TR = grad[:,6].unsqueeze(1) # repetition time assumed in the seventh position in ms
    
#     # the implementation works with TD (delay time) instead of TE, assuming a multi-echo sequence
#     TD = TE - torch.min(TE)

#     # parameters
#     theta = params[:,0].unsqueeze(1)
#     phi = params[:,1].unsqueeze(1)
#     # we transform into Cartesian coordinates
#     n = sphere2cart(theta,phi)
#     Dpar = params[:,2].unsqueeze(1)
#     kperp = params[:,3].unsqueeze(1)
#     Dperp = kperp*Dpar
#     T2star = params[:,4].unsqueeze(1)
#     T1 = params[:,5].unsqueeze(1)
#     S0 = params[:,6].unsqueeze(1)
    
#     # the implementation depends on tensor encoding. We assume linear tensor encoding
#     b_delta = 1.0

#     # signal representation: s = s0 * e^(-b:D) * abs(1 - 2*e^(-TI/T1) + e^(-TR/T1)) * e^(-TD/T2*)
#     # we begin obtaining -b:D, called b_D here for simplicity
#     b_D = b_delta/3.0 * bvals * (Dpar - Dperp) - bvals/3.0 * (Dperp + 2*Dpar) - bvals * b_delta * (torch.mm(g,n)**2) * (Dpar - Dperp)

#     S = S0 * torch.exp(b_D) * torch.abs(1.0 - 2.0 * torch.exp(-TI/T1) + torch.exp(-TR/T1)) * torch.exp(-TD/T2star)

#     return S


def t1_smdt(grad,params):
    # T1-spherical mean diffusion tensor representation from Grussu et al. (2021; Front Phys, doi: 10.3389/fphy.2021.752208)
    
    g = grad[:,0:3] # we assume that the first three columns contain the diffusion gradient direction in Cartesian coordinates
    bvals = grad[:,3].unsqueeze(1) # b-value assumed in the fourth position in s/mm^2
    bvals[bvals==0] = 0.01 # to potentially avoid divisions by 0
    bvals = bvals/1000.0 # b-values in ms/um^2
    TI = grad[:,5].unsqueeze(1) # inversion time assumed in the sixth position in ms
    TS = grad[:,4].unsqueeze(1) # saturation or preparation time assumed in the fifth position in ms

    # Constant factor employed in the equation
    sfac = 0.5 * np.sqrt(np.pi)

    # parameters
    Dpar = params[:,0].unsqueeze(1)
    kperp = params[:,1].unsqueeze(1)
    Dperp = kperp*Dpar
    T1 = params[:,2].unsqueeze(1)
    S0 = params[:,3].unsqueeze(1)

    # we obtain the signal
    S = sfac * S0 * torch.abs(1.0 - torch.exp(-TI/T1) - (torch.exp(-TS/T1)) * torch.exp(-TI/T1)) * torch.erf(torch.sqrt(bvals*(Dpar-Dperp)))/torch.sqrt(bvals*(Dpar-Dperp))

    return S

def Sphere(grad, params):

    _parameter_ranges = { 'Dpar': (.1, 3), 'theta': (0, torch.pi), 'phi': (-torch.pi, torch.pi) }

    _n_params = 3

    SPHERE_TRASCENDENTAL_ROOTS = np.r_[
        # 0.,
        2.081575978, 5.940369990, 9.205840145,
        12.40444502, 15.57923641, 18.74264558, 21.89969648,
        25.05282528, 28.20336100, 31.35209173, 34.49951492,
        37.64596032, 40.79165523, 43.93676147, 47.08139741,
        50.22565165, 53.36959180, 56.51327045, 59.65672900,
        62.80000055, 65.94311190, 69.08608495, 72.22893775,
        75.37168540, 78.51434055, 81.65691380, 84.79941440,
        87.94185005, 91.08422750, 94.22655255, 97.36883035,
        100.5110653, 103.6532613, 106.7954217, 109.9375497,
        113.0796480, 116.2217188, 119.3637645, 122.5057870,
        125.6477880, 128.7897690, 131.9317315, 135.0736768,
        138.2156061, 141.3575204, 144.4994207, 147.6413080,
        150.7831829, 153.9250463, 157.0668989, 160.2087413,
        163.3505741, 166.4923978, 169.6342129, 172.7760200,
        175.9178194, 179.0596116, 182.2013968, 185.3431756,
        188.4849481, 191.6267147, 194.7684757, 197.9102314,
        201.0519820, 204.1937277, 207.3354688, 210.4772054,
        213.6189378, 216.7606662, 219.9023907, 223.0441114,
        226.1858287, 229.3275425, 232.4692530, 235.6109603,
        238.7526647, 241.8943662, 245.0360648, 248.1777608,
        251.3194542, 254.4611451, 257.6028336, 260.7445198,
        263.8862038, 267.0278856, 270.1695654, 273.3112431,
        276.4529189, 279.5945929, 282.7362650, 285.8779354,
        289.0196041, 292.1612712, 295.3029367, 298.4446006,
        301.5862631, 304.7279241, 307.8695837, 311.0112420,
        314.1528990
    ]

    D = 2
    gamma = 2.67e2
    radius = r

    b_values = np.array([1e-6, 0.090, 1e-6, 0.500, 1e-6, 1.5, 1e-6, 2, 1e-6, 3])
    Delta = np.array([23.8, 23.8, 23.8, 31.3, 23.8, 43.8, 23.8, 34.3, 23.8, 38.8])
    delta = np.array([3.9, 3.9, 3.9, 11.4, 3.9, 23.9, 3.9, 14.4, 3.9, 18.9])

    gradient_strength = np.array([np.sqrt(b_values[i])/(gamma*delta[i]*np.sqrt(Delta[i]-delta[i]/3)) for i,_ in enumerate(b_values)])

    alpha = SPHERE_TRASCENDENTAL_ROOTS / radius
    alpha2 = alpha ** 2
    alpha2D = alpha2 * D

    first_factor = -2 * (gamma * gradient_strength) ** 2 / D
    
    summands = np.zeros((len(SPHERE_TRASCENDENTAL_ROOTS),len(b_values)))
    for i,_ in enumerate(delta):
        summands[:,i] = (
            alpha ** (-4) / (alpha2 * radius ** 2 - 2) *
            (
                2 * delta[i] - (
                    2 +
                    np.exp(-alpha2D * (Delta[i] - delta[i])) -
                    2 * np.exp(-alpha2D * delta[i]) -
                    2 * np.exp(-alpha2D * Delta[i]) +
                    np.exp(-alpha2D * (Delta[i] + delta[i]))
                ) / (alpha2D)
            )
        )
    
    S = np.exp(
        first_factor *
        summands.sum()
    )

    return S

def astrosticks(l):

    bvals = np.array([1e-6, 0.090, 1e-6, 0.500, 1e-6, 1.5, 1e-6, 2, 1e-6, 3])
    lambda_par = l
    S = np.ones_like(bvals)
    S = ((np.sqrt(np.pi) * erf(np.sqrt(bvals * lambda_par))) /
                (2 * np.sqrt(bvals * lambda_par)))

    return S

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


def msdki(grad,params):
    
    #D = torch.clamp(params[:, 0].unsqueeze(1), min = 0.01, max = 5)
    #K = torch.clamp(params[:, 1].unsqueeze(1), min= 0.001, max=3)
    D = params[:, 0].unsqueeze(1)
    K = params[:, 1].unsqueeze(1)
    
    bvals = grad[:,3]
        
    S = torch.exp(-bvals*D + (bvals**2 * D**2 * K / 6)) 
    
    return S


def get_model_nparams(model):
    if model=="ball_stick":
        return 5
    if model=="t2_adc":
        return 2
    if model=="msdki":
        return 2
    if model=="zeppelin":
        return 7
    if model=="t1_smdt":
        return 4




