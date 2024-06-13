from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from utils.util_function import sphere2cart, cart2sphere
from utils.utils_wm import WM_model, spherical_harmonics_directions

__all__ = [     
    'Ball',
    'Stick',
    'MSDKI',
    't2_adc',  
    't1_smdt',
    'get_model_nparams',
    'StandardWM']


class Ball:
    def __init__(self):
        self.parameter_ranges = [[.001, 3]]
        self.param_names = ['D']
        self.n_params = 1
        self.spherical_mean = False


    def __call__(self, grad, params):    
        
        D = params[:, 0].unsqueeze(1) # ADC

        b_values = grad.bvalues

        print(D.shape)
        print(b_values.shape)
        S = torch.exp(-b_values * D)
        

        return S


class Stick:
    def __init__(self):
        self.parameter_ranges = [[.001, 3], [0, torch.pi], [-torch.pi, torch.pi]]
        
        self.param_names = ['Dpar', 'theta', 'phi']
        self.n_params = 3
        self.spherical_mean = False


    def __call__(self, grad, params):                   
        bvecs = grad.bvecs
        b_values = grad.bvalues

        Dpar = params[:, 0].unsqueeze(1)
        theta = params[:, 1].unsqueeze(1)
        phi = params[:, 2].unsqueeze(1)

        n = sphere2cart(theta, phi)
        
        S = torch.exp(-b_values * Dpar * torch.mm(bvecs, n).t() ** 2)                          
        
     
        return S

class MSDKI:
    def __init__(self):        
        self.parameter_ranges = [[0.001, 3], [0.001, 2]]        
        self.param_names = ['D', 'K']        
        self.n_params = 2
        self.spherical_mean = True
    
    def __call__(self, grad, params):
        b_values = grad.bvalues
        
        D = params[:,0].unsqueeze(1)
        K = params[:,1].unsqueeze(1)
                
        S = torch.exp(-b_values*D + (b_values**2 * D**2 * K / 6)) 

        return S

class Sphere:
    def __init__(self):
        self.parameter_rangers = [[0.001, 15]]
        self.param_names = ['radius']
        self.n_params = 1
        self.spherical_mean = True

    def __call__(self, grad, params):
        b_values = grad.bvalues
        delta = grad.delta
        Delta = grad.Delta

        D = 2 # D_IC
        radius = params[:,0].unsqueeze(1)

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
        first_factor = -2*(gamma*self.gradient_strength)**2 / 2
 
        delta = self.delta.unsqueeze(0).unsqueeze(2)
        Delta = self.Delta.unsqueeze(0).unsqueeze(2)
        
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
            summands.sum()
        )

        return S
    
class Astrosticks:
    def __init__(self):
        self.parameter_ranges = [[0.5, 3]]
        self.param_names = ['D_par']
        self.n_params = 1
        self.spherical_mean = True

    def __call__(self, grad, params):
        b_values = grad.bvalues

        D_par = params[:, 0].unsqueeze(1)
        S = np.ones_like(b_values)
        S = ((np.sqrt(np.pi) * torch.erf(np.sqrt(b_values * D_par))) /
                    (2 * np.sqrt(b_values * D_par)))

        return S


class Zeppelin:
    def __init__(self):
        self.parameter_ranges = [[.001, 3], [.001, 1], [0, torch.pi], [-torch.pi, torch.pi]]
        
        self.param_names = ['Dpar', 'k', 'theta', 'phi']
        self.n_params = 4
        self.spherical_mean = False


    def __call__(self, grad, params):                   
        b_vectors = grad.bvecs
        b_values = grad.bvalues

        Dpar = params[:, 0].unsqueeze(1)
        k = params[:, 1].unsqueeze(1)
        Dper = k*Dpar
        theta = params[:, 2].unsqueeze(1)
        phi = params[:, 3].unsqueeze(1)

        n = sphere2cart(theta, phi)

        S = torch.exp(1/3.0 * b_values * (Dpar - Dper) - b_values/3.0 * (Dper + 2*Dpar) - b_values * (torch.mm(b_vectors),n)**2) * (Dpar - Dper)
             
        return S
    

class Standard_WM:

    def __init__(self): 

        self.order = 2 #have to figure something out for this
        order = 2
        nSH = int((order + 1) * (order + 2) / 2)
        self.parameter_ranges = [[0,1], [0, 3], [0, 3], [0, 3], [0, 1],[-0.5, 0.5],[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5] ]  # pas ranges aan      
        self.param_names = ['S0', 'Di', 'De', 'Dp', 'f', 'p2_2', 'p2_1', 'p20', 'p21', 'p22' ]  #consider order 2 for now
        self.n_params = 10
        self.spherical_mean = False ##output afhankelijk van gradient richting

    
    def __call__(self, grad, params):
        order = 2
        b_values = grad.bvalues
        b_vectors = grad.bvecs

        # is not really delta 
        if grad.bdelta == None:
            bdelta = 1
        else:
            bdelta = grad.bdelta
        
        p00 = 1/torch.sqrt(torch.tensor(4)*torch.pi)*torch.ones_like(params[:,4].unsqueeze(1))

        S0 = params[:,0].unsqueeze(1)
        Di = params[:,1].unsqueeze(1)
        De = params[:,2].unsqueeze(1)
        Dp = params[:,3].unsqueeze(1)
        f  = params[:,4].unsqueeze(1)
        
        fODF = [p00,params[:,5].unsqueeze(1),params[:,6].unsqueeze(1),params[:,7].unsqueeze(1),params[:,8].unsqueeze(1),params[:,9].unsqueeze(1)]

        # Compute spherical harmonics
        Ysh = torch.from_numpy(spherical_harmonics_directions(b_vectors, order)) #this might be a problem. Regarding torch/scipy. Can maybe also be used as input?
                
        S = WM_model(order, b_values, bdelta, Ysh, f, Di, De, Dp, fODF, S0)

        return S
    

class t1_smdt:

    def __init__(self):

        self.parameter_ranges = [[0.5, 3], [.001, 1], [0, 100000],  [0.001,1000000]] #no idea if the ranges are solid
        self.param_names = ['D_par', 'k', 'T1', 'S0' ]
        self.n_params = 4
        self.spherical_mean = False


    def __call__(self, grad, params):
        

        b_vecs = grad.bvecs # we assume that the first three columns contain the diffusion gradient direction in Cartesian coordinates
        b_values = grad.bvalues # b-value assumed in the fourth position in s/mm^2

        # b_values [b_values ==0] = 0.01 # to potentially avoid divisions by 0
        b_values = b_values /1000.0 # b-values in ms/um^2
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
        S = sfac * S0 * torch.abs(1.0 - torch.exp(-TI/T1) - (torch.exp(-TS/T1)) * torch.exp(-TI/T1)) * torch.erf(torch.sqrt(b_values*(Dpar-Dperp)))/torch.sqrt(b_values*(Dpar-Dperp))

        return S
'''   
    class Cylinder:

        def __init__(self, grad, params):

            self.parameter_ranges = [[0, torch.pi], [-torch.pi, torch.pi], [.001, 3], [.001, 10]] 
            self.param_names = ['theta', 'phi', 'D_par', 'radius']
            self.n_params = 3
            self.spherical_mean = False

        def __call__(self, grad, params):

'''







