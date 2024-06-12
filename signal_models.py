from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from scipy.special import sph_harm,erf

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
        self.spherical_mean = True


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

class Sphere:
    def __init__(self):
        self.parameter_rangers = [[0.001, 15]]
        self.param_names = ['radius']
        self.n_params = 1
        self.spherical_mean = True

    def __call__(self, grad, params):
        b_values = grad[:, 3]
        delta = grad[:, 4]
        Delta = grad[:, 5]
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
        alpha2D = alpha2 * 2
        alpha = alpha.unsqueeze(1)
        alpha2 = alpha2.unsqueeze(1)
        alpha2D = alpha2D.unsqueeze(1)

        gamma = 2.675987e2
        gradient_strength = np.array([np.sqrt(b_values[i])/(gamma*delta[i]*np.sqrt(Delta[i]-delta[i]/3)) for i,_ in enumerate(b_values)])
        first_factor = -2*(gamma*gradient_strength)**2 / 2
        
        summands = np.zeros((len(SPHERE_TRASCENDENTAL_ROOTS),len(b_values)))
        for i,_ in enumerate(delta):
            summands[:,i] = (
                alpha ** (-4) / (alpha2 * radius ** 2 - 2) *
                (
                    2 * delta[i] - (
                        2 +
                        torch.exp(-alpha2D * (Delta[i] - delta[i])) -
                        2 * torch.exp(-alpha2D * delta[i]) -
                        2 * torch.exp(-alpha2D * Delta[i]) +
                        torch.exp(-alpha2D * (Delta[i] + delta[i]))
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
        b_values = grad[:, 3]
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
        g = grad[:, 0:3]
        b_values = grad[:, 3]

        Dpar = params[:, 0].unsqueeze(1)
        k = params[:, 1].unsqueeze(1)
        Dper = k*Dpar
        theta = params[:, 2].unsqueeze(1)
        phi = params[:, 3].unsqueeze(1)

        n = sphere2cart(theta, phi)

        S = torch.exp(1/3.0 * b_values * (Dpar - Dper) - b_values/3.0 * (Dper + 2*Dpar) - b_values * (torch.mm(g,n)**2) * (Dpar - Dper))
             
        return S
    

class Standard_WM:

    def __init__(self, order): 

        self.order = order #have to figure something out for this
        nSH = int((order + 1) * (order + 2) / 2)
        self.parameter_ranges = [[0,1], [0, 3], [0, 3], [0, 3], [0, 1],[-0.5, 0.5],[-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5] ]  # pas ranges aan      
        self.param_names = ['S0', 'Di', 'De', 'Dp', 'f', 'p2_2', 'p2_1', 'p20', 'p21', 'p22' ]  #consider order 2 for now
        self.n_params = 10
        self.spherical_mean = False ##output afhankelijk van gradient richting

    
    def __call__(self, grad, order, params):
        b_values = grad[:, 3] 
        b_vectors = grad[:,0:3]

        bdelta =grad[:,4]
        p00 = 1/torch.sqrt(torch.tensor(4)*torch.pi)*torch.ones_like(params[:,4].unsqueeze(1))

        S0 = params[:,0].unsqueeze(1)
        Di = params[:,1].unsqueeze(1)
        De = params[:,2].unsqueeze(1)
        Dp = params[:,3].unsqueeze(1)
        f  = params[:,4].unsqueeze(1)
        
        fODF = [p00,params[:,5].unsqueeze(1),params[:,6].unsqueeze(1),params[:,7].unsqueeze(1),params[:,8].unsqueeze(1),params[:,9].unsqueeze(1)]

        # Compute spherical harmonics
        Ysh = torch.from_numpy(spherical_harmonics_directions(b_vectors, order)) #this might be a problem. Regarding torch/scipy. Can maybe also be used as input?
                
        S = model(order, b_values, bdelta, Ysh, f, Di, De, Dp, fODF, S0)

        return S


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

def standardwm(grad, order, params):
    b_values = grad[:, 3] 
    b_vectors = grad[:,0:3]
    bdelta = grad[:,4] #just for now maybe adjust to input parameter later
    p00 = 1/torch.sqrt(torch.tensor(4)*torch.pi)*torch.ones_like(params[:,4].unsqueeze(1))

    S0 = params[:,0].unsqueeze(1)
    Di = params[:,1].unsqueeze(1)
    De = params[:,2].unsqueeze(1)
    Dp = params[:,3].unsqueeze(1)
    f  = params[:,4].unsqueeze(1)
    fODF = [p00,params[:,5].unsqueeze(1),params[:,6].unsqueeze(1),params[:,7].unsqueeze(1),params[:,8].unsqueeze(1),params[:,9].unsqueeze(1)]

    # Compute spherical harmonics
    Ysh = torch.from_numpy(spherical_harmonics_directions(b_vectors, order)) #this might be a problem. Regarding torch/scipy. Can maybe also be used as input?

    S = model(order, b_values, bdelta, Ysh, f, Di, De, Dp, fODF, S0)

    return S


def get_model_nparams(model): ## lets change this
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
    if model=='standardwm':
        return 10
    if model=="verdict":
        return 4


##check if matlab model and python model correspond. Take bvalues from hcp. order 2, random sample Di,De,Dp [0,3]. Di>De>Dp. fODF: use watson distribution to sample from. Simulate one voxel first
def model(order, bvals, bdelta, Ysh, f, Di, De, Dp, fODF, S0):

    """
    calculation of Signal intensity using parameters and other defined functions

    input:
    see WM standard for parameters
    :param Ysh: Something with spherical harmonics Y
    :param: fODF: not really sure yet

    :return S
    """
    nSh = int((order + 1) * (order + 2) / 2) #not really know what this is yet
    nb = len(bvals) #number of bvalues
    nsamples = len(S0)
    K = K2comp_fast(order, bvals, bdelta, S0, Di, De, Dp, f)
    #reserve arrays with zeros
    Slm = torch.zeros((nSh, nsamples, nb))

    #print(np.shape(fODF))
    #multiply signal with corresponding orientation weigth (pii) # order l=1 is constant, falls into S0
    #only works for order 2
    Slm[0,:,:] = K[0] #l=0, m=0
    Slm[1,:,:] = fODF[1]*K[2] #l=2, m=-2
    Slm[2,:,:] = fODF[2]*K[2] #l=2, m=-1
    Slm[3,:,:] = fODF[3]*K[2] #l=2, m=0
    Slm[4,:,:] = fODF[4]*K[2] #l=2, m= 1
    Slm[5,:,:] = fODF[5]*K[2] #l=2, m= 2


    #normalize according to SMI
    Nl = torch.tensor([torch.sqrt(torch.tensor(4)*torch.pi),torch.sqrt(torch.tensor(20)*torch.pi),torch.sqrt(torch.tensor(20)*torch.pi),torch.sqrt(torch.tensor(20)*torch.pi),torch.sqrt(torch.tensor(20)*torch.pi), torch.sqrt(torch.tensor(20)*torch.pi) ])
    Nl = Nl.view(-1, 1, 1)
    Slm = Slm*Nl
    # Transpose Ysh to match dimensions for matrix multiplication
    Ysh_transposed = Ysh.transpose(0, 1)  
    Ysh_transposed = Ysh_transposed.float()
    Slm = Slm.float()

    # Perform matrix multiplication
    mm = torch.matmul(Ysh_transposed[None], Slm.swapaxes(0,1))
    #multiply with corresponding spherical harmonic value and take the diagonal elements

    S = torch.diagonal(mm,dim1=1, dim2=2) #I would say that this is the signal
    

    return S


def K2comp_fast(order, bvalues, bdelta, S0, Di, De, Dp, f):

    """
    not really sure yet what we are calculating here. Probably something with cross terms?
    Maybe we are calculating k up to second order. Look at expression k_l0/A.10 from Tax et al.

    input:
    see WM standard for parameters


    :return [K, dK]
    """
    
    #create dictionary for exponential factors due to diffusion signal decay
    K = {}   
    b = bvalues
    bd = bdelta
    for l in [0,2]:
        
        y = b*bdelta*Di

        # Linearly interpolate clot and dclot at points y # why do we need to interpolate. 
        #tmp = np.interp(y, np.squeeze(bD), np.squeeze(clot[l]))
        c_ias = analytical_sol(y,l)  # Extract the first column as c_ias
        
        # Calculate y = b * bd * (De - Dp)
        y_new = b * bdelta * (De - Dp)
    
        # Linearly interpolate clot and dclot at points y_new
        #tmp = np.interp(y_new, np.squeeze(bD), np.squeeze(clot[l]))
        #calculate analytically
        c_eas = analytical_sol(y_new,l)  # Extract the first column as c_eas
        
        # do not really see what K and dK are. K seems logical but what is dK doing. Doesnt it miss a factor 2? # check with chantal if this expression is correct vs tax et al
        K[l] = S0*(f * torch.exp((Di * b * bd) / 3 - (Di * b) / 3) * c_ias + (1 - f) * torch.exp((De * b * bd) / 3 - (De * b) / 3 - (2 * Dp * b) / 3 - (Dp * b * bd) / 3) * c_eas)



    return K


def analytical_sol(a,n):

    #function is not continious in a = 0, so approx by a -> 0.00001
    a[a == 0.] = 0.001
    a = torch.tensor(a,dtype=torch.complex64)

    if n ==0:
        analytical_sol = (torch.sqrt(torch.tensor(torch.pi)) * erf(torch.sqrt(a))) / (2 * torch.sqrt(a))
    if n ==2:
        analytical_sol = (-6 * torch.sqrt(a) * torch.exp(-a) + (3 - 2 * a) * torch.sqrt(torch.tensor(torch.pi)) * erf(torch.sqrt(a))) / (8 * a*torch.sqrt(a))
    if n ==4:
        term1 = (3 * torch.sqrt(torch.tensor(torch.pi)) * (4 * a**2 - 20 * a + 35) * erf(torch.sqrt(a))) / (64 * a**(2)*torch.sqrt(a))
        term2 = (5 * (2 * a + 21) * torch.exp(-a)) / (32 * a**2)
        analytical_sol = term1 - term2
    
    #maybe also make a torch version of erf
    
    return torch.nan_to_num(analytical_sol)



def spherical_harmonics_directions(directions, l):
    # Extract spherical coordinates
    phi, theta = cart2sph(directions[:,0], directions[:,1], directions[:,2]) #theta is elevation, phi is elevation
    theta = np.pi/2-theta

    
    # Initialize array for spherical harmonic values
    num_samples = directions.shape[0]
    N = int(0.5 * (1 + l) * (2 + l))
    Ysh = np.zeros((N, num_samples))

    # Compute spherical harmonic values for each direction
    col = 0
    for l_val in range(0, l + 1, 2):
        for m in range(-l_val, l_val + 1, 1):
            for idx in range(num_samples):

                Ysh[col, idx] = real_spherical_harmonics(l_val, m, theta[idx], phi[idx]) #scipy uses a different convention for azimuthal and polar input so watch out
            col += 1
    
    return Ysh

def real_spherical_harmonics(l, m, phi, theta):
    # Compute associated Legendre polynomial
    
    if m < 0:
        Ysh_m = np.sqrt(2)*np.imag(sph_harm(abs(m),l,theta,phi))
    elif m > 0:
        Ysh_m = np.sqrt(2)*np.real(sph_harm(m,l,theta, phi))
    else:
        Ysh_m = np.real(sph_harm(0,l,theta,phi))
    
    return Ysh_m

def cart2sph(x, y, z):
    phi = np.arctan2(z , np.sqrt(x**2 + y**2)) #elevation
    theta = np.arctan2(y, x) # azimuth

    
    return np.nan_to_num(theta), np.nan_to_num(phi)

from scipy.special import sph_harm,erf
    
def erf_approx_torch(x, n_terms=10):
    """
    Approximate the error function using Taylor series expansion. --- does torch.erf not work??

    Parameters:
    - x (float): Input value.
    - n_terms (int): Number of terms in the Taylor series expansion. Default is 10.

    Returns:
    - result (float): Approximation of the error function.

    Does not work yet....
    """
    result = x
    for n in range(1, n_terms):
        result += (-1)**n * x**(2*n+1) / (n * (2*n + 1) * torch.tensor(torch.math.factorial(n), dtype=torch.float))
    return 2 / torch.sqrt(torch.tensor(torch.pi)) * result



