import numpy as np
import torch
from scipy.special import sph_harm, erf, gamma,eval_laguerre
from .util_function import cart2sphere

##check if matlab model and python model correspond. Take bvalues from hcp. order 2, random sample Di,De,Dp [0,3]. Di>De>Dp. fODF: use watson distribution to sample from. Simulate one voxel first
def WM_model(order, bvals, bdelta, Ysh, f, Di, De, Dp, fODF, S0):

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
    Nl = torch.tensor([torch.sqrt(torch.tensor(4*torch.pi)),torch.sqrt(torch.tensor(4/5)*torch.pi),torch.sqrt(torch.tensor(4/5)*torch.pi),torch.sqrt(torch.tensor(4/5)*torch.pi),torch.sqrt(torch.tensor(4/5)*torch.pi), torch.sqrt(torch.tensor(4/5)*torch.pi) ])

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
        
        K[l] = (2* torch.sqrt(torch.pi * (2 * torch.tensor(l) + 1))* S0* (
                f * torch.exp((Di * b * bd) / 3 - (Di * b) / 3) * c_ias
                + (1 - f)
                * torch.exp(
                    (De * b * bd) / 3
                    - (De * b) / 3
                    - (2 * Dp * b) / 3
                    - (Dp * b * bd) / 3
                )
                * c_eas
            )
        )



    return K


def analytical_sol(a, n):
    # function is not continious in a = 0, so approx by a -> 0.00001
    a_limit = 0.5*gamma(n+0.5)/gamma(2*n + 3/2)*(-a)**n
    a_eps_idx = (a <= 1e-6)
    a[a_eps_idx] = 1e-6  # TODO: is this the way?
    # a = a.clone().detach().to(dtype=torch.complex64) # why do we detach and cast to complex here?

    if n == 0:
        analytical_sol = (
            torch.sqrt(torch.tensor(torch.pi)) * torch.erf(torch.sqrt(a))
        ) / (2 * torch.sqrt(a))

    if n == 2:
        analytical_sol = (
            -6 * torch.sqrt(a) * torch.exp(-a)
            + (3 - 2 * a)
            * torch.sqrt(torch.tensor(torch.pi))
            * torch.erf(torch.sqrt(a))
        ) / (8 * a * torch.sqrt(a))


    analytical_sol[a_eps_idx] = a_limit[a_eps_idx]

    return torch.nan_to_num(analytical_sol)



def spherical_harmonics_directions(directions, l):
    # Extract spherical coordinates
    phi, theta = cart2sph(directions[:,0], directions[:,1], directions[:,2]) #theta is elevation, phi is elevation
    theta = np.pi/2-theta

    phi = phi.cpu().numpy()
    theta = theta.cpu().numpy()

    
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

    rho = torch.sqrt(x**2 + y**2)
    phi = torch.atan2(z, rho)  # elevation
    theta = torch.atan2(y, x)  # azimuth

    # Replace NaNs with zero (if any)
    theta = torch.nan_to_num(theta)
    phi = torch.nan_to_num(phi)

    return theta, phi

from scipy.special import sph_harm,erf

    
def erf_torch(x): #also still need a well working replacement for this
    """Compute the error function for real inputs using scipy."""
    # Convert the input tensor to numpy, compute erf, and then convert back to tensor
    x_np = x.cpu().numpy()  # Convert to numpy array on CPU
    erf_np = erf(x_np)  # Compute the error function using scipy
    return torch.tensor(erf_np, dtype=x.dtype, device=x.device) 