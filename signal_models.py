import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils

__all__ = [
    'ball_stick',
    't2_adc',
    'msdki',
    'ball',
    'stick',
    'zeppelin',
    't1_smdt',
    'get_model_nparams']

def ball_stick(grad,params):
    
    #nparams = len(params)
    #need to force this shape?
    #params = torch.zeros([1,nparams])
     
    
    # extract the parameters
    f = params[:,0].unsqueeze(1)
    Dpar = params[:, 1].unsqueeze(1)
    D = params[:, 2].unsqueeze(1)
    theta = params[:, 3].unsqueeze(1)
    phi = params[:, 4].unsqueeze(1)


    g = grad[:,0:2]
    bvals = grad[:,3]  
    
    print(stick(grad, Dpar, theta,phi).size())
    print(ball(grad, D).size())
    
    S = f * stick(grad, Dpar, theta,phi) + (1 - f) * ball(grad, D)

    return S

def t2_adc(grad,params):
    # extract the parameters
    T2 = params[:,0].unsqueeze(1)
    D = params[:, 1].unsqueeze(1)   
    
    bvals = grad[:,3]
    te = grad[:,4]
    S = torch.exp(-grad[:,3]*D) * torch.exp(-(te - torch.min(te))/T2) 
    
    return S

def msdki(grad,params):
    
    #D = torch.clamp(params[:, 0].unsqueeze(1), min = 0.01, max = 5)
    #K = torch.clamp(params[:, 1].unsqueeze(1), min= 0.001, max=3)
    D = params[:, 0].unsqueeze(1)
    K = params[:, 1].unsqueeze(1)
    
    bvals = grad[:,3]
        
    S = torch.exp(-bvals*D + (bvals**2 * D**2 * K / 6)) 
    
    return S



def ball(grad, D):
    bvals = grad[:, 3].unsqueeze(1)

    S = torch.exp(-bvals * D)
    return S


def stick(grad, Dpar, theta, phi):
    
    g = grad[:, 0:3]
    bvals = grad[:, 3].unsqueeze(1)

    n = sphere2cart(theta,phi)
    
    S = torch.exp(-bvals * Dpar * torch.mm(g, n) ** 2)
    
    return S


def zeppelin(grad,params):
    # This zeppelin model is based on the implementation shown in Tax et al. (2021; NeuroImage, doi: 10.1016/j.neuroimage.2021.117967)
    # input:
    # grad: acquisition parameters
    # params: parameters estimated for signal prediction. Order: 
    # 1. theta 
    # 2. phi (angles of the first eigenvector with theta) 
    # 3. Dpar (parallel diffusivity)
    # 4. kperp (k*Dpar = Dperp or perpendicular diffusivity). k is a parameter between 0 and 1 to make sure that Dperp <= Dpar
    # 5. T2*
    # 6. T1
    # 7. S0

    g = grad[:,0:3] # we assume that the first three columns contain the diffusion gradient direction in Cartesian coordinates
    bvals = grad[:,3].unsqueeze(1) # b-value assumed in the fourth position in s/mm^2
    bvals = bvals/1000.0 # b-values in ms/um^2
    TI = grad[:,5].unsqueeze(1) # inversion time assumed in the sixth position in ms
    TE = grad[:,4].unsqueeze(1) # echo time assumed in the fifth position in ms
    TR = grad[:,6].unsqueeze(1) # repetition time assumed in the seventh position in ms
    
    # the implementation works with TD (delay time) instead of TE, assuming a multi-echo sequence
    TD = TE - torch.min(TE)

    # parameters
    theta = params[:,0].unsqueeze(1)
    phi = params[:,1].unsqueeze(1)
    # we transform into Cartesian coordinates
    n = sphere2cart(theta,phi)
    Dpar = params[:,2].unsqueeze(1)
    kperp = params[:,3].unsqueeze(1)
    Dperp = kperp*Dpar
    T2star = params[:,4].unsqueeze(1)
    T1 = params[:,5].unsqueeze(1)
    S0 = params[:,6].unsqueeze(1)
    
    # the implementation depends on tensor encoding. We assume linear tensor encoding
    b_delta = 1.0

    # signal representation: s = s0 * e^(-b:D) * abs(1 - 2*e^(-TI/T1) + e^(-TR/T1)) * e^(-TD/T2*)
    # we begin obtaining -b:D, called b_D here for simplicity
    b_D = b_delta/3.0 * bvals * (Dpar - Dperp) - bvals/3.0 * (Dperp + 2*Dpar) - bvals * b_delta * (torch.mm(g,n)**2) * (Dpar - Dperp)

    S = S0 * torch.exp(b_D) * torch.abs(1.0 - 2.0 * torch.exp(-TI/T1) + torch.exp(-TR/T1)) * torch.exp(-TD/T2star)

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
    
    sintheta = torch.sin(theta)
    
    n[0,:] = torch.squeeze(sintheta * torch.cos(phi))
    n[1,:] = torch.squeeze(sintheta * torch.sin(phi))
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




