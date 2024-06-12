import numpy as np
import torch

class ModelMaker:
    """
    for a given model string, returns the appropriate model class
    the model class is a callable object that takes the gradient directions and parameters as input and returns the signal
    the model class also contains:
    self.parameter_ranges: tuple containing the ranges of the parameters for each compartment
    self.param_names: tuple containing the names of the parameters for each compartment
    self.n_params: the number of non-volume fraction parameters for each compartment
    self.n_frac: the number of volume fractions
    self.spherical_mean: boolean indicating whether the compartments are spherical mean or not spherical mean 
    """
    
    def __init__(self,comps):

        self.comps = comps 

        #check that all compartments are either spherical mean or not spherical mean
        if not (all(comp.spherical_mean==True for comp in comps) or all(comp.spherical_mean==False for comp in comps)):
            raise ValueError("Invalid input. All compartments must have the same spherical mean property, either all spherical mean or all not spherical mean.")

        #initialise the parameter ranges, parameter names, compartment names, and number of parameters
        parameter_ranges = []
        param_names = []
        comp_names = []
        n_params = 0
        #extract the spherical mean binary from the first compartment
        spherical_mean = comps[0].spherical_mean

        for comp in comps:  #loop through the compartments adding the parameter ranges, names and number of parameters for each compartment
            parameter_ranges.extend(comp.parameter_ranges)
            param_names.extend(comp.param_names)
            comp_names = comp_names + [comp.__class__.__name__]
            n_params += comp.n_params

        parameter_ranges = np.array(parameter_ranges) #convert to numpy array

        self.parameter_ranges = parameter_ranges
        self.param_names = param_names
        self.comp_names = comp_names
        self.n_params = n_params #the number of non-volume fraction parameters
        self.n_frac = len(comps) - 1 #the number of volume fractions
        self.spherical_mean = spherical_mean
        
        self.param_ind = get_parameter_indices(self) #get the indices of the parameters in the parameter vector for each compartment
        self.comp_ind = get_comp_indices(self) #get the indices of the compartment that each parameter at a given index belongs to

        #add the volume fraction names to the parameter names
        param_names.extend([f'f_{i}' for i in range(self.n_frac)])

    def __call__(self, grad, params):   
        
        if type(self.comps) != tuple or len(self.comps) == 1: 
            f = 1                                        
        elif len(self.comps) >= 2:
            k = len(self.comps) - 1 #number of volume fraction parameters
            f = params[0,-k:]  #extracts the volume fraction parameters from the end of the parameter vector 
                 
         
        if not isinstance(self.comps, tuple):  # single compartment defined by a string
            S = self.comps(grad, params)
        else:
            num_comps = len(self.comps)
        if num_comps == 1:  # single compartment defined by a tuple containing one compartment
            S = self.comps[0](grad, params)
        else: # signal equation for two or more compartments        
            #sum the signals from each compartment weighted by the volume fractions except for the last compartment
            S = sum(f[i] * self.comps[i](grad, params[:, self.param_ind[i]]) for i in range(num_comps - 1)) 
            #add the signal from the last compartment weighted by the remaining volume fraction
            S += (1 - sum(f[:num_comps-1])) * self.comps[num_comps - 1](grad, params[:, self.param_ind[num_comps - 1]])
               
        #return the signal for the appropriate model
        return S

    
def get_parameter_indices(self):
    #calculate a tuple of indices of where the parameters for each compartment are in the parameter vector
    param_ind = (list(range(0,self.comps[0].n_params)) ,) #initialise tuple from the first compartment
    for i in range(1,len(self.comps)):  
        #get the index of the last compartment's last parameter
        last_param_ind = 1 + param_ind[i-1][-1]
        #add the indices of the parameters for the next compartment to the tuple
        param_ind += (list(range(last_param_ind, last_param_ind + self.comps[i].n_params)), ) 
        
    return param_ind


def get_comp_indices(self):
    #returns the indices of the compartment that each parameter belongs to
    comp_ind = []
    for param_index in range(self.n_params):
        for i in range(len(self.param_ind)):
            if param_index in self.param_ind[i]:
                comp_ind.append(i)
    #add the indices of the volume fraction parameters
    for i in range(self.n_frac):
        comp_ind.append(i)
    return comp_ind