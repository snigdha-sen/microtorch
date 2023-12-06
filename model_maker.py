import numpy as np
import torch
import signal_models

def ModelMaker(comps): #makes a model FUNCTION
    #    
    #check that all compartments are either spherical mean or not
    if not (all(comp.spherical_mean==True for comp in comps) or all(comp.spherical_mean==False for comp in comps)):
        raise ValueError("Invalid input. All compartments must have the same spherical mean property, either all spherical mean or all not spherical mean.")
        
    class ModelFuncMaker:      
        def __init__(self,comps):
            self.comps = comps                    

            parameter_ranges = []
            param_names = []
            n_params = 0
            #can just take the spherical mean property from the first compartment as have checked that all are spherical mean or all not spherical mean
            spherical_mean = comps[0].spherical_mean

            for comp in comps:            
                parameter_ranges.extend(comp.parameter_ranges)
                param_names.extend(comp.param_names)
                n_params += comp.n_params
             
            #convert to numpy arrays
            parameter_ranges = np.array(parameter_ranges)
            
            #add information to the full model structure
            self.parameter_ranges = parameter_ranges
            self.param_names = param_names
            self.n_params = n_params #the number of non-volume fraction parameters
            self.n_frac = len(comps) - 1 #the number of volume fractions
            self.spherical_mean = spherical_mean
            
            
        def __call__(self, grad, params):   
            
            f = get_f(comps,params)   
            
            param_ind = get_param_ind(comps)    
                        
            if not type(comps) == tuple: 
                #signal equation for one compartment model                                                        
                S = comps(grad,params)                        
        
            elif len(comps) == 1:
                #signal equation for one compartment model                                                        
                S = comps[0](grad,params)
                
            elif len(comps) == 2:                           
                #signal equation for two compartment model
                S = f * comps[0](grad, params[:,param_ind[0]]) \
                    + (1-f) * comps[1](grad, params[:,param_ind[1]])   
                                    
            elif len(comps) == 3:
                #signal equation for three compartment model
                
                S = f[0] * comps[0](grad, params[:,param_ind[0]]) \
                    + f[1] * comps[1](grad, params[:,param_ind[1]]) \
                    + (1 - f[0] - f[1]) * comps[2](grad, params[:,param_ind[2]])                                                
            
            elif len(comps) == 4:
                #signal equation for four compartment model                   
                
                S = f[0] * comps[0](grad, params[:,param_ind[0]]) \
                    + f[1] * comps[1](grad, params[:,param_ind[1]]) \
                    + f[2] * comps[2](grad, params[:,param_ind[2]]) \
                    + (1 - f[1] - f[2] - f[3]) * comps[3](grad, params[:,param_ind[3]])

            #return the signal for the appropriate model
            return S
        
    #make an instance of the model function
    modelfunc = ModelFuncMaker(comps) 
    return modelfunc
    
#function that gets the volume fractions
def get_f(comps,params):
    if not type(comps) == tuple: 
        f = 1                         

    elif len(comps) == 1:
        f = 1
            
    elif len(comps) >= 2:
        k = len(comps) - 1
        f = params[0,-k:]                                 
    
    return f
    
#function that gets the indicies of the parameters for each compartment
def get_param_ind(comps):
    param_ind = (list(range(0,comps[0].n_params)) ,) #initialise tuple from the first compartment
    for i in range(1,len(comps)):  
        #get the index of the last compartment's last parameter
        last_param_ind = 1 + param_ind[i-1][-1]
        param_ind += (list(range(last_param_ind, last_param_ind + comps[i].n_params)), )
        
        
    return param_ind

        
      


            