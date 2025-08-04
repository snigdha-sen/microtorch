import numpy as np
import torch
import re
import model_code.signal_models as signal_models_module

# Import the signal models module dynamically
#import importlib
#signal_models_module = importlib.import_module("signal_models")

class ModelMaker:
    """
    For a given model string, returns the appropriate model class.
    The model class is a callable object that takes the gradient directions and parameters as input and returns the signal.
    The model class also contains:
    - self.parameter_ranges: tuple containing the ranges of the parameters for each compartment.
    - self.param_names: tuple containing the names of the parameters for each compartment.
    - self.n_params: the number of non-volume fraction parameters for each compartment.
    - self.n_frac: the number of volume fractions.
    - self.spherical_mean: boolean indicating whether the compartments are spherical mean or not spherical mean.
    """
    
    def __init__(self, modelname):
        self.comps = self.model_compartments(modelname)


        ## Comparments must have the same spherical mean property, if spherical mean isnt relevant for a compartment then it is set to None
        spherical_means = [comp.spherical_mean for comp in self.comps if comp.spherical_mean is not None]
        assert all(spherical_means) == True, "All compartments must have the same spherical mean property, either all spherical mean or all not spherical mean."


        # Initialize the parameter ranges, parameter names, compartment names, and number of parameters
        self.parameter_ranges = []
        self.param_names = []
        self.comp_names = []
        self.n_params = 0
        print('###########', self.comps)
        self.spherical_mean = spherical_means[0]

        for comp in self.comps:
            self.parameter_ranges.extend(comp.parameter_ranges)
            self.param_names.extend(comp.param_names)
            self.comp_names.append(comp.__class__.__name__)
            self.n_params += comp.n_params

        self.parameter_ranges = np.array(self.parameter_ranges)  # Convert to numpy array

        self.n_frac = len(self.comps) - 1  # The number of volume fractions
        self.param_names.extend([f'f_{i}' for i in range(self.n_frac)])

        self.param_ind = self.get_parameter_indices()  # Get the indices of the parameters in the parameter vector for each compartment
        self.comp_ind = self.get_comp_indices()  # Get the indices of the compartment that each parameter at a given index belongs to

    def __call__(self, grad, params):
        if len(self.comps) == 1:
            S = self.comps[0](grad, params)
        else:
            #k = self.n_frac  # Number of volume fraction parameters
            #f = params[:, -k:]  # Extracts the volume fraction parameters from the end of the parameter vector
            f = params[:, self.n_params:] # Extracts the volume fraction parameters from the end of the parameter vector         

            num_comps = len(self.comps)
            # Sum the signals from each compartment weighted by the volume fractions except for the last compartment        
            S = sum(f[:, i:i+1] * self.comps[i](grad, params[:, self.param_ind[i]]) for i in range(num_comps - 1))
            # Add the signal from the last compartment weighted by the remaining volume fraction
            S += (1 - f.sum(dim=1, keepdim=True)) * self.comps[-1](grad, params[:, self.param_ind[-1]])


        return S

    
            
    def convert_model_string_to_compartments(model):
        #converts the model string to a tuple of compartment strings for input to
        if model == "MSDKI":
            comps = ("MSDKI",)
        elif model == "NEXI":
            comps = ("NEXI",)
        elif model == "BallStick":
            comps = ("Ball","Stick")
        elif model == "StickBall":
            comps = ("Stick","Ball")        
        return comps            
                


    def import_compartments(comps):
        #given tuple of compartment strings import compartment classes dynamically based on the chosen model
        import importlib
        signal_models_module = importlib.import_module("signal_models")

        comps_classes = ()
        for comp in comps:
            #get the class
            this_class = getattr(signal_models_module, comp) #add to the tuple
            #create an instance of the class and add to the tuple
            comps_classes += (this_class(),)
            
        return comps_classes  
        
        
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
        # Returns the indices of the compartment that each parameter belongs to
        comp_ind = []
        for param_index in range(self.n_params):
            for i in range(len(self.param_ind)):
                if param_index in self.param_ind[i]:
                    comp_ind.append(i)
        # Add the indices of the volume fraction parameters
        comp_ind.extend(range(self.n_frac))
        return comp_ind

    @staticmethod
    def model_compartments(modelname):
        comps_classes = []
        compartment_list = []
        
        if modelname == "VERDICT":
            compartment_list = ["Ball", "Sphere", "Astrosticks_fixed"]
        elif modelname == "SANDI":
            compartment_list = ["Ball", "Sphere", "Astrosticks"]
        elif modelname == "IVIM":
            compartment_list = ["Ball", "Ball"]
        elif modelname == "NEXI":
            compartment_list = ["NEXI",]
        else:
            compartment_list = re.findall('([A-Z][a-z]+)', modelname)
        for comp in compartment_list:
            this_class = getattr(signal_models_module, comp)
            comps_classes.append(this_class())


        return tuple(comps_classes)

