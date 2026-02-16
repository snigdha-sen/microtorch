import numpy as np
import re

import torch
import src.signal_models as signal_models_module


class ModelMaker:
    """
    A class to construct multi-compartment microstructure models based on their compartment or model names.

    For a given model name, the class:
    - Maps the model to appropriate compartment classes.
    - Validates the consistency of spherical mean properties across compartments.
    - Manages parameters, including their ranges, names, and indices for each compartment.

    Attributes:
        compartments (tuple): Instances of the compartment classes corresponding to the model.
        parameter_ranges (np.ndarray): Ranges for each parameter across all compartments.
        parameter_names (list): Names of the parameters for all compartments.
        compartment_names (list): Names of the compartment classes in the model.
        n_parameters (int): Total number of parameters across all compartments.
        n_fractions (int): Number of volume fraction parameters.
        spherical_mean (bool): Indicates if all compartments are spherically averaged.
        parameter_indices (list): Indices of parameters in the vector for each compartment.
        compartment_indices (list): Indices mapping each parameter to its compartment.

    Methods:
        __init__(modelname): Initializes the model and its compartments.
        __call__(gradients, parameters): Computes the model signal given gradients and parameters.
        model_compartments(modelname): Maps the model name to its compartments.
        get_parameter_indices(): Computes parameter indices for each compartment.
        get_compartment_indices(): Maps each parameter index to its compartment.
    """

    def __init__(self, modelname):
        """
        Initializes the ModelMaker instance by creating compartments and managing parameters.

        Args:
            modelname (str): The name of the model.

        Raises:
            ValueError: If the compartments have inconsistent spherical mean properties.
        """

        self.compartments = self.model_compartments(modelname)

        ## Comparments must have the same spherical mean property, if spherical mean isnt relevant for a compartment then it is set to None
        spherical_flags = [
            c.spherical_mean for c in self.compartments
            if c.spherical_mean is not None
        ]

        if spherical_flags: 
            if not (all(spherical_flags) or all(not f for f in spherical_flags)):
                raise ValueError(
                    "Invalid input: either all compartments are spherically averaged, "
                    "or none of them should be."
        )

        # Initialize the parameter ranges, parameter names, compartment names, and number of parameters
        self.parameter_ranges = []
        self.parameter_names = []
        self.compartment_names = []
        self.n_parameters = 0
        print('###########', self.compartments)
        self.spherical_mean = self.compartments[0].spherical_mean

        for comp in self.compartments:
            self.parameter_ranges.extend(comp.parameter_ranges)
            self.parameter_names.extend(comp.parameter_names)
            self.compartment_names.append(comp.__class__.__name__)
            self.n_parameters += comp.n_parameters

        self.parameter_ranges = np.array(self.parameter_ranges)  # Convert to numpy array

        self.n_fractions = len(self.compartments) - 1  # The number of volume fractions
        self.parameter_names.extend([f'f_{i}' for i in range(self.n_fractions)])

        self.parameter_indices = self.get_parameter_indices()  # Get the indices of the parameters in the parameter vector for each compartment
        self.compartment_indices = self.get_comp_indices()  # Get the indices of the compartment that each parameter at a given index belongs to

    def __call__(self, grad, parameters):
        """
        Computes the model signal for given gradients and parameters.

        Args:
            grad (torch.Tensor): Gradient directions.
            parameters (torch.Tensor): Parameter vector of shape [num_samples, n_parameters + n_fractions - 1].

        Returns:
            torch.Tensor: The computed signal of shape [num_samples, num_measurements].
        """
    
        if len(self.compartments) == 1:
            return self.compartments[0](grad, parameters)
        
        # Extract volume fractions
        f = parameters[:, self.n_parameters:]  # shape [num_samples, n_fractions-1]
        last_fraction = 1 - f.sum(dim=1, keepdim=True)  # shape [num_samples, 1]

        num_comps = len(self.compartments)
        
        # Initialize signal to zeros
        S = torch.zeros(
            parameters.size(0),
            grad.number_of_measurements,  # assumes grad has this attribute
            dtype=parameters.dtype,
            device=parameters.device
        )
        
        # Add contributions from all compartments except last
        for i in range(num_comps - 1):
            fraction = f[:, i:i+1]
            S += fraction * self.compartments[i](grad, parameters[:, self.parameter_indices[i]])

        # Add last compartment
        S += last_fraction * self.compartments[-1](grad, parameters[:, self.parameter_indices[-1]])
        
                
        return S



    def convert_model_string_to_compartments(model):
        #converts the model string to a tuple of compartment strings for input to
        if model == "MSDKI":
            compartments = ("MSDKI",)
        elif model == "NEXI":
            compartments = ("NEXI",)
        elif model == "BallStick":
            compartments = ("Ball","Stick")
        elif model == "StickBall":
            compartments = ("Stick","Ball")
        return compartments


    def import_compartments(compartments):
        #given tuple of compartment strings import compartment classes dynamically based on the chosen model

        comps_classes = ()
        for comp in compartments:
            #get the class
            this_class = getattr(signal_models_module, comp) #add to the tuple
            #create an instance of the class and add to the tuple
            comps_classes += (this_class(),)

        return comps_classes


    def get_parameter_indices(self):
        #calculate a tuple of indices of where the parameters for each compartment are in the parameter vector
        param_ind = (list(range(0,self.compartments[0].n_parameters)) ,) #initialise tuple from the first compartment
        for i in range(1,len(self.compartments)):
            #get the index of the last compartment's last parameter
            last_param_ind = 1 + param_ind[i-1][-1]
            #add the indices of the parameters for the next compartment to the tuple
            param_ind += (list(range(last_param_ind, last_param_ind + self.compartments[i].n_parameters)), )

        return param_ind


    def get_comp_indices(self):
        # Returns the indices of the compartment that each parameter belongs to
        compartment_indices = []
        for parameter_index in range(self.n_parameters):
            for i in range(len(self.parameter_indices)):
                if parameter_index in self.parameter_indices[i]:
                    compartment_indices.append(i)
        # Add the indices of the volume fraction parameters
        compartment_indices.extend(range(self.n_fractions))
        return compartment_indices

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
        elif modelname == "Standard_wm":
            compartment_list = ["Standard_wm",]
        elif modelname == "t1_smdt":
            compartment_list = ["t1_smdt",]
        else:
            compartment_list = re.findall('([A-Z][a-z]+)', modelname)
        for comp in compartment_list:
            this_class = getattr(signal_models_module, comp)
            comps_classes.append(this_class())


        return tuple(comps_classes)

