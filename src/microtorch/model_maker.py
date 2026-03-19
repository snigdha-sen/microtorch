import numpy as np
import re
import yaml

import torch
import microtorch.signal_models as signal_models_module

from microtorch.utils.paths import MODELS_CONF_PATH


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

        # Determine the overall spherical mean property of the model based on its compartments
        vals = [c.spherical_mean for c in self.compartments]
        if any(v is True for v in vals):
            self.spherical_mean = True
        elif any(v is False for v in vals):
            self.spherical_mean = False
        else:
            self.spherical_mean = None

        for comp in self.compartments:
            self.parameter_ranges.extend(comp.parameter_ranges)
            self.parameter_names.extend(comp.parameter_names)
            self.compartment_names.append(comp.__class__.__name__)
            self.n_parameters += comp.n_parameters


        self.parameter_ranges = np.array(self.parameter_ranges)  # Convert to numpy array

        self.n_compartments = len(self.compartments) # The number of compartments in the model
        if self.n_compartments > 1:
            self.n_fractions = len(self.compartments) # The number of ALL volume fractions
        elif self.n_compartments == 1:
            self.n_fractions = 0 # if only one compartment, no volume fractions needed

        # If more than one compartment, add n_fractions volume fraction parameters
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
        
        if self.n_compartments > 1: # Extract volume fractions for multicompartment models
            frac_start = self.n_parameters
            frac_end = frac_start + self.n_fractions
            f = parameters[:, frac_start:frac_end] # shape [num_samples, n_fractions]
            #last_fraction = 1 - fractions.sum(dim=1, keepdim=True)  # shape [num_samples, 1]
        elif self.n_compartments == 1: # Set f to 1 single compartment models 
            f = torch.ones(parameters.size(0), 1, device=parameters.device) # shape [num_samples, 1]

                            
        # Initialize signal to zeros
        S = torch.zeros(
            parameters.size(0),
            grad.number_of_measurements,  # assumes grad has this attribute
            dtype=parameters.dtype,
            device=parameters.device
        )
        
        # Add contributions from all compartments 
        for i in range(self.n_compartments):
            fraction = f[:, i:i+1]
            S += fraction * self.compartments[i](grad, parameters[:, self.parameter_indices[i]])

        # Add last compartment
        #S += last_fraction * self.compartments[-1](grad, parameters[:, self.parameter_indices[-1]])
        
                
        return S


    def get_parameter_indices(self):
        """
         Computes the indices of the parameters in the parameter vector for each compartment.
         Returns:
             param_ind (tuple): A tuple of lists, where each list contains the indices of the parameters for a specific compartment.
         """
        
        param_ind = (list(range(0,self.compartments[0].n_parameters)) ,) #initialise tuple from the first compartment
        for i in range(1,len(self.compartments)):
            #get the index of the last compartment's last parameter
            last_param_ind = 1 + param_ind[i-1][-1]
            #add the indices of the parameters for the next compartment to the tuple
            param_ind += (list(range(last_param_ind, last_param_ind + self.compartments[i].n_parameters)), )

        return param_ind


    def get_comp_indices(self):
        """
         Computes the indices of the compartment that each parameter belongs to.
         Returns:
             compartment_indices (list): A list where each element is the index of the compartment that the corresponding parameter belongs to.
         """
        
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

        model_file = MODELS_CONF_PATH / f"{modelname}.yaml"

        if model_file.exists():
            with open(model_file, "r") as f:
                config = yaml.safe_load(f) or {}

            compartment_specs = config.get("compartments", [])
            print(f"Found YAML configuration for {modelname} model.")
            print("Using specified compartments and applying any YAML-defined parameter range overrides.")
        else:
            # fallback to parsing modelname if no yaml exists
            compartment_list = re.findall(r"([A-Z][a-z]*\d*)", modelname)
            compartment_specs = [{"class": comp} for comp in compartment_list]

            print(f"No YAML configuration found for {modelname} model.")
            print(f"Falling back to parsing model name for compartments: {compartment_list}.")
            print("Parameter ranges will be the default compartment values.")

        for spec in compartment_specs:
            class_name = spec["class"]
            init_kwargs = spec.get("init_kwargs", {})
            parameter_ranges = spec.get("parameter_ranges", None)

            cls = getattr(signal_models_module, class_name)
            obj = cls(**init_kwargs)

            # Only override parameter ranges if they are explicitly given in YAML
            if parameter_ranges is not None:
                default_parameter_ranges = obj.parameter_ranges

                print(parameter_ranges)
                print(default_parameter_ranges)
                
                if len(parameter_ranges) != len(default_parameter_ranges):
                    raise ValueError(
                        f"Invalid number of parameter ranges for compartment '{class_name}' "
                        f"in model '{modelname}'. Expected {len(default_parameter_ranges)}, "
                        f"got {len(parameter_ranges)}."
                    )

                for i, param_range in enumerate(parameter_ranges):
                    if len(param_range) != 2:
                        raise ValueError(
                            f"Invalid parameter range at index {i} for compartment '{class_name}' "
                            f"in model '{modelname}'. Each parameter range must be [min, max]."
                        )

                obj.parameter_ranges = parameter_ranges

            comps_classes.append(obj)

        print("-----------")
        print("########### Making model: ", modelname)
        print("########### Compartments:", [comp.__class__.__name__ for comp in comps_classes])
        print("########### Parameter names:", [comp.parameter_names for comp in comps_classes])
        print("########### Parameter ranges:", [comp.parameter_ranges for comp in comps_classes])
        print("-----------")

        return tuple(comps_classes)



                
        



