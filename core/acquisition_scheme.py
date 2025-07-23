import warnings

import numpy as np
from data.load_data import load_grad
import torch

#from messy_examples.try_acq_scheme import bvalues


#This file generates acquisition schemes - i.e the parameters which the model runs on.
#Currently works but could be improved by integrating the methods of loading into the AS class



class AquisitionScheme():

    def __init__(self):


        self.bvalues = None
        self.bvecs = None
        self.gradient_strengths = None
        self.small_delta = None
        self.Delta = None #This is only staying this way because a lot of previous code uses capital D
        self.TE = None
        self.bdelta = None
        self.number_of_measurements = None

        self.loaded = False


    #Setter for scheme info
    def set_scheme(self,
                   bvalues = None,
                   bvecs = None,
                   small_delta = None,
                   Delta = None,
                   TE = None,
                   bdelta = None,
                   number_of_measurements = None,
                   ): #sets a scheme into memory based on params
        ##This could be done using *kwargs if there are way more params in the future!
        #Setter should receive exact required data!


        ##checking if values are torch tensors
        check_vals = [bvalues,bvecs,small_delta,Delta,TE,bdelta]
        for val in check_vals:
            if val is not None:
                assert isinstance(val,torch.Tensor), "All inputs should be tensors"


        #This can definetely be done better but leaving as it works for now and not a huge computational drag
        self.bvalues = bvalues if bvalues is not None else self.bvalues
        self.bvecs = bvecs if bvecs is not None else self.bvecs
        #self.gradient_strengths = gradient_strengths if gradient_strengths is not None else self.gradient_strengths
        self.small_delta = small_delta if small_delta is not None else self.small_delta
        self.Delta = Delta if Delta is not None else self.Delta
        self.TE = TE if TE is not None else self.TE
        self.bdelta = bdelta if bdelta is not None else self.bdelta
        self.set_number_measurements()

        return

    def set_number_measurements(self):
        #num of measurements is based off the bvals
        self.number_of_measurements = torch.tensor(len(self.bvalues.flatten()))

    def set_bvalues(self, bval):
        self.bvalues = bval
        self.set_number_measurements()



    ##Save scheme (todo)
    def save_scheme_to_file(self,filepath_acquisition_scheme): ##future function to save schemes

        return


    ##Loaders ==> For loading Schemes from various formats
    def load_scheme_from_args(self,args): #this loads a scheme if you have an args parser
        #If the format of args changes pls change this code
        arg_dict = self.get_aq_dict(args.bvals, args.bvecs, args.smalldelta, args.delta, args.TE, args.bdelta)
        for key, data in arg_dict.items():
            arg_dict[key] = self.parse_argument_str_or_num(data)



        ##Checks and alteration
        #We will check if the array is either a single number, or the at least the size of the number of measurements, if neither throw error
        #Grab number of measurements
        num_measurements = self.get_num_measurements(arg_dict["bvalues"])
        for key, data in arg_dict.items():
            arg_dict[key] = self.format_array_length(arg_dict[key], num_measurements)
            arg_dict[key] = self.convert_numpy_to_tensor(arg_dict[key])
            arg_dict[key] = self.sanitize_tensor(arg_dict[key])



        #self.set_scheme(bvalues,bvecs,Delta,small_delta,TE,bdelta)
        self.set_scheme(**arg_dict)

        return self

    def load_scheme_from_file(self,filepath_acquisition_scheme):
        #Loads a Aq Scheme from a file
        #This code is using an older grad scheme layout (the matrix one)
        #Should be deprecated once we stop using that way

        acq_scheme = np.loadtxt(filepath_acquisition_scheme)
        bvalues = np.reshape(acq_scheme[:, 3], (1, len(acq_scheme[:, 3])))

        if max(bvalues[0, :]) > 100: #Normalisation?
            bvalues = bvalues / 1000

        if np.any(bvalues < 0):
            raise ValueError("bvals contains negative values")

        bvecs = acq_scheme[:, 0:3]

        Delta = self.grab_matrix_val_for_grad_matrix_aq(acq_scheme,4)
        small_delta = self.grab_matrix_val_for_grad_matrix_aq(acq_scheme, 5)
        gradient_strengths = self.grab_matrix_val_for_grad_matrix_aq(acq_scheme, 6) ##Getting rid of this because it was never used in previous code
        TE = self.grab_matrix_val_for_grad_matrix_aq(acq_scheme, 7)
        bdelta = self.grab_matrix_val_for_grad_matrix_aq(acq_scheme, 8)


        # check_acquisition_scheme(bvalues, bvecs, delta, Delta, TE)

        self.set_scheme(bvalues, bvecs, Delta, small_delta, TE, bdelta)






    ##Error Checking
    def format_array_length(self, array, num_measurements):
        if array is None:
            warnings.warn("One of the parameters is a Nonetype obj, it may be intentional (i.e gradient strengths) this is just a warning in case things break")

        size = np.size(array)
        if size > 1: #This will almost definetely break if the number of measurements is one, soooo dont do that. I am doing it like this because bvecs is shape 266,3, which messes with np.size
            return array
        elif size == 1:
            return np.repeat(array, num_measurements)
        else:
            AttributeError("One of the aquisition parameters is neither a number nor an array equivalent to the number of measurements")

    ##Static methods => for simple ops
    @staticmethod
    def get_num_measurements(bvalues): # should be bvalues array
        return len(bvalues.flatten())
    @staticmethod
    def convert_numpy_to_tensor(matrix):
        return torch.from_numpy(matrix.astype(np.float32))
    @staticmethod
    def parse_argument_str_or_num(value):
        if type(value) is str: #If the arguement is a string
            ##load the numpy string data
            ##This numpy array should be for each measurement!
            data = load_grad(value)
            data = np.transpose(data)
        elif type(value) is int or type(value) is float:
            data = np.array(value)
        else:
            TypeError("argument is not a valid type")
        return data

    @staticmethod
    def grab_matrix_val_for_grad_matrix_aq(matrix, column=int):
        try:
            data = matrix[:, column]
        except:
            data = None
        return data

    @staticmethod
    def get_aq_dict(
            bvalues,
            bvecs,
            small_delta,
            Delta,
            TE,
            bdelta,


    ):
        return {
            "bvalues": bvalues,
            "bvecs": bvecs,
            "small_delta": small_delta,
            "Delta": Delta,
            "TE": TE,
            "bdelta": bdelta,
        }

    @staticmethod
    def sanitize_tensor(  #This is just a testing function, to remove nans and 0s from my test data
            input_tensor: torch.Tensor,
            replace_nan_with_zero: bool = True,
            clamp_min: bool = True,
            min_clamp_value: float = 0.0,
            replace_zeros_with_epsilon: bool = False,
            epsilon: float = 1e-6
    ) -> torch.Tensor:
        """
        Sanitizes a PyTorch tensor by handling NaNs, zeros, and clamping values.

        Args:
            input_tensor (torch.Tensor): The input tensor to sanitize.
            replace_nan_with_zero (bool): If True, replaces NaN (Not a Number) values with 0.0.
                                          Defaults to True.
            clamp_min (bool): If True, clamps all values in the tensor to be at least
                              `min_clamp_value`. This ensures non-negativity or a minimum floor.
                              Defaults to True.
            min_clamp_value (float): The minimum value for clamping. Only effective if `clamp_min` is True.
                                     Defaults to 0.0.
            replace_zeros_with_epsilon (bool): If True, replaces any exact zero values in the tensor
                                               with `epsilon`. This is particularly useful for values
                                               that might appear in denominators (e.g., to prevent division by zero)
                                               or as arguments to functions like `sqrt` where zero or negative
                                               values are problematic. Defaults to False.
            epsilon (float): The small positive value to use when replacing zeros. Only effective if
                             `replace_zeros_with_epsilon` is True. Defaults to 1e-6.

        Returns:
            torch.Tensor: A new, sanitized tensor. The original input tensor is not modified.
        """
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor.")

        # Create a copy to avoid modifying the original tensor in-place
        sanitized_tensor = input_tensor.clone()

        # 1. Handle NaNs: Replace NaN values with 0.0
        if replace_nan_with_zero:
            sanitized_tensor = torch.nan_to_num(sanitized_tensor, nan=0.0)

        # 2. Clamp minimum values: Ensure values are at least `min_clamp_value`
        if clamp_min:
            sanitized_tensor = torch.clamp_min(sanitized_tensor, min_clamp_value)

        # 3. Replace exact zeros with epsilon: For strict positivity requirements
        if replace_zeros_with_epsilon:
            # Only replace zeros that are exactly 0.0 after clamping
            sanitized_tensor[sanitized_tensor == 0.0] = epsilon

        return sanitized_tensor