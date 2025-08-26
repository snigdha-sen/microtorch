import warnings
import numpy as np
import torch

#This file generates acquisition schemes - i.e the parameters which the model runs on.


class AquisitionScheme():

    def __init__(self):

        '''
        Note: This class will store a the parameters as listed in key_params AND store a numpy version when using the setter i.e bvalues will be a tensor and bvalues_numpy will be a numpy array
        The job of this class is to store the aquisition parameters for each aquisition - sounds simple enough?!
        Paramaters are defined as listed in key_params - this class is fairly flexible - so you can add and take away params as needed
        A successfully loaded scheme will have each parameter loaded as a tensor array, the arrays should allll be the same length as the number of measurements! as it captures the aq at any given measurement
        This make it easy to access from the model


        NOTE: If you add new paramaeters to key_params ensure the rest of the code is updated to handle them, also i would ensure the order is still the same as it is now!
        '''

        # These are the key params that are used in the aq scheme, if you add new ones please add them to the list below - syntax of each param is case sensitive (althought could theoretically be converted to not?)
        self.key_params = ['bvecs','bvalues','Delta', 'small_delta', 'TE', 'bdelta']
        for param in self.key_params: #Cycles through the above list
            setattr(self, param, None) #sets an attribute by each name in the list to None

        self.number_of_measurements = None
        self.loaded = False
        self.image = None #Future placeholder to store image with params
        self.gradient_strengths = None #This is not used in the current code, just adding for compatability with older code


    #Setter for scheme info
    def set_scheme(self,
                   **kwargs #Can take any of the aq params as kwargs, i.e bvalues, bvecs, small_delta, Delta, TE, bdelta (Must be case sensitive!)
                   ):
        ##Check if all new params are tensors
        new_params = self.get_aq_dict(**kwargs) #This function conveniently returns a dictionary of the aq params, filtering out any garbage data!
        for key, data in new_params.items():
            if data is not None: #If the data is none we just ignore it
                assert isinstance(data,torch.Tensor) or isinstance(data, np.ndarray), "All inputs should be tensors or nparrays"

        ## Set the attributes - doing this in a seperate loop so we check params before setting them
        for key, data in new_params.items():
            if data is not None:
                setattr(self,key,data)
                setattr(self,(key+"_numpy"),data.numpy()) #Also set the numpy version of the data, this is useful for saving to files

        self.update_number_measurements()

        return

    def update_number_measurements(self):
        #num of measurements is based off the bvals
        self.number_of_measurements = torch.tensor(len(self.bvalues.flatten()))

    ##Save scheme (todo)
    def save_scheme_to_file(self,filepath_acquisition_scheme): ##future function to save schemes

        return


    ##Loaders ==> For loading Schemes from various formats
    def load_scheme_from_args(self,args): #this loads a scheme if you have an args parser
        #If the format of args changes pls change this code
        arg_dict = self.get_aq_dict(
            bvalues = args.bvals,
            bvecs = args.bvecs,
            small_delta = args.smalldelta,
            Delta = args.delta,
            TE = args.TE,
            bdelta = args.bdelta)
        for key, i in arg_dict.items():
            arg_dict[key] = self.parse_argument_str_or_num(i)



        ##Checks and alteration
        #We will check if the array is either a single number, or the at least the size of the number of measurements, if neither throw error
        #Grab number of measurements
        num_measurements = self.get_num_measurements(arg_dict["bvalues"])
        for key, data in arg_dict.items():
            arg_dict[key] = self.format_array_length(data, num_measurements)
            #arg_dict[key] = data.astype(np.float32) if isinstance(data, np.ndarray) else data #Convert to float32, this is the default for torch tensors
            arg_dict[key] = self.convert_numpy_to_tensor(arg_dict[key])
            arg_dict[key] = self.sanitize_tensor(arg_dict[key])



        #self.set_scheme(bvalues,bvecs,Delta,small_delta,TE,bdelta)
        self.set_scheme(**arg_dict)

        return self

    def load_scheme_from_file(self,filepath_acquisition_scheme):
        #Loads a Aq Scheme from a file
        #This code is using an older grad scheme layout (the matrix one)
        #Should be deprecated once we stop using that way
        #NOTE LATEST - THIS FUNCTION SHOULD BE DEPRECATED SOON OR REWRITTEN

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

        self.set_scheme(bvalues = bvalues, bvecs = bvecs, Delta = Delta, small_delta=small_delta, TE = TE, bdelta=bdelta)






    ##Error Checking
    def format_array_length(self, array, num_measurements):
        if array is None:
            warnings.warn("One of the parameters is a Nonetype obj, it may be intentional (i.e gradient strengths) this is just a warning in case things break")
            return np.repeat(np.array([None]), num_measurements)

        size = np.size(array)
        if size > 1: #This will almost definetely break if the number of measurements is one, soooo dont do that. I am doing it like this because bvecs is shape 266,3, which messes with np.size
            return array
        elif size == 1:
            return np.repeat(array, num_measurements)
        else:
            AttributeError("One of the aquisition parameters is neither a number nor an array equivalent to the number of measurements")



    ##Used for direction averaging - ONLY CALL ONCE SCHEME IS SET
    ##We are separating the computation of the scheme and the image, so we can do it multiple times without recomputing the scheme
    def compute_direction_averaged_scheme(self):
        #Direction averaging finds unique shells (where all parameters are the same), computes the direction average, and creates a "shortened" acquisition scheme

        grad_matrix = self.get_scheme_as_matrix()
        self.unique_shells = torch.unique(grad_matrix[:, 3:], dim=0)  # Get unique shells based on parameters after bvecs IMPORTANT NEW PARAM - WILL BE CALLED DURING PREPROCESS IF DAVG IS TRUE
        self.shell_idxs = []  #stores the indices of the unique shells in grad_matrix
        new_grad = torch.zeros(self.unique_shells.shape[0], grad_matrix.shape[1], dtype=grad_matrix.dtype)
        for i, shell in enumerate(self.unique_shells):
            # Indices of grad file for this shell
            shell_index = torch.all(grad_matrix[:, 3:] == shell, dim=1)
            self.shell_idxs.append(shell_index)
            # Fill in this row of the direction-averaged grad file
            new_grad[i, 3:] = shell

        self.set_scheme_from_matrix(new_grad)
        self.update_number_measurements()


        return
    def get_scheme_as_matrix(self): #This is effectively a rewrite of get_grad_matrix function that was in preprocessing.py - fingers crossed this would work if any more params are added in the future
        #This function assumes that the scheme has been set, and that all of it is in the correct format

        param_dict = self.get_aq_dict()
        as_matrix = torch.zeros(self.number_of_measurements, len(param_dict)+2) #Setting a framework
        as_matrix[:, :3] = param_dict["bvecs"]  # bvecs are the first three columns [Rows, First thre cols]

        #Yes having the order of columns of the matrix dependent on the order of the key_params list is a bit hacky, but its easy
        del param_dict["bvecs"]  # Remove bvecs from the param_dict, as it is already added to the matrix

        for key, value in param_dict.items():
            param_flattened = value.view(-1)
            column_idx = list(param_dict.keys()).index(key) + 3  # Get the index of the parameter in the dictionary
            as_matrix[:, column_idx] = param_flattened

        as_matrix = torch.nan_to_num(as_matrix)  # Replace NaNs with 0s

        return as_matrix

    def set_scheme_from_matrix(self, matrix):
        ## effectively the above but reversed

        current = self.get_aq_dict()
        current["bvecs"] = matrix[:, :3]
        keys = self.key_params.copy()
        keys.remove("bvecs")

        for i, key in enumerate(keys):
            current[key] = matrix[:, i + 3]  # +3 because first three are bvecs

        self.set_scheme(**current)
        return

    def get_aq_dict(self, **kwargs): ## Return aquisition parameters as a dictionary
        return {param: kwargs.get(param, getattr(self, param, None)) for param in self.key_params}


    def parse_argument_str_or_num(self,value):
        if type(value) is str and not "":  # If the arguement is a string
            ##load the numpy string data
            ##This numpy array should be for each measurement!
            data = self.load_grad(value)
            data = np.transpose(data)
            # data = np.add(data, 1)
            return data

        elif type(value) is int or type(value) is float:
            data = np.array(value)
            return data
        else:
            TypeError("argument is not a valid type")

    def apply_direction_average_to_image(self, img):

        # Find unique shells - all parameters except gradient directions are the same
        unique_shells = self.unique_shells
        shell_idxs = self.shell_idxs

        # Preallocate
        new_img  = torch.zeros(img.shape[0:3] + (unique_shells.shape[0],), dtype=img.dtype)

        for i, shell in enumerate(unique_shells):
            # Indices of grad file for this shell
            shell_index = shell_idxs[i]
            # Calculate the spherical mean of this shell - average along final axis
            new_img[..., i] = torch.squeeze(torch.mean(img[..., shell_index], axis=img.ndim-1))


        return new_img

    ##Static methods => for simple ops
    @staticmethod
    def get_num_measurements(bvalues): # should be bvalues array
        return len(bvalues.flatten())
    @staticmethod
    def convert_numpy_to_tensor(matrix):
        return torch.from_numpy(matrix.astype(np.float32).squeeze())

    @staticmethod
    def load_grad(grad_filename): #migrated from utils/load_grad_data.py
        # TO DO: replace with something that finds the file e.g. pkg_resources.resource_filename
        # grad_files_path = '/Users/paddyslator/python/self-qmri/data'
        try:
            # grad = torch.tensor(np.loadtxt(grad_filename), dtype=torch.float32)
            grad = np.loadtxt(grad_filename)

            if len(grad.shape) < 2:
                grad = grad[:, None]

            return grad  # np.transpose(grad)
        except Exception as e:
            print(e)
            print(grad_filename)
            return None


    @staticmethod
    def grab_matrix_val_for_grad_matrix_aq(matrix, column=int):
        try:
            data = matrix[:, column]
        except:
            data = None
        return data


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