import numpy as np
import re
import src.signal_models as signal_models_module


class ModelMaker:
    def __init__(self, modelname, debug: bool = False):
        self.compartments = self.model_compartments(modelname)

        # Validate spherical_mean consistency, ignoring None
        sm_flags = [c.spherical_mean for c in self.compartments if c.spherical_mean is not None]
        if sm_flags and not (all(sm_flags) or not any(sm_flags)):
            raise ValueError(
                "Invalid input: either all relevant compartments are spherically averaged, or none are."
            )

        self.spherical_mean = self.compartments[0].spherical_mean

        # Collect metadata
        self.parameter_ranges = []
        self.parameter_names = []
        self.compartment_names = []
        self.n_parameters = 0

        for comp in self.compartments:
            self.parameter_ranges.extend(comp.parameter_ranges)
            self.parameter_names.extend(comp.parameter_names)
            self.compartment_names.append(comp.__class__.__name__)
            self.n_parameters += comp.n_parameters

        self.parameter_ranges = np.asarray(self.parameter_ranges)

        # Fractions live at the end: N compartments -> N-1 explicit fractions
        self.n_fractions = max(len(self.compartments) - 1, 0)
        self.parameter_names.extend([f"f_{i}" for i in range(self.n_fractions)])

        # Use slices rather than lists-of-indices
        self.parameter_slices = self.get_parameter_slices()

        # Optional debugging
        if debug:
            print("Compartments:", self.compartment_names)
            print("Total parameters (excluding fractions):", self.n_parameters)
            print("Fractions:", self.n_fractions)
            print("Slices:", self.parameter_slices)

    def __call__(self, grad, parameters):
        """
        grad: torch.Tensor
        parameters: torch.Tensor shaped (batch, n_parameters + n_fractions)
        """
        if len(self.compartments) == 1:
            # If your compartment expects only its own parameters, use the slice:
            slc = self.parameter_slices[0]
            return self.compartments[0](grad, parameters[:, slc])

        f = parameters[:, self.n_parameters:]  # (batch, n_fractions)

        # Weighted sum
        S = 0.0
        for i, cp in enumerate(self.compartments[:-1]):
            slc = self.parameter_slices[i]
            S = S + f[:, i:i+1] * cp(grad, parameters[:, slc])

        last_slc = self.parameter_slices[-1]
        last_weight = 1.0 - f.sum(dim=1, keepdim=True)
        S = S + last_weight * self.compartments[-1](grad, parameters[:, last_slc])

        return S

    def get_parameter_slices(self):
        slices = []
        start = 0
        for cp in self.compartments:
            end = start + cp.n_parameters
            slices.append(slice(start, end))
            start = end
        return tuple(slices)

    @staticmethod
    def model_compartments(modelname):
        if modelname == "VERDICT":
            compartment_list = ["Ball", "Sphere", "Astrosticks_fixed"]
        elif modelname == "SANDI":
            compartment_list = ["Ball", "Sphere", "Astrosticks"]
        elif modelname == "IVIM":
            compartment_list = ["Ball", "Ball"]
        elif modelname == "NEXI":
            compartment_list = ["NEXI"]
        elif modelname == "Standard_wm":
            compartment_list = ["Standard_wm"]
        else:
            # Split CamelCase words like BallStick -> ["Ball", "Stick"]
            compartment_list = re.findall(r"([A-Z][a-z]+)", modelname)

        compartment = []
        for comp in compartment_list:
            cls = getattr(signal_models_module, comp)
            compartment.append(cls())

        return tuple(compartment)


