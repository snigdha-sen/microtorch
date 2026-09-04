import torch

from microtorch.utils.acquisition_scheme import AcquisitionScheme


class T2Relaxation:
    """Add T2 relaxation to an existing signal model compartment."""

    def __init__(
        self,
        base_compartment,
        T2_range=(0.01, 0.5),
    ):

        self.base_model = base_compartment

        base_name = getattr(
            base_compartment,
            "name",
            base_compartment.__class__.__name__
        )

        #make new model compartment name by adding T2 to the base compartment name
        self.name = f"{base_name}T2"

        self.parameter_ranges = (
            base_compartment.parameter_ranges
            + [list(T2_range)]
        )

        self.parameter_names = (
            base_compartment.parameter_names
            + ["T2"]
        )

        self.n_parameters = base_compartment.n_parameters + 1
        self.spherical_mean = base_compartment.spherical_mean


    def __call__(
        self,
        grad: AcquisitionScheme,
        parameters: torch.Tensor,
    ) -> torch.Tensor:

        # parameters in the original diffusion model
        diffusion_parameters = parameters[:, :-1]
        # T2 relaxation parameter
        T2 = parameters[:, -1].unsqueeze(1)

        #original diffusion model signal
        S = self.base_model(
            grad,
            diffusion_parameters
        )

        TE = grad.TE

        return S * torch.exp(
            -(TE - torch.min(TE)) / T2
        )






class T1InversionRecovery:
    """Add T1 inversion recovery weighting to an existing signal model."""

    def __init__(
        self,
        base_model,
        T1_range=(0.05, 5.0),
        IE_range=(0.5, 2.0),
    ):
        self.base_model = base_model

        base_name = getattr(
            base_model,
            "name",
            base_model.__class__.__name__
        )

        self.name = f"{base_name}T1"

        self.parameter_ranges = (
            base_model.parameter_ranges
            + [list(T1_range), list(IE_range)]
        )

        self.parameter_names = (
            base_model.parameter_names
            + ["T1", "IE"]
        )

        self.n_parameters = base_model.n_parameters + 2
        self.spherical_mean = base_model.spherical_mean

    def __call__(
        self,
        grad: AcquisitionScheme,
        parameters: torch.Tensor,
    ) -> torch.Tensor:

        # Parameters for the original model
        base_parameters = parameters[:, :-2]

        # T1 and inversion efficiency
        T1 = parameters[:, -2].unsqueeze(1)
        IE = parameters[:, -1].unsqueeze(1)

        # Original model signal
        S = self.base_model(
            grad,
            base_parameters
        )

        TI = grad.TI
        TR = grad.TR

        # Inversion recovery weighting
        ir = torch.abs(
            1
            - IE * torch.exp(-TI / T1)
            + torch.exp(-TR / T1)
        )

        return S * ir