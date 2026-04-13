from typing import Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from microtorch.utils.acquisition_scheme import AcquisitionScheme
from microtorch.model_maker import ModelMaker

from microtorch.networks import build_network
from microtorch.utils.network_constraints import squash, fraction_squash

class Net(nn.Module):

    def __init__(
        self,
        grad: AcquisitionScheme,
        modelfunc: ModelMaker,
        input_neurons: int,
        layer_dims: int,
        n_layers: int,
        dropout_fraction: float,
        network_type: str = "hidden_dropout_mlp",
        clipping_method: str = "clamp",
        clipping_method_fraction: str = "clamp",
        activation: nn.Module = nn.PReLU()
    ) -> None:
        """
        Define the network architecture.
        
        Args:
            grad: gradient table/sequence details
            model: compartmental model to fit
            layer_dims: number of units in each hidden layer
            n_layer: number of hidden layers
            dropout_frac: dropout fraction
            activation: activation function for each layer. Defaults to nn.PReLU().
        """

        super(Net, self).__init__()

        self.grad = grad
        self.modelfunc = modelfunc
        self.clipping_method = clipping_method
        self.clipping_method_fraction = clipping_method_fraction
        self.network_type = network_type

        self.post_param_dropout = (
            nn.Dropout(dropout_fraction) if dropout_fraction > 0 else None
        )

        if modelfunc.n_fractions > 1:
            dim_out = modelfunc.n_parameters + modelfunc.n_fractions
        else:
            dim_out = modelfunc.n_parameters

        self.encoder = build_network(
            network_type,
            input_neurons=input_neurons,
            layer_dims=layer_dims,
            n_layers=n_layers,
            dim_out=dim_out,
            activation=activation,
            dropout=dropout_fraction,
        )

    def forward(
        self,
        X: torch.Tensor,
        return_latent: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],]:       
        
        #get the model function and clipping method
        modelfunc = self.modelfunc
        clipping_method = self.clipping_method
        clipping_method_fraction = self.clipping_method_fraction

        # Get the start and end indices for the volume fraction parameters
        frac_start = modelfunc.n_parameters
        frac_end   = frac_start + modelfunc.n_fractions  # all the fractions

        params = self.encoder(X)

        if self.network_type == "dev_mlp": # can make softmax mlp by choosing clipping method to be softmax
            params = F.softplus(params)

        if (
            self.clipping_method_fraction == "softmax"
            and self.post_param_dropout is not None
        ):
            params[:, :frac_start] = self.post_param_dropout(params[:, :frac_start])


        for i in range(modelfunc.n_parameters):
            params[:, i] = squash(
                params[:, i].clone().unsqueeze(1),
                clipping_method,
                modelfunc.parameter_ranges[i, 0],
                modelfunc.parameter_ranges[i, 1],
                T=1.0
            )

        #set min/max of volume fraction parameters and enforce sum to 1 across fractions
        if modelfunc.n_fractions > 1:
            fractions = fraction_squash(
                clipping_method_fraction,
                params[:, frac_start:frac_end], 
                modelfunc,
                tau=1.0
                )
            #store all the fractions 
            params[:, frac_start:frac_end] = fractions
        
        # compute the predicted signal using the model function with the current parameters
        X = modelfunc(self.grad, params)
                    
        return X.to(torch.float32), params
    