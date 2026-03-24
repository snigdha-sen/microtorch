import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from microtorch.networks import build_network
from microtorch.utils.network_constraints import squash, fraction_squash

class Net(nn.Module):

    def __init__(self, 
                 grad, 
                 modelfunc, 
                 input_neurons, 
                 layer_dims, 
                 n_layers, 
                 dropout_fraction, 
                 network_type="hidden_dropout_mlp", 
                 clipping_method='clamp',
                 clipping_method_fraction="clamp", 
                 activation=nn.PReLU()):
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
        '''
        # ----------------------------------------------------------------
        # dev_MLP / softmax_MLP — original flat encoder (no per-layer dropout)
        # ----------------------------------------------------------------
        fc_layers = []
        fc_layers.extend([nn.Linear(input_neurons, layer_dims), copy.deepcopy(activation)])
        for _ in range(n_layers - 1):
            fc_layers.extend([nn.Linear(layer_dims, layer_dims), copy.deepcopy(activation)])
        self.encoder = nn.Sequential(*fc_layers, nn.Linear(layer_dims, dim_out))

        # optional dropout for dev_MLP / softmax_MLP
        if dropout_fraction > 0:
            self.dropout = nn.Dropout(dropout_fraction)

        # ----------------------------------------------------------------
        # hidden_dropout_MLP — dropout after every hidden activation
        # Structure per layer: Linear -> Activation -> Dropout
        # ----------------------------------------------------------------
        hidden_layers = []

        # first layer: input -> hidden
        hidden_layers.extend([nn.Linear(input_neurons, layer_dims), copy.deepcopy(activation)])
        if dropout_fraction > 0:
            hidden_layers.append(nn.Dropout(dropout_fraction))

        # remaining hidden layers
        for _ in range(n_layers - 1):
            hidden_layers.extend([nn.Linear(layer_dims, layer_dims), copy.deepcopy(activation)])
            if dropout_fraction > 0:
                hidden_layers.append(nn.Dropout(dropout_fraction))

        self.hidden = nn.Sequential(*hidden_layers)
        self.head = nn.Linear(layer_dims, dim_out)  # output head, no dropout
        '''

        self.encoder = build_network(
            network_type,
            input_neurons=input_neurons,
            layer_dims=layer_dims,
            n_layers=n_layers,
            dim_out=dim_out,
            activation=activation,
            dropout=dropout_fraction,
        )

    def forward(self, X, return_latent=False):        
        
        #get the model function and clipping method
        modelfunc = self.modelfunc
        clipping_method = self.clipping_method
        clipping_method_fraction = self.clipping_method_fraction

        # Get the start and end indices for the volume fraction parameters
        frac_start = modelfunc.n_parameters
        frac_end   = frac_start + modelfunc.n_fractions  # all the fractions

        '''
        #choose which network - messy for now but will clean up after testing different options
        #network = "dev_MLP"
        #network = "softmax_MLP"
        network = "hidden_dropout_MLP"

        if network == "dev_MLP":
            
            if self.dropout_fraction > 0:
                X = self.dropout(X)

            #params = self.encoder(X)              
            params = F.softplus(self.encoder(X))
            #params = torch.abs(self.encoder(X))
            #get the signal model function        
            #modelfunc = getattr(models, model)
                                        
            for i in range(modelfunc.n_parameters): #set min/max of non-volume fraction parameters       
                params[:,i] = Net.squash(params[:, i].clone().unsqueeze(1), clipping_method, modelfunc.parameter_ranges[i,0], modelfunc.parameter_ranges[i,1])

                        
        if network == "softmax_MLP":
            params = self.encoder(X)

            if self.dropout_fraction > 0:
                # only do dropout on the non-fraction parameters 
                params[:, :frac_start] = self.dropout(params[:, :frac_start])  # only non-fraction params       
                                
            for i in range(modelfunc.n_parameters): #set min/max of non-volume fraction parameters       
                params[:,i] = Net.squash(params[:, i].clone().unsqueeze(1), clipping_method, modelfunc.parameter_ranges[i,0], modelfunc.parameter_ranges[i,1])

        elif network == "hidden_dropout_MLP":
            h = self.hidden(X)  # dropout already applied after every hidden activation
            params = self.head(h)  # clean linear projection to output
            for i in range(modelfunc.n_parameters):
                params[:, i] = Net.squash(
                    params[:, i].clone().unsqueeze(1),
                    clipping_method,
                    modelfunc.parameter_ranges[i, 0],
                    modelfunc.parameter_ranges[i, 1],
                )
        '''

        params_out = self.encoder(X)

        if self.network_type == "vae":
            params, mu, logvar = params_out
        else:
            params = params_out
            mu = logvar = None

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
                    
        if return_latent and self.network_type == "vae":
            return X.to(torch.float32), params, mu, logvar
        else:
            return X.to(torch.float32), params
    