import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from networks import build_network

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

    def forward(self, X):        
        
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

        params = self.encoder(X)

        if (
            self.clipping_method_fraction == "softmax"
            and self.post_param_dropout is not None
        ):
            params[:, :frac_start] = self.post_param_dropout(params[:, :frac_start])

        if self.network_type == "dev_mlp": # can make softmax mlp by choosing clipping method to be softmax
            params = F.softplus(params)

        for i in range(modelfunc.n_parameters):
            params[:, i] = Net.squash(
                params[:, i].clone().unsqueeze(1),
                clipping_method,
                modelfunc.parameter_ranges[i, 0],
                modelfunc.parameter_ranges[i, 1],
            )

        #set min/max of volume fraction parameters and enforce sum to 1 across fractions
        if modelfunc.n_fractions > 1:
            fractions = Net.fraction_squash(
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
    

    @staticmethod
    def squash(param, method, p_min, p_max):
        """
        Constrain the parameter values to be within the specified range 
        [p_min, p_max] using the specified method.

        Args:
            param (torch.Tensor): The parameter tensor to be constrained.
            method (str): The method to use for constraining the parameters. 
            Options are 'clamp' (simple clipping) or 'sigmoid' (sigmoid squashing).
            p_min (float): The minimum value for the parameter.
            p_max (float): The maximum value for the parameter.

        Returns:
            unsqueezed_param (torch.Tensor): The constrained parameter tensor, 
            with the same shape as the input param but squeezed to remove the extra dimension.
        """

        if method == 'clamp':

            squashed_param_tensor = torch.clamp(param, min=p_min, max=p_max)
            unsqueezed_param = squashed_param_tensor.squeeze(1)

        elif method == 'sigmoid':

            T = 1.0

            sigmoid_param = torch.sigmoid(param / T)
            scaled_param = p_min + (p_max - p_min) * sigmoid_param
            unsqueezed_param = scaled_param.squeeze(1)
        elif method == 'free': #no squashing 
            unsqueezed_param = param.squeeze(1)

        else:    
            raise ValueError("Unsupported method: {}".format(method))

        return unsqueezed_param
    
    @staticmethod
    def fraction_squash(method, logits_all, modelfunc, tau=1.0):

        if method == 'softmax':
            fractions = torch.softmax(logits_all / tau, dim=1)

        elif method == 'clamp':
            if modelfunc.n_fractions == 1:
                # Two compartments: clip the single free fraction to [0,1]
                fractions = Net.squash(
                    logits_all[:, 0].clone().unsqueeze(1),
                    method,
                    0, 1
                )
                fractions = torch.cat([fractions, 1 - fractions], dim=1)  # implicit second fraction
            else:
                #More than two compartments
                # extract free fractions and make sure in [0,1]
                f_free = torch.relu(logits_all[:, :-1])  # all but last fraction are free 

                # Compute implicit last fraction
                final_f = 1 - f_free.sum(dim=1, keepdim=True)
                final_f = torch.clamp(final_f, min=1e-8)  # prevent negative last fraction

                # Concatenate free fractions + last fraction
                fractions = torch.cat([f_free, final_f], dim=1)

                # Normalize all fractions so sum = 1 
                sum_all = fractions.sum(dim=1, keepdim=True)
                fractions = fractions / torch.clamp(sum_all, min=1e-8)
        elif method == 'free': #no squashing 
            fractions = logits_all  
        else:
            raise ValueError("Unsupported method: {}".format(method))

        return fractions