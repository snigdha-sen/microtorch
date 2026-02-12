import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self,grad, modelfunc, layer_dims, n_layers, dropout_fraction, clipping_method = 'clamp', activation=nn.PReLU()):  

        """
        Define the network architecture
        
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
        self.modelfunc  = modelfunc
        self.clipping_method = clipping_method

        dim_in          = layer_dims
        self.fc_layers  = nn.ModuleList()
        self.fc_layers.extend([nn.Linear(dim_in, layer_dims), activation])
        
        # Get the number of signal model parameters
        dim_out = modelfunc.n_parameters + modelfunc.n_fractions
        
        # Add fully connected hidden layers 
        for _ in range(n_layers - 1): 
            self.fc_layers.extend([nn.Linear(layer_dims, layer_dims), activation])
        
        # Add the last linear layer for regression    
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(layer_dims, dim_out)) 
        
        self.dropout_fraction = dropout_fraction
        if dropout_fraction > 0:
            self.dropout = nn.Dropout(dropout_fraction)

    def forward(self, X):        
        
        if self.dropout_fraction > 0:
            X = self.dropout(X)

        #params = self.encoder(X)              
        params = F.softplus(self.encoder(X))
        #params = abs(self.encoder(X))
        #get the signal model function        
        #modelfunc = getattr(models, model)
        modelfunc = self.modelfunc
        clipping_method = self.clipping_method
                               
        for i in range(modelfunc.n_parameters): #set min/max of non-volume fraction parameters       
            params[:,i] = Net.squash(params[:, i].clone().unsqueeze(1), clipping_method, modelfunc.parameter_ranges[i,0], modelfunc.parameter_ranges[i,1])
         
       # Enforce volume fraction parameters 
        frac_start = modelfunc.n_parameters
        frac_end   = frac_start + modelfunc.n_fractions  # only the free fractions

        if modelfunc.n_fractions == 1:
            # Two compartments: clip the single free fraction to [0,1]
            params[:, frac_start] = Net.squash(
                params[:, frac_start].clone().unsqueeze(1),
                clipping_method,
                0, 1
            ).squeeze(1)
        else:
            # extract free fractions and make sure in [0,1]
            f_free = torch.relu(params[:, frac_start:frac_end])

            # Compute implicit last fraction
            final_f = 1 - f_free.sum(dim=1, keepdim=True)
            final_f = torch.clamp(final_f, min=1e-8)  # prevent negative last fraction

            # Concatenate free fractions + last fraction
            all_f = torch.cat([f_free, final_f], dim=1)

            # Normalize all fractions so sum = 1 
            sum_all = all_f.sum(dim=1, keepdim=True)
            all_f = all_f / torch.clamp(sum_all, min=1e-8)

            # Store back only the free fractions
            params[:, frac_start:frac_end] = all_f[:, :-1]






        
        X = self.modelfunc(self.grad, params)
        return X.to(torch.float32), params
    


    def squash(param, method, p_min, p_max):

        if method == 'clamp':

            squashed_param_tensor =torch.clamp(param, min=p_min, max=p_max)
            unsqueezed_param = squashed_param_tensor.squeeze(1)

        elif method == 'sigmoid':

            sigmoid_param = torch.sigmoid(param)
            scaled_param = p_min + (p_max - p_min) * sigmoid_param
            unsqueezed_param = scaled_param.squeeze(1)

        else:
            raise ValueError("Unsupported method: {}".format(method))

        return unsqueezed_param