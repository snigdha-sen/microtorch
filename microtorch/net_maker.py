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
         
        #set min/max of volume fraction parameters  
        if modelfunc.n_fractions == 1: #if just two compartments, then constraining one volume fraction to [0,1] also constrains the other to [0,1]
            params[:, modelfunc.n_parameters] = Net.squash(params[:, modelfunc.n_parameters].clone().unsqueeze(1), clipping_method, 0, 1)
        else:    
            params[:, modelfunc.n_parameters:modelfunc.n_parameters + modelfunc.n_fractions] = \
                torch.relu(params[:, modelfunc.n_parameters:modelfunc.n_parameters + modelfunc.n_fractions])
            sum_params = torch.sum(
                params[:, modelfunc.n_parameters:modelfunc.n_parameters + modelfunc.n_fractions],
                dim=1,
                keepdim=True
            )
            sum_params = torch.clamp(sum_params, min=1e-8)
            params[:, modelfunc.n_parameters:modelfunc.n_parameters + modelfunc.n_fractions] /= sum_params

        
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