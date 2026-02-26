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
        
        #get the signal model function               
        modelfunc = self.modelfunc
        clipping_method = self.clipping_method

        # Get the start and end indices for the volume fraction parameters
        frac_start = modelfunc.n_parameters
        frac_end   = frac_start + modelfunc.n_fractions  # all the fractions

        # if self.dropout_fraction > 0:
        #     X = self.dropout(X)
        
        params = self.encoder(X)

        if self.dropout_fraction > 0:
            # params = self.dropout(params)
            params[:, :frac_start] = self.dropout(params[:, :frac_start])  # only non-fraction params       
                               
        for i in range(modelfunc.n_parameters): #set min/max of non-volume fraction parameters       
            params[:,i] = Net.squash(params[:, i].clone().unsqueeze(1), clipping_method, modelfunc.parameter_ranges[i,0], modelfunc.parameter_ranges[i,1])
         
        # Enforce volume fraction parameters 
        logits_all = params[:, frac_start:frac_end]
                        
        #softmax across the fractions to get valid fractions that sum to 1
        tau = 1.0
        fractions = torch.softmax(logits_all / tau, dim=1)
        
        #store all the fractions 
        params[:, frac_start:frac_end] = fractions

       
        

        X = self.modelfunc(self.grad, params)
        
        return X.to(torch.float32), params
    


    def squash(param, method, p_min, p_max):

        if method == 'clamp':

            squashed_param_tensor =torch.clamp(param, min=p_min, max=p_max)
            unsqueezed_param = squashed_param_tensor.squeeze(1)

        elif method == 'sigmoid':

            T = 3.0

            sigmoid_param = torch.sigmoid(param / T)
            scaled_param = p_min + (p_max - p_min) * sigmoid_param
            unsqueezed_param = scaled_param.squeeze(1)

        else:
            raise ValueError("Unsupported method: {}".format(method))

        return unsqueezed_param