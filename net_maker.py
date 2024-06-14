import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, grad, modelfunc, dim_hidden, num_layers, dropout_frac, activation=nn.PReLU()):  
        """
        Define the network architecture
        
        Args:
            grad: gradient table/sequence details
            model: compartmental model to fit
            dim_hidden: number of units in each hidden layer
            num_layers: number of hidden layers
            dropout_frac: dropout fraction
            activation: activation function for each layer. Defaults to nn.PReLU().
        """
        
        super(Net, self).__init__()
        self.grad       = grad
        self.modelfunc  = modelfunc
        dim_in          = dim_hidden
        self.fc_layers  = nn.ModuleList()
        self.fc_layers.extend([nn.Linear(dim_in, dim_hidden), activation])
        
        # Get the number of signal model parameters
        dim_out = modelfunc.n_params + modelfunc.n_frac
        
        # Add fully connected hidden layers 
        for _ in range(num_layers-1): 
            self.fc_layers.extend([nn.Linear(dim_hidden, dim_hidden), activation])
        
        # Add the last linear layer for regression    
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(dim_hidden, dim_out)) 
        
        self.dropout_frac = dropout_frac
        if dropout_frac > 0:
            self.dropout = nn.Dropout(dropout_frac)

    def forward(self, X):        
        
        if self.dropout_frac > 0:
            X = self.dropout(X)

        params = F.softplus(self.encoder(X))
              
        # Get the signal model function        
        modelfunc = self.modelfunc
                       
        # Set min/max of non-volume fraction parameters               
        for i in range(modelfunc.n_params): 
            this_param_clamped = torch.clamp(params[:, i].clone().unsqueeze(1), min = modelfunc.parameter_ranges[i,0], max =  modelfunc.parameter_ranges[i,1])  
            params[:,i] = this_param_clamped.squeeze()
         
        # Set min/max of volume fraction parameters  
        for i in range(modelfunc.n_params, modelfunc.n_params + modelfunc.n_frac):
            # Set negative values to 0
            params[:, i] = torch.relu(params[:, i]) 

        # Normalize to make the sum of params[:, i] equal to 1
        sum_params = torch.sum(params[:, modelfunc.n_params:modelfunc.n_params + modelfunc.n_frac], dim=1, keepdim=True)
        for i in range(modelfunc.n_params, modelfunc.n_params + modelfunc.n_frac):
            params[:, modelfunc.n_params:modelfunc.n_params + modelfunc.n_frac] /= sum_params

        X = self.modelfunc(self.grad, params)

        return X.to(torch.float32), params