import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import signal_models

# WRAPPER FUNCTION MAKE NETWORK 
# make_net((grad, model, dim_hidden, dim_out, num_layers, dropout_frac, activation=nn.PReLU()))
#
# grad: gradient table/sequence details
# model: compartmental model to fit
# dim_hidden: number of units in each hidden layer
# num_layers: number of hidden layers
# dropout_frac: dropout fraction
# activation: activation function for each layer

#def net_maker(grad, modelfunc, dim_hidden, num_layers, dropout_frac, activation=nn.PReLU()):
class Net(nn.Module):
    def __init__(self, grad, modelfunc, dim_hidden, num_layers, dropout_frac, activation=nn.PReLU()):  
        super(Net, self).__init__()
        #add gradient table
        self.grad = grad
        self.modelfunc = modelfunc
        dim_in = grad.shape[0]
        self.fc_layers = nn.ModuleList()
        #create the first layer - input layer
        self.fc_layers.extend([nn.Linear(dim_in, dim_hidden), activation])
        #get the number of signal model parameters
        dim_out = modelfunc.n_params + modelfunc.n_frac
        
        for i in range(num_layers-1): # num_layers fully connected hidden layers - number of nodes defined by the user as dim_hidden
            self.fc_layers.extend([nn.Linear(dim_hidden, dim_hidden), activation])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(dim_hidden, dim_out)) # Add the last linear layer for regression
        
        self.dropout_frac = dropout_frac
        if dropout_frac > 0:
            self.dropout = nn.Dropout(dropout_frac)

    def forward(self, X):        
        
        if self.dropout_frac > 0:
            X = self.dropout(X)
        #params = self.encoder(X)               
        params = F.softplus(self.encoder(X))
              
        #get the signal model function        
        #modelfunc = getattr(models, model)
        modelfunc = self.modelfunc
                               
        for i in range(modelfunc.n_params): #set min/max of non-volume fraction parameters       
            this_param_clamped = torch.clamp(params[:, i].clone().unsqueeze(1), min = modelfunc.parameter_ranges[i,0], max =  modelfunc.parameter_ranges[i,1])  
            params[:,i] = this_param_clamped.squeeze()
            
        for i in range(modelfunc.n_params, modelfunc.n_params + modelfunc.n_frac): #set min/max of volume fraction parameters  
            this_frac_clamped = torch.clamp(params[:, i].clone().unsqueeze(1), min = 0, max =  1) #TO DO: need to change this so it makes sum(frac) = 1 
            params[:,i] = this_frac_clamped.squeeze()

        
        X = self.modelfunc(self.grad, params)

        
        
        return X.to(torch.float32), params
    #return Net(grad, modelfunc, dim_hidden, num_layers, dropout_frac, activation=nn.PReLU())
