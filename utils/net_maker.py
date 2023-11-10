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
        #dim_out = model.nparams
        dim_out = 1
        for i in range(num_layers-1): # num_layers fully connected hidden layers - number of nodes defined by the user as dim_hidden
            self.fc_layers.extend([nn.Linear(dim_hidden, dim_hidden), activation])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(dim_hidden, dim_out)) # Add the last linear layer for regression
        self.dropout = nn.Dropout(dropout_frac)

    def forward(self, X):        
        
        X = self.dropout(X)
        #params = self.encoder(X)               
        params = F.softplus(self.encoder(X))

        #get the signal model function        
        #modelfunc = getattr(models, model)
        #X = self.modelfunc(self.grad, params, 0.5)
        
        D = torch.clamp(params[:, 0].unsqueeze(1), min = 0, max =  3)
        X = torch.exp(-self.grad[:,3]*D)

        
        #for i in range(model.parameter_ranges.shape[0]):
        #    params[:,i] = torch.clamp(params[:, i].unsqueeze(1), min = model.parameter_ranges[i,1], max =  model.parameter_ranges[i,2])
        
        return X.to(torch.float32), D
    #return Net(grad, modelfunc, dim_hidden, num_layers, dropout_frac, activation=nn.PReLU())
