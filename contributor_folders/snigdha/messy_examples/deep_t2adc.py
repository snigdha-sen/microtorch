# import libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm



#define the t2-adc neural network


class Net(nn.Module):
    def __init__(self, b_values, TE ):
        super(Net, self).__init__()
        
        nparams = 2

        self.b_values = b_values
        self.TE = TE
        self.fc_layers = nn.ModuleList()
        for i in range(3): # 3 fully connected hidden layers
            self.fc_layers.extend([nn.Linear(len(b_values), len(b_values)), nn.ELU()])
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(len(b_values), nparams))

    def forward(self, X):
        params = torch.abs(self.encoder(X)) # Dp, Dt, Fp
        print(params.size())
        D = params[:, 0].unsqueeze(1)
        T2 = params[:, 1].unsqueeze(1)
        
        X = torch.exp(-self.b_values*D) * torch.exp(-(self.TE - torch.min(self.TE))/T2) 

        return X, D, T2