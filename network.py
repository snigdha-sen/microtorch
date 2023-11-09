import numpy as np
import torch
import torch.nn as nn

class Net(nn.Module):

  def __init__(self, b_values, Delta, delta, gradient_strength, nparams):
      super(Net, self).__init__()

      self.b_values = b_values
      self.Delta = Delta
      self.delta = delta
      self.gradient_strength = gradient_strength

      self.layers = nn.ModuleList()
      for i in range(3): # 3 fully connected hidden layers
          self.layers.extend([nn.Linear(len(b_values), len(b_values)), nn.PReLU()])
      self.encoder = nn.Sequential(*self.layers, nn.Linear(len(b_values), nparams))
      self.dropout = nn.Dropout(0.5)

  def forward(self, X):
      
      X = self.dropout(X)
      params = torch.nn.functional.softplus(self.encoder(X))

      # constrain parameters to biophysically-realistic ranges

      X = #signal

      return X, #fitted params

        
