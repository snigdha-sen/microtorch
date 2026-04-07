import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(
        self,
        input_neurons,
        layer_dims,
        n_layers,
        dim_out,
        kernel_size=3,
        activation=None,
        dropout=None,
    ):
        super().__init__()

        self.input_neurons = input_neurons
        
        layers = []
        in_channels = input_neurons
        n_filters = layer_dims

        for _ in range(n_layers):
            layers.append(
                nn.Conv1d(
                    in_channels,
                    n_filters,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(copy.deepcopy(activation))
            layers.append(nn.BatchNorm1d(n_filters))
            if dropout is not None and dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = n_filters

        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(n_filters, dim_out)

    def forward(self, x):
        """
        x: (batch, channels, kernel_size)
        """
        #ensure correct shape for 1D conv: (batch_size, input_neurons, kernel_size) 
        #input_neurons = number of signal measurements, kernel_size = 1 for voxel-wise input, or >1 for patch-wise input

        # If input is (input_neurons, kernel_size), add a batch dimension to make it (1, input_neurons, kernel_size)
        if x.ndim == 2:
            x = x.unsqueeze(0)

        # If input is (batch_size, kernel_size, input_neurons), transpose to (batch_size, input_neurons, kernel_size)
        if x.shape[1] != self.input_neurons and x.shape[2] == self.input_neurons:
            x = x.transpose(1, 2)

        h = self.conv(x)
        
        #h = h.mean(dim=2)  # global average pooling

        center_idx = h.shape[2] // 2
        h = h[:, :, center_idx]  # use centre voxel features

        params = self.head(h)
        return params
    


    