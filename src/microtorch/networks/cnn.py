import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels, dim_out, n_filters=32, kernel_size=3, n_layers=3, activation=nn.ReLU(), dropout=0.0):
        super().__init__()

        layers = []
        in_channels = input_channels
        for _ in range(n_layers):
            layers.append(nn.Conv1d(in_channels, n_filters, kernel_size, padding=kernel_size//2))
            layers.append(activation)
            layers.append(nn.BatchNorm1d(n_filters))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_channels = n_filters

        self.conv = nn.Sequential(*layers)
        self.head = nn.Linear(n_filters, dim_out)

    def forward(self, x):
        """
        x: (batch, channels, sequence_length)
        """
        h = self.conv(x)
        h = h.mean(dim=2)  # global average pooling
        params = self.head(h)
        return params