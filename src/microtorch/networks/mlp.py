from typing import Optional, Union, Sequence, Callable
import torch
import torch.nn as nn
import copy


class DevMLP(nn.Module):
    def __init__(
        self, 
        input_neurons: int, 
        layer_dims: int, 
        n_layers: int, 
        dim_out: int, 
        activation: nn.Module, 
        dropout: float
    ) -> None:
        super().__init__()

        layers = []
        layers.extend([nn.Linear(input_neurons, layer_dims), copy.deepcopy(activation)])

        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(layer_dims, layer_dims), copy.deepcopy(activation)])

        self.net = nn.Sequential(*layers, nn.Linear(layer_dims, dim_out))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout:
            x = self.dropout(x)
        return self.net(x)


class HiddenDropoutMLP(nn.Module):
    def __init__(
        self, 
        input_neurons: int, 
        layer_dims: int, 
        n_layers: int, 
        dim_out: int, 
        activation: nn.Module, 
        dropout: float
    ) -> None:
        super().__init__()

        layers = []

        layers.extend([nn.Linear(input_neurons, layer_dims), copy.deepcopy(activation)])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(layer_dims, layer_dims), copy.deepcopy(activation)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(layer_dims, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return self.head(h)