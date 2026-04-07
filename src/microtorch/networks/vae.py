import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_neurons, layer_dims, n_layers=None, dim_out=None, activation=None, dropout=None, latent_dim=64):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        # Encoder
        self.fc1 = nn.Linear(input_neurons, layer_dims)
        self.fc_mu = nn.Linear(layer_dims, latent_dim)
        self.fc_logvar = nn.Linear(layer_dims, latent_dim)
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, layer_dims),
            self.activation,
            nn.Linear(layer_dims, dim_out)
        )

    def encode(self, x):
        h = self.activation(self.fc1(x))
        if self.dropout:
            h = self.dropout(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        params = self.decoder(z)
        return params, mu, logvar


     
