from .mlp import DevMLP, HiddenDropoutMLP
#from .cnn import SimpleCNN
#from .vae import VAE

NETWORK_REGISTRY = {
    "dev_mlp": DevMLP,
    "hidden_dropout_mlp": HiddenDropoutMLP,
    #"cnn": SimpleCNN,
    #"vae": VAE
}

def build_network(name, **kwargs):
    if name not in NETWORK_REGISTRY:
        raise ValueError(f"Unknown network: {name}")
    return NETWORK_REGISTRY[name](**kwargs)