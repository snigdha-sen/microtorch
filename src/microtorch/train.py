import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import torch
import optuna

import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import torch
import optuna

def train(net, img, lossfunc, lr=1e-3, batch_size=256, num_iters=10, patience=10, trial=None, beta=1.0):
    """
    Train a network (MLP, CNN, or VAE) on input data with a given loss function.

    Args:
        net (torch.nn.Module): The neural network model to train.
        img (torch.Tensor): Input data.
        lossfunc (torch.nn.Module): Reconstruction/physics-based loss function.
        lr (float): Learning rate.
        batch_size (int): Batch size.
        num_iters (int): Number of epochs.
        patience (int): Early stopping patience.
        trial (optuna.Trial, optional): For hyperparameter tuning.
        beta (float): Scaling factor for KL loss (VAE).

    Returns:
        X_real_pred (torch.Tensor): Predicted signal/image.
        params (torch.Tensor): Predicted parameters.
        best_loss (float): Best training loss achieved.
    """

    trainloader = utils.DataLoader(
        img,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )

    optimizer = optim.Adam(net.parameters(), lr=lr)
    best_loss = float("inf")
    num_bad_epochs = 0

    for epoch in range(num_iters):
        print("-----------------------------------------------------------------")
        print(f"Epoch: {epoch}; Bad epochs: {num_bad_epochs}")
        net.train()
        running_loss = 0.0

        for i, X_batch in enumerate(tqdm(trainloader), 0):
            optimizer.zero_grad()

            # Forward pass: return latent if VAE
            if net.network_type.lower() == "vae":
                X_pred, pred_params, mu, logvar = net(X_batch, return_latent=True)
            else:
                X_pred, pred_params = net(X_batch)
                mu = logvar = None

            # Reconstruction / physics loss
            recon_loss = lossfunc(X_pred, X_batch)

            # KL divergence for VAE
            if mu is not None and logvar is not None:
                kl_loss = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp())
            else:
                kl_loss = 0.0

            # Total loss
            loss = recon_loss + beta * kl_loss

            if torch.isnan(loss) or torch.isinf(loss):
                print("Loss is NaN or Inf! Debugging needed.")
                break

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 200 == 0:
                print("lr:", optimizer.param_groups[0]["lr"])

        print(f"Epoch loss: {running_loss}")

        # Optuna reporting
        if trial is not None:
            trial.report(running_loss, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Early stopping
        if running_loss < best_loss:
            print("########## New best epoch, saving model")
            final_model = net.state_dict()
            best_loss = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs += 1
            if num_bad_epochs >= patience:
                print(f"Early stopping triggered. Best loss: {best_loss}")
                break

    print("Training complete.")
    net.load_state_dict(final_model)

    # Evaluate full dataset
    net.eval()
    with torch.no_grad():
        if net.network_type.lower() == "vae":
            X_real_pred, params, mu, logvar = net(img, return_latent=True)
        else:
            X_real_pred, params = net(img)

    return X_real_pred, params, best_loss
