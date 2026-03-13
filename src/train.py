import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm
import torch

def train(net, img, lossfunc, lr=1e-3, batch_size=256, num_iters=10, patience=10):
    """
    A function to train a MLP neural network on the image data, using 
    the specified loss function and training parameters.

    Args:
        net (torch.nn.Module): The neural network model to be trained.
        img (torch.Tensor): The input image data for training.
        lossfunc (torch.nn.Module): The loss function to optimize.
        lr (float): Learning rate for the optimizer.
        batch_size (int): The size of each training batch.
        num_iters (int): The number of training iterations (epochs).

    Returns:
        X_real_pred (torch.Tensor): The predicted image after training.
        params (torch.Tensor): The predicted parameters from the network.
    """

    # create batch queues for data
    num_batches = len(img) // batch_size
    trainloader = utils.DataLoader(img,
                                    batch_size = batch_size,
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = True)

    # loss function and optimizer
    my_optim =  optim.Adam(net.parameters(), lr=lr)
    # my_optim =  optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)  
    # my_scheduler = optim.lr_scheduler.OneCycleLR(
    #     my_optim, 
    #     max_lr=1e-2, 
    #     steps_per_epoch=len(trainloader), 
    #     epochs=num_iters, 
    #     pct_start=0.1,
    #     anneal_strategy="cos",
    #     div_factor=25.0,
    #     final_div_factor=1e4,
    # )

    # best loss
    best = 1e16
    num_bad_epochs = 0
    patience = 10

    for epoch in range(num_iters):
        print("-----------------------------------------------------------------")
        print("epoch: {}; bad epochs: {}".format(epoch, num_bad_epochs))
        net.train()
        running_loss = 0.

        for i, X_batch in enumerate(tqdm(trainloader), 0):
            
            # zero the parameter gradients
            my_optim.zero_grad()

            # forward + backward + optimize
            X_pred, pred_params = net(X_batch)

            loss = lossfunc(X_pred, X_batch)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Loss is NaN or Inf! Debugging needed.")
                break  # Or raise an error
            loss.backward()
            my_optim.step()
            # my_scheduler.step()  # Update learning rate
            if i % 200 == 0:
                print("lr:", my_optim.param_groups[0]["lr"])
            running_loss += loss.item()

        print("loss: {}".format(running_loss))

        # early stopping
        if running_loss < best:
            print("########## This epoch's loss is better than the best loss so far, saving model")
            final_model = net.state_dict()
            best = running_loss
            num_bad_epochs = 0
        else:
            num_bad_epochs = num_bad_epochs + 1
            if num_bad_epochs == patience:
                print("done, best loss: {}".format(best))
                break
    print("done")
    # restore best model
    net.load_state_dict(final_model)

    net.eval()
    with torch.no_grad():
        X_real_pred, params = net(img)
    return X_real_pred, params
