import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm

def train(net, img, grad, modelfunc, lr=1e-3, batch_size=256, num_iters=10):

    # create batch queues for data
    num_batches = len(img) // batch_size
    trainloader = utils.DataLoader(img,
                                    batch_size = batch_size, 
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = True)

    # loss function and optimizer
    criterion =  nn.MSELoss()
    my_optim =  optim.Adam(net.parameters(), lr=lr)

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
            loss = criterion(X_pred, X_batch)
            loss.backward()
            my_optim.step()
            running_loss += loss.item()
    
        print("loss: {}".format(running_loss))
        # early stopping
        if running_loss < best:
            print("####################### saving good model #######################")
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
