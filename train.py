import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils
from tqdm import tqdm

from network import Net
import net_maker
from dataloader import data_loader

# define network
img, b_values, Delta, delta, gradient_strength = data_loader()
grad = grad_maker(b_values, b_vecs, Delta, delta,)# <---- NEED TO BE CREATED

class trainer():
    def __init__(self, grad, model, lr=1e-3, print_freq=1, batch_size=256, num_iters=10000, dim_hidden=30, num_layers=3, dropout_frac=0.5, activation=nn.PReLU()):

        self.optimizer = torch.optim.Adam(self.representation.parameters(), lr=lr)
        self.print_freq = print_freq
        self.steps = 0  # Number of steps taken in training
        self.loss_func = torch.nn.MSELoss()
        # Store parameters of best model (in terms of highest PSNR achieved)
        self.best_model = OrderedDict((k, v.detach().clone()) for k, v in self.representation.state_dict().items())

        net = net_maker(grad, model, dim_hidden=grad.shape[0], num_layers=3, dropout_frac=0.5, activation=nn.PReLU())

        # create batch queues for data
        num_batches = len(img) // batch_size
        trainloader = utils.DataLoader(torch.from_numpy(img.astype(np.float32)),
                                        batch_size = batch_size, 
                                        shuffle = True,
                                        num_workers = 2,
                                        drop_last = True)

        # loss function and optimizer
        criterion =  self.loss_fun
        my_optim =  self.optimizer

        # best loss
        best = 1e16
        num_bad_epochs = 0
        patience = 10

        # train
        def train(self, trainloader, num_iters, num_bad_epochs, my_optim):
            for epoch in range(num_iters): 
                print("-----------------------------------------------------------------")
                print("epoch: {}; bad epochs: {}".format(epoch, num_bad_epochs))
                net.train()
            running_loss = 0.

            for i, X_batch in enumerate(tqdm(trainloader), 0):
                # zero the parameter gradients
                my_optim.zero_grad()

                # forward + backward + optimize
                X_pred, ... = net(X_batch)
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
            X_real_pred, ... = net(torch.from_numpy(img.astype(np.float32)))
