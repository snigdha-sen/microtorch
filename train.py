import torch.optim as optim
import torch.utils.data as utils


from model_code.torch_blocks import *
from core.trackers import EarlyStoppage
from utils.params import params

def train_single_scan_using_blocks(net,  ##Training off a single iamge using the blocks from core.torch_blocks
                                   data,
                                   criterion,
                                   lr=1e-3,
                                   batch_size=256,
                                   epochs=10,
                                   device = "cpu"
                                   ):

    num_batches = len(data) // batch_size
    trainloader = utils.DataLoader(data,
                                   batch_size = batch_size,
                                   drop_last=True,
                                   shuffle=True,
                                   )

    optimizer = optim.Adam(net.parameters(), lr=lr)
    scaler = torch.amp.GradScaler(device=device)
    early_stoppage = EarlyStoppage(trigger_epoch=10)
    # best loss
    best = 1e16
    num_bad_epochs = 0
    patience = 10

    for e in range(epochs):
        print("-----------------------------------------------------------------")
        print("epoch: {}; bad epochs: {}".format(e, num_bad_epochs))
        net, loss_average = train_block(net,
                                          trainloader,
                                          criterion,
                                          optimizer,
                                          scaler,
                                          device=device)

        early_stoppage.update(loss_average)
        if early_stoppage.improved:
            print("Saving Model")
            torch.save(net.state_dict(), "results/model.pth")
        print(f"Best loss: {early_stoppage.loss_track}")


    ##Evaluate
    net.eval()
    ##Use the test function from core.torch_blocks if doing a more substantial test
    with torch.no_grad():
        X_real_pred, params = net(data)
    return X_real_pred, params, net









#Old train function, will be deprecated in future versions
def train(net, img, lossfunc, lr=1e-3, batch_size=256, num_iters=10):

    # create batch queues for data
    num_batches = len(img) // batch_size
    trainloader = utils.DataLoader(img,
                                    batch_size = batch_size,
                                    shuffle = True,
                                    num_workers = 2,
                                    drop_last = True)

    # loss function and optimizer
    # criterion =  nn.MSELoss() # loss function now input as a argument "lossfunc" - ssFit updated to specificy argument as input to train()
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

            #X_batch = AquisitionScheme.sanitize_tensor(X_batch)
            X_pred, pred_params = net(X_batch)
            #print(X_pred.max(), X_pred.min(), X_batch.max(), X_batch.min())

            loss = lossfunc(X_pred, X_batch)
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                print("Loss is NaN or Inf! Debugging needed.")
                break  # Or raise an error
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
