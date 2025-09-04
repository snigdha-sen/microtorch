
### Building blocks for basic training and validation loops, each function can train on a single epoch!

import torch
from torchmetrics.aggregation import MeanMetric
from tqdm import tqdm


def test_block(model, loader, criterion,device = "cuda"):
    model.eval()

    loss_average = MeanMetric()

    with torch.no_grad():
        for i,data in enumerate(loader):
            inputs = data
            inputs = inputs.to(device)

            prediction = model(inputs).to(device)
            loss = criterion(prediction, inputs)
            loss_average.update(loss.item())

    loss_average = loss_average.compute().item()
    return model, loss_average

def train_block(model, loader, criterion, optimizer, scaler, device="cuda"):


    ##Training for ONE epoch / pass through dataset
    model.train()
    loss_average = MeanMetric() #Tracks the loss over the dataset

    for i, data in enumerate(tqdm(loader), 0):
        inputs = data["image"] if type(data) is dict else data
        #inputs = inputs.to(device)

        with torch.autocast(device_type=device):
            prediction, prediction_parameters = model(inputs)


            ##DEBUGGING LINE
            #prediction = torch.clamp(prediction,  min=1e-8, max=1e6)
            #inputs = torch.clamp(inputs,  min=1e-8, max=1e6)

            loss = criterion(prediction, inputs) ##Will need to check if this is correct, old code also puts in the prediction and inputs here, but i am used to using ground truth?

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        scaler.step(optimizer)
        scaler.update()
        loss_average.update(loss.item())

    loss_average = loss_average.compute().item()
    return model, loss_average
