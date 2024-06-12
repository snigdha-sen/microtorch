# Import packages
import torch
import torch.nn as nn

# Rician Loss Function
class RicianLoss(nn.Module):
    def __init__(self, sigma=0.05):
        super(RicianLoss, self).__init__()
        self.sigma = sigma
    #
    def forward(self, predictions, inputs):
        # Rician loss
        term1 = torch.log(inputs / (self.sigma ** 2))
        term2 = -(inputs ** 2 + predictions ** 2) / (2 * (self.sigma ** 2))
        #
        z = (inputs * predictions) / (self.sigma ** 2)
        I0e = torch.special.i0e(z)
        lI0e = torch.log(I0e)
        term3 = lI0e + z
        #
        log_pdf = term1 + term2 + term3
        #
        n_batch = inputs.shape[0]
        loss = -torch.sum(log_pdf) / n_batch
        return loss

#
# # Example Usage
# import RicianLoss
# loss_fun = RicianLoss()
# loss = loss_fun.forward(predictions, inputs)
# print(loss.item())
