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


class RicianLossStable(nn.Module):  #New Rician Loss with added stability
    def __init__(self, sigma=0.05, eps=1e-8):
        super(RicianLossStable, self).__init__()
        self.sigma = sigma
        self.eps = eps  #Epsilon Param for avoiding 0/NaN

    def forward(self, predictions, inputs):
        # Ensure inputs and predictions are positive and non-zero
        inputs = torch.clamp(inputs, min=self.eps)
        predictions = torch.clamp(predictions, min=self.eps)

        # Compute terms with numerical stability
        sigma_squared = self.sigma ** 2

        # Add eps to prevent log(0)
        term1 = torch.log(inputs / sigma_squared + self.eps)

        # Compute squared terms
        term2 = -(inputs ** 2 + predictions ** 2) / (2 * sigma_squared)

        # Compute z with clipping to prevent overflow
        z = torch.clamp((inputs * predictions) / sigma_squared, max=100)

        # Compute modified Bessel function and its log
        I0e = torch.special.i0e(z)
        # Add eps to prevent log(0)
        lI0e = torch.log(I0e + self.eps)

        # Combine terms
        log_pdf = term1 + term2 + lI0e + z

        # Handle any remaining NaN values
        log_pdf = torch.nan_to_num(log_pdf, nan=0.0, posinf=0.0, neginf=0.0)

        # Compute final loss
        n_batch = inputs.shape[0]
        loss = -torch.sum(log_pdf) / n_batch

        return loss

#
# # Example Usage
# import RicianLoss
# loss_fun = RicianLoss()
# loss = loss_fun.forward(predictions, inputs)
# print(loss.item())
