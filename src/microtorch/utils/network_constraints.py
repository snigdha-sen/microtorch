from typing import Any
import torch

def squash(
    param: torch.Tensor,
    method: str,
    p_min: float,
    p_max: float,
    T: float = 1.0
) -> torch.Tensor:
    """
    Constrain the parameter values to be within the specified range 
    [p_min, p_max] using the specified method.

    Args:
        param (torch.Tensor): The parameter tensor to be constrained.
        method (str): The method to use for constraining the parameters. 
        Options are 'clamp' (simple clipping) or 'sigmoid' (sigmoid squashing).
        p_min (float): The minimum value for the parameter.
        p_max (float): The maximum value for the parameter.

    Returns:
        unsqueezed_param (torch.Tensor): The constrained parameter tensor, 
        with the same shape as the input param but squeezed to remove the extra dimension.
    """

    if method == 'clamp':
        squashed_param_tensor = torch.clamp(param, min=p_min, max=p_max)
        unsqueezed_param = squashed_param_tensor.squeeze(1)

    elif method == 'sigmoid':
        sigmoid_param = torch.sigmoid(param / T)
        scaled_param = p_min + (p_max - p_min) * sigmoid_param
        unsqueezed_param = scaled_param.squeeze(1)

    elif method == 'free': #no squashing 
        unsqueezed_param = param.squeeze(1)

    else:    
        raise ValueError("Unsupported method: {}".format(method))

    return unsqueezed_param


def fraction_squash(
    method: str,
    logits_all: torch.Tensor,
    modelfunc: Any,
    tau: float = 1.0
) -> torch.Tensor:
    """
    Constrain the fraction parameters to be valid fractions (between 0 and 1, and sum to 1) using the specified method. 
    Args:
        method (str): The method to use for constraining the fractions. Options are 'softmax' (softmax squashing), 'clamp' (clamping free fractions and computing implicit  last fraction), or 'free' (no squashing, raw logits).
        logits_all (torch.Tensor): The raw output tensor from the network containing both non-fraction parameters and fraction parameters. Shape: (batch_size, n_parameters + n_fractions).
        modelfunc (ModelMaker): The model function object containing information about the number of parameters and fractions.  
        tau (float): Temperature parameter for softmax squashing (only used if method='softmax').
    Returns:
        fractions (torch.Tensor): The constrained fraction tensor, with shape (batch_size, n_fractions), where each row sums to 1 and each element is in [0, 1] (if method is 'softmax' or 'clamp').
    Note:
        - For 'softmax', the fraction parameters are obtained by applying softmax to the relevant logits, ensuring they sum to 1 and are between 0 and 1.
        - For 'clamp', the first n_fractions-1 parameters are treated as free fractions that are clamped to [0, 1], and the last fraction is computed implicitly as 1 minus the sum of the free fractions. All fractions are then normalized to ensure they sum to 1.
        - For 'free', no squashing is applied and the raw logits for the fractions are returned, which may not be valid fractions.
    """

    if method == 'softmax':
        fractions = torch.softmax(logits_all / tau, dim=1)

    elif method == 'clamp':
        if modelfunc.n_fractions == 1:
            # Two compartments: clip the single free fraction to [0,1]
            fractions = squash(
                logits_all[:, 0].clone().unsqueeze(1),
                method,
                0, 1
            )
            fractions = torch.cat([fractions, 1 - fractions], dim=1)  # implicit second fraction
        else:
            #More than two compartments
            # extract free fractions and make sure in [0,1]
            f_free = torch.relu(logits_all[:, :-1])  # all but last fraction are free 

            # Compute implicit last fraction
            final_f = 1 - f_free.sum(dim=1, keepdim=True)
            final_f = torch.clamp(final_f, min=1e-8)  # prevent negative last fraction

            # Concatenate free fractions + last fraction
            fractions = torch.cat([f_free, final_f], dim=1)

            # Normalize all fractions so sum = 1 
            sum_all = fractions.sum(dim=1, keepdim=True)
            fractions = fractions / torch.clamp(sum_all, min=1e-8)
    elif method == 'free': #no squashing 
        fractions = logits_all  
    else:
        raise ValueError("Unsupported method: {}".format(method))

    return fractions