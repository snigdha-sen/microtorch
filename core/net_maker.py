import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self,grad, modelfunc, dim_hidden, num_layers, dropout_frac, clipping_method = 'clamp', activation=nn.PReLU()):  

        """
        Define the network architecture
        
        Args:
            grad: gradient table/sequence details
            model: compartmental model to fit
            dim_hidden: number of units in each hidden layer
            num_layers: number of hidden layers
            dropout_frac: dropout fraction
            activation: activation function for each layer. Defaults to nn.PReLU().
        """
        
        super(Net, self).__init__()
        self.grad = grad
        self.modelfunc  = modelfunc
        self.clipping_method = clipping_method

        dim_in          = dim_hidden
        self.fc_layers  = nn.ModuleList()
        self.fc_layers.extend([nn.Linear(dim_in, dim_hidden), activation])
        
        # Get the number of signal model parameters
        dim_out = modelfunc.n_params + modelfunc.n_frac
        
        # Add fully connected hidden layers 
        for _ in range(num_layers-1): 
            self.fc_layers.extend([nn.Linear(dim_hidden, dim_hidden), activation])
        
        # Add the last linear layer for regression    
        self.encoder = nn.Sequential(*self.fc_layers, nn.Linear(dim_hidden, dim_out)) 
        
        self.dropout_frac = dropout_frac
        if dropout_frac > 0:
            self.dropout = nn.Dropout(dropout_frac)

    def forward(self, X):        
        
        if self.dropout_frac > 0:
            X = self.dropout(X)

        #params = self.encoder(X)              
        params = F.softplus(self.encoder(X))
        #params = abs(self.encoder(X))
        #get the signal model function        
        #modelfunc = getattr(models, model)
        modelfunc = self.modelfunc
        clipping_method = self.clipping_method
                               
        for i in range(modelfunc.n_params): #set min/max of non-volume fraction parameters       
            params[:,i] = Net.squash(params[:, i].clone().unsqueeze(1), clipping_method, modelfunc.parameter_ranges[i,0], modelfunc.parameter_ranges[i,1])
         
        #set min/max of volume fraction parameters  
        if modelfunc.n_frac == 1: #if just two compartments, then constraining one volume fraction to [0,1] also constrains the other to [0,1]
            params[:, modelfunc.n_params] = Net.squash(params[:, modelfunc.n_params].clone().unsqueeze(1), clipping_method, 0, 1)
        else:    
            ValueError("This isn't correct for more than two compartments at the moment. It doesn't consider the final volume fraction parameter, which is 1-sum(frac). Not sure where best to calculate the final volume fraction parameter so not trying to fix this yet (Paddy).")    
            for i in range(modelfunc.n_params, modelfunc.n_params + modelfunc.n_frac):
                # Set negative values to 0
                params[:, i] = torch.relu(params[:, i]) 

            sum_params = torch.sum(params[:, modelfunc.n_params:modelfunc.n_params + modelfunc.n_frac], dim=1, keepdim=True)
            for i in range(modelfunc.n_params, modelfunc.n_params + modelfunc.n_frac): #set min/max of volume fraction parameters  
                # Normalize to make the sum of params[:, i] equal to 1
                params[:, modelfunc.n_params:modelfunc.n_params + modelfunc.n_frac] /= sum_params


        
        X = self.modelfunc(self.grad, params)
        X = X.to(torch.float32)
        return torch.nan_to_num(X,nan=0.5), params
    


    def squash(param, method, p_min, p_max):

        if method == 'clamp':

            squashed_param_tensor =torch.clamp(param, min=p_min, max=p_max)
            unsqueezed_param = squashed_param_tensor.squeeze(1)

        elif method == 'sigmoid':

            sigmoid_param = torch.sigmoid(param)
            scaled_param = p_min + (p_max - p_min) * sigmoid_param
            unsqueezed_param = scaled_param.squeeze(1)

        else:
            raise ValueError("Unsupported method: {}".format(method))

        return unsqueezed_param