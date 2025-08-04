#This script will eventually become


from model_code.model_maker import ModelMaker
from model_code.net_maker import Net

from core.acquisition_scheme import AquisitionScheme
from core.dataset import qMRIDataset
from core.preprocessing import PreProcess

from train import train_single_scan_using_blocks


import torch

###This file will be testing the usage of the newer classes for dataset loading and training


mlp_activation = {'relu': torch.nn.ReLU(), 'prelu': torch.nn.PReLU(), 'tanh': torch.nn.Tanh(), 'elu': torch.nn.ELU()}



def fit_model_wand():



    return

def fit_model_single_image_test(image_path, mask_path, args):

    model_name = args.model
    signal_model = ModelMaker(model_name)

    grad_scheme = AquisitionScheme().load_scheme_from_args(args)

    i_paths = [image_path]
    m_paths = [mask_path]

    pp = PreProcess(grad_scheme, signal_model.spherical_mean)
    dataset = qMRIDataset(i_paths, m_paths, pp, grad_scheme)

    criterion = torch.nn.MSELoss()
    net = Net(grad_scheme, signal_model, dim_hidden=grad_scheme.number_of_measurements, num_layers=3, dropout_frac=args.dropout_frac, clipping_method=args.clip, activation=mlp_activation[args.activation])

    _, params = train_single_scan_using_blocks(net, dataset, criterion, lr=args.learning_rate, batch_size=256, epochs=args.num_iters)

    #batch_train(net, dataset, criterion, lr=args.learning_rate, batch_size=256, epochs=args.num_iters)