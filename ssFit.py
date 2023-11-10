import argparse
import getpass
import os
import random
import sys
import torch
import util
import numpy as np
import nibabel as nib
from siren import Siren
from siren import MLP
from training import Trainer
from sklearn.preprocessing import MinMaxScaler
import pickle
import py7zr
from train import trainer


parser = argparse.ArgumentParser()
#parser.add_argument("-ld", "--logdir", help="Path to save output", default=f"/tmp/{getpass.getuser()}")
#parser.add_argument("-lm", "--log_measures", help="Save measures for each epoch", action='store_true')
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=2000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=3e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-img", "--image", help="Image to train on", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=256)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=3)
parser.add_argument("-m", "--model", help="Compartmental Model to use. Implemented are verdict, sandi, or user defined ones form combinations of ball; sphere, stick; astrosticks; cylinder; astrocylinders; zeppelin; astrozeppelins; dot.", default="verdict")
parser.add_argument("-a", "--activation", help="Activation function to use with mlp: relu, prelu or tanh.", default="prelu")
parser.add_argument("-op", "--operation", help="Operation to perform (train+fit, train, fit).", default="train+fit")
parser.add_argument("-bvals", "--bvals", help="bval file in FSL format and in [s/mm2]", default="")
parser.add_argument("-bvecs", "--bvecs", help="bvec file in FSL format", default="")
parser.add_argument("-d", "--delta", help="gradient pulse separation in ms", default="")
parser.add_argument("-sd", "--smalldelta", help="gradient pulse duration in ms", default="")
parser.add_argument("-TE", "--TE", help="echo time in ms", default="")
parser.add_argument("-TR", "--TR", help="repetition time in ms", default="")
parser.add_argument("-TI", "--TI", help="inversion time in ms", default="")

args = parser.parse_args()
mlp_activation = {'relu': torch.nn.ReLU(),'prelu': torch.nn.PReLU, 'tanh': torch.nn.Tanh()}

# Set up torch and cuda
deviceinuse = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Set the inputs
img = args.image
bvals = args.bvals
bvecs = args.bvecs
delta = args.delta
smalldel = args.smalldel
lr = args.learning_rate
num_iters = args.num_iters

trainer = trainer(grad, model, lr=1e-3, print_freq=1, batch_size=256, num_iters=10000, dim_hidden=30, num_layers=3, dropout_frac=0.5, activation=nn.PReLU())