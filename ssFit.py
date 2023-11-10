import argparse
import getpass
import os
import random
import sys
import torch
import utils
import numpy as np
import nibabel as nib
from sklearn.preprocessing import MinMaxScaler
import pickle
from train import train
from model_maker import ModelMaker
from utils.net_maker import Net
import torch.nn as nn
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
#parser.add_argument("-ld", "--logdir", help="Path to save output", default=f"/tmp/{getpass.getuser()}")
#parser.add_argument("-lm", "--log_measures", help="Save measures for each epoch", action='store_true')
parser.add_argument("-ni", "--num_iters", help="Number of iterations to train for", type=int, default=2000)
parser.add_argument("-lr", "--learning_rate", help="Learning rate", type=float, default=3e-4)
parser.add_argument("-se", "--seed", help="Random seed", type=int, default=random.randint(1, int(1e6)))
parser.add_argument("-img", "--image", help="Image to train on", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-ma", "--mask", help="Mask to use for image", default=f"/tmp/{getpass.getuser()}")
parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=256)
parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=3)
parser.add_argument("-m", "--model", help="Compartmental Model to use. Implemented are verdict, sandi, or user defined ones form combinations of ball; sphere, stick; astrosticks; cylinder; astrocylinders; zeppelin; astrozeppelins; dot.", default="verdict")
parser.add_argument("-a", "--activation", help="Activation function to use with mlp: relu, prelu or tanh.", default="prelu")
parser.add_argument("-op", "--operation", help="Operation to perform (train+fit, train, fit).", default="train+fit")
parser.add_argument("-bvals", "--bvals", help="bval file in FSL format and in [s/mm2]", default="")
parser.add_argument("-bvecs", "--bvecs", help="bvec file in FSL format", default="")
parser.add_argument("-d", "--delta", help="gradient pulse separation in ms", default=24, type=float)
parser.add_argument("-sd", "--smalldelta", help="gradient pulse duration in ms", default=8, type=float)
parser.add_argument("-TE", "--TE", help="echo time in ms", default="")
parser.add_argument("-TR", "--TR", help="repetition time in ms", default="")
parser.add_argument("-TI", "--TI", help="inversion time in ms", default="")

args = parser.parse_args()
mlp_activation = {'relu': torch.nn.ReLU(),'prelu': torch.nn.PReLU, 'tanh': torch.nn.Tanh()}

# Set up torch and cuda
#deviceinuse = 'cuda' if torch.cuda.is_available() else 'cpu'
#dtype = torch.float32
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

# Set random seeds
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

# Set the inputs
img = args.image
mask = args.mask
bvals = args.bvals
bvecs = args.bvecs
delta = args.delta
smalldel = args.smalldelta
TE = args.TE
TR = args.TR
TI = args.TI
lr = args.learning_rate
num_iters = args.num_iters


from signal_models import Ball, Stick
ball = Ball()
stick = Stick()
comps = ball


def grad_maker(img, bvals, bvecs, delta, smalldel):

    bvals = np.loadtxt(bvals)
    bvecs = np.loadtxt(bvecs)
    bvals = bvals * 1e-3 #in ms/um2
    bvecs = np.transpose(bvecs)
    gamma = 2.67e2 #ms^-1mT-1
    G = (np.sqrt(bvals/(delta-(smalldel/3))))/(gamma*smalldel) #mT/um
    '''
    if TE:
        grad = np.concatenate((bvecs,bvals[:,None],delta,smalldel,G,TE),axis=1)
    if TR and TI:
        grad = np.concatenate((bvecs,bvals[:,None],delta,smalldel,G,TE=None,TR,TI),axis=1)
    if TE and TR and TI:
        grad = np.concatenate((bvecs,bvals[:,None],delta,smalldel,G,TE,TR,TI),axis=1)
    '''
    grad = np.concatenate((bvecs,bvals[:,None]),axis=1)

    return torch.tensor(grad)

def img_masker(imgfile, maskfile):

    img = nib.load(imgfile).get_fdata()
    mask = nib.load(maskfile).get_fdata()
    imgdim = np.shape(img)
    maskm = np.reshape(mask,np.prod(imgdim[0:3]))
    imgr = np.reshape(img,(np.prod(imgdim[0:3]),imgdim[3]))
    imgm = imgr[maskm==1,:]
    imgm = imgm/np.expand_dims(imgm[:,0],axis=1)

    return imgm

grad = grad_maker(img, bvals, bvecs, delta, smalldel)
modelfunc = ModelMaker(comps)

imgm = img_masker(img, mask)
bunique = np.unique(grad[:,3])
imgm_ave = np.zeros((imgm.shape[0],len(bunique)))
for i in range(len(bunique)):
    imgm_ave[:,i] = np.mean(imgm[:,grad[:,3]==bunique[i]], axis=1)

print(imgm_ave)

grad_ave = np.zeros((len(bunique), 4))
grad_ave[:,3] = bunique
grad_ave = torch.tensor(grad_ave)

net = Net(grad_ave, modelfunc, dim_hidden=grad_ave.shape[0], num_layers=3, dropout_frac=0.5, activation=nn.PReLU())
signal, D = train(net, imgm_ave, grad_ave, modelfunc, lr=1e-3, batch_size=256, num_iters=10000)

plt.figure()
plt.plot(signal, bunique)
plt.show()