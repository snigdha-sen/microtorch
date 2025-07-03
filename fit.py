from multiprocessing import freeze_support
from utils.preprocessing import *
from acquisition_scheme import *
import argparse
import os
import random
import torch
import numpy as np
import nibabel as nib
from train import train
from model_maker import ModelMaker
from net_maker import Net
import torch.nn as nn
from acquisition_scheme import txt_file_loader, acquisition_scheme_loader
import matplotlib.pyplot as plt
from pathlib import Path
from utils.util_function import strip_filename 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-ni",  "--num_iters",  help="Number of iterations to train for", type=int, default=20)
    parser.add_argument("-lr",  "--learning_rate", help="Learning rate",            type=float,default=3e-4)
    parser.add_argument("-se",  "--seed",       help="Random seed",                 type=int,  default=random.randint(1, int(1e6)))
    parser.add_argument("-f",   "--folder",     help="Folder where image & mask are stored",   default=os.getcwd())
    parser.add_argument("-img", "--image",      help="Path of the image to train on", required=True)
    parser.add_argument("-ma",  "--mask",       help="Path of the mask to apply to image", default=None, type=str)
    parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int,  default=256)
    parser.add_argument("-nl",  "--num_layers", help="Number of layers", type=int, default=3)
    parser.add_argument("-m",   "--model", type=str, help="Compartmental Model to use. Implemented are verdict, sandi, or user defined ones form combinations of ball; sphere, stick; astrosticks; cylinder; astrocylinders; zeppelin; astrozeppelins; dot.", default="verdict")
    parser.add_argument("-a",   "--activation", type=str, help="Activation function to use with mlp: elu, relu, prelu or tanh.", default="prelu")
    parser.add_argument("-op",  "--operation",  help="Operation to perform (train+fit, train, fit).", default="train+fit")
    parser.add_argument("-bvals", "--bvals",    help="bvals file in FSL format and in [s/mm2]",      default=None,      type=str)
    parser.add_argument("-bvecs", "--bvecs",    help="bvecs file in FSL format",                     default=None,      type=str)
    parser.add_argument("-grad", "--grad",      help="acquisition scheme file in FSL format and in [s/mm2]", default=None,  type=str)
    parser.add_argument("-d",   "--delta",      help="txt file with gradient pulse separation (ms)", default="data/grad_files/delta.txt",      type=str)
    parser.add_argument("-sd",  "--smalldelta", help="txt file with gradient pulse duration (ms)",   default="data/grad_files/smalldelta.txt", type=str)
    parser.add_argument("-TE",  "--TE",         help="echo time in ms", default="")
    parser.add_argument("-TR",  "--TR",         help="repetition time in ms", default="")
    parser.add_argument("-TI",  "--TI",         help="inversion time in ms", default="")
    parser.add_argument("-df",  "--dropout_frac", help="dropout fraction", type=float, default=0.2)
    parser.add_argument("-lmax","--lmax",       help="max order used for spherical harmonics", default = 2)
    parser.add_argument("-bd",  "--bdelta",     help="shape of gradient pulse", default=1, type=float)
    parser.add_argument("-c",   "--clip", type=str, help="Clipping method to go to parameter space. Options are clamp and sigmoid", default="clamp")
    args = parser.parse_args()      

    mlp_activation = {'relu': torch.nn.ReLU(), 'prelu': torch.nn.PReLU(), 'tanh': torch.nn.Tanh(), 'elu': torch.nn.ELU()}

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Set the inputs
    model = args.model
    modelfunc = ModelMaker(model)

    # Load acquisition parameters
    if args.bvals is not None:
        grad = txt_file_loader(args.bvals, args.bvecs, args.delta, args.smalldelta, args.TE, args.bdelta)
        
    if args.grad is not None:
        grad = acquisition_scheme_loader(args.grad)
    
    # Load the image and mask
    img  = torch.from_numpy(nib.load(os.path.join(args.folder, args.image)).get_fdata().astype(np.float32))
    if args.mask is None:
        # No mask provided; use whole image
        mask = torch.ones(img.shape[:3], dtype=torch.float32)
    else:
        # Load mask from file
        mask = torch.from_numpy(nib.load(os.path.join(args.folder, args.mask)).get_fdata().astype(np.float32))
    
    # # OPTIONAL: make a smaller mask for testing
    # tmpmask  = torch.zeros_like(mask)
    # zslice   = 5
    # #make a smaller mask for testing
    # tmpmask = torch.zeros_like(mask)
    # zslice = 0
    # tmpmask[:,:,zslice] = mask[:,:,zslice]
    # mask     = tmpmask

    #need to put a check in here to see if the data needs to be direction averaged
    if modelfunc.spherical_mean:        
        #direction average the data. img, grad now become the direction-averaged versions
        print(grad)
        img,grad = direction_average(img,grad)
        
    #convert to "voxel-form" i.e. flatten
    X_train, maskvox = img2voxel(img,mask)
    
    #this ensures that there wont be any NaNs
    X_train = X_train + 1e-16
        
    #normalise using the function
    X_train = normalise(X_train,grad)


    # Define network
    torch.autograd.set_detect_anomaly(True) 
    lossfunc = nn.MSELoss()
    net = Net(grad, modelfunc, dim_hidden=grad.number_of_measurements, num_layers=3, dropout_frac=args.dropout_frac, clipping_method=args.clip, activation=mlp_activation[args.activation])
    print(grad.bvecs.shape)
    # Train network
    _, params = train(net, X_train, grad, modelfunc, lossfunc, lr=args.learning_rate, batch_size=256, num_iters=args.num_iters)
        

    # Reconstruct parameter maps from network outputs
    param_map = np.zeros((*np.shape(mask),modelfunc.n_params + modelfunc.n_frac))
    for i in range(0,modelfunc.n_params + modelfunc.n_frac):
        param_map[...,i] = voxel2img(params[:,i], maskvox, mask.shape)

    # Create folder to store results
    output_folder = "./results"
    Path(output_folder).mkdir(parents=True, exist_ok=True)  

    # Save output maps as NIFTI 
    img     = nib.load(os.path.join(args.folder, args.image))
    new_img = nib.Nifti1Image(param_map, img.affine, img.header)  
    nib.save(new_img, os.path.join(output_folder, strip_filename(args.image) + '_param_maps.nii.gz'))
    
    # Visualise output maps
    zslice = 0
    _, ax = plt.subplots(1, modelfunc.n_params + modelfunc.n_frac ,figsize=(5 * (modelfunc.n_params + modelfunc.n_frac), 2))
    for i in range(0,modelfunc.n_params + modelfunc.n_frac):
        im = ax[i].imshow(param_map[:, :, zslice, i])
        _  = plt.colorbar(im, ax=ax[i])
        ax[i].set_title(modelfunc.param_names[i] + ' (' + modelfunc.comp_names[modelfunc.comp_ind[i]] + ')')
    
    plt.show()

    
if __name__ == '__main__':
    freeze_support()
    main()
