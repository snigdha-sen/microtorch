from multiprocessing import freeze_support
import os
import nibabel as nib
from train import train_single_scan_using_blocks
import torch.nn as nn

import matplotlib.pyplot as plt
from pathlib import Path
from utils.util_function import strip_filename 


from core.args import gen_args, gen_args
from model_code.model_maker import ModelMaker
from model_code.net_maker import Net
from core.acquisition_scheme import *
from core.preprocessing import *

def fit_model(args):

    mlp_activation = {'relu': torch.nn.ReLU(), 'prelu': torch.nn.PReLU(), 'tanh': torch.nn.Tanh(), 'elu': torch.nn.ELU()}

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Set the inputs
    model = args.model
    modelfunc = ModelMaker(model) ##Model Func is the function that generates the qmri model i.e SANDI

    # Load acquisition parameters
    #if args.bvals is not None:
    #    grad = as_auto_loader(args.bvals, args.bvecs, args.delta, args.smalldelta, args.TE, args.bdelta)
        
    #if args.grad is not None:
    #    grad = acquisition_scheme_loader(args.grad)

    grad = AquisitionScheme().load_scheme_from_args(args)


    # Load the image and mask
    img  = torch.from_numpy(nib.load(os.path.join(args.folder, args.image)).get_fdata().astype(np.float32))
    if args.mask is None:
        # No mask provided; use whole image
        mask = torch.ones(img.shape[:3], dtype=torch.float32)
    else:
        # Load mask from file
        mask = torch.from_numpy(nib.load(os.path.join(args.folder, args.mask)).get_fdata().astype(np.float32))


    ##This is a debug step because my test data has NaNs in it (i.e it sucks)


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
        #grad.compute_direction_averaged_scheme()
        img = direction_average(img,grad) ##This function leaves Nan's in the data so will need to be fixed in the future.

        
    #convert to "voxel-form" i.e. flatten
    X_train, maskvox = img2voxel(img,mask)
    
    #this ensures that there wont be any NaNs
    X_train = X_train + 1e-16
        
    #normalise using the function
    X_train = normalise(X_train,grad)


    # Define network
    #torch.autograd.set_detect_anomaly(True)
    lossfunc = nn.MSELoss()
    net = Net(grad, modelfunc, dim_hidden=grad.number_of_measurements, num_layers=3, dropout_frac=args.dropout_frac, clipping_method=args.clip, activation=mlp_activation[args.activation])
    print(grad.bvecs.shape)
    # Train network
    _, params, __ = train_single_scan_using_blocks(net, X_train, lossfunc, lr=args.learning_rate, batch_size=256, epochs=args.num_iters)
        

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



def main():
    args = gen_args()
    fit_model(args)


    return
if __name__ == '__main__':
    freeze_support()
    main()
