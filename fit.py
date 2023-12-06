from multiprocessing import freeze_support

            
def main():
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
    parser.add_argument("-img", "--image", help="Filename of the image to train on", default=f"/tmp/{getpass.getuser()}")
    parser.add_argument("-ma", "--mask", help="Filename of the mask to apply to image", default=f"/tmp/{getpass.getuser()}")
    parser.add_argument("-lss", "--layer_size", help="Layer sizes as list of ints", type=int, default=256)
    parser.add_argument("-nl", "--num_layers", help="Number of layers", type=int, default=3)
    parser.add_argument("-m", "--model", type=str, help="Compartmental Model to use. Implemented are verdict, sandi, or user defined ones form combinations of ball; sphere, stick; astrosticks; cylinder; astrocylinders; zeppelin; astrozeppelins; dot.", default="verdict")
    parser.add_argument("-a", "--activation", type=str, help="Activation function to use with mlp: elu, relu, prelu or tanh.", default="prelu")
    parser.add_argument("-op", "--operation", help="Operation to perform (train+fit, train, fit).", default="train+fit")
    parser.add_argument("-bvals", "--bvals", help="bval file in FSL format and in [s/mm2]", default="")
    parser.add_argument("-bvecs", "--bvecs", help="bvec file in FSL format", default="")
    parser.add_argument("-d", "--delta", help="gradient pulse separation in ms", default=24, type=float)
    parser.add_argument("-sd", "--smalldelta", help="gradient pulse duration in ms", default=8, type=float)
    parser.add_argument("-TE", "--TE", help="echo time in ms", default="")
    parser.add_argument("-TR", "--TR", help="repetition time in ms", default="")
    parser.add_argument("-TI", "--TI", help="inversion time in ms", default="")
    parser.add_argument("-df","--dropout_frac", help="dropout fraction", type=float, default=0)

    args = parser.parse_args()
    mlp_activation = {'relu': torch.nn.ReLU(),'prelu': torch.nn.PReLU, 'tanh': torch.nn.Tanh(), 'elu': torch.nn.ELU()}

    # Set up torch and cuda
    #deviceinuse = 'cuda' if torch.cuda.is_available() else 'cpu'
    #dtype = torch.float32
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #torch.set_default_tensor_type('torch.cuda.FloatTensor' if torch.cuda.is_available() else 'torch.FloatTensor')

    # Set random seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Set the inputs
    imgfile = args.image
    maskfile = args.mask
    bvals = args.bvals
    bvecs = args.bvecs
    delta = args.delta
    smalldel = args.smalldelta
    TE = args.TE
    TR = args.TR
    TI = args.TI
    lr = args.learning_rate
    num_iters = args.num_iters
    model = args.model
    act = args.activation
    dropout_frac = args.dropout_frac

    #need to write a big function that does this for all models 
    if model == "MSDKI":
        comps = ("MSDKI",)
    elif model == "BallStick":
        comps = ("Ball","Stick")
    elif model == "StickBall":
        comps = ("Stick","Ball")

    #import compartment classes dynamically based on the chosen model (write a function to do this!)
    import importlib
    signal_models_module = importlib.import_module("signal_models")

    comps_classes = () #initialise tuple
    for comp in comps:
        #get the class
        this_class = getattr(signal_models_module, comp) #add to the tuple
        #create an instance of the class and add to the tuple
        comps_classes += (this_class(),)

    #make the model function that will be incorporated into the net
    from model_maker import ModelMaker
    modelfunc = ModelMaker(comps_classes)

    def grad_maker(bvals, bvecs, delta, smalldel):

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

        return grad




    # def img_masker(imgfile, maskfile):

    #     img = nib.load(imgfile).get_fdata()
    #     mask = nib.load(maskfile).get_fdata()
    #     imgdim = np.shape(img)
    #     maskm = np.reshape(mask,np.prod(imgdim[0:3]))
    #     imgr = np.reshape(img,(np.prod(imgdim[0:3]),imgdim[3]))
    #     imgm = imgr[maskm==1,:]
    #     imgm = imgm/np.expand_dims(imgm[:,0],axis=1)

    #     return imgm

    grad = grad_maker(bvals, bvecs, delta, smalldel)

    # imgm = img_masker(img, mask)
    
    #load the image and mask
    img = nib.load(imgfile).get_fdata()
    mask = nib.load(maskfile).get_fdata()
    
    #make a smaller mask for testing
    tmpmask = np.zeros_like(mask)
    zslice = 70
    tmpmask[:,:,zslice] = mask[:,:,zslice]
    mask=tmpmask

    #need to put a check in here to see if the data needs to be direction averaged
    if modelfunc.spherical_mean:        
        from utils.preprocessing import direction_average
        #direction average the data. img, grad now become the direction-averaged versions
        img,grad = direction_average(img,grad)
        
    #convert to "voxel-form" i.e. flatten
    from utils.preprocessing import img2voxel
    X_train, maskvox = img2voxel(img,mask)
    
    #this ensures that there wont be any NaNs
    X_train = X_train + 1e-16
        
    #normalise using the function
    from utils.preprocessing import normalise
    X_train = normalise(X_train,grad)

    
    # bunique = np.unique(grad[:,3])
    # imgm_ave = np.zeros((imgm.shape[0],len(bunique)))
    # for i in range(len(bunique)):
    #     imgm_ave[:,i] = np.mean(imgm[:,grad[:,3]==bunique[i]], axis=1)


    # grad_ave = np.zeros((len(bunique), 4))
    # grad_ave[:,3] = bunique
    # grad_ave = torch.tensor(grad_ave)
    
    #convert grad and data to tensor ready for training
    grad_torch = torch.tensor(grad, dtype=torch.float32)
    Xtrain_torch = torch.from_numpy(X_train.astype(np.float32))
    
    torch.autograd.set_detect_anomaly(True)
    
    net = Net(grad_torch, modelfunc, dim_hidden=grad_torch.shape[0], num_layers=3, dropout_frac=dropout_frac, activation=mlp_activation[act])
        
    signal, params = train(net, Xtrain_torch, grad_torch, modelfunc, lr=lr, batch_size=256, num_iters=1000)
        
    from utils.preprocessing import voxel2img        
    
    print(modelfunc.n_params)
    
    print(np.shape(params))
    
    param_map = np.zeros((*np.shape(mask),modelfunc.n_params + modelfunc.n_frac))
    for i in range(0,modelfunc.n_params + modelfunc.n_frac):
        tmpparams = np.zeros_like(maskvox)
        tmpparams[maskvox == 1] = params[:,i]
        param_map[...,i] = np.reshape(tmpparams, np.shape(mask))

    print(np.shape(param_map))


    fig, ax = plt.subplots(modelfunc.n_params + modelfunc.n_frac ,1 ,figsize=(5, 5 * (modelfunc.n_params + modelfunc.n_frac)))
    
    # Iterate through subplots
    for i in range(0,modelfunc.n_params + modelfunc.n_frac):
        im = ax[i].imshow(param_map[:, :, zslice, i])
        cbar = plt.colorbar(im, ax=ax[i])
        #ax[i].set_title(modelfunc.param_names[i])
    
    plt.show()

    
if __name__ == '__main__':
    freeze_support()
    main()