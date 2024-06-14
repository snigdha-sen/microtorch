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
    #from sklearn.preprocessing import MinMaxScaler
    #import pickle
    from train import train
    from model_maker import ModelMaker
    from net_maker import Net
    import torch.nn as nn
    import matplotlib.pyplot as plt
    from acquisition_scheme import acquisition_scheme_loader, txt_file_loader
                    
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
    parser.add_argument("-grad", "--grad", help="grad file in FSL format", default=None)
    parser.add_argument("-bvals", "--bvals", help="bval file in FSL format and in [s/mm2]", default=None)
    parser.add_argument("-bvecs", "--bvecs", help="bvec file in FSL format", default=None)
    parser.add_argument("-d", "--delta", help="gradient pulse separation in ms", default=24, type=float)
    parser.add_argument("-sd", "--smalldelta", help="gradient pulse duration in ms", default=8, type=float)
    parser.add_argument("-TE", "--TE", help="echo time in ms", default="")
    parser.add_argument("-TR", "--TR", help="repetition time in ms", default="")
    parser.add_argument("-TI", "--TI", help="inversion time in ms", default="")
    parser.add_argument("-df","--dropout_frac", help="dropout fraction", type=float, default=0)
    parser.add_argument("-lmax", "--lmax", help="max order used for spherical harmonics", default = 2)
    parser.add_argument("-bd", "--bdelta", help="shape of gradient pulse", default=1, type=float)
    parser.add_argument("-c", "--clip", type=str, help="Clipping method of parameters. Options are clamp, sigmoid, ", default="clamp")
    parser.add_argument("-f",   "--folder",     help="Folder where image & mask are stored",   default="./data/test_images")


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
    model = args.model


    #need to write a big function that does this for all models 
    if model == "MSDKI":
        comps = ("MSDKI",)
    elif model == "BallStick":
        comps = ("Ball","Stick")
    elif model == "StickBall":
        comps = ("Stick","Ball")
    elif model == "StandardWM":
        comps = ("Standard_WM",)
    elif model == "Ball":
        comps = ("Ball",)

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


    # def img_masker(imgfile, maskfile):

    #     img = nib.load(imgfile).get_fdata()
    #     mask = nib.load(maskfile).get_fdata()
    #     imgdim = np.shape(img)
    #     maskm = np.reshape(mask,np.prod(imgdim[0:3]))
    #     imgr = np.reshape(img,(np.prod(imgdim[0:3]),imgdim[3]))
    #     imgm = imgr[maskm==1,:]
    #     imgm = imgm/np.expand_dims(imgm[:,0],axis=1)

    #     return imgm

    # grad = grad_maker(bvals, bvecs, delta, smalldel)
    if args.bvals is not None:
        grad = txt_file_loader(args.bvals, args.bvecs, args.delta, args.smalldelta, args.TE, args.bdelta)

    if args.grad is not None:
        grad = acquisition_scheme_loader(os.path.join(args.folder, args.grad))
    # imgm = img_masker(img, mask)
    
    #load the image and mask
    img  = torch.from_numpy(nib.load(os.path.join(args.folder, args.image)).get_fdata().astype(np.float32))
    mask = torch.from_numpy(nib.load(os.path.join(args.folder, args.mask)).get_fdata().astype(np.float32))


    #make a smaller mask for testing
    # tmpmask = np.zeros_like(mask)
    # zslice = 70
    # tmpmask[:,:,zslice] = mask[:,:,zslice]
    # mask=tmpmask

    #what does this do??
    # #need to put a check in here to see if the data needs to be direction averaged
    if modelfunc.spherical_mean:      
        from utils.preprocessing import direction_average
        #direction average the data. img, grad now become the direction-averaged versions
        img,grad = direction_average(img,grad)
        
    #convert to "voxel-form" i.e. flatten
    from utils.preprocessing import img2voxel
    X_train, maskvox = img2voxel(img,mask)

    #this ensures that there wont be any NaNs
    #X_train = X_train + 1e-16

    #normalise using the function
    from utils.preprocessing import normalise

    X_train = normalise(X_train,grad)

    plt.figure()
    for i in range(0,5):

        plt.scatter(np.array(grad.bvalues).flatten(), X_train[i,:])
    plt.show()
    #convert grad and data to tensor ready for training
    
    torch.autograd.set_detect_anomaly(True)

    lossfunc = nn.MSELoss(reduction='mean')
    net = Net(grad, modelfunc, dim_hidden=len(grad.bvalues[0,:]), num_layers=args.num_layers, dropout_frac=args.dropout_frac,clipping_method=args.clip, activation=mlp_activation[args.activation])
        
    signal, params_pred = train(net, X_train, grad, modelfunc, lossfunc, lr=args.learning_rate, batch_size=256, num_iters=args.num_iters)
    
    #load in ground truths
    import scipy
    gt_params = scipy.io.loadmat(r'R:\C_PersonalData\Leon\qMRI_gradientcorrection\tryout_shit_for_microtorch\out_sameDelta_8384.mat')['out']['kernel'][0][0]
    gt_plm = scipy.io.loadmat(r'R:\C_PersonalData\Leon\qMRI_gradientcorrection\tryout_shit_for_microtorch\out_sameDelta_8384.mat')['out']['plm'][0][0]
    gt_s0 = scipy.io.loadmat(r'R:\C_PersonalData\Leon\qMRI_gradientcorrection\tryout_shit_for_microtorch\out_sameDelta_8384.mat')['out']['RotInvs'][0][0]['S0'][0][0][:,:,:,0] #take S0 fitted with first shell
    


    # fig, ax = plt.subplots(5, 2, figsize=(8,16))


    # ax[0,0].plot(params_plm_wm[:,0],params_pred[:,0],'o',markersize=1)
    # ax[0, 0].set_xlim(params_plm_wm[:,0].min(), params_plm_wm[:,0].max())
    # ax[0, 0].set_ylim(params_plm_wm[:,0].min(), params_plm_wm[:,0].max())
    # ax[0, 0].set_title('S0')

    # ax[0,1].plot(params_plm_wm[:,1],params_pred[:,1],'o',markersize=1)
    # # ax[0, 1].set_xlim(params[:,1].min(), params[:,1].max())
    # # ax[0, 1].set_ylim(params[:,1].min(), params[:,1].max())
    # ax[0, 1].set_title('Di')


    # ax[1,0].plot(params_plm_wm[:,2],params_pred[:,2],'o',markersize=1)
    # # ax[1, 0].set_xlim(params[:,2].min(), params[:,2].max())
    # # ax[1, 0].set_ylim(params[:,2].min(), params[:,2].max())
    # ax[1, 0].set_title('De')


    # ax[1,1].plot(params_plm_wm[:,3],params_pred[:,3],'o',markersize=1)
    # # ax[1, 1].set_xlim(params[:,3].min(), params[:,3].max())
    # # ax[1, 1].set_ylim(params[:,3].min(), params[:,3].max())
    # ax[1, 1].set_title('De,perp')

    # ax[2,0].plot(params_plm_wm[:,4],params_pred[:,4],'o',markersize=1)
    # ax[2, 0].set_xlim(params_plm_wm[:,4].min(), params_plm_wm[:,4].max())
    # ax[2, 0].set_ylim(params_plm_wm[:,4].min(), params_plm_wm[:,4].max())
    # ax[2, 0].set_title('f')

    # ax[2,1].plot(params_plm_wm[:,5],params_pred[:,5],'o',markersize=1)
    # ax[2, 1].set_xlim(params_plm_wm[:,5].min(), params_plm_wm[:,5].max())
    # ax[2, 1].set_ylim(params_plm_wm[:,5].min(), params_plm_wm[:,5].max())
    # ax[2, 1].set_title('p2_2')

    # ax[3,0].plot(params_plm_wm[:,6],params_pred[:,6],'o',markersize=1)
    # ax[3, 0].set_xlim(params_plm_wm[:,6].min(), params_plm_wm[:,6].max())
    # ax[3, 0].set_ylim(params_plm_wm[:,6].min(), params_plm_wm[:,6].max())
    # ax[3, 0].set_title('p2_1')

    # ax[3,1].plot(params_plm_wm[:,7],params_pred[:,7],'o',markersize=1)
    # ax[3, 1].set_xlim(params_plm_wm[:,7].min(), params_plm_wm[:,7].max())
    # ax[3, 1].set_ylim(params_plm_wm[:,7].min(), params_plm_wm[:,7].max())
    # ax[3, 1].set_title('p20')

    # ax[4,0].plot(params_plm_wm[:,8],params_pred[:,8],'o',markersize=1)
    # ax[4, 0].set_xlim(params_plm_wm[:,8].min(), params_plm_wm[:,8].max())
    # ax[4, 0].set_ylim(params_plm_wm[:,8].min(), params_plm_wm[:,8].max())
    # ax[4, 0].set_title('p21')

    # ax[4,1].plot(params_plm_wm[:,9],params_pred[:,9],'o',markersize=1)
    # ax[4, 1].set_xlim(params_plm_wm[:,9].min(), params_plm_wm[:,9].max())
    # ax[4, 1].set_ylim(params_plm_wm[:,9].min(), params_plm_wm[:,9].max())
    # ax[4, 1].set_title('p22')
    

    plt.figure()
    plt.scatter(signal, X_train)
    plt.show()

    plt.figure()
    plt.plot(params_pred[:,0])
    plt.show()
    # param_map = np.zeros((*np.shape(mask),modelfunc.n_params + modelfunc.n_frac))
    # for i in range(0,modelfunc.n_params + modelfunc.n_frac):
    #     tmpparams = np.zeros_like(maskvox)
    #     tmpparams[maskvox == 1] = params[:,i]
    #     param_map[...,i] = np.reshape(tmpparams, np.shape(mask))



    # fig, ax = plt.subplots(1, modelfunc.n_params + modelfunc.n_frac ,figsize=(5 * (modelfunc.n_params + modelfunc.n_frac), 2))
    
    # # Iterate through subplots
    # for i in range(0,modelfunc.n_params + modelfunc.n_frac):
    #     im = ax[i].imshow(param_map[0, :, :, i])
    #     cbar = plt.colorbar(im, ax=ax[i])
    #     ax[i].set_title(modelfunc.param_names[i] + ' (' + modelfunc.comp_names[modelfunc.comp_ind[i]] + ')')
    
    # plt.show()

    
if __name__ == '__main__':
    freeze_support()
    main()