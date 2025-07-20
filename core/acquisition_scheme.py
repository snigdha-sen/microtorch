import numpy as np
from data.load_data import load_grad
import torch

#This file generates acquisition schemes - i.e the parameters which the model runs on.
#Currently works but could be improved by integrating the methods of loading into the AS class





class AcquisitionScheme: #Renamed for better readability
    def __init__(self,
                 bvalues, ##Bvals
                 bvecs, ##bvecs
                 gradient_strengths,
                 small_delta,
                 Delta,
                 TE,
                 bdelta):


        #Redoing this section for clarity - assuming it operates on a simmilar principle to the grad matrix, where the params should actually be lists which align to each image taken

        self.bvalues = torch.from_numpy(bvalues.astype(np.float32))
        self.number_of_measurements = torch.tensor(len(self.bvalues.flatten()))
        self.bvecs = torch.from_numpy(bvecs.astype(np.float32))

        gradient_strengths = gradient_strengths if np.size(gradient_strengths) == self.number_of_measurements else np.repeat(gradient_strengths,self.number_of_measurements)
        small_delta = small_delta if np.size(small_delta) == self.number_of_measurements else np.repeat(small_delta,self.number_of_measurements)
        Delta = Delta if np.size(Delta) == self.number_of_measurements else np.repeat(Delta,self.number_of_measurements)
        TE = TE if np.size(TE) == self.number_of_measurements else np.repeat(TE,self.number_of_measurements)
        bdelta = bdelta if np.size(bdelta) == self.number_of_measurements else np.repeat(bdelta,self.number_of_measurements)

        #This chunk of code could be redone to be even shorter+elegantly but leaving it like this incase individual variables need to be changed in the future
        #A better way to do this would probably be a dictionary, but its far too intertwined in everything else to change at this point


        self.gradient_strengths = torch.from_numpy(gradient_strengths.astype(np.float32)) if gradient_strengths is not None else None
        self.small_delta = torch.from_numpy(small_delta.astype(np.float32)) if small_delta is not None else None
        self.Delta = torch.from_numpy(Delta.astype(np.float32)) if Delta is not None else None
        self.TE = torch.from_numpy(TE.astype(np.float32)) if TE is not None else None
        self.bdelta = torch.from_numpy(bdelta.astype(np.float32)) if bdelta is not None else None


        


def acquisition_scheme_loader(filepath_acquisition_scheme):
    r"""
    Creates an acquisition scheme object from bvalues, gradient directions,
    pulse duration $\delta$ and pulse separation time $\Delta$.

    Note: this function as far as i can tell loads an aquisition scheme from a saved file, although theres no where in this code showing how to generate on

    """
    acq_scheme = np.loadtxt(filepath_acquisition_scheme)
    bvalues = np.reshape(acq_scheme[:,3], (1, len(acq_scheme[:,3])))

    if max(bvalues[0,:]) >100:
        bvalues = bvalues/1000

    if np.any(bvalues < 0):
        raise ValueError("bvals contains negative values")
    
    bvecs = acq_scheme[:,0:3]

    try:
        Delta = acq_scheme[:,4]

    except:
        Delta = None
        
    try:
        delta = acq_scheme[:,5]
    except:
        delta = None

    try:
        gradient_strengths = acq_scheme[:,6]
    except:
        gradient_strengths = None

    try:
        TE = acq_scheme[:,7]
    except:
        TE = None

    try:
        bdelta = acq_scheme[:,8]
    except:
        bdelta = None
    
    #check_acquisition_scheme(bvalues, bvecs, delta, Delta, TE)


    return AcquisitionScheme(bvalues, bvecs,
                             gradient_strengths, Delta, delta, TE, bdelta)


def check_acquisition_scheme(
        b_values, bvecs, delta, Delta, TE):
    "function to check the validity of the input parameters."
    if b_values.ndim > 1:
        msg = "b/q/G input must be a one-dimensional array. "
        msg += "Currently its dimensions is {}.".format(
            b_values.ndim
        )
        raise ValueError(msg)
    if len(b_values) != len(bvecs):
        msg = "b/q/G input and gradient_directions must have the same length. "
        msg += "Currently their lengths are {} and {}.".format(
            len(b_values), len(bvecs)
        )
        raise ValueError(msg)
    if delta is not None:
        if len(b_values) != len(delta):
            msg = "b/q/G input and delta must have the same length. "
            msg += "Currently their lengths are {} and {}.".format(
                len(b_values), len(delta)
            )
            raise ValueError(msg)
        if delta.ndim > 1:
            msg = "delta must be one-dimensional array. "
            msg += "Currently its dimension is {}".format(
                delta.ndim
            )
            raise ValueError(msg)
        if np.min(delta) < 0:
            msg = "delta must be zero or positive. "
            msg += "Currently its minimum value is {}.".format(
                np.min(delta)
            )
            raise ValueError(msg)
    if Delta is not None:
        if len(b_values) != len(Delta):
            msg = "b/q/G input and Delta must have the same length. "
            msg += "Currently their lengths are {} and {}.".format(
                len(b_values), len(Delta)
            )
            raise ValueError(msg)
        if Delta.ndim > 1:
            msg = "Delta must be one-dimensional array. "
            msg += "Currently its dimension is {}.".format(
                Delta.ndim
            )
            raise ValueError(msg)
        if np.min(Delta) < 0:
            msg = "Delta must be zero or positive. "
            msg += "Currently its minimum value is {}.".format(
                np.min(Delta)
            )
            raise ValueError(msg)

    if bvecs.ndim != 2 or bvecs.shape[1] != 3:
        msg = "gradient_directions n must be two dimensional array of shape "
        msg += "[N, 3]. Currently its shape is {}.".format(
            bvecs.shape)
        raise ValueError(msg)
    if np.min(b_values) < 0.:
        msg = "b/q/G input must be zero or positive. "
        msg += "Minimum value found is {}.".format(b_values.min())
        raise ValueError(msg)
    gradient_norms = np.linalg.norm(bvecs, axis=1)
    zero_norms = gradient_norms == 0.
    if not np.all(abs(gradient_norms[~zero_norms] - 1.) < 0.001):
        msg = "gradient orientations n are not unit vectors. "
        raise ValueError(msg)
    if TE is not None and len(TE) != len(b_values):
        msg = "If given, TE must be same length b/q/G input."
        msg += "Currently their lengths are {} and {}.".format(
            len(TE), len(bvecs)
        )


def as_auto_loader(bvals, bvecs, delta, small_delta,TE,bdelta): ##Aquisition Scheme Auto Loader -> refined version of txt loader

    if type(bvals) is str:
        bvals = load_grad(bvals)
        bvals = np.transpose(bvals)

    else:
        TypeError("Bvals Should be a path")

    if type(bvecs) is str:
        bvecs = load_grad(bvecs)
        bvecs = np.transpose(bvecs)
    else:
        TypeError("Bvals Should be a path")#

    if type(delta) is str:
        delta = load_grad(delta)
        delta = np.transpose(delta)
    elif type(delta) is int or type(delta) is float:
        delta = np.array(delta)

    if type(small_delta) is str:
        small_delta = load_grad(small_delta)
        small_delta = np.transpose(small_delta)
    elif type(small_delta) is int or type(small_delta) is float:
        small_delta = np.array(small_delta)

    if type(TE) is str:
        TE = load_grad(TE)
        TE = np.transpose(TE)
    elif type(TE) is int or type(TE) is float:
        TE = np.array(TE)

    if type(bdelta) is str:
        bdelta = load_grad(bdelta)
        bdelta = np.transpose(bdelta)
    elif type(bdelta) is int or type(bdelta) is float:
        bdelta = np.array(bdelta)


    if np.any(bvals < 0):
        raise ValueError("bvals contains negative values")

    if max(bvals[0,:]) >100:
        bvals = bvals/1000 # some reduction function

    gradient_strengths = None  # for now just name this none, can be input or calcualted with deltas

    return AcquisitionScheme(bvals, bvecs,
                             gradient_strengths,
                             small_delta,
                             delta,
                             TE,
                             bdelta
                             )

def txt_file_loader(bvals, bvecs, Delta, delta,TE,bdelta): ##Deprecated

    bvals = load_grad(bvals)
    bvals = np.transpose(bvals)
    bvecs = load_grad(bvecs)
    bvecs = np.transpose(bvecs)
    Delta = load_grad(Delta)
    Delta = np.transpose(Delta)
    smalldelta = load_grad(delta)
    smalldelta = np.transpose(smalldelta)
    TE = load_grad(TE)
    TE = np.transpose(TE)
    bdelta = load_grad(bdelta)
    bdelta = np.transpose(bdelta)
    gradient_strengths = None #for now just name this none, can be input or calcualted with deltas

    if np.any(bvals < 0):
        raise ValueError("bvals contains negative values")

    if max(bvals[0,:]) >100:
        bvals = bvals/1000

    '''
    if TE:
        grad = np.concatenate((bvecs,bvals[:,None],delta,smalldel,G,TE),axis=1)
    if TR and TI:
        grad = np.concatenate((bvecs,bvals[:,None],delta,smalldel,G,TE=None,TR,TI),axis=1)
    if TE and TR and TI:
        grad = np.concatenate((bvecs,bvals[:,None],delta,smalldel,G,TE,TR,TI),axis=1)
    '''

    '''
    if TE:
        grad = np.concatenate((bvecs,bvals[:,None],delta,smalldel,G,TE),axis=1)
    if TR and TI:
        grad = np.concatenate((bvecs,bvals[:,None],delta,smalldel,G,TE=None,TR,TI),axis=1)
    if TE and TR and TI:
        grad = np.concatenate((bvecs,bvals[:,None],delta,smalldel,G,TE,TR,TI),axis=1)
    '''

    return AcquisitionScheme(bvals, bvecs,
                             gradient_strengths, smalldelta, Delta, TE, bdelta
                             )

