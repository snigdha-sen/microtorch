import numpy as np
from data.load_data import load_grad
import torch

class acquisitions_scheme:


    def __init__(self, bvalues, bvecs, 
                 gradient_strengths, Delta, delta, TE, bdelta):
        
        self.bvalues = torch.from_numpy(bvalues.astype(np.float32))
        self.bvecs   = torch.from_numpy(bvecs.astype(np.float32))
        self.number_of_measurements = torch.tensor(len(self.bvalues))

        self.gradient_strengths = None
        if gradient_strengths is not None:
            self.gradient_strengths = torch.from_numpy(gradient_strengths.astype(np.float32))
        self.Delta = None
        if Delta is not None:
            self.Delta = torch.from_numpy(Delta.astype(np.float32))
        self.delta = None
        if delta is not None:
            self.delta = torch.from_numpy(delta.astype(np.float32))
        self.TE = None
        if TE is not None:
            self.TE = torch.from_numpy(TE.astype(np.float32))
        self.bdelta = None
        if bdelta is not None:
            self.bdelta = torch.from_numpy(bdelta.astype(np.float32))
        
        
def acquisition_scheme_loader(filepath_acquisition_scheme):
    r"""
    Creates an acquisition scheme object from bvalues, gradient directions,
    pulse duration $\delta$ and pulse separation time $\Delta$.

    """
    acq_scheme = np.loadtxt(filepath_acquisition_scheme)
    bvalues = acq_scheme[:,3]

    if max(bvalues) >100:
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
        gradient_strengths = acq_scheme[:,6] ##not sure yet if this is the right order in which schemes are ordered
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
    
    check_acquisition_scheme(bvalues, bvecs, delta, Delta, TE)

    return acquisitions_scheme(bvalues, bvecs, 
                                gradient_strengths, Delta, delta, TE, bdelta)



def check_acquisition_scheme(b_values, bvecs, delta, Delta, TE):
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


def txt_file_loader(bvals, bvecs, Delta, delta, TE,bdelta):

    bvals  = load_grad(bvals)
    bvecs  = load_grad(bvecs)
    Delta  = load_grad(Delta)
    delta  = load_grad(delta)
    TE     = load_grad(TE)
    bdelta = load_grad(bdelta)
    gradient_strengths = None #for now just name this none, can be input or calcualted with deltas

    if np.any(bvals < 0):
        raise ValueError("bvals contains negative values")

    if max(bvals[0,:]) >100:
        bvals = bvals/1000

    return acquisitions_scheme(bvals, bvecs,
                                  gradient_strengths, Delta, delta, TE, bdelta)