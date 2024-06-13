import numpy as np

class acquisitions_scheme:


    def __init__(self, bvalues, bvecs, 
                 gradient_strengths, delta, Delta, TE, bdelta):
        
        self.bvalues = bvalues.astype(float)
        self.number_of_measurements = len(self.bvalues)
        self.bvecs = bvecs.astype(float)

        self.gradient_strengths = None
        if gradient_strengths is not None:
            self.gradient_strengths = gradient_strengths.astype(float)
        self.delta = None
        if delta is not None:
            self.delta = delta.astype(float)
        self.Delta = None
        if Delta is not None:
            self.Delta = Delta.astype(float)
        self.TE = None
        if TE is not None:
            self.TE = TE.astype(float)
        self.bdelta = None
        if bdelta is not None:
            self.bdelta = bdelta.astype(float)


        self.spherical_mean_scheme = AcquisitionScheme(
        self.bvalues,
        self.gradient_strengths,
        self.Delta,
        self.delta,
        self.bdelta)



class AcquisitionScheme:
    "Acquisition scheme for isotropic spherical mean models."

    def __init__(self, bvalues,
                 gradient_strengths, Deltas, deltas, bdeltas):
        self.bvalues = bvalues
        self.gradient_strengths = gradient_strengths
        self.Delta = Deltas
        self.delta = deltas
        self.bdelta = bdeltas
        self.number_of_measurements = len(bvalues)


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
        delta = acq_scheme[:,4]
    except:
        delta = None

    try:
        Delta = acq_scheme[:,5]
    except:
        Delta = None

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
    
    check_acquisition_scheme(
        bvalues, bvecs, delta, Delta, TE)

    return acquisitions_scheme(bvalues, bvecs,
                                  gradient_strengths, delta, Delta, TE, bdelta
                                    )



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


