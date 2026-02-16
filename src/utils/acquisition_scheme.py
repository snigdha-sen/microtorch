import numpy as np
import torch
from src.utils.load_data import load_grad

class AcquisitionScheme:
    def __init__(
        self,
        bvalues,
        bvecs,
        gradient_strengths=None,
        delta=None,
        Delta=None,
        TE=None,
        bdelta=None,
    ):
        """
        Container for acquisition scheme details.

        Args:
            bvalues (array-like): 1D array of b-values
            bvecs (array-like): 2D array of shape (N, 3) containing the gradient directions as unit vectors
            gradient_strengths (array-like, optional): 1D array of gradient strengths in T/m
            delta (array-like, optional): 1D array of gradient pulse durations in seconds
            Delta (array-like, optional): 1D array of diffusion times in seconds
            TE (array-like, optional): 1D array of echo times in seconds
            bdelta (array-like, optional): 1D array of bdelta values (for advanced models)
        """

        self.bvalues = torch.as_tensor(bvalues, dtype=torch.float32)
        self.bvecs = torch.as_tensor(bvecs, dtype=torch.float32)

        self.number_of_measurements = int(self.bvalues.size(-1))

        self.gradient_strengths = _to_tensor_or_none(gradient_strengths)
        self.delta = _to_tensor_or_none(delta)
        self.Delta = _to_tensor_or_none(Delta)
        self.TE = _to_tensor_or_none(TE)
        self.bdelta = _to_tensor_or_none(bdelta)


def _to_tensor_or_none(x):
    """
    Helper function to convert input to a PyTorch tensor or return None if input is None.

    Args:
        x (array-like or None): The input to convert to a tensor, or None.
    """

    if x is None:
        return None
    return torch.as_tensor(x, dtype=torch.float32)


def _process_bvalues(bvals):
    """
    Process b-values to ensure they are in the correct format and units.
    Args:
        bvals (array-like): The input b-values, which may be in various formats (e.g., 1D array, list) and units (e.g., s/mm^2 or s/m^2).
    Returns:
        bvals (numpy.ndarray): The processed b-values as a 1D array in s/mm^2.
    """

    bvals = np.asarray(bvals, dtype=np.float32)

    if np.any(bvals < 0):
        raise ValueError("bvals contains negative values")

    if np.max(bvals) > 100:
        bvals = bvals / 1000.0

    return bvals


def check_acquisition_scheme(bvalues, bvecs, delta=None, Delta=None, TE=None):
    """
    Validates the acquisition scheme parameters.
    Args:
        bvalues (array-like): 1D array of b-values
        bvecs (array-like): 2D array of shape (N, 3) containing the gradient directions as unit vectors
        delta (array-like, optional): 1D array of gradient pulse durations in seconds
        Delta (array-like, optional): 1D array of diffusion times in seconds
        TE (array-like, optional): 1D array of echo times in seconds
    Raises:
        ValueError: If any of the validation checks fail."""
    if bvalues.ndim != 1:
        raise ValueError("bvalues must be one-dimensional")

    if bvecs.ndim != 2 or bvecs.shape[1] != 3:
        raise ValueError("bvecs must have shape (N, 3)")

    if len(bvalues) != len(bvecs):
        raise ValueError("bvalues and bvecs must have the same length")

    if np.any(bvalues < 0):
        raise ValueError("bvalues must be non-negative")

    norms = np.linalg.norm(bvecs, axis=1)
    nonzero = norms > 0
    if not np.allclose(norms[nonzero], 1.0, atol=1e-3):
        raise ValueError("bvecs must be unit vectors")

    for name, arr in [("delta", delta), ("Delta", Delta), ("TE", TE)]:
        if arr is not None:
            if arr.ndim != 1:
                raise ValueError(f"{name} must be one-dimensional")
            if len(arr) != len(bvalues):
                raise ValueError(f"{name} must match bvalues length")
            if np.any(arr < 0):
                raise ValueError(f"{name} must be non-negative")

def acquisition_scheme_loader(filepath):
    """
    Load acquisition scheme from a single text file.
    Expected columns:
        0-2: bvecs
        3:   bvalues
        4+:  optional timing parameters
        
    Args:
        filepath (str): Path to the text file containing the acquisition scheme.
    Returns:
        AcquisitionScheme: An instance of the AcquisitionScheme class containing the loaded acquisition parameters.
    """
    data = np.loadtxt(filepath)

    bvecs = data[:, 0:3]
    bvalues = _process_bvalues(data[:, 3])

    Delta = data[:, 4] if data.shape[1] > 4 else None
    delta = data[:, 5] if data.shape[1] > 5 else None
    gradient_strengths = data[:, 6] if data.shape[1] > 6 else None
    TE = data[:, 7] if data.shape[1] > 7 else None
    bdelta = data[:, 8] if data.shape[1] > 8 else None

    check_acquisition_scheme(bvalues, bvecs, delta, Delta, TE)

    return AcquisitionScheme(
        bvalues=bvalues,
        bvecs=bvecs,
        gradient_strengths=gradient_strengths,
        delta=delta,
        Delta=Delta,
        TE=TE,
        bdelta=bdelta,
    )


def txt_file_loader(bvals, bvecs, Delta=None, delta=None, TE=None, bdelta=None):
    """
    Load acquisition scheme from separate text files.

    Args:
        bvals (str): Path to the text file containing b-values.
        bvecs (str): Path to the text file containing b-vectors.
        Delta (str, optional): Path to the text file containing diffusion times. Default is None.
        delta (str, optional): Path to the text file containing gradient pulse durations. Default is None.
        TE (str, optional): Path to the text file containing echo times. Default is None.
        bdelta (str, optional): Path to the text file containing bdelta values.
    Returns:
        AcquisitionScheme: An instance of the AcquisitionScheme class containing the loaded acquisition parameters.
    """
    bvals = _process_bvalues(load_grad(bvals).T.squeeze())
    bvecs = load_grad(bvecs).T

    Delta = load_grad(Delta).T.squeeze() if Delta else None
    delta = load_grad(delta).T.squeeze() if delta else None
    TE = load_grad(TE).T.squeeze() if TE else None
    bdelta = load_grad(bdelta).T.squeeze() if bdelta else None

    check_acquisition_scheme(bvals, bvecs, delta, Delta, TE)

    return AcquisitionScheme(
        bvalues=bvals,
        bvecs=bvecs,
        gradient_strengths=None,
        delta=delta,
        Delta=Delta,
        TE=TE,
        bdelta=bdelta,
    )

