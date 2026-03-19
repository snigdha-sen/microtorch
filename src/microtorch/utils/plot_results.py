import nibabel as nib
import matplotlib.pyplot as plt

def plot_param_maps(nifti_file, modelfunc, zslice=0):
    """
    Plots the parameter maps from a NIfTI file.
    
    Args:
        nifti_file (str): Path to the NIfTI file containing the parameter maps.
        modelfunc (ModelFunction): The model function object containing parameter and compartment information.
        zslice (int): The z-slice index to plot (default is 0).
    """

    img = nib.load(nifti_file).get_fdata()

    n_maps = img.shape[-1]
    fig, ax = plt.subplots(1, n_maps, figsize=(5 * n_maps, 2))
    if n_maps == 1:
        ax = [ax]  # make iterable

    for i in range(n_maps):
        im = ax[i].imshow(img[:, :, zslice, i])
        plt.colorbar(im, ax=ax[i])

        if i < modelfunc.n_parameters:
            cp_idx = next(
                (j for j, slc in enumerate(modelfunc.parameter_slices)
                 if slc.start <= i < slc.stop),
                None
            )
            cp_label = modelfunc.compartment_names[cp_idx] if cp_idx is not None else "UnknownCompartment"
        else:
            # It's a fraction parameter
            frac_idx = i - modelfunc.n_parameters
            cp_label = f"fraction f_{frac_idx}"

        ax[i].set_title(f"{modelfunc.parameter_names[i]} ({cp_label})")

    plt.tight_layout()
    plt.show()

