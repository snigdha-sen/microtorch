import nibabel as nib
import matplotlib.pyplot as plt

def plot_param_maps(nifti_file, modelfunc, zslice=0):
    img = nib.load(nifti_file).get_fdata()

    n_maps = img.shape[-1]
    _, ax = plt.subplots(1, n_maps, figsize=(5 * n_maps, 2))

    for i in range(n_maps):
        im = ax[i].imshow(img[:, :, zslice, i])
        plt.colorbar(im, ax=ax[i])
        ax[i].set_title(
            f"{modelfunc.parameter_names[i]} "
            f"({modelfunc.compartment_names[modelfunc.compartment_indices[i]]})"
        )

    plt.show()
