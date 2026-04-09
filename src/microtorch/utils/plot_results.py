import nibabel as nib
import matplotlib.pyplot as plt

from microtorch.model_maker import ModelMaker

def plot_param_maps(nifti_file: str, modelfunc: ModelMaker, zslice: int = 0) -> None:
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


import os
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from microtorch.model_maker import ModelMaker


def _get_param_indices_by_compartment(modelfunc, n_maps):
    return [
        [p for p in range(n_maps) if modelfunc.compartment_indices[p] == c]
        for c in range(modelfunc.n_compartments)
    ]


def _set_identity_line_and_limits(axis, p, n_maps, modelfunc):
    is_non_fraction = p < (n_maps - modelfunc.n_fractions) or modelfunc.n_fractions == 1

    if is_non_fraction:
        pr = modelfunc.parameter_ranges[p]
        axis.plot(pr, pr, "k--")
        axis.set_xlim(pr)
        axis.set_ylim(pr)
    elif modelfunc.n_fractions > 1:
        axis.plot([0, 1], [0, 1], "k--")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)


def _format_axis(axis, modelfunc, p):
    comp_idx = modelfunc.compartment_indices[p]
    axis.set_title(
        f"{modelfunc.parameter_names[p]} "
        f"({modelfunc.compartment_names[comp_idx]})"
    )
    axis.set_xlabel("Ground Truth")
    axis.set_ylabel("Fitted")
    axis.ticklabel_format(useOffset=False)
    axis.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axis.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def plot_fitted_vs_gt_for_model(gt, fit, model, save_path=None, title=None, show=True):
    modelfunc = ModelMaker(model)

    n_maps = gt.shape[1]
    param_indices_by_compartment = _get_param_indices_by_compartment(modelfunc, n_maps)

    n_compartments = modelfunc.n_compartments
    max_params = max((len(p_idxs) for p_idxs in param_indices_by_compartment), default=1)

    fig, ax = plt.subplots(
        n_compartments,
        max_params,
        figsize=(3 * max_params, 3 * n_compartments),
        squeeze=False,
    )

    for c, p_idxs in enumerate(param_indices_by_compartment):
        for j in range(max_params):
            axis = ax[c, j]

            if j >= len(p_idxs):
                axis.axis("off")
                continue

            p = p_idxs[j]
          
            axis.plot(gt[:, p], fit[:, p], "o", markersize=0.1)
            _format_axis(axis, modelfunc, p)
            _set_identity_line_and_limits(axis, p, n_maps, modelfunc)

    if title:
        fig.suptitle(title, fontsize=16)
    else:
        fig.suptitle(f"Fitted vs Ground Truth Parameters for {model} model", fontsize=16)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_fitted_vs_gt(gt,fit, simulation_data_models, save_dir=None, show=True):
    results = {}

    for model in simulation_data_models:
        save_path = None
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{model}_fitted_vs_gt.png")

        print(model)
        
        fig, ax = plot_fitted_vs_gt_for_model(
            gt=gt,
            fit=fit,
            model=model,
            save_path=save_path,
            show=show,
        )
        results[model] = (fig, ax)

    return results