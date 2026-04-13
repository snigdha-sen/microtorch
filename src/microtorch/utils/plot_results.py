from typing import Optional, Tuple, List, Dict, Union
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

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


def _get_param_indices_by_compartment(modelfunc: ModelMaker, n_maps: int) -> List[List[int]]:
    """
    Groups parameter indices by their corresponding compartments based on the model function's compartment indices.
    Args:
        modelfunc (ModelMaker): The model function object containing compartment information.
        n_maps (int): The total number of parameter maps.
    Returns:
        List[List[int]]: A list of lists, where each inner list contains the parameter indices corresponding to a specific compartment. The outer list is ordered by compartment index.
    """
    return [
        [p for p in range(n_maps) if modelfunc.compartment_indices[p] == c]
        for c in range(modelfunc.n_compartments)
    ]


def _set_identity_line_and_limits(axis: plt.Axes, p: int, n_maps: int, modelfunc: ModelMaker) -> None:
    """
    Sets the identity line and axis limits for a given parameter index, based on whether it's a non-fraction parameter or a fraction parameter.
    Args:
        axis (plt.Axes): The matplotlib axis to modify.
        p (int): The parameter index.
        n_maps (int): The total number of parameter maps.
        modelfunc (ModelMaker): The model function object containing parameter and fraction information.
    """

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


def _format_axis(axis: plt.Axes, modelfunc: ModelMaker, p: int) -> None:
    """
    Formats the axis title, labels, and tick formatting for a given parameter index.
    Args:
        axis (plt.Axes): The matplotlib axis to format.
        modelfunc (ModelMaker): The model function object containing parameter and compartment information.
        p (int): The parameter index.
    """
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


def plot_fitted_vs_gt_for_model(
    gt: np.ndarray,
    fit: np.ndarray,
    model: str,
    save_path: Optional[str] = None,
    title: Optional[str] = None,
    show: bool = True,
) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plots fitted parameters vs ground truth parameters for a specific model, grouping parameters by their corresponding compartments and formatting the plots accordingly.
    Args:
        gt (np.ndarray): Ground truth parameter values.
        fit (np.ndarray): Fitted parameter values.
        model (str): The name of the model to determine compartment grouping and parameter formatting.
        save_path (Optional[str]): If provided, the path to save the resulting figure.
        title (Optional[str]): Optional title for the figure. If not provided, a default title based on the model name will be used.
        show (bool): Whether to display the figure after plotting. If False, the figure will be closed after saving (if save_path is provided) and not shown.
    Returns:
        Tuple[plt.Figure, np.ndarray]: The matplotlib figure and axes array containing the plots for the fitted vs ground truth parameters, organized by compartment and parameter index.
    """
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


def plot_fitted_vs_gt(
    gt: np.ndarray,
    fit: np.ndarray,
    simulation_data_models: List[str],
    save_dir: Optional[str] = None,
    show: bool = True,
) -> Dict[str, Tuple[plt.Figure, np.ndarray]]:
    """
    Plots fitted parameters vs ground truth parameters for multiple models, saving the resulting figures if a save directory is provided.
    Args:
        gt (np.ndarray): Ground truth parameter values.
        fit (np.ndarray): Fitted parameter values.
        simulation_data_models (List[str]): A list of model names corresponding to the columns in gt and fit, used to determine how to group parameters by compartment and format the plots.
        save_dir (Optional[str]): If provided, the directory where the resulting figures will be saved. Each figure will be named "{model}_fitted_vs_gt.png" based on the model name. If None, the figures will not be saved.
        show (bool): Whether to display the figures after plotting. If False, the figures will be closed after saving (if save_dir is provided) and not shown.
    Returns:
        Dict[str, Tuple[plt.Figure, np.ndarray]]: A dictionary mapping each model name to a tuple containing the matplotlib figure and axes array for the fitted vs ground truth parameter plots corresponding
    """
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