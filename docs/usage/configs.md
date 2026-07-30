# Guide to microTorch Configuration Files

microTorch uses **Hydra** for configuration management. Rather than placing all settings in a single file, Hydra combines multiple configuration files into one experiment configuration. This makes experiments easier to organise, reproduce, and modify.

The top-level configuration file, `config.yaml`, specifies which configuration file to use from each of five categories.

Each configuration file defines default values for a particular aspect of an experiment. These defaults can be overridden using command-line arguments.

The top-level configuration file, `config.yaml`, references five categories of configuration files (alongside setting some other defaults):

- **acquisition:** Contains a `default.yaml` file specifying the default acquisition protocol for each microTorch model. Additional configuration files can be added to define acquisition settings for specific datasets or scans. See [Data Formats](data.md) for the acquisition-related command-line overrides.
- **data:** Contains a `default.yaml` file defining the input image and mask. Additional configuration files can be added for specific datasets or experiments. See [Data Formats](data.md).
- **model:** Contains YAML files defining individual models. Most models have their own configuration file (e.g. `IVIM.yaml`, `VERDICT.yaml`), although a YAML file is not required for models whose structure can be inferred directly from their compartment names. See [Models and Compartments](models.md) and [Adding a New Model](../developer/adding_models.md) for more details.
- **training:** Contains YAML files defining training hyperparameters. `default.yaml` provides sensible default settings for training, while model-specific configurations (e.g. `IVIM_best_hyperparams.yaml`) contain hyperparameters optimised for individual models using Optuna. See [Training Parameters](training.md).
- **tuning:** Contains a `default.yaml` file defining the Optuna hyperparameter search space. See [Training Parameters](training.md).

All configuration files are located in `src/microtorch/conf`.

See below for an annotated version of the top level `config.yaml` file.

```yaml
# The order in which configuration files are loaded.
# Settings in later files can override those loaded earlier.
defaults:
  - _self_                 # Load the settings in this file.
  - training: default      # Training hyperparameters.
  - model: default         # Signal model configuration.
  - acquisition: default   # Acquisition protocol and gradient files.
  - data: default          # Input image and mask.
  - tuning: default        # Optuna hyperparameter search space.

hydra:
  run:
    # Directory where outputs from each run are saved.
    # ${now:...} inserts the current date and time.
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}

plot:
  # Whether to generate parameter map figures after fitting.
  enabled: false

  # Slice index to plot for 3D images.
  zslice: 0

paths:
  # Directory containing microTorch signal model implementations.
  models_dir: src/microtorch/signal_models

  # Default location for simulated datasets.
  sim_data_dir: simulation_data/data

  # Directory containing bundled MRI acquisition protocols.
  grad_dir: src/microtorch/resources/protocols
```

## Example: a documented model configuration file

Model configuration files (in `src/microtorch/conf/model/`) define which
compartments make up a model, and optionally override their default
parameter ranges or fix some of their parameters. Here is the full,
annotated `IVIM.yaml`:

```yaml
name: IVIM

compartments:
  # Tissue diffusion compartment (D).
  - class: Ball
    # Overrides this compartment's default parameter range(s). Must be a
    # list of [min, max] pairs, one per parameter, in the same order as the
    # `Ball` class's `parameter_names` (here just one: D).
    parameter_ranges:
      - [1.0e-03, 3.0]

  # Pseudo-diffusion / perfusion compartment (D*).
  - class: Ball
    parameter_ranges:
      - [3.0, 30.0]
```

This defines the IVIM model as two `Ball` compartments - one constrained to
tissue diffusivities, one to pseudo-diffusion/perfusion diffusivities. At
fit time, microTorch estimates a volume fraction for each compartment and
combines them into the predicted signal. To use it:

```bash
python -m microtorch.main model.name=IVIM data.image=/path/to/dwi.nii.gz acquisition.grad=/path/to/grad.txt
```

See [Adding a New Model](../developer/adding_models.md) for the full set of
options a model configuration file supports (e.g. `init_kwargs` for fixing
a compartment's parameters).
