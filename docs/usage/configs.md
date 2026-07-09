# Guide to microTorch Configuration Files

microTorch uses **Hydra** for configuration management. Rather than placing all settings in a single file, Hydra combines multiple configuration files into one experiment configuration. This makes experiments easier to organise, reproduce, and modify.

The top-level configuration file, `config.yaml`, specifies which configuration file to use from each of five categories.

Each configuration file defines default values for a particular aspect of an experiment. These defaults can be overridden using command-line arguments.

The top-level configuration file, `config.yaml`, references five categories of configuration files (alongside setting some other defaults):

- **acquisition:** Contains a `default.yaml` file specifying the default acquisition protocol for each microTorch model. Additional configuration files can be added to define acquisition settings for specific datasets or scans. LINK TO ACQUISITION MD FILE WITH CONFIG.
- **data:** Contains a `default.yaml` file defining the input image and mask. Additional configuration files can be added for specific datasets or experiments. LINK TO DATA MD FILE WITH CONFIG.
- **model:** Contains YAML files defining individual models. Most models have their own configuration file (e.g. `IVIM.yaml`, `VERDICT.yaml`), although a YAML file is not required for models whose structure can be inferred directly from their compartment names. See [Adding a New Model](developer/adding_models.md) for more details.
- **training:** Contains YAML files defining training hyperparameters. `default.yaml` provides sensible default settings for training, while model-specific configurations (e.g. `IVIM_best_hyperparameters.yaml`) contain hyperparameters optimised for individual models using Optuna. LINK TO TRAINING MD FILE WITH CONFIG.
- **tuning:** Contains a `default.yaml` file defining the Optuna hyperparameter search space. LINK TO TUNING MD FILE WITH CONFIG.

All configuration files are located in `src/microtorch/config`.

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
