# Training Parameters

microTorch also allows configuration of network training parameters, as well as hyperparameter tuning via Optuna.

## Common Parameters

``` bash
training.num_iters=1000
training.learning_rate=1e-3
training.activation=relu
training.seed=42
training.dropout_frac=0.1
training.layer_size=128
training.num_layers=4
training.clip=clamp
```

# Optuna Tuning

## Overview

Optuna is an automatic hyperparameter optimization framework. By selecting the optuna_tuner option in the conf/training/default.yaml file, you can enable Optuna-based tuning. This allows the model to search for the best hyperparameters, optimizing them during the training process. If the load_tuned option is selected, the best hyperparameters found during the search will be used for the final model fitting.

## Configuration
### Tuning Options

In the conf/training/default.yaml file, the tune option determines how the tuning process works:

- optuna_tuner: This option launches the Optuna tuner.
- load_tuned: Loads the best hyperparameters from the tuning process for final model fitting. These values are stored in a `training/*model_name*_best_hyperparams.yaml` file.
- default: If selected, all hyperparameters will be taken from the training/default.yaml file without any tuning.

The hyperparameters that are optimized by Optuna include:

- Activation function
- Dropout fraction
- Hidden size
- Learning rate
- Number of hidden layers
- Patience

Other hyperparameters that are not optimized will use the values defined in the training/default.yaml file.

### Trials

Optuna relies on a number of trials to explore the hyperparameter search space. The number of trials can be set under the n_trials option in the training/default.yaml file. It is recommended to use at least 40-50 trials to efficiently explore the hyperparameter space.

### Hyperparameter Search Space

The hyperparameter search space is defined in the tuning/default.yaml file. Each hyperparameter can have different types:

- Integer: Defined by specifying a range of integer values.
- Float: Defined by specifying a continuous range with lower and upper bounds.
- Categorical: A set of discrete choices.

#### Continuous Hyperparameters

For continuous values, you specify a lower and upper bound. For example:

```yaml
lr:
  type: float
  low: 1.0e-4
  high: 5.0e-3
  log: true
```

#### Discrete Hyperparameters

For discrete values, such as the size of hidden layers, you define possible values using the choices option. For example:

```yaml
hidden_size:
  type: categorical
  choices: [32, 64, 128, 256, 512]
```

### Example Configuration

Here is the full `tuning/default.yaml` file shipped with microTorch:

```yaml
num_layers:
  type: int
  low: 1
  high: 6

hidden_size:
  type: categorical
  choices: [32, 64, 128, 256, 512]

patience:
  type: categorical
  choices: [50, 100, 200]

dropout_frac:
  type: float
  low: 0.0
  high: 0.2

lr:
  type: float
  low: 1.0e-4
  high: 5.0e-3
  log: true

activation:
  type: categorical
  choices: ["relu", "prelu", "tanh", "elu"]
```

## Summary
To enable Optuna tuning, set tune: optuna_tuner in the conf/training/default.yaml file.
The best hyperparameters will be stored in a `*_best_hyperparams.yaml` file.
Use at least 40-50 trials to ensure efficient exploration of the search space.
Define the hyperparameter search space in tuning/default.yaml with appropriate types (float, integer, or categorical).
