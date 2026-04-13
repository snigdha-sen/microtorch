# Training Parameters

microTorch also allows configuration of network training parameters, as well as hyperparameter tuning via optuna ADD

## Common Parameters

``` bash
training.num_iters=1000
training.learning_rate=1e-3
training.activation=relu
training.seed=42
training.dropout_frac=0.1
training.layer_size=128
training.num_layers=4
training.clip=1.0
training.operation=fit
```

# Optuna Tuning

## Overview

Optuna is an automatic hyperparameter optimization framework. By selecting the optuna_tuner option in the conf/training/default.yaml file, you can enable Optuna-based tuning. This allows the model to search for the best hyperparameters, optimizing them during the training process. If the load_tuned option is selected, the best hyperparameters found during the search will be used for the final model fitting.

## Configuration
### Tuning Options

In the conf/training/default.yaml file, the tune option determines how the tuning process works:

- optuna_tuner: This option launches the Optuna tuner.
- load_tuned: Loads the best hyperparameters from the tuning process for final model fitting. These values will be stored in the training/*model_name*_best_hyperparameters.yaml file.
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

For continuous values, you can specify a lower and upper bound. For example:

##### learning_rate:

- type: float
- low: 1e-6
- high: 1e-2
  
#### Discrete Hyperparameters

For discrete values, such as the number of hidden layers or the size of hidden layers, you can define possible values using the choices option. For example:

##### hidden_size:

- type: categorical
- choices: [64, 128, 256, 512]

Example Configuration

Here is an example of a tuning/default.yaml file configuration:

### Hyperparameter search space

#### learning_rate:
  type: float
  
  low: 1e-6
  
  high: 1e-2

#### dropout_fraction:
  type: float
  
  low: 0.1
  
  high: 0.5

#### hidden_size:
  type: categorical
  
  choices: [64, 128, 256]

#### activation_function:
  type: categorical
  
  choices: ['relu', 'tanh', 'sigmoid']


## Summary
To enable Optuna tuning, set tune: optuna_tuner in the conf/training/default.yaml file.
The best hyperparameters will be stored in a *_best_hyperparameters.yaml file.
Use at least 40-50 trials to ensure efficient exploration of the search space.
Define the hyperparameter search space in tuning/default.yaml with appropriate types (float, integer, or categorical).
