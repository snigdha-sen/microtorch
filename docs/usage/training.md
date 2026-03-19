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