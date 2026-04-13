import os
import sys
import logging
import yaml
from pathlib import Path
import optuna
from typing import Any, Dict
from optuna.trial import Trial, TrialState
from omegaconf import DictConfig
import torch
from hydra.core.hydra_config import HydraConfig
from microtorch.net_maker import Net
from microtorch.train import train
import torch.nn as nn

from microtorch.utils.acquisition_scheme import AcquisitionScheme
from microtorch.model_maker import ModelMaker

def sample_hyperparams(trial: Trial, tuning_cfg: DictConfig) -> Dict[str, Any]:
    """Dynamically sample hyperparams from tuning config."""
    params = {}
    for name, spec in tuning_cfg.items():
        if spec.type == "int":
            params[name] = trial.suggest_int(name, spec.low, spec.high)
        elif spec.type == "float":
            params[name] = trial.suggest_float(
                name, spec.low, spec.high, log=spec.get("log", False)
            )
        elif spec.type == "categorical":
            params[name] = trial.suggest_categorical(name, list(spec.choices))
    return params

def run_hyperparams_tuning(
    grad: AcquisitionScheme,
    modelfunc: ModelMaker,
    mlp_activation: Dict[str, nn.Module],
    X_train: torch.Tensor,
    cfg: DictConfig,
) -> Dict[str, Any]:
    """
    Hyperparameter tuning routine using Optuna.

    Args:
        grad:           Acquisition scheme object (provides number_of_measurements).
        modelfunc:      Model function object.
        mlp_activation: Dict mapping activation name -> nn.Module.
        X_train:        Training data tensor.
        cfg:            Hydra DictConfig (used for fixed settings not being tuned).
        optuna_m:       Sampler strategy: 'tpe' or 'grid_search'.
        n_trials:       Number of Optuna trials.

    Returns:
        dict: Best hyperparameters found.
    """
    lossfunc = nn.MSELoss()

    def objective(trial: Trial) -> float:
        # ---- Sample hyperparameters ----
        hp = sample_hyperparams(trial, cfg.tuning)

        print(f"\nTrial {trial.number}: layers={hp['num_layers']},patience={hp['patience']}, dropout={hp['dropout_frac']:.3f}, "
              f"lr={hp['lr']:.2e}, activation={hp['activation']}, hidden_size={hp['hidden_size']}")

        # ---- Build network with sampled hyperparams ----
        net = Net(
            grad,
            modelfunc,
            input_neurons=grad.number_of_measurements,
            layer_dims=hp['hidden_size'],
            n_layers=hp['num_layers'],
            dropout_fraction=hp['dropout_frac'],
            clipping_method=cfg.training.clip,
            activation=mlp_activation[hp['activation']],
        )

        # ---- Train and get validation loss ----
        _, _, best_loss = train(
            net,
            X_train,
            lossfunc,
            lr=hp['lr'],
            batch_size=256,
            num_iters=cfg.training.num_iters,
            patience=hp['patience'],
            trial=trial,           # passed through so train() can call trial.report / trial.should_prune
        )

        return float(best_loss)


    sampler = optuna.samplers.TPESampler()

    # ---- Study setup ----
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    output_folder = Path(HydraConfig.get().run.dir)
    db_path       = output_folder / "db.sqlite3"
    storage_name  = f"sqlite:///{db_path}"
    study_name    = f"{cfg.model.name}"

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        storage=storage_name,
        study_name=study_name,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100),
    )

    study.optimize(objective, n_trials=cfg.training.n_trials)

    # ---- Report results ----
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])
    print("\nStudy statistics:")
    print(f"  Finished trials : {len(study.trials)}")
    print(f"  Complete trials : {len(complete_trials)}")

    best = study.best_trial
    print(f"\nBest trial (#{best.number}):")
    print(f"  Value : {best.value}")
    print("  Params:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")

    return dict(best.params)


def get_model_hyperparams(
    grad: AcquisitionScheme,
    modelfunc: ModelMaker,
    mlp_activation: Dict[str, nn.Module],
    X_train: torch.Tensor,
    cfg: DictConfig,
) -> Dict[str, Any]:
    """
    Entry point for hyperparameter resolution.

    Depending on cfg.training.tune:
      - 'optuna_tuner' : run Optuna search and save best params to YAML.
      - 'load_tuned'   : load previously saved params from YAML.
      - anything else  : fall back to values already in cfg.training.

    Args:
        grad, modelfunc, mlp_activation, X_train, cfg: forwarded to tuner.
    Returns:
        dict: Resolved hyperparameters.
    """
    output_folder = Path(HydraConfig.get().run.dir)
    output_folder.mkdir(parents=True, exist_ok=True)
    training_config_folder = Path(HydraConfig.get().runtime.config_sources[1].path) / "training"
    hyperparams_path = training_config_folder / f"{cfg.model.name}_best_hyperparams.yaml"

    method = cfg.training.tune

    #if loading tuned hyperparams but file doesn't exist, fall back to default (prevents crashes if user forgets to run tuner first)
    if method == "load_tuned" and not hyperparams_path.exists():
        method = "default"
        print("-" * 45)
        print(f"Warning: No tuned hyperparameters found at {hyperparams_path}.")    
        print("Falling back to default hyperparameters from config. "    \
              "To run hyperparameter tuning and save results, set cfg.training.tune=optuna_tuner")
        print("-" * 45)


    if method == "optuna_tuner":
        print("-" * 45)
        print("Starting Optuna hyperparameter search …")
        print("-" * 45)
        best_hyperparams = run_hyperparams_tuning(
            grad=grad,
            modelfunc=modelfunc,
            mlp_activation=mlp_activation,
            X_train=X_train,
            cfg=cfg
        )
        with open(hyperparams_path, "w") as f:
            yaml.safe_dump(best_hyperparams, f)
        print(f"Saved best hyperparameters → {hyperparams_path}")
        return best_hyperparams

    elif method == "load_tuned":
        print("-" * 45)
        print(f"Loading previously tuned hyperparameters for {cfg.model.name} model…")
        print("-" * 45)
        with open(hyperparams_path) as f:
            hyperparams = yaml.safe_load(f)
        print(f"Loaded hyperparameters from {hyperparams_path}")
        return hyperparams

    elif method == "default":
        # Fall back to config values — no tuning
        print("-" * 45)
        print("Warning: Using default hyperparameters from config (no tuning). " \
        "These hyperparamters have not been optimized for the " 
        + cfg.model.name + " model and may lead to suboptimal results.")
        print("To run hyperparameter tuning and save results, set cfg.training.tune=optuna_tuner")
        print("-" * 45)
        return {
            "num_layers":   cfg.training.num_layers,
            "hidden_size": cfg.training.layer_size,
            "dropout_frac": cfg.training.dropout_frac,
            "patience": cfg.training.patience,
            "lr":           cfg.training.learning_rate,
            "activation":   cfg.training.activation,
        }
    else:
        raise ValueError(f"Invalid tuning method: {method}. Must be one of 'optuna_tuner', 'load_tuned', or 'default'.")
