# Running microTorch from the Command Line

microTorch uses **Hydra** for configuration management. You can override
any parameter directly from the command line using dot-notation.

## Basic Usage

``` bash
python -m microtorch.main key=value
```

You can override multiple parameters:

``` bash
python -m microtorch.main training.num_iters=1000 training.learning_rate=1e-3
```

## Minimal Example

``` bash
python -m microtorch.main \
  model.name=SANDI \
  data.image=/data/dwi.nii \
  acquisition.grad=/data/grad.scheme \
  training.num_iters=2000 \
  training.learning_rate=5e-4
```

## Common Overrides

-   `model.name=...` → choose model\
-   `data.image=...` → input data\
-   `acquisition.grad=...` → acquisition parameters\
-   `training.num_iters=...` → training length

## Useful Information

### zsh shell autocompletion

In zsh, file path autocompletion does not always work after the `=` used in Hydra command-line overrides (e.g. `data.image=...`). To enable path completion in this case, add the following line to your `~/.zshrc` file:

```zsh
setopt magic_equal_subst
```
Then reload your shell

```
source ~/.zshrc
```

## Learn More

-   Models & Compartments → models.md
-   Data & Acquisition → data.md
-   Training Parameters → training.md