## Adding a New Model

You can define a new model (i.e. a combination of compartments with specified parameter ranges and settings) using **YAML configuration files** located in the `src/models/` directory. Each model should have its own YAML file named after the model (e.g. `IVIM.yaml`, `VERDICT.yaml`).

A YAML file is **not strictly required** to run a model that can be inferred directly from its compartment names. For example, a model named `BallBall` can be run without a configuration file. However, using a YAML file allows you to define model-specific settings such as parameter ranges or fixed parameters. For instance, the `IVIM.yaml` file enforces the condition **D\* > D** required by the IVIM model.

### Basic Structure

A model file defines a list of **compartments**. Each compartment specifies the signal model class and optional settings such as parameter ranges or fixed parameters.

Example (`VERDICT.yaml`):

```yaml
compartments:
  - class: Ball

  - class: Sphere
    init_kwargs:
      fixed_D: 2.0
      
  - class: Astrosticks
    init_kwargs:
      fixed_D_par: 8.0
```

Example (`IVIM.yaml`):

```yaml
compartments:
  - class: Ball
    parameter_ranges:
      - [1.0e-03, 3.0]
      
  - class: Ball
    parameter_ranges:
      - [3.0, 30.0]
```