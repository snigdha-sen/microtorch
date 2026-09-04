# Models and Compartments

### 1. Single-Compartment Models

To use a single compartment:

``` bash
python -m microtorch.main model.name=Ball
```

Available compartments include:

-   `Ball`
-   `Stick`
-   `Sphere`
-   `Astrosticks` (option to fix diffusivity)
-   `Zeppelin`
-   `Cylinder`

### 2. Multi-Compartment Models

You can combine compartments by concatenating their names in
**PascalCase**, with no spaces:

``` bash
python -m microtorch.main model.name=BallBallSphere
```

This example creates a model with: 
- 2 × Ball compartments
- 1 × Sphere compartment

**Important Rules**

-   Compartment names must start with an uppercase letter.
-   No spaces are allowed between compartments.
-   Order determines how compartments are constructed internally.

### 3. Predefined Models

microTorch also includes commonly used multicompartment models:

``` bash
python -m microtorch.main model.name=VERDICT
```

Available predefined models:

-   `VERDICT` → Ball + Sphere + fixed Astrosticks
-   `SANDI` → Ball + Sphere + Astrosticks
-   `IVIM` → Ball + Ball

### 4. Combined diffusion-relaxation models

Any compartment can be extended for combined diffusion-relaxation MRI acquisitions. Currently, two types of relaxation weighting are supported:

- **T2 relaxation:** For multi-echo spin-echo acquisitions, where diffusion-weighted measurements are acquired at multiple echo times (TEs), the compartment signal equation is multiplied by `exp(-TE/T2)`.

- **T1 inversion recovery:** For inversion-recovery acquisitions, where diffusion-weighted measurements are acquired at multiple inversion times (TIs) and repetition times (TRs) following a 180° inversion pulse, the compartment signal equation is multiplied by `|1 - IE exp(-TI/T1) + exp(-TR/T1)|`, where `IE` is the inversion efficiency.

Relaxation weighting is added to a compartment simply by appending `T1` and/or `T2` to its name. For example:

- `BallT2` — Ball compartment with T2 relaxation.
- `BallT1` — Ball compartment with T1 inversion recovery.
- `BallT1T2` — Ball compartment with both T1 inversion recovery and T2 relaxation.

The same convention can be applied to any supported compartment.

### 5. Adding new Models

For information on defining new models and/or compartments see [Adding a New Model](../developer/adding_models.md).

