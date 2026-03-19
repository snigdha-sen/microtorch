# Models and Compartments

### 1. Single-Compartment Models

To use a single compartment:

``` bash
python -m src.main model.name=Ball
```

Available compartments include:

-   `Ball`
-   `Stick`
-   `Sphere`
-   `Astrosticks` (option to fix diffusivity)
-   `Zeppelin`
-   `StandardWM`
-   `Cylinder`

### 2. Multi-Compartment Models

You can combine compartments by concatenating their names in
**PascalCase**, with no spaces:

``` bash
python -m src.main model.name=BallBallSphere
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
python -m src.main model.name=VERDICT
```

Available predefined models:

-   `VERDICT` → Ball + Sphere + fixed Astrosticks
-   `SANDI` → Ball + Zeppelin + Astrosticks
-   `IVIM` → Ball + Ball