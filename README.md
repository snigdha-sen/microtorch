# microTorch: microstructure model fitting with PyTorch


<img align="left" width="295" height="295" src="files/logo.jpeg" > 

The ```microTorch``` software package is designed to flexibly fit diffusion MRI (dMRI) microstructure models, using a self-supervised deep learning approach. The framework is designed to work with a variety of established dMRI multicompartment models, such as Ball-and-Stick, VERDICT and SANDI, but also allows users to combine compartment models as they wish.

We designed this framework to leverage the inference time gains of deep learning, whilst removing the requirement for explicit training data. Training and inference is performed simultaneously, for each dataset at a time, mimicking a traditional model fitting approach and reducing bias in the parameter estimates. Please see [1,2] for the theoretical underpinnings of this approach.

This work is by members of the UCL Centre for Medical Image Computing and the Cardiff University Brain Research Imaging Centre. We encourage contributions from the wider diffusion MRI community, and welcome requests for new features. Please contact snigdha.sen.20@ucl.ac.uk with any questions.  
&nbsp;  

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

# Installation

## 1. Clone the repository

``` bash
git clone https://github.com/snigdha-sen/microtorch.git
cd microtorch
```

## 2. Create and activate a virtual environment (recommended)

Create a virtual environment:

``` bash
python -m venv .venv
```

Activate it:

**macOS / Linux**

``` bash
source .venv/bin/activate
```

**Windows**

``` bash
.venv\Scripts\activate
```

Upgrade pip:

``` bash
pip install --upgrade pip
```

## 3. Install the package

Install MicroTorch and its dependencies (as defined in
`pyproject.toml`):

``` bash
pip install .
```

### Development Installation (editable mode)

If you plan to modify the code:

``` bash
pip install -e .
```

This installs the package in editable mode so changes to the source code
are immediately reflected.

### Install directly from GitHub

If you do not want to clone the repository:

``` bash
pip install git+https://github.com/snigdha-sen/microtorch.git
```

> ⚠️ **Note on PyTorch:**\
> Depending on your CUDA setup, you may need to install a specific
> version of PyTorch.\
> See: https://pytorch.org/get-started/locally/

&nbsp;   

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

# Running microTorch from the Command Line

microTorch uses **Hydra** for configuration management.\
Default configuration files are located in `src/conf/`, and any
parameter can be overridden directly from the command line using Hydra's
dot-notation syntax:

``` bash
python -m src.main key=value
```

Multiple parameters can be overridden in a single command.

## Choosing a Model

microTorch allows you to define models either by combining individual
compartments or by selecting a predefined model.

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

This example creates a model with: - 2 × Ball compartments\
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

## Data and File Paths

### Required

``` bash
data.image=/path/to/dwi.nii
```

Optional mask:

``` bash
data.mask=/path/to/mask.nii
```

## Acquisition Parameters

You can either provide a single gradient scheme file:

``` bash
acquisition.grad=/path/to/grad.scheme
```

**OR** specify acquisition parameters individually:

``` bash
acquisition.bvals=/path/to/bvals
acquisition.bvecs=/path/to/bvecs
acquisition.delta=/path/to/delta
acquisition.smalldelta=/path/to/smalldelta
acquisition.TE=/path/to/TE
acquisition.TR=/path/to/TR
acquisition.TI=/path/to/TI
acquisition.bdelta=/path/to/bdelta
```

You only need to provide the parameters required by your selected model.

## Training / Network Parameters

Training parameters can be overridden in the same way:

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


## Minimal Example

``` bash
python -m src.main \
  model.name=SANDI \
  data.image=/data/dwi.nii \
  acquisition.grad=/data/grad.scheme \
  training.num_iters=2000 \
  training.learning_rate=5e-4
```

For a full list of configurable parameters, see:

    src/conf/

    
<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

# Examples and Simulated Test Data

We have provided some test images to allow you to test if you have correctly set up all the dependencies:
```
python fit.py -m BallStick -img data/test_images/BallStick.nii.gz  -grad data/grad_files/grad_HCP.txt -a relu -lr 0.0001 -ni 20
```

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">


# Contributing

We welcome contributions from the diffusion MRI community.

To propose a new feature or improvement:

1.  Fork the repository.
2.  Create a new branch named after your feature
    (e.g. `feature/new-compartment`).
3.  Implement your changes.
4.  Open a Pull Request (PR) to the `main` branch.
5.  A maintainer will review your contribution.

Please ensure your code is well documented and tested before submitting
a PR.

## Adding a New Compartment

New microstructure compartments can be added by extending:

    src/signal_models.py

Each compartment must follow the structure below:

``` python
class CompartmentName:
    def __init__(self):
        self.parameter_ranges = [[min_A, max_A], ...]  # Acceptable range for each parameter
        self.param_names = ['A', ...]                  # Parameter names
        self.n_params = N                              # Number of parameters
        self.spherical_mean = True                     # Whether spherical mean is required

    def __call__(self, grad, params):

        # Acquisition parameters
        # The Grad class provides:
        # b_values, b_vecs, Delta, delta, gradient_strength, TE, bdelta
        ac_param = grad.ac_param

        # Extract parameters (i corresponds to index in param_names)
        param_A = params[:, i].unsqueeze(1)

        # Signal equation (must be fully differentiable)
        S = ...

        return S
```

### Requirements

-   The forward model must be **fully differentiable** (compatible with
    PyTorch autograd).
-   Parameter ordering must match `param_names`.
-   Parameter ranges should reflect physically meaningful bounds.
-   Output shape must match the expected signal shape.

## Adding Tests

All new compartments must include appropriate unit tests.

Tests should be added to:

    microtorch/tests/signal_models/

Please follow the structure and conventions of existing tests. Tests
should verify:

-   Correct parameter handling
-   Numerical stability
-   Expected output shape
-   Basic sanity checks of signal behaviour

We encourage contributors to open an issue first if they would like to
discuss substantial changes before implementation.

# Acknowledgements

## Authors

```microTorch``` is authored by Snigdha Sen, Rajib Ahmed, Gerrit Arends, Alvaro Planchuelo Gomez, Xiaoxiang Chen, Marta Masramon Masramon, Marco Palombo, Chris Parker, Chantal MW Tax, Eleftheria Panagiotaki and Paddy J Slator. 

Snigdha Sen undertook this work whilst being funded by the EPSRC-funded UCL Center for Doctoral Training in Intelligent, Integrated Imaging in Healthcare (i4health) (EP/S021930/1) and the Department of Health's NIHR-funded Biomedical Research Centre at University College London Hospitals.

## References

[1] Sen S, Singh S, Pye H, et al. ssVERDICT: Self-supervised VERDICT-MRI for enhanced prostate tumor characterization. Magn Reson Med. 2024; 92: 2181-2192. doi: 10.1002/mrm.30186
[2] Barbieri S, Gurney-Champion OJ, Klaassen R, Thoeny HC. Deep learning how to fit an intravoxel incoherent motion model to diffusion-weighted MRI. Magn Reson Med. 2020 Jan;83(1):312-321. doi: 10.1002/mrm.27910

