# microTorch: microstructure model fitting with PyTorch


<img align="left" width="295" height="295" src="files/logo.jpeg" > 

The microTorch software package is designed to flexibly fit diffusion MRI (dMRI) microstructure models, using a self-supervised deep learning approach. The framework is designed to work with a variety of established dMRI multicompartment models, such as Ball-and-Stick, VERDICT and SANDI, but also allows users to combine compartment models as they wish.

We designed this framework to leverage the inference time gains of deep learning, whilst removing the requirement for explicit training data. Training and inference is performed simultaneously, for each dataset at a time, mimicking a traditional model fitting approach and reducing bias in the parameter estimates. Please see [1,2] for the theoretical underpinnings of this approach.

This work is by members of the UCL Centre for Medical Image Computing and the Cardiff University Brain Research Imaging Centre. We encourage contributions from the wider diffusion MRI community, and welcome requests for new features. Please contact snigdha.sen.20@ucl.ac.uk with any questions.  
&nbsp;  

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

## Installation

### 1. Clone the repository

``` bash
git clone https://github.com/snigdha-sen/microtorch.git
cd microtorch
```

------------------------------------------------------------------------

### 2. Create and activate a virtual environment (recommended)

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

------------------------------------------------------------------------

### 3. Install the package

Install MicroTorch and its dependencies (as defined in
`pyproject.toml`):

``` bash
pip install .
```

------------------------------------------------------------------------

## Development Installation (editable mode)

If you plan to modify the code:

``` bash
pip install -e .
```

This installs the package in editable mode so changes to the source code
are immediately reflected.

------------------------------------------------------------------------

## Install directly from GitHub

If you do not want to clone the repository:

``` bash
pip install git+https://github.com/snigdha-sen/microtorch.git
```

------------------------------------------------------------------------

> ⚠️ **Note on PyTorch:**\
> Depending on your CUDA setup, you may need to install a specific
> version of PyTorch.\
> See: https://pytorch.org/get-started/locally/

&nbsp;   

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

## Running from the Command Line

microTorch uses **Hydra** for configuration management.\
Default configuration files are located in `src/conf/`, and any
parameter can be overridden directly from the command line using Hydra's
dot-notation syntax:

``` bash
python -m src.main key=value
```

Multiple parameters can be overridden in a single command.

------------------------------------------------------------------------

## Minimal Example

``` bash
python -m src.main \
  model.name=BallStick \
  data.image=/path/to/dwi.nii \
  acquisition.bvals=/path/to/bvals \
  acquisition.bvecs=/path/to/bvecs
```

------------------------------------------------------------------------

## Model Selection

Specify the model using:

``` bash
model.name=MODEL_NAME
```

Examples:

-   `BallStick`
-   `BallSphere`
-   `VERDICT`
-   `SANDI`

Model names should be written in **PascalCase**.

------------------------------------------------------------------------

## Data and File Paths

### Required

``` bash
data.image=/path/to/dwi.nii
```

Optional mask:

``` bash
data.mask=/path/to/mask.nii
```

------------------------------------------------------------------------

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

------------------------------------------------------------------------

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

------------------------------------------------------------------------

## Example with Custom Training Parameters

``` bash
python -m src.main \
  model.name=SANDI \
  data.image=/data/dwi.nii \
  acquisition.grad=/data/grad.scheme \
  training.num_iters=2000 \
  training.learning_rate=5e-4 \
  training.layer_size=256
```

------------------------------------------------------------------------

For a full list of configurable parameters, see the configuration files
in:

    src/conf/

## Choosing a Model

microTorch allows you to define models either by combining individual
compartments or by selecting a predefined model.

------------------------------------------------------------------------

## 1. Single-Compartment Models

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

------------------------------------------------------------------------

## 2. Multi-Compartment Models

You can combine compartments by concatenating their names in
**PascalCase**, with no spaces:

``` bash
python -m src.main model.name=BallBallSphere
```

This example creates a model with: - 2 × Ball compartments\
- 1 × Sphere compartment

### Important Rules

-   Compartment names must start with an uppercase letter.
-   No spaces are allowed between compartments.
-   Order determines how compartments are constructed internally.

------------------------------------------------------------------------

## 3. Predefined Models

microTorch also includes commonly used multicompartment models:

``` bash
python -m src.main model.name=VERDICT
```

Available predefined models:

-   `VERDICT` → Ball + Sphere + fixed Astrosticks\
-   `SANDI` → Ball + Zeppelin + Astrosticks\
-   `IVIM` → Ball + Ball

These models provide convenient presets for widely used diffusion MRI
frameworks.

------------------------------------------------------------------------

For full configuration options, see:

    src/conf/model/


### Examples and Simulated Test Data

We have provided some test images to allow you to test if you have correctly set up all the dependencies:
```
python fit.py -m BallStick -img data/test_images/BallStick.nii.gz  -grad data/grad_files/grad_HCP.txt -a relu -lr 0.0001 -ni 20
```

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">


## Adding a new compartment

You can add compartments that are not included in microTorch by modifying the ```signal_models.py``` file. Compartments must adhere to the following structure:

```
class <Compartment_name>:
    def __init__(self):
        self.parameter_ranges   = [[min_A, max_A], ...] # Acceptable range of values for each parameter.
        self.param_names        = ['A', ...]            # Name of each parameter.
        self.n_params           = N                     # Total number of parameters. In this case, N=1.
        self.spherical_mean     = True                  # Requires spherical mean (True/False).


    def __call__(self, grad, params):    
        
        # Get necessary acquisition parameters
        # Grad class includes b_values, b_vecs, Delta, delta, gradient_strength, TE and bdelta 
        ac_param = grad.ac_param

        # Get estimated parameters
        # i is the index of the parameter as defined in __init__. In this case, i=0.
        param_A  = params[:, i].unsqueeze(1) 

        # The signal equation must be fully differentiable.
        S = ...

        return S
```
