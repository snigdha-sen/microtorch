# microTorch: microstructure model fitting with PyTorch


<img align="left" width="250" height="250" src="microtorch/files/logo.jpeg" > 
The microTorch software package is designed to flexibly fit diffusion MRI microstructure models, using a self-supervised deep learning approach. 

This work is by members of the UCL Centre for Medical Image Computing and the Cardiff University Brain Research Imaging Centre. Please contact snigdha.sen.20@ucl.ac.uk with any questions.  

&nbsp;  
&nbsp;  
&nbsp;  
&nbsp;  
&nbsp;  
&nbsp;  
&nbsp;  

<img align="left" src="microtorch/files/torch.png" alt="icon" width="45" height="45">

## Installation


### Without virtual enviroment (not recommended)

Run the following in bash

```bash
git clone https://github.com/snigdha-sen/microtorch.git

pip install torch numpy nibabel tqdm scipy matplotlib torchmetrics

```
And the code might work!

### With virtual environment (recommended)

Clone the directory

```bash
git clone https://github.com/snigdha-sen/microtorch.git
```

Create a virtual environment in the cloned directory

```bash
python -m venv <MICROTORCH_DIR>/microtorch_env
```
where ```<MICROTORCH_DIR>``` is the location where you cloned the repo.

Activate the virtual environment

```bash
files <MICROTORCH_DIR>/microtorch_env/bin/activate
```

Then run 

```bash
pip install torch numpy nibabel tqdm scipy matplotlib torchmetrics

```

### Without virtual enviroment (not recommended)

Run the following in bash

```bash
git clone https://github.com/snigdha-sen/microtorch.git

pip install torch numpy nibabel tqdm scipy matplotlib torchmetrics

```

&nbsp;   

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

## Run from command line

The command line input takes in a number of parameters that allow personalisation of file paths, model type and network parameters. The package makes use of hydra for configuration management - defaults are set in src/conf and can be changed by the user or via the command line.

**Model type**
```
python3 -m src.main model.name=
```

Add the name of the model you wish to use after the equals - either a widely-used model such as VERDICT, SANDI or compartments in Pascal case e.g. BallStick or BallSphere.

**Folders and paths**

Add the paths to the image (and mask, if using) after the equals.
```
data.image= data.mask=
```
In the same way, set the path  to the gradient file, containing all the acquisition parameters...
```
acquisition.grad=
```
... or specify individual file paths for each acquisition parameter (b/delta/Delta/TE/TR/TI/bdelta)
```
acquisition.bvals= acquisition.bvecs= acquisition.delta= acquisition.smalldelta= acquisition.TE= acquisition.TR= acquisition.TI= acquisition.bdelta= 
```
**Network parameters**

To set the network training parameters, the same approach applies

```
training.num_iters= training.learning_rate= training.activation= training.seed= training.dropout_frac= training.layer_size= training.num_layers= training.clip= training.operation= 


<img align="left" src="microtorch/files/torch.png" alt="icon" width="45" height="45">


```

### Test image

We have provided some test images to allow you to test if you have correctly set up all the dependencies:
```
python fit.py -m BallStick -img data/test_images/BallStick.nii.gz  -grad data/grad_files/grad_HCP.txt -a relu -lr 0.0001 -ni 20
```

<img align="left" src="microtorch/files/torch.png" alt="icon" width="45" height="45">

## Choosing a model

To create a model comprising a single compartment, set 
```-m <compartment_name>```. There are a number of typical compartments included:
- Ball
- Stick
- Sphere
- Astrosticks (option to fix D)
- Zeppelin
- Standard WM
=======
- MSDKI
- Astrosticks (option to fix D)
- Zeppelin
- Standard WM
- T1 SMDT
- Cylinder

To create a model comprising **multiple** compartments, set ```-m <compartment_name1 compartment_name2...>``` 
- e.g. ```-m BallBallSphere``` will result in a model comprising of 2 Balls and 1 Sphere.
- Note: Compartment names must start with an uppercase letter and be followed by lowercase. There must be no spaces between compartments. 

There are also a number of predefined models to be used as ```-m <model_name>```:
- VERDICT (Ball, Sphere, Astrosticks - fixed)
- SANDI (Ball, Zeppelin, Astrosticks)
- IVIM (Ball, Ball)


<img align="left" src="microtorch/files/torch.png" alt="icon" width="45" height="45">

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
