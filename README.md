# microTorch: microstructure model fitting with PyTorch

<div style="display: flex; align-items: center;">
  <img src="logo.jpeg" alt="logo" style="width: 30%; margin-right: 20px;">
  <div>
    <p>The microTorch software package is designed to flexibly fit diffusion MRI microstructure models, using a self-supervised deep learning approach.</p>
    <p>This work is by members of the UCL Centre for Medical Image Computing and the Cardiff University Brain Research Imaging Centre. Please contact snigdha.sen.20@ucl.ac.uk with any questions.</p>
  </div>
</div>

<table>
<tr>
<td><img src="torch.jpeg" alt="icon" width="50" height="50"></td>
<td><h2>Installation</h2></td>
</tr>
</table>

### Without virtual enviroment (not recommended)

Run the following in bash

```bash
git clone https://github.com/snigdha-sen/microtorch.git

pip install torch numpy nibabel tqdm scipy matplotlib

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
source <MICROTORCH_DIR>/microtorch_env/bin/activate
```

Then run 

```bash
pip install torch numpy nibabel tqdm scipy matplotlib

```
And the code should work!


<table>
<tr>
<td><img src="torch.jpeg" alt="icon" width="50" height="50"></td>
<td><h2>Command line examples</h2></td>
</tr>
</table>

The command line input takes in a number of parameters that allow personalisation of file paths, model type and network parameters. 

**Model type**
```
python3 fit.py -m <model name>
```

**Folders and paths**

Set the right paths to the input images
```
python3 fit.py -f <data folder> -img <image file name> -ma <mask file name> 
```
Set the path to the gradient file, containing all the acquisition parameters...
```
python3 fit.py -grad <grad file path>
```
... or specify individual file paths for each acquisition parameter
```
python3 fit.py -bvals <bvals path> -bvecs <bvecs path> -d <Deltas path> -sd <deltas path> -TE <TEs path> -bd <bdelta path>
```
**Network parameters**
```
python3 fit.py -lr <learning rate> -se <seed> -lss <layer size> -nl <number of layers> -a <activation function> -df <dropout fraction> -ni <number of iterations>

```

### Test image

We have provided some test images to allow you to test if you have correctly set up all the dependencies:
```
python fit.py -m BallStick -img BallStick.nii.gz -ma mask.nii.gz -bvals data/grad_files/bvals -bvecs data/grad_files/bvecs -se 123 -a relu -lr 0.0001 -ni 20
```

<table>
<tr>
<td><img src="torch.jpeg" alt="icon" width="50" height="50"></td>
<td><h2>Choosing a model</h2></td>
</tr>
</table>

To create a model comprising a single compartment, set 
```-m <compartment_name>```. There are a number of typical compartments included:
- Ball
- Stick
- Sphere
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

<table>
<tr>
<td><img src="torch.jpeg" alt="icon" width="50" height="50"></td>
<td><h2>Adding a new compartment</h2></td>
</tr>
</table>

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