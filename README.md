### MicroTorch Library
Library of PyTorch implementations of microstructural and quantitative MRI models.

UCL Centre for Medical Image Computing and Cardiff University Brain Research Imaging Centre

## Dependencies (incomplete)
PyTorch
Numpy

## Installation

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

## Command line examples

*Ball-stick (doesn't quite work)*

python3 fit.py -img data.nii.gz -ma mask.nii.gz -bvals bvals.txt -bvecs bvecs.txt -d 24 -sd 8 -se 123 -m BallStick -a relu -lr 0.0001 

*MSDKI (works but this model doesn't seem to like dropout)*

python3 fit.py -img data.nii.gz -ma mask.nii.gz -bvals bvals.txt -bvecs bvecs.txt -d 24 -sd 8 -se 123 -m MSDKI -a elu -lr 0.01  
