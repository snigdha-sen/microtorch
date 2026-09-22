# microTorch: microstructure model fitting with PyTorch


<img align="left" width="295" height="295" src="files/logo.jpeg" > 

The ```microTorch``` software package[1] is designed to flexibly fit diffusion MRI (dMRI) microstructure models, using a self-supervised deep learning approach. The framework is designed to work with a variety of established dMRI multicompartment models, and also allows users to combine compartment models as they wish.

We designed this framework to leverage the inference time gains of deep learning, whilst removing the requirement for explicit training data. Training and inference is performed simultaneously for each dataset. Please see [2,3] for the theoretical underpinnings of this approach.

This work is by members of the UCL Centre for Medical Image Computing and the Cardiff University Brain Research Imaging Centre. Please contact snigdha.sen.20@ucl.ac.uk with any questions.  
&nbsp;  

MicroTorch is actively developed software and will contain bugs and issues. If you encounter a problem you can:

- Open an issue here: https://github.com/snigdha-sen/microtorch/issues 
- Fork or branch the repository, implement a fix, and submit a merge request

We appreciate any feedback or contributions that help improve the project.


<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

# Installation

## Quick install (source code)
Installs the core microTorch package from PyPI:

```bash
pip install microtorch-mri
```

This installs the microTorch package and its command line tools.

## Full installation (source code, notebooks, and tests)

Includes the package, example Jupyter notebooks, tests, and development utilities. All files will be cloned into a user-chosen `INSTALL_DIR`.

### macOS/Linux

```bash
cd INSTALL_DIR
git clone https://github.com/snigdha-sen/microtorch.git
cd microtorch
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install --upgrade pip
pip install -e ".[dev]"
```

### Windows
Replace ```source .venv/bin/activate``` with

```bash
.venv\Scripts\activate 
```    

### Running the tests

```bash
python -m pytest tests
```

With a coverage report:

```bash
python -m pytest tests --cov --cov-report=term-missing
```

See [docs/developer/testing.md](docs/developer/testing.md) for more, including what the integration test covers and which modules are excluded from the coverage target (and why).


<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

# Quickstart
After installation, make sure you're in the `microtorch` repository directory (`INSTALL_DIR/microtorch`, where you ran `pip install -e .`).

Next, make some simulated test data

```bash
microtorch-create-test-images
```    
This will create test datasets in ```INSTALL_DIR/microtorch/simulation_data/data```.

You can then fit a model (SANDI in this example) to one of the simulated datasets.

```bash
python -m microtorch.main \
model.name=SANDI \
data.image=simulation_data/data/SANDI/SANDI_BallSphereAstrosticks_data.nii.gz \
acquisition.grad=src/microtorch/resources/protocols/grad_sandi.txt
```

When the model fit has finished you can compare the fitted and ground truth parameters using [this notebook](examples/plot_test_images.ipynb).
  
To fit any supported model to your own diffusion MRI data, specify the model name together with the image and acquisition protocol:

```bash
python -m microtorch.main \
model.name=CHOSEN_MODEL \
data.image=/path/to/dwi.nii.gz \
acquisition.grad=/path/to/grad.txt
```

For full usage instructions and examples, see the [documentation](docs/index.md).

Useful starting points include:

- **General usage:** [here](docs/usage/cli.md)
- **Tutorials:** [here](docs/tutorials/simulation_data.md) 
- **Command line interface and configuration:** [here](docs/usage/configs.md)
- **Contributing:** [here](docs/developer/contributing.md)

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

# Contributing

We welcome contributions! Please fork the repo, create a branch for your feature, and submit a Pull Request. 
Ensure code is tested and documented. Full guidelines in [contributing](docs/developer/contributing.md).

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

# References
If you use ```microTorch``` in your work please cite the accompanying overview paper:

[1] Sen S, Ahmed R, Arends GC, et al. **microTorch: A Software Package for Fast and Flexible Self-Supervised Diffusion MRI Model Fitting.** Journal of Open Research Software, 2026; 14: 59. DOI: https://doi.org/10.5334/jors.736

If you use a specific method, please also cite the relevant paper. For example, for the VERDICT model cite:

[2] Sen S, Singh S, Pye H, et al. **ssVERDICT: Self-supervised VERDICT-MRI for enhanced prostate tumor characterization.** Magn Reson Med. 2024; 92: 2181-2192. doi: https://doi.org/10.1002/mrm.30186

and for the IVIM model cite:

[3] Barbieri S, Gurney-Champion OJ, Klaassen R, Thoeny HC. **Deep learning how to fit an intravoxel incoherent motion model to diffusion-weighted MRI.** Magn Reson Med. 2020 Jan;83(1):312-321. doi: https://doi.org/10.1002/mrm.27910

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

# Acknowledgements

## Authors

```microTorch``` is authored by Snigdha Sen, Rajib Ahmed, Gerrit Arends, Alvaro Planchuelo Gomez, Xiaoxiang Chen, Marta Masramon Masramon, Marco Palombo, Chris Parker, Chantal MW Tax, Eleftheria Panagiotaki and Paddy J Slator. 

## Funding
Snigdha Sen undertook this work whilst being funded by the EPSRC-funded UCL Center for Doctoral Training in Intelligent, Integrated Imaging in Healthcare (i4health) (EP/S021930/1) and the Department of Health's NIHR-funded Biomedical Research Centre at University College London Hospitals.

## Other
The following code repositories were helpful in development of ```microTorch```:

- **Deep Learning How to Fit an Intravoxel Incoherent Motion Model to Diffusion-Weighted MRI** Barbieri et al. [https://github.com/sebbarb/deep_ivim](https://github.com/sebbarb/deep_ivim)
- **Dmipy: Diffusion Microstructure Imaging in Python** [https://github.com/AthenaEPI/dmipy](https://github.com/AthenaEPI/dmipy)

