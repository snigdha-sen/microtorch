# microTorch: microstructure model fitting with PyTorch


<img align="left" width="295" height="295" src="files/logo.jpeg" > 

The ```microTorch``` software package is designed to flexibly fit diffusion MRI (dMRI) microstructure models, using a self-supervised deep learning approach. The framework is designed to work with a variety of established dMRI multicompartment models, and also allows users to combine compartment models as they wish.

We designed this framework to leverage the inference time gains of deep learning, whilst removing the requirement for explicit training data. Training and inference is performed simultaneously for each dataset. Please see [1,2] for the theoretical underpinnings of this approach.

This work is by members of the UCL Centre for Medical Image Computing and the Cardiff University Brain Research Imaging Centre. Please contact snigdha.sen.20@ucl.ac.uk with any questions.  
&nbsp;  

MicroTorch is actively developed software and will contain bugs and issues. If you encounter a problem you can:

- Open an issue here: https://github.com/snigdha-sen/microtorch/issues 
- Fork or branch the repository, implement a fix, and submit a merge request

We appreciate any feedback or contributions that help improve the project.


<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

# Installation

```bash
git clone https://github.com/snigdha-sen/microtorch.git
cd microtorch
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
pip install --upgrade pip
pip install -e .
```

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

# Quickstart

```bash
python -m microtorch.main \
model.name=SANDI \
data.image=/path/to/dwi.nii \
acquisition.grad=/path/to/grad.scheme
```

For full usage instructions and examples, see the [documentation](usage/cli.md).

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

# Contributing

We welcome contributions! Please fork the repo, create a branch for your feature, and submit a Pull Request. 
Ensure code is tested and documented. Full guidelines in [contributing](developer/contributing.md).

<img align="left" src="files/torch.png" alt="icon" width="45" height="45">

# Acknowledgements

## Authors

```microTorch``` is authored by Snigdha Sen, Rajib Ahmed, Gerrit Arends, Alvaro Planchuelo Gomez, Xiaoxiang Chen, Marta Masramon Masramon, Marco Palombo, Chris Parker, Chantal MW Tax, Eleftheria Panagiotaki and Paddy J Slator. 

Snigdha Sen undertook this work whilst being funded by the EPSRC-funded UCL Center for Doctoral Training in Intelligent, Integrated Imaging in Healthcare (i4health) (EP/S021930/1) and the Department of Health's NIHR-funded Biomedical Research Centre at University College London Hospitals.

## References

[1] Sen S, Singh S, Pye H, et al. **ssVERDICT: Self-supervised VERDICT-MRI for enhanced prostate tumor characterization.** Magn Reson Med. 2024; 92: 2181-2192. doi: 10.1002/mrm.30186

[2] Barbieri S, Gurney-Champion OJ, Klaassen R, Thoeny HC. **Deep learning how to fit an intravoxel incoherent motion model to diffusion-weighted MRI.** Magn Reson Med. 2020 Jan;83(1):312-321. doi: 10.1002/mrm.27910

The following code repositories were helpful in development of MicroTorch:

- **Deep Learning How to Fit an Intravoxel Incoherent Motion Model to Diffusion-Weighted MRI** Barbieri et al. [https://github.com/sebbarb/deep_ivim](https://github.com/sebbarb/deep_ivim)
- **Dmipy: Diffusion Microstructure Imaging in Python** [https://github.com/AthenaEPI/dmipy](https://github.com/AthenaEPI/dmipy)

## Citation
