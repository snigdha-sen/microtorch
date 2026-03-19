# microTorch Documentation

microTorch is a Python package for fitting diffusion MRI (dMRI) microstructure models using self-supervised deep learning. It allows flexible composition of multi-compartment models without requiring training data, and achieves fast inference by training and fitting simultaneously for each dataset.

Developed by members of the UCL Centre for Medical Image Computing and Cardiff University Brain Research Imaging Centre, microTorch is actively maintained and designed for both research and practical use in diffusion MRI modelling.

## Contents

- [Installation](getting_started.md) – Set up your environment and install microTorch.
- [Running microTorch](usage/cli.md) – Learn how to run experiments with Hydra and override configuration parameters.
- [Models & Compartments](usage/models.md) – Detailed reference for predefined and custom microstructure models.
- [Data Formats](usage/data.md) – How to provide input data, gradient schemes, and acquisition parameters.
- [Training Parameters](usage/training.md) – Customise training, network, and optimisation settings.
- [Simulation Data](tutorials/simulation_data.md) - How to create and use simulated data for various microstructural models.
- [Developer Guide](developer/contributions.md) – Instructions for adding new compartments or models.
