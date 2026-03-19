# microTorch Documentation

microTorch is a Python package for fitting diffusion MRI (dMRI) microstructure models using self-supervised deep learning. It allows flexible composition of multi-compartment models without requiring training data, and achieves fast inference by training and fitting simultaneously for each dataset.

Developed by members of the UCL Centre for Medical Image Computing and Cardiff University Brain Research Imaging Centre, microTorch is actively maintained and designed for both research and practical use in diffusion MRI modelling.

---

## Key Features

- **Self-supervised learning:** No external training dataset is required.
- **Flexible multi-compartment modeling:** Combine single-compartment models to create complex multi-compartment models.
- **Fast inference:** Training is performed simultaneously with fitting, reducing runtime.
- **Extensible:** Easily add new compartments or models.
- **Integrated with PyTorch:** Fully compatible with PyTorch autograd for differentiable modeling.

---

## Getting Started

- [Installation](getting_started.md) – Set up your environment and install microTorch.
- [Running microTorch](usage/cli.md) – Learn how to run experiments and override configuration parameters.
- [Models & Compartments](usage/models.md) – Detailed reference for predefined and custom microstructure models.
- [Data Formats](usage/data.md) – How to provide input data, gradient schemes, and acquisition parameters.
- [Training Parameters](usage/training.md) – Customize training, network, and optimization settings.
- [Developer Guide](developer/adding_compartments.md) – Instructions for adding new compartments or models.
- [Tutorials](tutorials/minimal_example.md) – Step-by-step tutorials for running experiments and testing synthetic datasets.

---

## Citation

If you use microTorch in your research, please cite:

[1] Sen et al. 2024  
[2] Barbieri et al. 2020

---

## Contributing

We welcome contributions! For guidance on contributing code, see [Developer Guide](developer/adding_compartments.md).

---
