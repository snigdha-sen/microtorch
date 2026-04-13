# Getting Started with microTorch

This guide will help you **install microTorch** and set up your environment for running experiments.

---

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

## 5. Verify Installation

After installation, you can verify that microTorch is available:

```bash
python -c "import microtorch; print(microtorch.__version__)"
```

If no errors appear, the installation is successful.

## 6. Next Steps

After installing microTorch, you can run your first experiment.  
See [Running microTorch](usage/cli.md) for instructions on using the command line and Hydra configuration system.