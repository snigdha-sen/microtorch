# Getting Started with microTorch

This guide will help you set up your environment, **install microTorch**, and run some simple simulation experiments to demonstrate microTorch fitting.

---

## 1. Create and activate a virtual environment (recommended)

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

## 2 (option A) Quick install of source code
To install the core microTorch package from PyPI run:

```bash
pip install --upgrade pip
pip install microtorch-mri
```

This installs the microTorch package and its command line tools. 


## 2 (option B) Clone the repository

To install the full package including source code, notebooks, and tests in editable mode.

Choose an installation location `INSTALL_DIR` and move to it.

```bash
cd INSTALL_DIR
```
Clone the microTorch repository

``` bash
git clone https://github.com/snigdha-sen/microtorch.git
cd microtorch
```
Move to the microtorch repository

```
cd microtorch
```
Next, Install MicroTorch and its dependencies (as defined in
`pyproject.toml`):

``` bash
pip install .
```

If you plan to modify the code run the following instead

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

## 3. Verify Installation

After installation, you can verify that microTorch is available:

```bash
python -c "import microtorch; print(microtorch.__version__)"
```

If no errors appear, the installation is successful.

## 4. Next Steps

After installing microTorch, you can run your first experiments:
  
* See [Testing with Simulated Data](tutorials/simulation_data.md) to generate simulated datasets with known ground truth and learn how microTorch fitting works. 
* See [Running microTorch](usage/cli.md) for instructions on fitting your own data using the using the command line interface and Hydra configuration system.