# `AFINNI` - Adaptive Affine Nearest Neighbor Interpolation
Code for the `AFINNI` algorithm.

# Installation
The project makes use a [`Makefile`](Makefile) for installing the necessary dependencies (support for both CPU-only and GPU-enabled).

`AFINNI` is currently tested on Linux and macOS. Windows is not currently tested or supported, especially for workflows involving MPI or GPU-enabled dependencies.

Run the necessary make command based on your system configuration to install the dependencies.

### Using UV
The project makes use of `uv` for python environment management, as does the `Makefile` for the required installation, benchmarking, linting, etc. 
 
If you want to install `uv`, you can refer to the [Installation UV Guide](https://docs.astral.sh/uv/getting-started/installation/).

Or if you cannot use `uv`, you can manually create a python virtual environment and install the required dependencies, making use of the `pyproject.toml` file via

```bash
python -m venv venv
source venv/bin/activate
pip install -e .
#  or pip install -e .[gpu, lint, test] for all optional dependencies
 ```
