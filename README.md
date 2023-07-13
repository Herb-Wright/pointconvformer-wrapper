
# PointConvFormer Wrapper

*Herbert Wright*

This project provides an easy interface for using PointConvFormer in Pytorch


## Getting Started

Make sure you have all the requirements

- CUDA compatable device
- conda/mamba (or pip if you know what you are doing)

Clone this repo

Create the conda environment

Install the project. This should also compile the cpp wrappers for pointconv

```bash
pip install -e . -v
```

Import and use the Modules:

```python
import torch
from pointconvformer_wrapper import PointConvFormerEncoder

model = PointConvFormerEncoder(
	in_dim=3,
	out_dim=256,
)

points = torch.randn((5, 3))
features = torch.randn((5, 3))
batch = torch.zeros((5,))

out = model(points, features, batch)  # should be (1, 256)
```


## Tests

To run tests, make sure you have the conda environment activated and install `pytest`:

```bash
conda activate pointconvformer_wrapper
pip install pytest
```

Then you can run all tests or a specific test like so:

```bash
pytest ./tests
```

or 

```bash
pytest ./tests/test_util.py
```


