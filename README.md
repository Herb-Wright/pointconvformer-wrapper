
# PointConvFormer Wrapper

*Herbert Wright*

This project provides an easy interface for using PointConvFormer in Pytorch. Right now only an Encoder with pooling is implemented. 

Here is the PointConvPaper by Wu et al:

https://arxiv.org/abs/2208.02879


## Getting Started

Make sure you have all the requirements

- CUDA compatable device
- conda/mamba (or pip if you know what you are doing)

Clone this repo

```bash
git clone https://github.com/Herb-Wright/pointconvformer-wrapper.git --recurse-submodules
cd pointconvformer-wrapper
```

Create the conda environment

```bash
conda env create -f environment.yml
conda activate pointconvformer_wrapper
```

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
	pool='max'
)

points = torch.randn((1000, 3))
features = torch.randn((1000, 3))
batch = torch.zeros((1000,))

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

All of the tests should be passing.

## Known Issues

- knn method needs to have batches with > K points or it will fail to reshape
  - this might be what we want. Perhaps sampling should be updated
	- (possible solution) I added a "hack" into the knn that will enforce that k indices are returned by randomly sampling
- running the setup.sh creates a `./data` folder that shouldn't exist


