from pointconvformer_wrapper.encoder import PointConvFormerEncoder
from torch import Tensor
import torch

def test_pointconvformerencoder():
	num_points = 10
	in_dim = 2
	out_dim = 4

	points = torch.randn((num_points, 3))
	feats = torch.randn((num_points, in_dim))
	batch = Tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])

	model = PointConvFormerEncoder(
		in_dim=in_dim,
		out_dim=out_dim
	)

	out = model(points, feats, batch)

	assert len(out.shape) == 2
	assert out.shape[0] == 3
	assert out.shape[1] == out_dim

