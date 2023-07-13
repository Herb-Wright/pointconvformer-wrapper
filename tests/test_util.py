from pointconvformer_wrapper.util import grid_subsample
from torch import Tensor
import torch


def test_grid_subsample():
	points = Tensor([
		[0, 0, 0],
		[0.5, 0.5, 0.5],
		[2, 2, 2],
		[0, 0, 0],
		[0, 0.6, 0],
		[0, 1.5, 0],
		[0, 1.7, 0],
	])
	batch = Tensor([0, 0, 0, 1, 1, 1, 1])
	sampled_points, sampled_batch = grid_subsample(points, batch, 1.0)  # <-- magic happens here
	expected_points = Tensor([
		[0.25, 0.25, 0.25],
		[2, 2, 2],
		[0, 0.3, 0],
		[0, 1.6, 0],
	])
	assert torch.allclose(sampled_points, expected_points)
	expected_batch = Tensor([0, 0, 1, 1])
	assert torch.allclose(sampled_batch, expected_batch)
	sampled_points, sampled_batch = grid_subsample(points, batch, 0.1)  # <-- magic happens here
	# if the grid size is small, there is not really any subsampling
	assert torch.allclose(sampled_points, points)
	assert torch.allclose(sampled_batch, batch)

from torch_cluster.knn import knn

def test_knn():
	points = Tensor([
		[0, 0, 0],
		[0.5, 0.5, 0.5],
		[2, 2, 2],
		[0, 0, 0],
		[0, 0.6, 0],
		[0, 1.5, 0],
		[0, 1.7, 0],
	])
	batch = Tensor([0, 0, 0, 1, 1, 1, 1])
	sampled_points, sampled_batch = grid_subsample(points, batch, 1.0)
	out = knn(sampled_points, points, 2, sampled_batch, batch)
	print(out[1].reshape(-1, 2))
	raise Exception()

