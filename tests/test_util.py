from pointconvformer_wrapper.util import grid_subsample, knn, hybrid_grid_subsample
from torch import Tensor
import torch
import pytest


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


def test_knn():
	K = 2
	points = Tensor([
		[0, 0, 0],
		[0.5, 0.5, 0.5],
		[2, 2, 2],
		[0, 0, 0],
		[0, 0.6, 0],
		[0, 1.5, 0],
		[0, 1.7, 0],
	])
	sampled_points = Tensor([
		[0.25, 0.25, 0.25],
		[2, 2, 2],
		[0, 0.3, 0],
		[0, 1.6, 0],
	])
	batch = Tensor([0, 0, 0, 1, 1, 1, 1])
	sampled_batch = Tensor([0, 0, 1, 1])
	out = knn(points, sampled_points, K, batch, sampled_batch)
	
	assert len(out.shape) == 2
	assert out.shape[0] == sampled_points.shape[0]
	assert out.shape[1] == K

	expected = torch.Tensor([
		[0, 1],
		[2, 1],
		[3, 4],
		[5, 6],
	]).to(torch.long)
	assert torch.allclose(out, expected)

def test_knn_few_points_throws():
	K = 20
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
	with pytest.raises(Exception):
		out = knn(points, sampled_points, K, sampled_batch, batch)

def test_hybrid_subsample():
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
	sampled_points, sampled_batch = hybrid_grid_subsample(points, batch, 1.0, 4)

	assert sampled_batch.shape[0] == sampled_points.shape[0]

	print(sampled_points)
	assert sampled_points.shape[0] > 5 + 5


