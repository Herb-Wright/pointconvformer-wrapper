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
		out = knn(points, sampled_points, K, batch, sampled_batch)

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


def test_knn_gpu():
	points = Tensor([
		[-0.0994,  0.0529, -0.1882],
		[ 0.0508,  0.0526, -0.1847],
		[ 0.1876,  0.0520, -0.1798],
		[-0.1861, -0.0143,  0.0466],
		[-0.0977, -0.0153,  0.0554],
		[ 0.0481, -0.0157,  0.0599],
		[ 0.1791, -0.0162,  0.0646],
		[-0.0993, -0.0795,  0.2712],
		[-0.0141,  0.0528, -0.1873],
		[-0.0571,  0.0530, -0.1884],
		[-0.0119, -0.0157,  0.0595],
		[ 0.0708, -0.0155,  0.0575],
		[ 0.0303, -0.0158,  0.0600],
		[-0.0542, -0.0156,  0.0586],
		[-0.0995,  0.0530, -0.1884],
		[ 0.2002, -0.0163,  0.0650],
		[ 0.2018,  0.0520, -0.1796],
		[ 0.2018,  0.0520, -0.1796],
		[ 0.1147, -0.0159,  0.0615],
		[ 0.1161,  0.0522, -0.1815],
		[ 0.2002, -0.0163,  0.0650],
		[ 0.0728,  0.0524, -0.1832],
		[-0.1070, -0.0796,  0.2722],
		[ 0.1592,  0.0521, -0.1801],
	]).to('cuda')
	batch = torch.zeros(points.shape[0]).to('cuda')
	out = knn(points, points, 16, batch, batch)

	assert out.shape[0] == points.shape[0]
	assert out.shape[1] == 16

def test_knn_hack():
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
	out = knn(points, sampled_points, K, batch, sampled_batch, hack=True)

	assert len(out.shape) == 2
	assert out.shape[0] == sampled_points.shape[0]
	assert out.shape[1] == K
