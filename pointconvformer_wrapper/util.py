from torch import Tensor
from typing import Tuple
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_geometric.nn import knn as pyg_knn
import torch

def grid_subsample(
	points: Tensor, 
	batch: Tensor, 
	subsample_grid_size: float,
) -> Tuple[Tensor, Tensor]:
	'''
	performs grid subsampling on points in packed format with a batch index tensor.
	inspired by various methods from the torch_points3d package
	
	args:
	- points (P, 3): points in packed format
	- batch (P,): batch indices
	- subsample_grid_size (float): grid size used
	
	returns:
	- sampled_points (P', 3)
	- sampled_batch (P'): batch indices
	'''
	cluster_ids = voxel_grid(points, subsample_grid_size, batch=batch)  # (P,)
	cluster_idxs, unique_pos_indices = consecutive_cluster(cluster_ids)
	sampled_points = pool_pos(cluster_idxs, points)
	sampled_batch = pool_batch(unique_pos_indices, batch)
	return sampled_points, sampled_batch
	
def grid_subsample_with_feats(
	points: Tensor, 
	batch: Tensor, 
	feats: Tensor,
	subsample_grid_size: float,
) -> Tuple[Tensor, Tensor, Tensor]:
	cluster_ids = voxel_grid(points, subsample_grid_size, batch=batch)  # (P,)
	cluster_idxs, unique_pos_indices = consecutive_cluster(cluster_ids)
	sampled_points = pool_pos(cluster_idxs, points)
	sampled_batch = pool_batch(unique_pos_indices, batch)
	sampled_feats = pool_pos(cluster_idxs, feats)
	return sampled_points, sampled_batch, sampled_feats

def hybrid_grid_subsample(
	points: Tensor, 
	batch: Tensor, 
	subsample_grid_size: float,
	num_random: int = 32,
):
	sampled_points, sampled_batch = grid_subsample(points, batch, subsample_grid_size)
	device = points.device
	for b in range(int(torch.amax(batch).item()) + 1):
		num_points = torch.sum(batch == b)
		size_mult = (1 + num_random // num_points) * num_points
		idxs_j = torch.randperm(size_mult, dtype=int, device=device) % num_points
		new_points = points[batch == b][idxs_j[:num_random]]
		sampled_points = torch.concatenate([sampled_points, new_points], dim=0)
		sampled_batch = torch.concatenate([sampled_batch, torch.ones((num_random,), device=device) * b], dim=0)
	return sampled_points, sampled_batch

def knn(x: Tensor, y: Tensor, k: int, x_batch: Tensor, y_batch: Tensor) -> Tensor:
	'''
	performs knn returns Tensor of indices (P', K) where y is (P', 3) and x is (P, 3)
	
	args:
	- x (P, 3)
	- y (P', 3)
	- k (int): the number of neighbors K
	- x_batch (P,)
	- y_batch (P',)
	
	returns:
	- idxs (P', K): the indices of the k nearest neighbors per each point in y
	'''
	P2 = y.shape[0]
	pairs = pyg_knn(x, y, k, x_batch, y_batch)
	idxs, mask = to_dense_batch(pairs[1], pairs[0])
	if torch.sum(~mask) > 0 or idxs.shape[0] != P2:
		raise Exception('too few neighbors found')
	return idxs
