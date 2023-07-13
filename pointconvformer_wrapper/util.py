from torch import Tensor
from typing import Tuple
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_geometric.nn import knn as pyg_knn
import torch

def grid_subsample(points: Tensor, batch: Tensor, subsample_grid_size: float) -> Tuple[Tensor, Tensor]:
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

def knn(
	x: Tensor, 
	y: Tensor, 
	k: int, 
	x_batch: Tensor, 
	y_batch: Tensor, 
	*, 
	hack: bool = False
) -> Tensor:
	'''
	performs knn returns Tensor of indices (P', K) where y is (P', 3) and x is (P, 3)
	
	args:
	- x (P, 3)
	- y (P', 3)
	- k (int): the number of neighbors K
	- x_batch (P,)
	- y_batch (P',)
	- hack (bool): if true will implement a hack that avoids errors where there are too
		little points for the knn
	
	returns:
	- idxs (P', K): the indices of the k nearest neighbors per each point in y
	'''
	P2 = y.shape[0]
	if hack:
		device = x.device
		for b in torch.unique(y_batch):
			if torch.sum(x_batch == b) < k:
				count = torch.sum(x_batch == b)
				diff = 2 + k - count
				pointcloud_size_multiple = (1 + diff // count) * count
				idxs_j = torch.randperm(pointcloud_size_multiple, dtype=int) % count
				x = torch.concatenate([x, x[x_batch == b][idxs_j[:diff]]])
				x_batch = torch.concatenate([x_batch, torch.ones((diff,), device=device) * b])
	pairs = pyg_knn(x, y, k, x_batch, y_batch)
	return pairs[1].reshape(P2, k)
