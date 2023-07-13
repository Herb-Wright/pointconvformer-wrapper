from torch import Tensor
from typing import Tuple
from torch_geometric.nn import voxel_grid
from torch_geometric.nn.pool.consecutive import consecutive_cluster
from torch_geometric.nn.pool.pool import pool_pos, pool_batch
from torch_geometric.nn import knn as pyg_knn

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

def knn(x: Tensor, y: Tensor, k: int, x_batch: Tensor, y_batch: Tensor) -> Tensor:
	'''performs knn returns Tensor of indices (P', K) where y is (P', 3) and x is (P, 3)'''
	P2 = y.shape[0]
	pairs = pyg_knn(x, y, k, x_batch, y_batch)
	return pairs[1].reshape(P2, k)
