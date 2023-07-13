from torch.nn import Module, Linear
from torch import Tensor
from .util import grid_subsample, knn
from .pcf_api import PCF_Backbone, get_default_configs
from easydict import EasyDict
from typing import List
from torch_geometric.nn.pool import global_mean_pool, global_max_pool




class PointConvFormerEncoder(Module):
	'''
	PointConvFormerEncoder module that encodes using pointconvformer
	then pools

	args:
	- in_dim (int): the input feature dimension D
	- feat_dims (list[int]): the dimensions of the features
	- out_dim (int): the output features dimension H
	- k_forward
	- k_self
	- pool (str): one of 'mean', 'max', or 'both'

	input:
	- points (P, 3)
	- feats (P, D)
	- batch (P,)
	- norms (P, 3) [Optional]

	output:
	- out_feats (N, H)
	'''

	def __init__(
		self,
		in_dim: int = 3,
		feat_dims: List[int] = [64, 128, 256, 512, 512],
		out_dim: int = 256,
		grid_size: List[float] = [0.02, 0.06, 0.15, 0.375, 0.9375],
		k_forward: List[int] = [16, 16, 16, 16, 16],
		k_self: List[int] = [16, 16, 16, 16, 16, 16],
		pool: str = 'mean',
	) -> None:
		super().__init__()
		num_levels = len(feat_dims)
		self.pool = pool
		cfg = EasyDict()
		cfg.feat_dim = feat_dims
		cfg: EasyDict = get_default_configs(cfg, num_level=num_levels, base_dim=feat_dims[0])
		self.backbone = PCF_Backbone(cfg, in_dim)
		self.grid_size = grid_size
		self.k_forward = k_forward
		self.k_self = k_self
		self.lin = Linear(feat_dims[-1], out_dim)

	def forward(self, points: Tensor, features: Tensor, batch: Tensor, norms: Tensor | None = None) -> Tensor:
		# (1) subsample + knn
		points_list = [points]
		batch_list = [batch]
		edges_self = [knn(points, points, self.k_self[0])]
		edges_forward = []
		if norms is None:
			sampled_points, sampled_batch = points, batch
			for i, gs in enumerate(self.grid_size):
				sampled_points, sampled_batch = grid_subsample(sampled_points, sampled_batch)
				edges_self.append(knn(sampled_points, sampled_points, self.k_self[i+1]))
				edges_forward.append(knn(sampled_points, points_list[-1], self.k_forward[i], sampled_batch, batch_list[-1]))
				points_list.append(sampled_points)
				batch_list.append(sampled_batch)
		else:
			raise Exception('Damn you and your norms')

		# (2) forward pass
		feats_list = self.backbone(
			features,
			points_list,
			edges_self,
			edges_forward,
			# maybe norms
		)

		final_feats = feats_list[-1]  # (P', H)
		final_batch = batch_list[-1]  # (P',)

		# (3) pool
		if self.pool == 'mean':
			out = global_mean_pool(final_feats, final_batch)
		elif self.pool == 'max':
			out = global_max_pool(final_feats, final_batch)

		return out







