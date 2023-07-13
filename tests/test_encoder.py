from pointconvformer_wrapper.encoder import PointConvFormerEncoder
import torch

def test_pointconvformerencoder():
	num_points = 2000
	in_dim = 2
	out_dim = 4

	points = torch.randn((num_points, 3))
	feats = torch.randn((num_points, in_dim))
	batch = torch.randint(0, 2, (num_points,))

	model = PointConvFormerEncoder(
		in_dim=in_dim,
		out_dim=out_dim,
	)

	out = model(points, feats, batch)

	assert len(out.shape) == 2
	assert out.shape[0] == 2
	assert out.shape[1] == out_dim


def test_encoder_backward():
	num_points = 2000
	in_dim = 2
	out_dim = 4

	points = torch.randn((num_points, 3), requires_grad=True)
	feats = torch.randn((num_points, in_dim), requires_grad=True)
	batch = torch.randint(0, 2, (num_points,))

	model = PointConvFormerEncoder(
		in_dim=in_dim,
		out_dim=out_dim,
	)

	out = model(points, feats, batch)

	loss = torch.mean(out ** 2)
	loss.backward()

	assert torch.sum(feats.grad ** 2) > 1e-6
	assert torch.sum(points.grad ** 2) > 1e-6

