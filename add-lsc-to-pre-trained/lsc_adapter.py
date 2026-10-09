import torch
import torch.nn as nn

class LSCAdapter(nn.Module):
	def __init__(self, dim):
		super(self).__init__()

		self.norm = nn.FP32LayerNorm(dim * 2, 1e-6, True)
		self.linear = nn.Linear(dim * 2, dim)

	def forward(self, x, skip):
		x = torch.cat([x, skip], dim=-1)
		x = self.norm(x)
		x = self.linear(x)
		return x

class LSCAdapterGroup(nn.Module):
	def __init__(self, dim, layer_count):
		super(self).__init__()

		self.all_lsc = nn.ModuleDict()
		for layer in range(layer_count):
			use_skip = layer > layer_count // 2
			if use_skip:
				self.all_lsc[layer] = LSCAdapter(dim)
		self.layer_count = layer_count

	def forward(self, x, skip, layer_idx):
		adapter = self.all_lsc[layer_idx]
		return adapter(x, skip)

def add_lsc(transformer, dim, layer_count):
	print('remember to manually modify the transformer code to use these LSCs')

	transformer.lsc = LSCAdapterGroup(dim, layer_count)
	return transformer
