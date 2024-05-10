import numpy as np


class LayerDifferenceResult:
	def __init__(self, d_layer_z: np.array, d_layer_weight: np.array, d_layer_bias: float):
		self.z = d_layer_z
		self.w = d_layer_weight
		self.b = d_layer_bias

	def __str__(self):
		return (f'z:	{self.z}\n'
				f'w:	{self.w}\n'
				f'b:    {self.b}')
