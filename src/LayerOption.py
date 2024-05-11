from src.activation.ActivationFunction import ActivationFunction


class LayerOption:
	def __init__(self, seed: int, neuron_number: int, activation_function: ActivationFunction):
		self.seed = seed
		self.neuronNumber = neuron_number
		self.activationFunction = activation_function
