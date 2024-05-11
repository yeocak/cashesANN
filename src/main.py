from typing import List

import numpy as np

from src.ArtificialNN import ArtificialNN
from src.Layer import Layer
from src.LayerOption import LayerOption
from src.Neuron import Neuron
from src.activation.ActivationReLU import ActivationReLU
from src.activation.ActivationSigmoid import ActivationSigmoid
from src.activation.ActivationSoftmax import ActivationSoftmax


def test_forward_backward():
	activation_relu = ActivationReLU.get_instance()
	activation_softmax = ActivationSoftmax.get_instance()
	activation_sigmoid = ActivationSigmoid.get_instance()

	example_input = np.array([-3, 5, 8, -9])

	seed = 0
	ann = ArtificialNN.create_random(seed, len(example_input), [
		LayerOption(seed, 4, activation_sigmoid),
		LayerOption(seed, 4, activation_softmax),
	])

	test_result_f = ann.execute_forward(example_input)
	print(test_result_f)

	test_result_b_1 = ann.execute_backward(test_result_f, [0.2, 0.5, 0.2, 0.1])
	for testing_b in test_result_b_1:
		print("-----")
		print(testing_b)
	print("---changed weights---")

	test_result_f_2 = ann.execute_forward(example_input)
	print(test_result_f_2)

	test_result_b_2 = ann.execute_backward(test_result_f_2, [0.3, 0.3, 0.3, 0.1])
	for testing_b in test_result_b_2:
		print("-----")
		print(testing_b)
	print("---changed weights---")

	test_result_f_3 = ann.execute_forward(example_input)
	print(test_result_f_3)


if __name__ == '__main__':
	test_forward_backward()