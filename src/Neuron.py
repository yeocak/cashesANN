from typing import List

import numpy as np

from src.activation.ActivationFunction import ActivationFunction


class Neuron:
    def __init__(self, activation: ActivationFunction, input_weights: np.array, bias: float):
        self.activation = activation
        self.inputWeights = input_weights
        self.bias = bias

    # Creates a new Neuron as all input weights are zero and bias zero.
    @staticmethod
    def create_with_zero_inputs(activation: ActivationFunction, input_number: int):
        input_weights: np.array = []
        for i in range(input_number):
            input_weights.append(0)

        return Neuron(activation, input_weights, 0)

    def get_number_of_inputs(self) -> int:
        return len(self.inputWeights)

    def execute(self, inputs: np.array) -> (float, float):
        if self.get_number_of_inputs() != len(inputs):
            raise Exception('Number of inputs does not match')

        linear_sum: float = self.bias
        for i in range(self.get_number_of_inputs()):
            linear_sum += inputs[i] * self.inputWeights[i]

        result = self.activation.execute(linear_sum)
        return result, linear_sum
