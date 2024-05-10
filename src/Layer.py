from typing import List

import numpy as np
from numpy import array

from src import Neuron
from src.activation.ActivationFunction import ActivationFunction


class Layer:

    def __init__(self):
        self.neuronList: List[Neuron] = []

    def get_weight_matrix(self) -> array:
        weight_list: array[array[float]] = []
        for neuron in self.neuronList:
            weight_list.append(np.array(neuron.inputWeights))
        return weight_list

    def set_weight_matrix(self, weights: array):
        for layer_weights_index in range(len(weights)):
            for weight_index in range(len(weights[layer_weights_index])):
                self.neuronList[layer_weights_index].inputWeights[weight_index] = weights[layer_weights_index][weight_index]

    def get_single_activation(self) -> ActivationFunction:
        return self.neuronList[0].activation

    def set_bias(self, bias: float):
        for neuron in self.neuronList:
            neuron.bias = bias

    def get_single_bias(self) -> float:
        return self.neuronList[0].bias
