import random
from typing import List

import numpy as np

from src.ann.Neuron import Neuron
from src.ann.activation.ActivationFunction import ActivationFunction


class Layer:

    def __init__(self):
        self.neuronList: List[Neuron] = []

    def get_weight_matrix(self) -> np.ndarray:
        weight_list = np.ndarray((len(self.neuronList), len(self.neuronList[0].inputWeights)))
        for neuron_index in range(len(self.neuronList)):
            current_neuron = self.neuronList[neuron_index]
            for weight_index in range(len(current_neuron.inputWeights)):
                current_weight = current_neuron.inputWeights[weight_index]
                weight_list[neuron_index][weight_index] = current_weight
        return weight_list

    def set_weight_matrix(self, weights: np.ndarray):
        for layer_weights_index in range(len(weights)):
            self.neuronList[layer_weights_index].inputWeights = weights[layer_weights_index]

    def get_single_activation(self) -> ActivationFunction:
        return self.neuronList[0].activation

    def set_bias(self, bias: float):
        for neuron in self.neuronList:
            neuron.bias = bias

    def get_single_bias(self) -> float:
        return self.neuronList[0].bias

    def get_input_count(self) -> int:
        return len(self.neuronList[0].inputWeights)

    @staticmethod
    def create_random(neuron_number: int, input_number: int, activation_function: ActivationFunction, seed: int, layer_index: int):
        result = Layer()

        for neuron_index in range(neuron_number):
            weights = np.empty(input_number)
            for input_index in range(input_number):
                weight_seed = int(f'{seed}{layer_index}{neuron_index}{input_index}')
                random.seed(weight_seed)
                weights[input_index] = random.random()

            bias_seed = int(f'{seed}{layer_index}{neuron_index}')
            random.seed(bias_seed)
            bias = random.random()
            result.neuronList.append(Neuron(activation_function, weights, bias))

        return result
