from multiprocessing import Array
from typing import List

import numpy as np

from src.ForwardResult import ForwardResult
from src.Layer import Layer
from src.LayerDifferenceResult import LayerDifferenceResult
from src.LayerOption import LayerOption
from src.activation.ActivationSoftmax import ActivationSoftmax


class ArtificialNN:
    def __init__(self, number_of_inputs: int):
        self.layers: List[Layer] = []
        self.numberOfInputs = number_of_inputs

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def execute_forward(self, inputs: np.array) -> ForwardResult:
        if (len(self.layers) == 0) or (len(inputs) == 0):
            raise Exception('Number of layers or inputs are can\'t be zero!')

        z_list: np.ndarray = np.empty((self.get_ann_matrix_height(), self.get_ann_matrix_width()))
        a_list: np.ndarray = np.empty((self.get_ann_matrix_height(), self.get_ann_matrix_width()))

        prev_layer_a = inputs
        for layer_index in range(len(self.layers)):
            current_layer = self.layers[layer_index]
            for neuron_index in range(len(current_layer.neuronList)):
                current_neuron = current_layer.neuronList[neuron_index]
                a, z = current_neuron.execute(prev_layer_a)
                z_list[layer_index][neuron_index] = z
                a_list[layer_index][neuron_index] = a

            current_activation = current_layer.neuronList[0].activation
            if isinstance(current_activation, ActivationSoftmax):
                new_a = ActivationSoftmax.get_instance().execute_list(z_list[layer_index])
                a_list[layer_index] = new_a

            prev_layer_a = a_list[layer_index]

        return ForwardResult(inputs, z_list, a_list)

    def execute_backward(self, forward_result: ForwardResult, true_values: np.array) -> np.array:
        differences = self.calculate_differences(forward_result, true_values)
        self.update_parameters(differences, 0.1)
        return differences

    # dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    # dW1 = 1 / m * dZ1.dot(X.T)
    # db1 = 1 / m * np.sum(dZ1)
    # returns list of LayerDifferenceResult
    def calculate_differences(self, forward_result: ForwardResult, true_values: np.array) -> np.array:
        if (len(self.layers) == 0) or (len(true_values) == 0) or (len(forward_result.a_list) == 0):
            raise Exception('Number of layers, true values, predict values are can\'t be zero!')

        if len(true_values) != len(forward_result.get_output_layer_a()):
            raise Exception('True values and predict values are not in same size!')

        layer_count = len(self.layers)
        output_class_count = len(self.layers[-1].neuronList)

        result_list: List[LayerDifferenceResult] = []

        d_last_layer_z: Array[float] = []

        # calculate for output first
        previous_result_a_list = forward_result.a_list[-2] if layer_count > 1 else forward_result.input_list

        d_output_z = np.subtract(forward_result.get_output_layer_a(), true_values)
        d_output_weight = 1 / output_class_count * np.dot(d_output_z, previous_result_a_list.T)
        d_output_bias = 1 / output_class_count * np.sum(d_output_z)

        d_last_layer_z = d_output_z.copy()

        result_list.append(LayerDifferenceResult(d_output_z, d_output_weight, d_output_bias))

        # loop is not going to calculate for output layer, input layer and first hidden layer
        # example: 4 layers -> 2,1
        last_hidden_layer_index = layer_count - 2
        first_hidden_layer_index = 0
        for forward_index in range(last_hidden_layer_index, first_hidden_layer_index, -1):
            current_layer = self.layers[forward_index]
            next_layer = self.layers[forward_index + 1]
            current_result_z_list = forward_result.z_list[forward_index]
            previous_result_a_list = forward_result.a_list[forward_index - 1]
            next_weight_transposed = next_layer.get_weight_matrix().T

            d_layer_z = (np.dot(next_weight_transposed, d_last_layer_z) *
                         current_layer.get_single_activation().execute_derivative_list(current_result_z_list))
            d_layer_weight = 1 / output_class_count * np.dot(d_layer_z, previous_result_a_list.T)
            d_layer_bias = 1 / output_class_count * np.sum(d_layer_z)

            d_last_layer_z = d_layer_z.copy()

            result_list.append(LayerDifferenceResult(d_layer_z, d_layer_weight, d_layer_bias))

        # calculate for first hidden layer
        if layer_count > 1:
            input_layer_transposed = forward_result.input_list.T
            current_layer = self.layers[0]
            next_layer = self.layers[1]
            current_result_z_list = forward_result.z_list[0]
            next_weight_transposed = next_layer.get_weight_matrix().T

            d_first_layer_z = (np.dot(next_weight_transposed, d_last_layer_z) *
                               current_layer.get_single_activation().execute_derivative_list(current_result_z_list))
            d_first_layer_weight = (1 / output_class_count * np.dot(d_first_layer_z, input_layer_transposed))
            d_first_layer_bias = 1 / output_class_count * np.sum(d_first_layer_z)

            result_list.append(LayerDifferenceResult(d_first_layer_z, d_first_layer_weight, d_first_layer_bias))

        result_list.reverse()
        return result_list

    def update_parameters(self, differences: List[LayerDifferenceResult], step_alpha: float):
        for difference_index in range(len(differences)):
            new_bias = self.layers[difference_index].get_single_bias() - step_alpha * differences[difference_index].b
            new_weights = (self.layers[difference_index].get_weight_matrix() - step_alpha * differences[difference_index].w)
            self.layers[difference_index].set_bias(new_bias)
            self.layers[difference_index].set_weight_matrix(new_weights)

    def get_ann_matrix_height(self) -> int:
        return len(self.layers)

    def get_ann_matrix_width(self) -> int:
        max_neuron_number = 0
        for layer in self.layers:
            neuron_number = len(layer.neuronList)
            if neuron_number > max_neuron_number:
                max_neuron_number = neuron_number
        return max_neuron_number

    @staticmethod
    def create_random(seed: int, input_number: int, layers: List[LayerOption]):
        result = ArtificialNN(input_number)
        for layer_index in range(len(layers)):
            current_layer = layers[layer_index]
            prev_layer_neuron_number = input_number if layer_index == 0 else layers[layer_index - 1].neuronNumber
            layer = Layer.create_random(current_layer.neuronNumber, prev_layer_neuron_number,
                                        current_layer.activationFunction, seed, layer_index)
            result.layers.append(layer)
        return result
