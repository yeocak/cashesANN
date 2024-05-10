from multiprocessing import Array
from typing import List

import numpy as np

from src.ForwardResult import ForwardResult
from src.Layer import Layer
from src.LayerDifferenceResult import LayerDifferenceResult


class ArtificialNN:
    def __init__(self, number_of_inputs: int):
        self.layers: List[Layer] = []
        self.numberOfInputs = number_of_inputs

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def execute_forward(self, inputs: np.array) -> ForwardResult:
        if (len(self.layers) == 0) or (len(inputs) == 0):
            raise Exception('Number of layers or inputs are can\'t be zero!')

        z_list: np.ndarray = []
        a_list: np.ndarray = []

        current = inputs
        for layer in self.layers:
            new_z: np.array = []
            new_a: np.array = []
            for neuron in layer.neuronList:
                a, z = neuron.execute(current)
                new_z.append(z)
                new_a.append(a)
            current = new_a
            z_list.append(new_z)
            a_list.append(new_a)

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
        test = np.transpose(previous_result_a_list)
        d_output_weight = 1 / output_class_count * np.dot(d_output_z, np.transpose(previous_result_a_list))
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

            next_weight_transposed = np.transpose(next_layer.get_weight_matrix())
            d_layer_z = (np.dot(next_weight_transposed, d_last_layer_z) *
                         current_layer.get_single_activation().execute_derivative_list(current_result_z_list))
            d_layer_weight = 1 / output_class_count * np.dot(d_layer_z, np.transpose(previous_result_a_list))
            d_layer_bias = 1 / output_class_count * np.sum(d_layer_z)

            d_last_layer_z = d_layer_z.copy()

            result_list.append(LayerDifferenceResult(d_layer_z, d_layer_weight, d_layer_bias))

        # calculate for first hidden layer
        if layer_count > 1:
            input_layer_transposed = np.transpose(forward_result.input_list)
            current_layer = self.layers[0]
            next_layer = self.layers[1]
            current_result_z_list = forward_result.z_list[0]
            next_weight_transposed = np.transpose(next_layer.get_weight_matrix())
            d_first_layer_z = (np.dot(next_weight_transposed, d_last_layer_z) *
                               current_layer.get_single_activation().execute_derivative_list(current_result_z_list))
            d_first_layer_weight = (1 / output_class_count *
                                    np.dot(d_first_layer_z, np.transpose(input_layer_transposed)))
            d_first_layer_bias = 1 / output_class_count * np.sum(d_first_layer_z)

            result_list.append(LayerDifferenceResult(d_first_layer_z, d_first_layer_weight, d_first_layer_bias))

        result_list.reverse()
        return result_list

    def update_parameters(self, differences: List[LayerDifferenceResult], step_alpha: float):
        for difference_index in range(len(differences)):
            new_bias = self.layers[difference_index].get_single_bias() - step_alpha * differences[difference_index].b
            new_weights = (self.layers[difference_index].get_weight_matrix() -
                           step_alpha * differences[difference_index].w)
            self.layers[difference_index].set_bias(new_bias)
            self.layers[difference_index].set_weight_matrix(new_weights)
