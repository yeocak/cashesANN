from typing import List, Optional

import numpy as np

from src.ann.Layer import Layer
from src.ann.activation.ActivationSoftmax import ActivationSoftmax
from src.ann.activation.CrossEntropy import CrossEntropy
from src.ann.options.ForwardResult import ForwardResult
from src.ann.options.LayerOption import LayerOption
from src.ann.properties.ANNProperties import ANNProperties
from src.utils.ListUtils import ListUtils


class ArtificialNN:
    def __init__(self):
        self.layers: List[Layer] = []

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def execute_forward(self, inputs: np.array) -> ForwardResult:
        if (len(self.layers) == 0) or (len(inputs) == 0):
            raise Exception('Number of layers or inputs are can\'t be zero!')

        # z_list: np.ndarray = np.full((self.get_ann_matrix_height(), self.get_ann_matrix_width()), np.nan)
        # a_list: np.ndarray = np.full((self.get_ann_matrix_height(), self.get_ann_matrix_width()), np.nan)
        z_list = list()
        a_list = list()

        prev_layer_a = inputs
        for layer_index in range(len(self.layers)):
            current_layer = self.layers[layer_index]
            neuron_count = len(current_layer.neuronList)
            new_z_list = np.empty(neuron_count)
            new_a_list = np.empty(neuron_count)
            for neuron_index in range(neuron_count):
                current_neuron = current_layer.neuronList[neuron_index]
                a, z = current_neuron.execute(prev_layer_a)
                new_z_list[neuron_index] = z
                new_a_list[neuron_index] = a
            z_list.append(new_z_list)
            a_list.append(new_a_list)

            current_activation = current_layer.neuronList[0].activation
            if isinstance(current_activation, ActivationSoftmax):
                not_nan_z = z_list[layer_index][~np.isnan(z_list[layer_index])]
                new_a = ActivationSoftmax.get_instance().execute_list(not_nan_z)
                a_list.pop()
                a_list.append(new_a)

            prev_layer_a = a_list[layer_index]

        return ForwardResult(inputs, z_list, a_list)

    def execute_backward(self, forward_result: ForwardResult, true_values: np.array, step_alpha: float) -> np.ndarray:
        differences = self.calculate_differences(forward_result, true_values)
        self.update_parameters(differences, forward_result, step_alpha)
        return differences

    # dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    # dW1 = 1 / m * dZ1.dot(X.T)
    # db1 = 1 / m * np.sum(dZ1)
    # returns list of LayerDifferenceResult
    def calculate_differences(self, forward_result: ForwardResult, true_values: np.array) -> np.ndarray:
        if (len(self.layers) == 0) or (len(true_values) == 0) or (len(forward_result.a_list) == 0):
            raise Exception('Number of layers, true values, predict values are can\'t be zero!')

        if len(true_values) != np.sum(~np.isnan((forward_result.get_output_layer_a()))):
            raise Exception('True values and predict values are not in same size!')

        delta_list = list()

        output_layer_index = len(self.layers) - 1
        first_hidden_layer_index = 0
        for layer_index in range(output_layer_index, first_hidden_layer_index - 1, -1):
            current_layer = self.layers[layer_index]

            errors = np.empty(len(current_layer.neuronList))
            # calculate for output layer
            if layer_index == output_layer_index:
                for neuron_index, neuron in enumerate(current_layer.neuronList):
                    neuron_result = forward_result.a_list[layer_index][neuron_index]
                    expected_result = true_values[neuron_index]
                    errors[neuron_index] = neuron_result - expected_result

            # calculate for hidden layers
            else:
                for neuron_index, neuron in enumerate(current_layer.neuronList):
                    current_error = 0.0
                    for next_layer_neuron_index, next_layer_neuron in enumerate(self.layers[layer_index + 1].neuronList):
                        current_error += next_layer_neuron.inputWeights[neuron_index] * delta_list[-1][
                            next_layer_neuron_index]
                    errors[neuron_index] = current_error

            # calculate deltas for next layer
            if isinstance(current_layer.get_single_activation(), ActivationSoftmax):
                # if activation is softmax
                next_layer = true_values if layer_index == output_layer_index else forward_result.a_list[
                    layer_index + 1]
                cross_entropy = CrossEntropy.execute(forward_result.a_list[layer_index], next_layer)
                # softmaxed_output = current_layer.get_single_activation().execute_derivative_list(forward_result.a_list[layer_index])
                new_deltas = np.empty(len(current_layer.neuronList))
                for neuron_index, neuron in enumerate(current_layer.neuronList):
                    delta = errors[neuron_index] * cross_entropy
                    new_deltas[neuron_index] = delta
                delta_list.append(new_deltas)
            else:
                # if activation is not softmax
                new_deltas = np.empty(len(current_layer.neuronList))
                for neuron_index, neuron in enumerate(current_layer.neuronList):
                    delta = (errors[neuron_index] *
                             neuron.activation.execute_derivative(forward_result.a_list[layer_index][neuron_index]))
                    new_deltas[neuron_index] = delta
                delta_list.append(new_deltas)

        delta_list.reverse()
        return ListUtils.convert_to_2d_array(delta_list)

    def update_parameters(self, delta_list: np.ndarray, forward_result: ForwardResult, step_alpha: float):
        for layer_index, layer in enumerate(self.layers):
            inputs = forward_result.input_list if layer_index == 0 else forward_result.a_list[layer_index - 1]
            for neuron_index, neuron in enumerate(layer.neuronList):
                for inputf_index, inputf in enumerate(inputs[~np.isnan(inputs)]):
                    neuron.inputWeights[inputf_index] -= step_alpha * delta_list[layer_index][neuron_index] * inputf
                neuron.bias -= step_alpha * delta_list[layer_index][neuron_index]

    def get_ann_matrix_height(self) -> int:
        return len(self.layers)

    def get_ann_matrix_width(self) -> int:
        max_neuron_number = 0
        for layer in self.layers:
            neuron_number = len(layer.neuronList)
            if neuron_number > max_neuron_number:
                max_neuron_number = neuron_number
        return max_neuron_number

    def get_input_count(self) -> int:
        return self.layers[0].get_input_count()

    @classmethod
    def create_random(cls, seed: int, input_number: int, layers: List[LayerOption]):
        result = cls()
        for layer_index in range(len(layers)):
            current_layer = layers[layer_index]
            prev_layer_neuron_number = input_number if layer_index == 0 else layers[layer_index - 1].neuronNumber
            layer = Layer.create_random(current_layer.neuronNumber, prev_layer_neuron_number,
                                        current_layer.activationFunction, seed, layer_index)
            result.layers.append(layer)
        return result

    @classmethod
    def create_from_properties(cls, properties: ANNProperties):
        layers = properties.layers
        result = cls()
        result.layers = layers
        return result

    def get_properties(self, seed: Optional[int]) -> ANNProperties:
        return ANNProperties(self.layers, seed)
