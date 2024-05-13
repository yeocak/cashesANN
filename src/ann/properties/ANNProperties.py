from typing import List, Optional

from src.ann.Layer import Layer
from src.ann.Neuron import Neuron
from src.ann.activation.ActivationFunction import ActivationFunction
from src.ann.activation.ActivationReLU import ActivationReLU
from src.ann.activation.ActivationSigmoid import ActivationSigmoid
from src.ann.activation.ActivationSoftmax import ActivationSoftmax
from src.ann.activation.ActivationTanH import ActivationTanH


class ANNProperties:
    def __init__(self, layers: List[Layer], seed: Optional[int]):
        self.layers = layers
        self.seed = seed

    # string format is like:
    # seed/{seed here if exists}
    # {activation}/{bias}/{weight1.1}_{weight1.2}_{weight1.3}/{bias}/{weight2.1}_{weight2.2}_{weight2.3}
    # {activation}/{bias}/{weight1.1}_{weight1.2}/{bias}/{weight2.1}_{weight2.2}
    def get_stringfied(self) -> str:
        result = ""
        if self.seed is not None:
            result += f'seed/{self.seed}\n'
        for layer_index, layer in enumerate(self.layers):
            result += f'{ANNProperties.activation_to_str(layer.get_single_activation())}'
            for neuron_index, neuron in enumerate(layer.neuronList):
                result += f'/{neuron.bias}/'
                for weight_index, weight in enumerate(neuron.inputWeights):
                    result += f'{weight}'
                    if weight_index != len(neuron.inputWeights) - 1:
                        result += '_'

            if layer_index != len(self.layers) - 1:
                result += '\n'
        return result

    @classmethod
    def get_property_from_stringfied(cls, stringfied: str):
        lines = stringfied.split('\n')
        seed: Optional[int] = None
        layers = list()
        if lines[0][:4] == "seed":
            seed = int(lines[0][5:])
            lines.pop(0)
        for line in lines:
            line_parts = line.split("/")
            activation = ANNProperties.str_to_activation(line_parts[0])
            layer = Layer()
            for neuron_index in range(1, len(line_parts) - 1, + 2):
                bias = float(line_parts[neuron_index])
                weights = line_parts[neuron_index + 1].split("_")
                weights_float = [float(item) for item in weights]
                layer.neuronList.append(Neuron(activation, weights_float, bias))
            layers.append(layer)
        return ANNProperties(layers, seed)

    @staticmethod
    def activation_to_str(activation: ActivationFunction) -> str:
        return activation.get_name()

    @staticmethod
    def str_to_activation(text: str) -> ActivationFunction:
        all_functions = ANNProperties.get_all_activations()
        all_named = list(filter(lambda func: func.get_name() == text, all_functions))
        return all_named[0]

    @staticmethod
    def get_all_activations() -> List:
        result = list()
        result.append(ActivationReLU.get_instance())
        result.append(ActivationSigmoid.get_instance())
        result.append(ActivationSoftmax.get_instance())
        result.append(ActivationTanH.get_instance())
        return result
