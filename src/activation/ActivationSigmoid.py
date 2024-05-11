import numpy as np

from src.activation.ActivationFunction import ActivationFunction


class ActivationSigmoid(ActivationFunction):
    _instance = None

    def __init__(self):
        raise RuntimeError('Use ActivationReLU.get_instance() to instantiate.')

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def execute(self, inputf: float) -> float:
        return 1 / (1 + np.exp(-inputf))

    def execute_derivative(self, inputf: float) -> float:
        return self.execute(inputf) * (1 - self.execute(inputf))

    def execute_list(self, inputs: np.array) -> np.array:
        result = []
        for single_input in inputs:
            result.append(self.execute(single_input))
        return np.array(result)

    def execute_derivative_list(self, inputs: np.array) -> np.array:
        result = []
        for single_input in inputs:
            result.append(self.execute_derivative(single_input))
        return np.array(result)
