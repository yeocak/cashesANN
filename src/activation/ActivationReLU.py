from typing import List

import numpy as np

from src.activation.ActivationFunction import ActivationFunction


class ActivationReLU(ActivationFunction):
    _instance = None

    def __init__(self):
        raise RuntimeError('Use ActivationReLU.get_instance() to instantiate.')

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def execute(self, inputf: float) -> float:
        return max(0.0, inputf)

    def execute_derivative(self, inputf: float) -> float:
        return inputf > 0

    def execute_list(self, inputs: np.array) -> np.array:
        result = []
        for single_input in inputs:
            result.append(self.execute(single_input))
        return result

    def execute_derivative_list(self, inputs: np.array) -> np.array:
        result = []
        for single_input in inputs:
            result.append(self.execute_derivative(single_input))
        return result
