from typing import List

import numpy as np

from src.activation.ActivationFunction import ActivationFunction


class ActivationSoftmax(ActivationFunction):
    _instance = None

    def __init__(self):
        raise RuntimeError('Use ActivationReLU.get_instance() to instantiate.')

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def execute(self, inputs: np.array) -> np.array:
        return None

    def execute_derivative(self, inputs: np.array) -> np.array:
        return None

    def execute_list(self, inputs: np.array) -> np.array:
        e = np.exp(inputs)
        return e / np.sum(e)

    def execute_derivative_list(self, inputs: np.array) -> np.array:
        n = len(inputs)
        d = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    d[i, j] = inputs[i] * (1 - inputs[i])
                else:
                    d[i, j] = -inputs[i] * inputs[j]

        return d
