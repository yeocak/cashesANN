from abc import abstractmethod

import numpy as np


class ActivationFunction:
    @abstractmethod
    def execute(self, inputf: float) -> float:
        pass

    @abstractmethod
    def execute_derivative(self, inputf: float) -> float:
        pass

    @abstractmethod
    def execute_list(self, inputs: np.array) -> np.array:
        pass

    @abstractmethod
    def execute_derivative_list(self, inputs: np.array) -> np.array:
        pass

    @abstractmethod
    def get_name(self):
        pass
