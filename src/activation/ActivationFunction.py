from abc import abstractmethod
from typing import List

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
