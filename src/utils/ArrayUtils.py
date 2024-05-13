import numpy as np


class ArrayUtils:

    @staticmethod
    def eliminate_nan(array: np.array) -> np.array:
        return array[~np.isnan(array)]
