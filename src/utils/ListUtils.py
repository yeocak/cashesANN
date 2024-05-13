from typing import List

import numpy as np


class ListUtils:
    @staticmethod
    def convert_to_2d_array(list: List) -> np.ndarray:
        vertical_length = len(list)
        horizontal_length = 0
        for items in list:
            if len(items) > horizontal_length:
                horizontal_length = len(items)
        result = np.ndarray((vertical_length, horizontal_length))
        for vertical_index in range(vertical_length):
            for horizontal_index in range(horizontal_length):
                if len(list[vertical_index]) <= horizontal_index:
                    result[vertical_index][horizontal_index] = None
                else:
                    result[vertical_index][horizontal_index] = list[vertical_index][horizontal_index]

        return result
