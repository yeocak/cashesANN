from typing import List, Optional

import numpy as np

from src.ann.ArtificialNN import ArtificialNN
from src.ann.options.ForwardResult import ForwardResult
from src.ann.options.LayerOption import LayerOption
from src.ann.properties.ANNProperties import ANNProperties
from src.utils.ArrayUtils import ArrayUtils


class RecurrentNN:
    def __init__(self, ann: ArtificialNN):
        self.ann = ann

    def execute_forward(self, inputs: List[np.array]) -> ForwardResult:
        last_result: ForwardResult
        for pure_input_index, pure_input in enumerate(inputs):
            if pure_input_index == 0:
                fill_count = self.ann.get_input_count() - len(inputs[0])
                first_input = np.concatenate((inputs[0], np.zeros(fill_count)))
                last_result = self.ann.execute_forward(first_input)
            else:
                last_output = ArrayUtils.eliminate_nan(last_result.a_list[-1])
                fill_count = self.ann.get_input_count() - len(inputs[0]) - len(last_output)
                word_input = np.concatenate((inputs[0], np.zeros(fill_count)))
                new_input = np.concatenate((word_input, last_output))
                last_result = self.ann.execute_forward(new_input)

        return last_result

    def execute_backward(self, forward_result: ForwardResult, true_values: np.array, step_alpha: float) -> np.array:
        return self.ann.execute_backward(forward_result, true_values, step_alpha)

    @classmethod
    def create_random(cls, seed: int, input_number: int, layers: List[LayerOption]):
        # input number is in RNN = output number + data input length
        input_number = input_number + layers[-1].neuronNumber
        result = ArtificialNN.create_random(seed, input_number, layers)
        return cls(result)

    def get_properties(self, seed: Optional[int]) -> ANNProperties:
        return ANNProperties(self.ann.layers, seed)

    def get_properties_as_str(self, seed: Optional[int]) -> str:
        return ANNProperties.get_stringfied(self.get_properties(seed))
