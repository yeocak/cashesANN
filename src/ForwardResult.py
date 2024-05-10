import numpy as np


class ForwardResult:

    def __str__(self):
        return (f'input_list:   {self.input_list}\n'
                f'a_list:       {self.a_list}\n'
                f'z_list:       {self.z_list}')

    # z_list is (∑wx)+b
    # a_list is g(zList) (could be any activation function)
    def __init__(self, input_list: np.array, z_list: np.ndarray, a_list: np.ndarray):
        self.input_list = input_list
        self.a_list = a_list
        self.z_list = z_list

    def get_output_layer_z(self) -> np.array:
        return self.z_list[-1]

    def get_output_layer_a(self) -> np.array:
        return self.a_list[-1]
