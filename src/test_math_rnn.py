import random
from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.core.defchararray import isnumeric

from src.ann.activation.ActivationReLU import ActivationReLU
from src.ann.activation.ActivationSigmoid import ActivationSigmoid
from src.ann.activation.ActivationSoftmax import ActivationSoftmax
from src.ann.activation.ActivationTanH import ActivationTanH
from src.ann.options.LayerOption import LayerOption
from src.ann.properties.ANNProperties import ANNProperties
from src.rnn.RecurrentNN import RecurrentNN
from src.utils.FileUtils import FileUtils

seed = 0


def read_math_data():
    df = pd.read_csv("assets/selected_veri.csv")

    random.seed(seed)
    df_shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_shuffle


def test_math_data():
    activation_sigmoid = ActivationSigmoid.get_instance()
    activation_relu = ActivationReLU.get_instance()
    activation_softmax = ActivationSoftmax.get_instance()
    activation_tanh = ActivationTanH.get_instance()

    rnn = RecurrentNN.create_random(seed, 1, [
        LayerOption(seed, 4, activation_relu),
        LayerOption(seed, 4, activation_relu),
        LayerOption(seed, 1, activation_relu),
    ])

    difference_list = list()
    epoch = 100
    df = read_math_data()
    for iteration in range(epoch):
        for index, row in df.head(12000).iterrows():
            inputs = np.array([[float(row["x1"])], [float(row["x2"])], [float(row["x3"])], [float(row["x4"])]])
            forward_result = rnn.execute_forward(inputs)
            # print(forward_result)
            backward_result = rnn.execute_backward(forward_result, [float(row["y"])], 0.001)
            # print(backward_result)

            if index % 1000 == 0:
                print(iteration + 1,
                      "\t-\ttraining\t|\tinputs: ", forward_result.input_list,
                      "\t|\tpredict: ", forward_result.a_list[-1][0],
                      "\t|\ttrue value: ", float(row["y"]),
                      "\t|\tdifference: ", forward_result.a_list[-1][0] - float(row["y"]))
                difference_list.append(abs(forward_result.a_list[-1][0] - float(row["y"])))

        rnn_model = rnn.get_properties_as_str(seed)
        FileUtils.write_to_txt(f"rnn_a_epoch_{iteration + 1}", rnn_model)

    plt.scatter(range(len(difference_list)), difference_list, color='red', label='Loss')
    plt.title('Loss Graph')
    plt.ylim(0, 0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_input_data():
    numbers_string = input("Write your numbers (separate by space):")
    numbers = [[float(number_string)] for number_string in numbers_string.split(" ")]
    rnn_stringfied = FileUtils.read_from_txt(f"rnn_a_epoch_20")
    rnn_property = ANNProperties.get_property_from_stringfied(rnn_stringfied)
    rnn = RecurrentNN.create_from_properties(rnn_property)
    test_result = rnn.execute_forward(numbers)
    print(test_result.a_list[-1])

if __name__ == '__main__':
    #test_math_data()
    test_input_data()
