import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.ann.ArtificialNN import ArtificialNN
from src.ann.options.LayerOption import LayerOption
from src.ann.activation.ActivationReLU import ActivationReLU
from src.ann.activation.ActivationSigmoid import ActivationSigmoid

seed = 0


def read_math_data():
    df = pd.read_csv("assets/selected_veri.csv")

    random.seed(seed)
    df_shuffle = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    return df_shuffle


def test_math_data():
    activation_sigmoid = ActivationSigmoid.get_instance()
    activation_relu = ActivationReLU.get_instance()

    ann = ArtificialNN.create_random(seed, 4, [
        LayerOption(seed, 4, activation_relu),
        LayerOption(seed, 1, activation_relu),
    ])

    difference_list = list()
    epoch = 4
    df = read_math_data()
    for iteration in range(epoch):
        for index, row in df.head(12000).iterrows():
            inputs = np.array([float(row["x1"]), float(row["x2"]), float(row["x3"]), float(row["x4"])])
            forward_result = ann.execute_forward(inputs)
            # print(forward_result)
            backward_result = ann.execute_backward(forward_result, [float(row["y"])], 0.01)
            # print(backward_result)

            if index % 25 == 0:
                print(iteration + 1,
                      "\t-\ttraining\t|\tinputs: ", forward_result.input_list,
                      "\t|\tpredict: ", forward_result.a_list[-1][0],
                      "\t|\ttrue value: ", float(row["y"]),
                      "\t|\tdifference: ", forward_result.a_list[-1][0] - float(row["y"]))
                difference_list.append(abs(forward_result.a_list[-1][0] - float(row["y"])))

    plt.scatter(range(len(difference_list)), difference_list, color='red', label='Loss')
    plt.title('Loss Graph')
    plt.ylim(0, 0.5)
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    test_math_data()
