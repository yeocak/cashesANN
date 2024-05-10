from typing import List

import numpy as np

from src.ArtificialNN import ArtificialNN
from src.Layer import Layer
from src.Neuron import Neuron
from src.activation.ActivationReLU import ActivationReLU
from src.activation.ActivationSoftmax import ActivationSoftmax


def test_forward_backward():
    activation = ActivationReLU.get_instance()

    ann = ArtificialNN(4)

    # hidden layer
    hidden_layer = Layer()
    hl_neuron_1 = Neuron.create_with_zero_inputs(activation, 4)
    hl_neuron_2 = Neuron.create_with_zero_inputs(activation, 4)
    hl_neuron_3 = Neuron.create_with_zero_inputs(activation, 4)
    hl_neuron_4 = Neuron.create_with_zero_inputs(activation, 4)
    hidden_layer.neuronList.append(hl_neuron_1)
    hidden_layer.neuronList.append(hl_neuron_2)
    hidden_layer.neuronList.append(hl_neuron_3)
    hidden_layer.neuronList.append(hl_neuron_4)
    ann.add_layer(hidden_layer)

    # output layer
    output_layer = Layer()
    ol_neuron_1 = Neuron.create_with_zero_inputs(activation, 4)
    ol_neuron_2 = Neuron.create_with_zero_inputs(activation, 4)
    ol_neuron_3 = Neuron.create_with_zero_inputs(activation, 4)
    ol_neuron_4 = Neuron.create_with_zero_inputs(activation, 4)
    output_layer.neuronList.append(ol_neuron_1)
    output_layer.neuronList.append(ol_neuron_2)
    output_layer.neuronList.append(ol_neuron_3)
    output_layer.neuronList.append(ol_neuron_4)
    ann.add_layer(output_layer)

    test_result_f = ann.execute_forward([3, 5, 8, 9])
    print(test_result_f)
    test_result_b = ann.execute_backward(test_result_f, [0.2, 0.5, 0.2, 0.1])
    for testing_b in test_result_b:
        print("-----")
        print(testing_b)
    print("-----")
    test_result_f_2 = ann.execute_forward([-3, -5, 8, 9])
    print(test_result_f_2)


if __name__ == '__main__':
    test_forward_backward()
    #test1 = [[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    #test2 = np.transpose([[1,2,3,4],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    #print(test2)
    #test3 = np.dot(test1,test2)
    #print(test3)
