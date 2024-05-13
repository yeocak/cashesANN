from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.ann.activation.ActivationReLU import ActivationReLU
from src.ann.activation.ActivationSigmoid import ActivationSigmoid
from src.ann.activation.ActivationSoftmax import ActivationSoftmax
from src.ann.activation.ActivationTanH import ActivationTanH
from src.ann.options.LayerOption import LayerOption
from src.rnn.RecurrentNN import RecurrentNN
from src.utils.FileUtils import FileUtils

seed = 0


def read_movie_review_data(count: int):
    df = pd.read_csv('assets/processed_reviews.csv').head(count)
    return df


def get_word_ascii_array(word: str) -> np.array:
    result = np.empty(len(word))
    for char_index, char in enumerate(word):
        result[char_index] = (float(ord(char)) / 100) - 1.5
    return result


def get_review_ascii_array(review: str) -> List[np.array]:
    text = review.split()
    result = list()
    for word_index, word in enumerate(text):
        word_ascii = get_word_ascii_array(word)
        result.append(word_ascii)
    return result


def one_hot_encode(num):
    encoded = np.zeros(10)

    if 1 <= num <= 10:
        encoded[num - 1] = 1.0
    else:
        raise ValueError("should between 1-10")

    return encoded


def calculate_optimization(predictions: np.array, true_values: np.array):
    predicted_class = np.argmax(predictions)

    # Confusion Matrix (Karmaşıklık Matrisi) oluşturma
    tp = 1 if true_values[predicted_class] == 1 else 0  # True Positive
    fp = 1 if true_values[predicted_class] == 0 else 0  # False Positive
    tn = np.sum(true_values) - tp  # True Negative: Diğer tüm sınıfların toplamı (true_label'deki 1'lerin sayısı) - TP
    fn = 1 - tp  # False Negative: Gerçek değer 1, ancak tahmin edilen sınıf farklı

    accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return [accuracy, precision, recall]


def calculate_optimization_basic(predictions: np.array, true_value: int):
    prediction_ratio = predictions[true_value - 1]

    return 1 - prediction_ratio


def test_math_data():
    activation_sigmoid = ActivationSigmoid.get_instance()
    activation_relu = ActivationReLU.get_instance()
    activation_softmax = ActivationSoftmax.get_instance()
    activation_tanh = ActivationTanH.get_instance()

    max_word_length = 20

    rnn = RecurrentNN.create_random(seed, max_word_length, [
        LayerOption(seed, 20, activation_tanh),
        LayerOption(seed, 10, activation_tanh),
        LayerOption(seed, 1, activation_tanh),
    ])

    difference_list = list()
    plt.ion()

    epoch = 10
    print_frequance = 500
    number_of_data = 20000
    df = read_movie_review_data(number_of_data)
    for iteration in range(epoch):
        optimization_sum = 0
        for index, row in df.iterrows():
            true_rating = float(int(row['rating'])) / 10
            true_review = str(row['review'])
            true_title = str(row['title'])

            raw_input = true_title + " " + true_review
            ascii_review_sentences = get_review_ascii_array(raw_input)
            forward_result = rnn.execute_forward(ascii_review_sentences)
            # print(forward_result)
            #true_result_one_hot = one_hot_encode(true_rating)
            backward_result = rnn.execute_backward(forward_result, [true_rating], 0.0001)

            #optimization_sum += calculate_optimization_basic(forward_result.a_list[-1], true_rating)
            optimization_sum += abs(true_rating-forward_result.a_list[-1][0])
            #print("true value:\t", true_rating, "\tpredictions:\t", np.array(forward_result.a_list[-1]))
            if index > print_frequance and index % print_frequance == 0:
               loss = float(optimization_sum) / print_frequance
               optimization_sum = 0
               difference_list.append(loss)
               plt.plot(difference_list)
               plt.pause(0.001)
               plt.clf()
               print("Epoch: ", iteration, "\tLoss: ", loss)

        rnn_model = rnn.get_properties_as_str(seed)
        FileUtils.write_to_txt(f"rnn_epoch_{iteration + 1}", rnn_model)
        #print("Epoch: ", iteration + 1, "\tLoss: ", loss)

    # plt.show()


if __name__ == '__main__':
    test_math_data()
