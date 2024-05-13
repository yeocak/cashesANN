import numpy as np


class CrossEntropy:
    @staticmethod
    def execute(predictions: np.array, targets: np.array) -> float:
        epsilon = 1e-12
        better_predictions = np.clip(predictions, epsilon, 1. - epsilon)
        neuron_count = len(better_predictions)

        return -np.sum(targets * np.log(better_predictions)) / neuron_count
