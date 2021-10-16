import numpy as np


class Metrics:
    @staticmethod
    def accuracy(y: np.array, y_pred: np.array) -> float:
        return y[y == y_pred].shape[0] / y.shape[0]

    @staticmethod
    def mse(y, y_pred):
        return (sum((y - y_pred) ** 2)) / len(y)

    @staticmethod
    def e_metrics(v1, v2):
        return np.linalg.norm(v1 - v2, ord=2)
