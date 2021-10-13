import numpy as np
from KNN import KNNClassifier
from metrics import Metrics
from sklearn import model_selection

from sklearn.datasets import load_iris
from standardization import standardization


def lose_information(eig_values: np.array, final_dimension: int):
    """
    Метод для подсчета потеряной информации после PCA преобразования

    :param eig_values: Вектор собственных значений
    :param final_dimension: Значение размерности после преобразования
    :return: Процент потеряной информации
    """
    eig_sum = sum(eig_values)
    return round(100 - sum([(i / eig_sum) * 100 for i in sorted(eig_values, reverse=True)][:final_dimension]), 2)


def pca(data: np.array, final_dimension: int = 2, method='svd'):
    """
    Метод для уменьшения размерности при помощи PCA

    :param data: Исходный массив данных
    :param final_dimension: Значение размерности после преобразования
    :param method: Метод используемый для преобразования. По-умалчанию svd
    :return: возвращает преобразованный массив данных и количество потеряной информации в %
    """
    if method == 'svd':
        u, s, vh = np.linalg.svd(data)
        w = vh[:, :final_dimension]
        return data.dot(w), lose_information(s, final_dimension)


if __name__ == '__main__':
    # (*) Написать свою рализацию метода главных компонент с помощью сингулярного разложения с использованием функции
    # numpy.linalg.svd()
    iris = load_iris()

    data = iris['data']
    target = iris['target']

    data = standardization(data)
    #
    data, lose_percent = pca(data)
    # print(f'I lose {lose_percent}% information')

    # (*) Обучить любую модель классификации на датасете IRIS до применения PCA и после него. Сравнить качество
    # классификации по отложенной выборке.

    X, y = load_iris(return_X_y=True)
    X_train_full, X_test_full, y_train_full, y_test_full = model_selection.train_test_split(X, y, test_size=0.2,
                                                                                            random_state=1)
    X_train_pca, X_test_pca, y_train_pca, y_test_pca = model_selection.train_test_split(data, y, test_size=0.2,
                                                                                        random_state=1)

    X_train = [X_train_full, X_train_pca]
    y_train = y_train_full
    X_test = [X_test_full, X_test_pca]
    y_test = y_test_full
    description = [' for full data', ' for PCA data']

    for i in range(len(X_train)):
        model = KNNClassifier()
        model.fit(X_train[i], y_train)
        print(f'Accuracy is {Metrics.accuracy(y_test, model.predict(X_test[i]))}' + description[i])
