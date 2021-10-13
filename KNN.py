import numpy as np
from sklearn import model_selection
from sklearn.datasets import load_iris
from metrics import Metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class DimensionException(Exception):
    pass


class KNNClassifier:
    def __init__(self, count_of_nearest_neighbors=5, weights=None):
        """

        :param count_of_nearest_neighbors: Число ближайших точек обучающей выборки на основании которых принимается
        решение о принадлежности из к тому или иному классу
        :param weights: None - алгоритм не учитывает расстояние от наблюдения до обьекта обучающей выборки
        'num' - соответствует добавлению весов для соседей в зависимости от номера соседа,
        'dist' - соответствует добавление весов для соседей в зависимости от расстояния до соседа
        """
        self.count_of_nearest_neighbors = count_of_nearest_neighbors
        self.X = None
        self.y = None
        self.weights = weights

    def fit(self, X, y):
        """
        Метод для обучения модели
        :param X: матрица признаков
        :param y: вектор меток
        :return: None
        """
        self.X = X
        self.y = y

    def predict(self, observations):
        answers = []
        for x in observations:
            test_distances = []

            for i in range(len(self.X)):
                # расчет расстояния от классифицируемого объекта до
                # объекта обучающей выборки
                distance = Metrics.e_metrics(x, self.X[i])

                # Записываем в список значение расстояния и ответа на объекте обучающей выборки
                test_distances.append((distance, self.y[i]))

            # создаем словарь со всеми возможными классами
            classes = {class_item: 0 for class_item in set(self.y)}

            # Сортируем список и среди первых k элементов подсчитаем частоту появления разных классов
            if not self.weights:
                for d in sorted(test_distances)[0:self.count_of_nearest_neighbors]:
                    classes[d[1]] += 1

                # Записываем в список ответов наиболее часто встречающийся класс
            if self.weights == 'num':
                for i, d in enumerate(sorted(test_distances)[0:self.count_of_nearest_neighbors], start=1):
                    classes[d[1]] += 1 / i

            if self.weights == 'dist':
                for d in sorted(test_distances)[0:self.count_of_nearest_neighbors]:
                    classes[d[1]] += 1 / (d[0] + 0.1)

            answers.append(sorted(classes, key=classes.get)[-1])

        return answers

    def create_figure(self):
        """
        Метод для построения разделяющей плоскости с нанесением обучающей выборки
        :return:
        """
        if self.X[:, :2].shape[1] > 2:
            raise DimensionException("I can't create figure of it, because dimension more then 2")

        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#00AAFF'])
        cmap = ListedColormap(['red', 'green', 'blue'])
        h = .02

        # Расчет пределов графика
        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        # Получим предсказания для всех точек
        dots = self.predict(np.c_[xx.ravel(), yy.ravel()])

        # Построим график
        Z = np.array(dots).reshape(xx.shape)
        plt.figure(figsize=(7, 7))
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # Добавим на график обучающую выборку
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap=cmap)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.title(f"Трехклассовая kNN классификация при k = {self.count_of_nearest_neighbors}")
        plt.show()


if __name__ == '__main__':
    X, y = load_iris(return_X_y=True)
    X = X[:, :2]
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
    # model = KNNClassifier(count_of_nearest_neighbors=3)
    # model.fit(X_train, y_train)
    # model.create_figure()

    # Задание № 1 и 2
    model = KNNClassifier(count_of_nearest_neighbors=3, weights='num')
    model.fit(X_train, y_train)
    y_pred = model.predict(y_test)
    print(Metrics.accuracy(y_test, y_pred))
    model.create_figure()
