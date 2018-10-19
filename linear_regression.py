# -*- coding: utf-8 -*-

from util import random_matrix
import numpy as np
from scipy.optimize import minimize

# Наша линейная регресия
class LinearRegression:
    """
    Линейная регрессия
    """
    def __init__(self, ins):
        """
        Конструктор класса
        Параметры:
          ins - количество входов (не считая неявного входа со значением 1)
        """
        self.ins = ins
        self.weights = random_matrix(ins + 1, 1)

        self.regularization_lambda = 0.1

    def get(self, inputs, weights = None):
        """
        Предсказать значение при данных входных данных
        Параметры:
          inputs - входные данные (без неявного входа со значением 1)
          weights - (опционально) использовать данные веса вместо сохранённых
        """
        if weights is None:
            weights = self.weights

        # Получаем входные данные в форме, удобной для NumPy и добавляем неявную единицу
        input_values = np.array([1] + inputs, dtype=float).transpose()

        # Одно простае умножение матриц, и ответ уже у нас
        output_value = np.matmul(input_values, weights)
        return output_value

    def set_weights(self, weights):
        """
        Сохранить заданные веса
        Параметры:
          weights - веса. Список длиной ins + 1, где ins - количество входов (см. __init__)
        """
        self.weights = np.array(weights, dtype=float)

    def cost(self, dataset, weights = None):
        """
        Посчитать функцию стоимости при данных весах и её градиент
        Попозже приведу формулы, по которым это считается
        Параметры:
          dataset - данные, на которых надо оценить точность предсказания
          weights - веса
        """
        if weights is None:
            weights = self.weights

        # m - количество примеров, n - количество входов и весов (включая неявную единицу)
        m = len(dataset)
        n = len(weights)

        # Дальше чёрная магия
        base_cost = 0
        gradient = np.zeros(n, dtype=float)
        for x, y in dataset:
            predicted_y = self.get(x, weights)
            error = predicted_y - y
            base_cost += error ** 2
            tx = [1] + x
            for j in range(n):
                gradient[j] += error * tx[j]

        # Больше чёрной магии: регуляризация
        regularization = 0
        for j in range(1, n):
            regularization += weights[j] ** 2
            gradient[j] += self.regularization_lambda * weights[j]

        gradient /= m
        cost = 1 / (2*m) * base_cost + self.regularization_lambda / (2*m) * regularization
        # Конец чёрной магии

        print('\x1b[A\x1b[2K' 'cost = {:.10f}'.format(cost))
        return cost, gradient

    def learn(self, dataset):
        print()
        func = lambda weights: self.cost(dataset, weights)
        # grad_func = lambda weights: np.array(self.cost(dataset, weights)[1])
        # self.set_weights(minimize(func, np.array(self.weights), jac=grad_func, method='CG').x)
        self.set_weights(minimize(func, np.array(self.weights), jac=True, method='CG', options={'maxiter':2000}).x)
