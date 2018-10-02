#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random as rd
import json

import fmt

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

    def cost(self, dataset, weights):
        """
        Посчитать функцию стоимости при данных весах и её градиент
        Попозже приведу формулы, по которым это считается
        Параметры:
          dataset - данные, на которых надо оценить точность предсказания
          weights - веса
        """
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

        return cost, list(gradient)

def random_matrix(h, w):
    """
    Сгенерировать матрицу размером h × w со случайными значениями от 0 до 1
    """
    return np.array([[rd.random() for i in range(w)] for i in range(h)], dtype=float)

def numeric_gradient(func, args):
    """
    Численно посчитать градиент функции
    """

    # Заполним его начальным значением
    grad = [0] * len(args)

    # Выберем маленькое число
    eps = 1e-5

    # Поехали!
    for i in range(len(args)):
        # Хитрый трюк: скопировать список по значению, а не по ссылке
        current_args = list(args)

        # Считаем с args[i] + eps
        current_args[i] += eps
        y1 = func(current_args)

        # Считаем с args[i] - eps
        current_args[i] -= 2 * eps
        y2 = func(current_args)

        # Частная производная вычисляется (почти) по определению
        grad[i] = (y1 - y2) / (2 * eps)
    return grad

def main():
    print(fmt.format('Checking cost function', 'bold'))

    # Это наша линейная регрессия
    reg = LinearRegression(2)

    # Наш большой датасет
    dataset = [
        ([3, 1], 5),
        ([2, 3], 4),
        ([1, 1], 2),
    ]

    # Пока зададим веса вручную
    weights = [1.2, 6.1, -2.3]

    # Считаем функцию стоимости и её градиент при заданных весах
    cost, grad = reg.cost(dataset, weights)
    print('Cost function:  {}'.format(cost))
    print('Cost gradient:  {}'.format(grad))

    # Проверяем правильность градиента: считаем его численно
    num_grad = numeric_gradient(lambda x: reg.cost(dataset, x)[0], weights)
    print('Num. gradient:  {}'.format(num_grad))

    # Считаем средний квадрат разности полученных значений
    average_error_sq = np.average((np.array(num_grad) - np.array(grad)) ** 2)
    print('Avg. error sq:  {}'.format(average_error_sq))

    # Если он маленький, то всё верно
    if average_error_sq < 1e-3:
        print(fmt.format('Gradient seems to be correct', 'green'))
    else:
        print(fmt.format('Gradient seems to be incorrect', 'red'))

if __name__ == '__main__':
    main()
