#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import fmt
from util import numeric_gradient
from linear_regression import LinearRegression

def check_gradient(reg, dataset, weights):
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

def main():

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

    # Проверяем работу
    print(fmt.format('Checking cost function before learning', 'bold'))
    check_gradient(reg, dataset, weights)
    print()

    # Учимся воспринимать этот датасет
    reg.learn(dataset)

    # И повторяем
    print(fmt.format('Checking cost function after learning', 'bold'))
    check_gradient(reg, dataset, reg.weights)

if __name__ == '__main__':
    main()
