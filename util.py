# -*- coding: utf-8 -*-

import numpy as np
import random as rd

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
