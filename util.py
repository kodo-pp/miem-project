# -*- coding: utf-8 -*-
import math

import numpy as np
import random as rd

import config as conf

def random_matrix(h, w):
    """
    Сгенерировать матрицу размером h × w со случайными значениями от 0 до 1
    """
    return np.array([[rd.random() - 0.5 for i in range(w)] for i in range(h)], dtype=float)

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

def scale_array(arr, n):
    eps = 1e-7
    m = len(arr)
    # arr = list(_arr) + [_arr[-1]]
    blen = m / n
    offset = 0
    new_arr = []
    for i in range(n):
        # bi = math.ceil(offset)
        # ei = math.floor(offset + blen)
        xi = math.ceil(offset + blen - eps) - 1
        yi = math.floor(offset + eps) + 1
        if xi < yi:
            accum = blen * arr[math.floor(offset)]
        elif xi == yi:
            accum = (yi - offset) * arr[math.floor(offset)] + (offset + blen - xi) * arr[xi]
        else:
            accum = (yi - offset) * arr[math.floor(offset)] + (offset + blen - xi) * arr[xi] + sum(arr[yi:xi])
        new_arr.append(accum / blen)
        offset += blen
    return new_arr

def sigmoid(z):
    return 1 / (1 + math.exp(-z))
