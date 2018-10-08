#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from argparse import ArgumentParser

import fmt
import audio
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
    count = 0
    time_quant = 0.2
    with audio.AudioRecorder() as recorder:
        while True:
            for i in recorder.bytes_to_numseq(recorder.record(seconds=time_quant)):
                count += 1
            print(count)

if __name__ == '__main__':
    main()