#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
from argparse import ArgumentParser

import fmt
import config as conf
from sliding_diff import sliding_diff
from audio import AudioRecorder
from util import numeric_gradient
from linear_regression import LinearRegression
from audio_analyzer import AudioAnalyzer

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
    if '--learn' in sys.argv:
        learn()
        return
    with AudioRecorder(rate=conf.sampling_rate) as recorder:
        analyzer = AudioAnalyzer(recorder)
        analyzer.analyze()

def learn():
    lr = LinearRegression(conf.lr_inputs)
    dataset = []
    while True:
        print('Type? [1/0/q]: ', end='')
        t = input()
        if t == '0':
            y = -1
        elif t == '1':
            y = 1
        else:
            break

        with AudioRecorder(rate=conf.sampling_rate) as recorder:
            print('--- RECORDING ---')
            data = list(sliding_diff(
                list(recorder.bytes_to_numseq(recorder.record(conf.block_size))),
                conf.sliding_diff_winsize
            ))
            print('--- FINISHED ---')
            dataset.append((data, y))
    print('Learning...')
    lr.learn(dataset)
    print('Learnt')
    with open('learnt.txt', 'w') as wf:
        wf.write(' '.join(map(str, lr.weights)))
    print('Weigts written to learnt.txt')

if __name__ == '__main__':
    main()
