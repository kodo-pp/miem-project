#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import numpy as np
from argparse import ArgumentParser

import fmt
import config as conf
from sliding_diff import sliding_diff
from audio import AudioRecorder
from util import numeric_gradient, scale_array
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
        lr = LinearRegression(conf.lr_inputs)
        lr.set_weights([float(i) for i in open('learnt.txt').read().split()])
        while True:
            print()
            print('\x1b[A\x1b[2K' 'Press Enter to start recording or Ctrl+C to exit')
            input()
            recorder.start_recording()
            print('\x1b[A\x1b[2K' '\x1b[31mREC\x1b[0m Press Enter to stop recording')
            input()
            data = list(recorder.bytes_to_numseq(recorder.finish_recording()))
            transformed_data = list(sliding_diff(data, conf.sliding_diff_winsize))
            scaled_data = scale_array(transformed_data, conf.lr_inputs)
            print('\x1b[A\x1b[2K' 'Predicted output: {}'.format(lr.get(scaled_data)))

def learn():
    lr = LinearRegression(conf.lr_inputs)
    dataset = []
    with AudioRecorder(rate=conf.sampling_rate) as recorder:
        while True:
            print('Type? [1/0/q]: ', end='')
            t = input()
            if t == '0':
                y = 0
            elif t == '1':
                y = 1
            else:
                break

            print('--- RECORDING --- (Press Enter to stop recording)')
            recorder.start_recording()
            input()
            data = list(recorder.bytes_to_numseq(recorder.finish_recording()))
            print('--- FINISHED RECORDING ---')
            transformed_data = list(sliding_diff(data, conf.sliding_diff_winsize))
            scaled_data = scale_array(transformed_data, conf.lr_inputs)
            dataset.append((scaled_data, y))
    print('Learning...')
    lr.learn(dataset)
    print('Learnt')
    for x, y in dataset:
        print('y = {}, predicted_y = {}'.format(y, lr.get(x)))
    with open('learnt.txt', 'w') as wf:
        wf.write(' '.join(map(str, lr.weights)))
    print('Weigts written to learnt.txt')

if __name__ == '__main__':
    main()
