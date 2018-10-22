#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import gzip
import json
import hashlib
import math

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
    if (len(sys.argv) < 2):
        print('Usage: {} {{collect | learn | recognize}}'.format(sys.argv[0]))
        return
    command = sys.argv[1]
    if command == 'learn':
        learn()
    elif command == 'collect':
        collect()
    elif command == 'recognize':
        recognize()
    else:
        print('Usage: {} {{collect | learn | recognize}}'.format(sys.argv[0]))
        return

def recognize():
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
            raw_pred = lr.get(scaled_data)
            pred = round(raw_pred)
            print('\x1b[A\x1b[2K' 'Predicted output: {:.3f} ({})'.format(raw_pred, pred))

def write_example(data):
    data_to_write = gzip.compress(json.dumps(data).encode())
    data_hash = hashlib.sha256(data_to_write).hexdigest()
    if not os.path.isdir('data/'):
        os.mkdir('data/')
    filename = 'data/{}.json.gz'.format(data_hash[:20])
    with open(filename, 'wb') as f:
        f.write(data_to_write)

def read_examples():
    for basename in os.listdir('data/'):
        fullname = 'data/' + basename
        with open(fullname, 'rb') as f:
            encoded = f.read()
        x, y = json.loads(gzip.decompress(encoded).decode())
        yield x, y

def collect():
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
            write_example((scaled_data, y))

def learn():
    lr = LinearRegression(conf.lr_inputs)
    dataset = list(read_examples())
    print('Learning...')
    lr.learn(dataset)
    print('Learnt')
    for x, y in dataset:
        pred = lr.get(x)
        if abs(y - pred) < 0.5:
            ok = '\x1b[32m(  OK  )\x1b[0m'
        else:
            ok = '\x1b[31m( FAIL )\x1b[0m'
        print('y = {}, predicted_y = {:.3f}   {}'.format(y, pred, ok))
    with open('learnt.txt', 'w') as wf:
        wf.write(' '.join(map(str, lr.weights)))
    print('Weigts written to learnt.txt')

if __name__ == '__main__':
    main()
