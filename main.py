#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import gzip
import json
import hashlib
import math
import random as rd

import numpy as np

import fmt
import config as conf
from sliding_diff import sliding_diff
from audio import AudioRecorder
from util import numeric_gradient, scale_array
from svm import Svm

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
    elif command == 'set-labels':
        set_labels()
    elif command == 'collect':
        collect()
    elif command == 'transform':
        transform()
    elif command == 'recognize':
        recognize()
    else:
        print('Usage: {} {{set-labels | collect | transform | learn | recognize}}'.format(sys.argv[0]))
        return

def set_labels():
    print('Enter the space-separated label list: ')
    labels = input()
    with open('labels.txt', 'w') as f:
        f.write(labels)
    print('Labels written to labels.txt')

def get_labels():
    with open('labels.txt') as f:
        return f.read().split()

def recognize():
    with AudioRecorder(rate=conf.sampling_rate) as recorder:
        svm = Svm()
        with open('trained_data.pickle.b64') as f:
            svm.import_model(f.read())
        while True:
            print()
            print('\x1b[A\x1b[2K' 'Press Enter to start recording or Ctrl+C to exit')
            input()
            recorder.start_recording()
            print('\x1b[A\x1b[2K' '\x1b[31mREC\x1b[0m Press Enter to stop recording')
            input()
            data = list(recorder.bytes_to_numseq(recorder.finish_recording()))
            transformed_data = list(sliding_diff(data, conf.sliding_diff_winsize))
            scaled_data = scale_array(transformed_data, conf.svm_inputs)
            pred = svm.get(scaled_data)
            print('\x1b[A\x1b[2K' 'Predicted output: {}'.format(pred))

def transform():
    for data, y in read_examples('raw_data'):
        transformed_data = list(sliding_diff(data, conf.sliding_diff_winsize))
        scaled_data = scale_array(transformed_data, conf.svm_inputs)
        write_example((scaled_data, y), 'data')

def write_example(data, dirname):
    data_to_write = gzip.compress(json.dumps(data).encode())
    data_hash = hashlib.sha256(data_to_write).hexdigest()
    if not os.path.isdir(dirname):
        os.mkdir(dirname)
    filename = os.path.join(dirname, '{}.json.gz'.format(data_hash[:20]))
    with open(filename, 'wb') as f:
        f.write(data_to_write)

def read_examples(dirname):
    for basename in os.listdir(dirname):
        fullname = os.path.join(dirname, basename)
        with open(fullname, 'rb') as f:
            encoded = f.read()
        x, y = json.loads(gzip.decompress(encoded).decode())
        yield x, y

def collect():
    import readline
    labels = get_labels()
    with AudioRecorder(rate=conf.sampling_rate) as recorder:
        while True:
            y = input('Enter a label or press Enter to quit: ')
            if y == '':
                break
            elif y not in labels:
                print('No such label: {}'.format(y))
                continue

            print('--- RECORDING --- (Press Enter to stop recording)')
            recorder.start_recording()
            input()
            data = list(recorder.bytes_to_numseq(recorder.finish_recording()))
            print('--- FINISHED RECORDING ---')
            # transformed_data = list(sliding_diff(data, conf.sliding_diff_winsize))
            # scaled_data = scale_array(transformed_data, conf.lr_inputs)
            write_example((data, y), 'raw_data')

def learn():
    labels = get_labels()
    # mlr = MultiLogisticRegression(labels, conf.lr_inputs)
    svm = Svm()
    dataset = list(read_examples('data'))

    rd.shuffle(dataset)
    m = len(dataset)
    train_m = int(0.7 * m)
    train_set = dataset[:train_m]
    test_set = dataset[train_m:]
    test_m = len(test_set)
    print('Learning...')
    svm.learn(train_set)
    print('Learnt')


    correct = 0
    for x, y in train_set:
        pred = svm.get(x)
        if pred == y:
            correct += 1
    print('Results on train set: {:2f}% accuracy'.format(100 * correct / train_m))

    correct = 0
    for x, y in test_set:
        pred = svm.get(x)
        if pred == y:
            correct += 1
    print('Results on test set: {:2f}% accuracy'.format(100 * correct / test_m))
    with open('trained_data.pickle.b64', 'w') as wf:
        wf.write(svm.export_model())
    print('Weights written to trained_data.pickle.b64')



if __name__ == '__main__':
    main()
