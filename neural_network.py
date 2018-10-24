# -*- coding: utf-8 -*-

import pickle
import math
from base64 import b64encode, b64decode

import numpy as np
from scipy.optimize import minimize

import config as conf
from util import random_matrix, sigmoid, sigmoid_gradient

class NeuralNetwork:
    def __init__(self, ins, hids, outs):
        self.ins = ins
        self.hids = hids
        self.outs = outs
        self.weights = [random_matrix(ins + 1, hids), random_matrix(hids + 1, outs)]

        self.regularization_lambda = conf.regularization

    def get(self, inputs, weights=None):
        if weights is None:
            weights = self.weights

        # Получаем входные данные в форме, удобной для NumPy и добавляем неявную единицу
        input_values = np.array([[1] + inputs], dtype=float)

        # Умножаем матрицы, и получаем значения скрытых нейронов
        hidden_values_z = np.matmul(input_values, weights[0])[0]
        # Теперь применяем к ним сигмоид и добавляем единицу
        hidden_values_a = np.array([1] + list(sigmoid(hidden_values_z)), dtype=float)

        # Считаем значения на выходе
        output_values_z = np.matmul(hidden_values_a, weights[1])
        # Применяем к ним сигмоид и добавляем единицу
        output_values_a = sigmoid(output_values_z)

        return list(output_values_a)

    def set_weights(self, weights):
        dim = self.weights[0].shape
        w1a = weights[:dim[0]*dim[1]]
        w2a = weights[dim[0]*dim[1]:]
        w1 = np.reshape(np.array())
        self.weights = [np.array(w1, dtype=float), np.array(w2, dtype=float)]

    def import_weights(self, s):
        base_encoded_w1, base_encoded_w2 = json.loads(s)
        encoded_w1 = b64decode(base_encoded_w1)
        encoded_w2 = b64decode(base_encoded_w2)
        self.weights = [pickle.loads(encoded_w1), pickle.loads(encoded_w2)]

    def export_weights(self):
        w1, w2 = self.weights
        encoded_w1 = w1.dumps()
        encoded_w2 = w2.dumps()
        base_encoded_w1 = b64encode(encoded_w1)
        base_encoded_w2 = b64encode(encoded_w2)
        return json.dumps([base_encoded_w1, base_encoded_w2])

    def cost(self, dataset, weights=None):
        m = len(dataset)

        J_main = 0
        J_regularization = 0

        Delta = [0, 0]

        # grad = np.zeros(len(self.unroll_weights()))
        gradient = [np.zeros(self.weights[0].shape), np.zeros(self.weights[0].shape)]

        for xi, yi in dataset:
            K = len(yi)
            if weights is None:
                weights = self.weights

            # Получаем входные данные в форме, удобной для NumPy и добавляем неявную единицу
            input_values = np.array([[1] + xi], dtype=float)

            # Умножаем матрицы, и получаем значения скрытых нейронов
            hidden_values_z = np.matmul(input_values, weights[0])[0]
            # Теперь применяем к ним сигмоид и добавляем единицу
            hidden_values_a = np.array([1] + list(sigmoid(hidden_values_z)), dtype=float)

            # Считаем значения на выходе
            output_values_z = np.matmul(hidden_values_a, weights[1])
            # Применяем к ним сигмоид и добавляем единицу
            output_values_a = sigmoid(output_values_z)
            pred = output_values_a

            # Main term
            for k in range(K):
                J_main += -yi[k] * math.log(pred[k]) - (1 - yi[k]) * math.log(1 - pred[k])

            output_values_d = output_values_a - yi
            hidden_values_d = np.matmul(weights[1], output_values_d)[1:] * sigmoid_gradient(hidden_values_z)
            hidden_values_d = hidden_values_d[1:]
            Delta[1] += np.matmul(hidden_values_a, output_values_d.transpose())
            Delta[0] += np.matmul(input_values_a, hidden_values_d.transpose())

        w0 = np.array(self.weights[0])
        w1 = np.array(self.weights[1])
        dim0 = w0.shape
        dim1 = w1.shape
        for j in range(dim0[1]):
            w0[0][j] = 0
        for j in range(dim1[1]):
            w1[0][j] = 0
        gradient[0] = (Delta[0] + conf.regularization * w0) / m;
        gradient[1] = (Delta[1] + conf.regularization * w1) / m;

        # Regularization term
        for w in self.unroll_nonbias_weights():
            J_regularization += w**2

        J = 1 / m * J_main + conf.regularization / (2*m) * J_regularization
        grad = self.unroll_weights(gradient)
        return J, grad

    def learn(self, dataset):
        print()
        func = lambda weights: self.cost(dataset, weights)
        # grad_func = lambda weights: np.array(self.cost(dataset, weights)[1])
        # self.set_weights(minimize(func, np.array(self.weights), jac=grad_func, method='CG').x)
        self.set_weights(minimize(func, np.array(self.weights), jac=True, method='CG', options={'maxiter':conf.fmin_max_iter}).x)

    def unroll_weights(self, weights=None):
        def func(weights):
            for i in weights:
                for j in i:
                    for k in j:
                        yield k
        return list(func(self.weights if weights is None else weights))

    def unroll_nonbias_weights(self, weights=None):
        def func(weights):
            for i in weights:
                for j in i[1:]:
                    for k in j:
                        yield k
        return list(func(self.weights if weights is None else weights))
