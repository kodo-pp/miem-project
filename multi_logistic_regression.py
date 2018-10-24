# -*- coding: utf-8 -*-

import json

from linear_regression import LinearRegression

class MultiLogisticRegression:
    def __init__(self, labels, ins):
        self.lrs = {label: LinearRegression(ins) for label in labels}

    def at(self, label):
        return self.lrs[label]

    def get(self, x, debug=False):
        predicted = {label: self.lrs[label].get(x) for label in self.lrs.keys()}
        predicted_tuples = list(reversed(sorted([(j, i) for i, j in predicted.items()])))
        if debug:
            print('MLR.get(): predicted_tuples = {}'.format(predicted_tuples))
        return max(predicted_tuples)

    def getlabel(self, x, treshold=0.0, debug=False):
        q, label = self.get(x, debug=debug)
        if q < treshold:
            return None
        return label

    def learn(self, multidataset):
        datasets = {label: list(multidataset) for label in self.lrs.keys()}
        for label in self.lrs.keys():
            for i, example in enumerate(datasets[label]):
                datasets[label][i] = example[0], (1 if example[1] == label else 0)

        for i, (label, dset) in enumerate(datasets.items()):
            print('Training: {} / {}'.format(i + 1, len(datasets)))
            self.lrs[label].learn(dset)

    def export_weights(self):
        return json.dumps({i: list(j.weights) for i, j in self.lrs.items()})

    def import_weights(self, s):
        weights = json.loads(s)
        for i, j in weights.items():
            self.lrs[i].set_weights(j)
