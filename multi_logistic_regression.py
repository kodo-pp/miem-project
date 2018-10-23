# -*- coding: utf-8 -*-

from linear_regression import LinearRegression

class MultiLogisticRegression:
    def __init__(self, labels, ins):
        self.lrs = {label: LinearRegression(ins) for label in labels}

    def at(self, label):
        return self.lrs[label]

    def get(self, x):
        predicted = {label: self.lrs[label].get(x) for label in self.lrs.keys()}
        predicted_tuples = list(reversed(sorted([(j, i) for i, j in predicted.items()])))
        return max(predicted_tuples)

    def getlabel(self, x, treshold=0.5):
        q, label = self.get(x)
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

# TEST
mlr = MultiLogisticRegression(['foo', 'bar', 'baz'], 2)
mlr.learn([
    ([4, 7], 'foo'),
    ([2, 4], 'baz'),
    ([8, 1], 'foo'),
    ([2, 6], 'foo'),
    ([6, 8], 'bar'),
    ([2, 2], 'baz'),
    ([7, 3], 'foo'),
])
print(mlr.get([4, 7]))
print(mlr.get([2, 4]))
print(mlr.get([8, 1]))
print(mlr.get([2, 6]))

print(mlr.getlabel([4, 7]))
print(mlr.getlabel([2, 4]))
print(mlr.getlabel([8, 1]))
print(mlr.getlabel([2, 6]))
