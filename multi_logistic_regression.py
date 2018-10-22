# -*- coding: utf-8 -*-

from linear_regression import LinearRegression

class MultiLogisticRegression:
    def __init__(self, labels):
        self.lrs = {label: LinearRegression for label in labels}

    def at(self, label):
        return self.lrs[label]

    def get(self, x):
        predicted = {label: self.lrs[label].get(x) for label in labels}
        predicted_tuples = reversed(sorted([(j, i) for i, j in predicted.items()]))
        return predicted_tuples[0][0]

    def learn(self, multidataset):
        datasets = {label: list(multidataset) for label in self.lrs.keys()}
        for label in self.lrs.keys():
            for i, example in enumerate(datasets[label]):
                datasets[label][i] = example[0], (1 if example[1] == label else 0)
        print(datasets)

# TEST
mlr = MultiLogisticRegression(['foo', 'bar', 'baz'])
mlr.learn([
    ([4, 7], 'foo'),
    ([2, 4], 'baz'),
    ([8, 1], 'foo'),
    ([2, 6], 'foo'),
    ([6, 8], 'bar'),
    ([2, 2], 'baz'),
    ([7, 3], 'foo'),
])
