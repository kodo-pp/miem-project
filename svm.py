# -*- coding: utf-8 -*-

from base64 import b64encode, b64decode

from sklearn import svm
import pickle

import config as conf

class Svm:
    def __init__(self):
        self.svc = svm.SVC(gamma='scale', C=conf.C)

    def learn(self, dataset):
        X, y = list(zip(*dataset))
        self.trained = self.svc.fit(X, y)

    def get(self, data):
        return self.trained.predict([data])

    def export_model(self):
        return b64encode(pickle.dumps(self.trained)).decode()

    def import_model(self, model):
        self.trained = pickle.loads(b64decode(model))
