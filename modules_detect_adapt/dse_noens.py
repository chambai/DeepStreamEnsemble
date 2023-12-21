import numpy as np

from joblib import dump, load
import os

# no ensemble
class noens:
    def __init__(self, base_classifier):
        self.ensemble_classifier = base_classifier
        self.correct_cnt = 0
        self.n_samples = 0
        self.base_classifier = base_classifier

    def load(self, dir, filename):
        is_loaded = False
        if os.path.exists(os.path.join(dir,filename)):
            self.ensemble_classifier = load(os.path.join(dir,filename))
            is_loaded = True
        return is_loaded


    def save(self, dir, filename):
        if not os.path.exists(dir):
            os.mkdir(dir)
        dump(self.ensemble_classifier, os.path.join(dir,filename))

    def fit(self, x, y):
        self.ensemble_classifier.fit(x,y)


    def partial_fit(self, x, y):
        self.ensemble_classifier.partial_fit(x, y)
        pass

    def predict(self, X):
        X = np.array([x for x in X])
        y_pred = self.ensemble_classifier.predict(X)
        y_pred = y_pred.astype(int)
        return y_pred

    def eval(self, y_pred, y_true):
        for p, t in zip(y_pred, y_true):
            if t == p:
                self.correct_cnt += 1
            self.n_samples += 1
        # self.ensemble_classifier.partial_fit(X, y)
        acc = self.correct_cnt / self.n_samples
        return acc

    def eval_window(self, y_pred, y_true):
        correct_cnt = 0
        n_samples = 0
        for p, t in zip(y_pred, y_true):
            if t == p:
                correct_cnt += 1
            n_samples += 1
        acc = correct_cnt / n_samples
        return acc
