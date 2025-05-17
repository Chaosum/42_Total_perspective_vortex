from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.1, max_iter=1000, tolerance=1e-8):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.theta_ = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.c_[np.ones(X.shape[0]), X]  # Ajout biais x₀ = 1
        self.thetas = np.zeros(X.shape[1])

        for epoch in range(self.max_iter):
            for i in np.random.permutation(len(X)):
                x_i = X[i]
                y_hat = self.sigmoid(np.dot(self.thetas, x_i))
                error = y_hat - y[i]
                gradient = error * x_i
                self.thetas -= self.learning_rate * gradient
        return self

    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        probs = self.sigmoid(np.dot(X, self.thetas))
        return (probs >= 0.5).astype(int)
