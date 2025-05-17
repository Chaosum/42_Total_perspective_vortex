from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class MyLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.1, max_iter=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.theta_ = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X_bias = np.hstack((np.ones((n_samples, 1)), X))  # Ajouter biais (x0 = 1)
        self.theta_ = np.zeros(X_bias.shape[1])

        for _ in range(self.max_iter):
            z = X_bias @ self.theta_
            y_hat = self.sigmoid(z)
            gradient = (X_bias.T @ (y_hat - y)) / n_samples

            old_theta = self.theta_.copy()
            self.theta_ -= self.learning_rate * gradient

            if np.linalg.norm(self.theta_ - old_theta) < self.tolerance:
                break

        return self

    def predict_proba(self, X):
        n_samples = X.shape[0]
        X_bias = np.hstack((np.ones((n_samples, 1)), X))
        z = X_bias @ self.theta_
        return self.sigmoid(z)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
