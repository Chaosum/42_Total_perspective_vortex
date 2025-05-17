from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class MyCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4):
        self.n_components = n_components
        self.filters_ = None

    def compute_cov(self, epoch):
        cov = epoch @ epoch.T
        return cov / np.trace(cov)
    
    def fit(self, X, y=None):
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly two classes.")

        class_1 = X[y == classes[0]]
        class_2 = X[y == classes[1]]

        cov_1 = np.mean([self.compute_cov(e) for e in class_1], axis=0)
        cov_2 = np.mean([self.compute_cov(e) for e in class_2], axis=0)

        cov_total = cov_1 + cov_2
        eigvals, eigvecs = np.linalg.eigh(cov_total)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
        P = eigvecs @ D_inv_sqrt @ eigvecs.T

        S1 = P @ cov_1 @ P.T
        eigvals, eigvecs = np.linalg.eigh(S1)
        filters = eigvecs.T @ P

        self.filters_ = filters[:self.n_components]
        return self

    def transform(self, X, y=None):
        transformed = []
        for epoch in X:
            Z = self.filters_ @ epoch  # shape: (n_components, n_times)
            var = np.var(Z, axis=1)
            log_var = np.log(var)
            transformed.append(log_var)
        return np.array(transformed)
