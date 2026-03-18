from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import shrunk_covariance
import numpy as np


class MyCSP(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=4, shrinkage=0.1, reg=1e-6):
        """
        n_components: number of spatial filters to return (features = n_components)
        shrinkage: float in [0,1] applied to class covariance matrices (0 = empirical)
        reg: small diagonal regularization added for numerical stability
        """
        self.n_components = n_components
        self.shrinkage = shrinkage
        self.reg = reg
        self.filters_ = None

    def _mean_normalized_cov(self, Xc):
        # Xc: (n_epochs, n_channels, n_times)
        covs = np.einsum('e c t, e d t -> e c d', Xc, Xc)
        traces = np.einsum('e i i -> e', covs)
        traces[traces == 0] = 1.0
        covs = covs / traces[:, None, None]
        return covs.mean(axis=0)

    def fit(self, X, y=None):
        X = np.asarray(X)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("CSP requires exactly two classes.")

        class_1 = X[y == classes[0]]
        class_2 = X[y == classes[1]]

        cov_1 = self._mean_normalized_cov(class_1)
        cov_2 = self._mean_normalized_cov(class_2)

        # apply shrinkage if requested
        if self.shrinkage is not None and 0.0 < self.shrinkage <= 1.0:
            cov_1 = shrunk_covariance(cov_1, shrinkage=self.shrinkage)
            cov_2 = shrunk_covariance(cov_2, shrinkage=self.shrinkage)

        # numerical regularization
        n_ch = cov_1.shape[0]
        cov_1 = cov_1 + self.reg * np.eye(n_ch)
        cov_2 = cov_2 + self.reg * np.eye(n_ch)

        cov_total = cov_1 + cov_2

        # eigen-decomposition and whitening
        eigvals, eigvecs = np.linalg.eigh(cov_total)
        # sort descending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # avoid non-positive eigenvalues
        eigvals[eigvals <= 0] = 1e-12
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))

        P = D_inv_sqrt @ eigvecs.T

        S1 = P @ cov_1 @ P.T
        eigvals_s1, eigvecs_s1 = np.linalg.eigh(S1)
        order_s1 = np.argsort(eigvals_s1)[::-1]
        eigvecs_s1 = eigvecs_s1[:, order_s1]

        filters = eigvecs_s1.T @ P

        # select components from both ends (largest and smallest eigenvalues)
        total_filters = filters.shape[0]
        n = min(self.n_components, total_filters)
        selected = []
        left = 0
        right = total_filters - 1
        while len(selected) < n:
            selected.append(left)
            left += 1
            if len(selected) < n:
                selected.append(right)
                right -= 1

        self.filters_ = filters[selected]
        return self

    def transform(self, X, y=None):
        X = np.asarray(X)
        if self.filters_ is None:
            raise ValueError("MyCSP not fitted yet")

        # project: result Z shape: (n_epochs, n_times, n_components)
        Z = np.tensordot(X, self.filters_.T, axes=([1], [0]))
        # variance over time axis
        var = np.var(Z, axis=1)
        # normalize per-epoch to be scale invariant, then log
        var_sum = var.sum(axis=1, keepdims=True)
        var_sum[var_sum == 0] = 1.0
        features = np.log((var / var_sum) + 1e-12)
        return features
