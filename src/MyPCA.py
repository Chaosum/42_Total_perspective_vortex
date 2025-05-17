import numpy as np

class MyPCA:
	def __init__(self, n_components=None):
		self.n_components = n_components
		self.components_ = None
		self.mean_ = None

	def fit(self, X):
		# Center the data
		self.mean_ = np.mean(X, axis=0)
		X_centered = X - self.mean_

		# covariance matrix
		cov_matrix = (X_centered.T @ X_centered) / X_centered.shape[0]
		 # 3. Décomposition : valeurs eig_vals (coef)/vecteurs propres eig_vecs (directions)
		eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
		# 4. Tri décroissant
		sorted_indices = np.argsort(eig_vals)[::-1] # donne les indices des valeurs propres triées par ordre décroissant
		eig_vals = eig_vals[sorted_indices] # tri les valeurs propres eig_vals[une liste d'indices] = la liste triée selon les indices
		eig_vecs = eig_vecs[:, sorted_indices] # pour chaque ligne de eig_vecs, on prend les colonnes triées selon les indices
		# Select the first n_components
		if self.n_components is not None:
			eig_vecs = eig_vecs[:, :self.n_components]
		self.components_ = eig_vecs

	def transform(self, X):
		X_centered = X - self.mean_
		return X_centered @ self.components_ # return np.dot = @ = produit matriciel

	def fit_transform(self, X):
		self.fit(X)
		return self.transform(X)