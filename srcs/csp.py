import numpy as np


class CSP:
	def	__init__(self, n_components=4):
		self.n_components = n_components
		self.filters = None
	
	def	compute_covariance(self, X):
		X -= X.mean(axis=1)[:, None]
		N = X.shape[1]
		return np.dot(X, X.T.conj())
	
	def	compute_covariance_matrix(self, X, y):
		_
