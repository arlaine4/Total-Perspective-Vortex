import numpy as np
from scipy.linalg import eigh

class CSP:
	def	__init__(self, n_components):
		self.n_components = n_components
		self.filters = None

	def	fit(self, X, y):
		"""
		1. Calculates covariances matrices for each class	
		2. Computes the average covariance matrix
		3. performs eigenvalue decomposition
		4. select CSP filters based on the largest eigenvalues and stores
		   those filters for later use

		-> This method prepares the CSP model by extracting the spatial
		   filters that maximize the differences in signal variance between
		   classes.
		"""
		# Compute class-wise covariance matrices
		# Covariance matrix list for each class
		covs = []
		classes = np.unique(y)
		# Iterate over each unique class and extracts EEG data corresponding
		# 	to that class in order to compute covariance for each class
		for cls_label in classes:
			cls_data = X[y == cls_label]
			cov = self.compute_covariance(cls_data)
			covs.append(cov)

		# Compute average covariance matrix over all classes
		avg_cov = np.mean(covs, axis=0)

		# Perform eigenvalue decomposition
		eigen_values, eigen_vectors = eigh(avg_cov)

		# Sort eigenvalues in descending order
		sorted_idx = np.argsort(eigen_values)[::-1]
		eigen_vectors = eigen_vectors[:, sorted_idx]

		# Select CSP filters
		# -> selecting the first n_components (most importent since we ordered
		#	 the eigen_vectors by ascending order earlier) and storing them in
		#	 filter variable.
		filters = eigen_vectors[:, :self.n_components]

		self.filters = filters
	
	def	transform(self, X):
		"""
		Apply CSP filters to input data
		-> Filters serve to spatially transform EEG signals,
			maximize the differences in signal variance between classes, and 
			identify discriminative spatial patterns.
		--> They play a crucial role in enhancing the separability of the data 
			and extracting features that capture the most relevant information
			for classification.
		---> CSP enables effective discrimination of different brain states or
			 conditions from EEG data.
		"""
		n_trials, n_channels, n_samples = X.shape
		transformed_data = np.zeros((n_trials, self.n_components, n_samples))
		for i in range(n_trials):
			trial_data = X[i]
			transformed_data[i] = np.dot(self.filters.T, trial_data)
		return transformed_data.reshape(n_trials, -1)
		# Apply CSP filters to input data
		#transformed_data = np.dot(X, self.filters[:self.n_components])
		#print('done transforming')
		#return transformed_data
	
	def	fit_transform(self, X, y):
		self.fit(X, y)
		transformed_data = self.transform(X)
		return transformed_data
	
	def	compute_covariance(self, X):
		"""
		Compute covariance -> statistical measure that quantifies the
		relationship between variables, here it's the relationship between EEG
		signals recorded at different electrodes.

		The covariance matrix represents the pairwaise covariances between EEG
		channels -> provides informations about how the EEG signals at different
		channels vary together.
		-> A higher positive cov indicates that the signals tend to increase or
			decrease together, while a negative cov indicates an inverse
			relationship between the signals.
		--> Cov matrix is a crucial component in identifying the spatial filters
			that maximize the differences in signal variance between diff classes
			or conditions. By analyzing the cov matrices of diff classes, CSP can
			identify spatial patterns that are most discriminative between
			those classes.
		"""
		n_trials, n_channels, n_samples = X.shape
		# Creating empty covariance matrix
		covariance = np.zeros((n_channels, n_channels))

		for i in range(n_trials):
			trial_data = X[i]
			# Compute covariance matrix for each trial in X
			# -> adding up the values so we can compute the average of
			#	 this matrix after iterating over every trial/
			covariance += np.dot(trial_data, trial_data.T) / n_samples
		# Compute the average covariance across all trials.
		# -> provides an estimation of the covariance matrix that represents
		# 	 the overall statistical relationship between signals recorded at
		#	 different channels.
		covariance /= n_trials
		return covariance

