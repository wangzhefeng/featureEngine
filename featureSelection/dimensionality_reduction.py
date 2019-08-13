#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import SparsePCA

from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis





class pca:
	def __init__(self,
				 X, n_components,
				 whiten = False,
				 copy = True,
				 svd_solver = "auto",
				 tol = 0.0,
				 iterated_power = "auto",
				 random_state = None,
				 batch_size = None):
		self.X = X
		self.n_components = n_components
		self.copy = copy
		self.whiten = whiten
		self.svd_solver = svd_solver
		self.tol = tol
		self.iterated_power = iterated_power
		self.random_state = random_state
		self.batch_size = batch_size

	def pca(self):
		pca = PCA(n_components = self.n_components,
				  copy = self.copy,
				  whiten = self.whiten,
				  svd_solver = self.svd_solver,
				  tol = self.tol,
				  iterated_power = self.iterated_power,
				  random_state = self.random_state)
		pca.fit_transform(self.X)

		return pca

	def incremental_pca(self):
		pca = IncrementalPCA(n_components = self.n_components,
							 whiten = self.whiten,
							 copy = self.copy,
							 batch_size = self.batch_size)
		# pca.partial_fit(self.X)
		pca.fit_transform(self.X)

		return pca







class lda:
	def __init__(self, X, n_components):
		self.X = X
		self.n_components = n_components

	def lda(self):
		lda = LinearDiscriminantAnalysis(n_components = self.n_components)
		lda.fit_transform(self.X)

		return lda