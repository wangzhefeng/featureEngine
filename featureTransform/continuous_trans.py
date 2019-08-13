#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import numpy as np
from sklearn import preprocessing
# from sklearn.preprocessing import KBinDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import binarize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize


def k_bins(data, n_bins, encoder = "ordinal", strategy = "quantile"):
	"""
	分箱
	:param data:
	:param n_bins:
	:param encoder:
	:param strategy:
	:return:
	"""
	est = preprocessing.KBinsDiscretizer(n_bins = n_bins,
										 encoder = encoder,
										 strategy = strategy)
	est.fit_transform(data)


def binarization(data, threshold = 0.0, is_copy = True):
	"""
	二值化
	:param feat:
	:param threshold:
	:return:
	"""
	transformed_data = binarize(X = data, threshold = threshold, copy = is_copy)

	return transformed_data


def standard_center(data, is_copy = True, with_mean = True, with_std = True):
	"""
	标准化,中心化
	:return:
	"""
	ss = StandardScaler(copy = is_copy, with_mean = with_mean, with_std = with_std)
	transformed_data = ss.fit_transform(data)

	return transformed_data


def normal(data):
	"""
	正规化：将特征变量的每个值正规化到某个区间，比如:[0, 1]
	:param data:
	:return:
	"""
	pass


def normalizer(data, norm, axis, is_copy = True, return_norm = False):
	"""
	正则化:将每个样本或特征正则化为L1, L2范数
	:return:
	"""
	transformed_data = normalize(X = data,
								 norm = norm,
								 axis = axis,
								 copy = is_copy,
								 return_norm = return_norm)

	return transformed_data


def main():
	X = np.array([[-3.0, 5.0, 15],
				  [0.0, 6.0, 14],
				  [6.0, 3.0, 11]])
	est = preprocessing.KBinsDiscretizer(n_bins = [3, 2, 2],
										 encoder = "ordinal",
										 strategy = "quantile")
	est.fit_transform(X)




if __name__ == "__main__":
	main()