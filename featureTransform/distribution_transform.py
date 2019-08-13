#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer


def feature_hist(feat):
	mpl.rcParams['figure.figsize'] = (12.0, 6.0)
	prices = pd.DataFrame({
		'%s' % feat: feat,
		'log(1 + %s)' % feat: log_trans_norm(feat)
	})
	prices.hist()


def normality_transform(feature):
	"""
	# Map data from any distribution to as close to Gaussian distribution as possible
	# in order to stabilize variance and minimize skewness:
	# 	- log(1 + x) transform
	# 	- Yeo-Johnson transform
	# 	- Box-Cox transform
	# 	- Quantile transform
	:param feature:
	:return:
	"""
	pass



def log_trans_norm(feat):
	feat_trans = np.log1p(feat)

	return feat_trans


def box_cox(feat):
	bc = PowerTransformer(method="box-cox", standardize=False)
	feat_trans = bc.fit_transfrom(feat)

	return feat_trans

def yeo_johnson():
	yj = PowerTransformer(method = "yeo-johnson", standardize = False)
	feat_trans = yj.fit_transfrom(feat)
	
	return feat_trans


def quantileNorm(feat):
	qt = QuantileTransformer(output_distribution = "normal", random_state = 0)
	feat_trans = qt.fit_transform(feat)

	return feat_trans

def quantileUniform(feat, feat_test = None):
	qu = QuantileTransformer(random_state = 0)
	feat_trans = qu.fit_transform(feat)
	feat_trans_test = qu.transform(feat_test)

	return feat, feat_trans_test





