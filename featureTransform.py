#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-13 21:39:03
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from sklearn import preprocessing
# from sklearn.preprocessing import KBinDiscretizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import binarize
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import label_binarize
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer





def oneHotEncoding(data, limit_value = 10):
	"""
	One-Hot Encoding: pandas get_dummies
	:param data:
	:param limit_value:
	:return:
	"""
	feature_cnt = data.shape[1]
	class_index = []
	class_df = pd.DataFrame()
	normal_index = []
	for i in range(feature_cnt):
		if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) < limit_value:
			class_index.append(i)
			class_df = pd.concat([class_df, pd.get_dummies(data.iloc[:, i], prefix = data.columns[i])], axis = 1)
		else:
			normal_index.append(i)
	data_update = pd.concat([data.iloc[:, normal_index], class_df], axis = 1)
	return data_update


def order_encoder(cate_feats):
	enc = OrdinalEncoder()
	encoded_feats = enc.fit_transform(cate_feats)

	return encoded_feats


def one_hot_encoder(cate_feats):
	"""
	One-Hot Encoding: sklearn.preprocessing.OneHotEncoder
	:param cate_feats:
	:return:
	"""
	enc = OneHotEncoder()
	encoded_feats = enc.fit_transform(cate_feats)

	return encoded_feats

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

def feature_dtype(data):
	with pd.option_context('display.max_rows', None, 'display.max_columns', None):
		print(data.dtypes)

def numeric_categorical_features(data, limit_value = 0):
	columns = data.columns

	num_feature_idx = []
	cate_feature_idx = []
	for i in columns:
		if (data[i].dtypes != "object") & (len(set(data[i])) >= limit_value):
			num_feature_idx.append(i)
		else:
			cate_feature_idx.append(i)

	num_feat_index = data[num_feature_idx].columns
	num_feat = data[num_feature_idx]
	cate_feat_index = data[cate_feature_idx].columns
	cate_feat = data[cate_feature_idx]

	return num_feat, num_feat_index, cate_feat, cate_feat_index


def skewed_features(data, num_feat_idx, limit_value = 0.75):
	skewed_feat = data[num_feat_idx].apply(lambda x: skew(x.dropna()))
	skewed_feat = skewed_feat[np.abs(skewed_feat) > limit_value]
	skewed_feat_index = skewed_feat.index

	return skewed_feat, skewed_feat_index

def targetTransfromer():
	trans = QuantileTransformer(output_distribution = "normal")
	return trans

def binarize_label(y, classes_list):
	y = label_binarize(y, classes = classes_list)

	return y





def main():
	X = np.array([["male"], ['female'], ['male']])
	X = pd.DataFrame(X)
	encoded_X = one_hot_encoder(X)
	print(encoded_X)

	data = oneHotEncoding(X)
	print(data)

if __name__ == "__main__":
	main()