#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import numpy as np
import pandas as pd
from scipy.stats import skew

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


