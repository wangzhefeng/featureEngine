#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


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




def main():
	X = np.array([["male"], ['female'], ['male']])
	X = pd.DataFrame(X)
	encoded_X = one_hot_encoder(X)
	print(encoded_X)

	data = oneHotEncoding(X)
	print(data)

if __name__ == "__main__":
	main()