#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, MissingIndicator



class NaTool(object):
	"""
	缺失值填充:
		- 分位数填充
		- 固定值填充
		- 众数填充
	"""

	def __init__(self):
		pass

	def QuantileImpute(self, data_input, key_value = 0.95):
		data_union = []
		data_union = pd.DataFrame(data_union)
		x = data_input
		y = key_value
		for i in range(len(x.columns)):
			data1 = x.iloc[:, i].dropna(how = 'any')
			key = data1.quantile(y)
			data2 = x.iloc[:, i]
			data2 = data2.fillna(value = key)
			data2[data2 > key] = key
			data_union = pd.concat([data_union, data2], axis = 1)
		return data_union

	def ValueImpute(self, data_input, Value):
		data_union = []
		data_union = pd.DataFrame(data_union)
		x = data_input
		y = Value
		for i in range(len(x.columns)):
			key = y
			data2 = x.iloc[:, i]
			data2 = data2.fillna(value = key)
			data2[data2 > key] = key
			data_union = pd.concat([data_union, data2], axis = 1)
		return data_union

	def ModeImpute(self, data_input, key_value = 0.95):
		data_union = []
		data_union = pd.DataFrame(data_union)
		x = data_input
		y = key_value
		for i in range(len(x.columns)):
			data1 = x.iloc[:, i].dropna(how = 'any')
			data1 = data1.copy()
			key = data1.value_counts().argmax()
			data2 = data1.copy()
			key1 = data2.quantile(y)
			data3 = x.iloc[:, i]
			data3[data3 > key1] = key1
			data3 = data3.fillna(value = key)
			data_union = pd.concat([data_union, data3], axis = 1)
		return data_union

	def nan_fill(self, data, limit_value = 10, continuous_dealed_method = "mean"):
		"""
		# 当存在空值且每个feature下独立的样本数小于limit_value，认为是class feature采取one_hot_encoding；
		# 当存在空值且每个feature下独立的样本数大于limit_value，认为是continuous feature采取mean,min,max方式
		"""
		feature_cnt = data.shape[1]
		normal_index = []
		continuous_feature_index = []
		class_feature_index = []
		continuous_feature_df = pd.DataFrame()
		class_feature_df = pd.DataFrame()
		for i in range(feature_cnt):
			if np.isnan(np.array(data.iloc[:, i])).sum() > 0:
				# continue variables
				if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) >= limit_value:
					if continuous_dealed_method == "mean":
						continuous_feature_df = pd.concat([continuous_feature_df,
														   data.iloc[:, i].fillna(data.iloc[:, i].mean())],
														  axis = 1)
						continuous_feature_index.append(i)
					elif continuous_dealed_method == "max":
						continuous_feature_df = pd.concat([continuous_feature_df,
														   data.iloc[:, i].fillna(data.iloc[:, i].max())],
														  axis = 1)
						continuous_feature_index.append(i)
					elif continuous_dealed_method == "min":
						continuous_feature_df = pd.concat([continuous_feature_df,
														   data.iloc[:, i].fillna(data.iloc[:, i].min())],
														  axis = 1)
						continuous_feature_index.append(i)
				# categorical variables
				elif len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) > 0 and len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) < limit_value:
					class_feature_df = pd.concat([class_feature_df,
												  pd.get_dummies(data.iloc[:, i], prefix = data.columns[i])],
												 axis = 1)
					class_feature_index.append(i)
			else:
				normal_index.append(i)
		data_update = pd.concat([data.iloc[:, normal_index], continuous_feature_df, class_feature_df], axis = 1)

		return data_update




