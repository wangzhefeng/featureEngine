#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import pandas as pd
import numpy as np

class check_deal(object):

	def __init__(self, data):
		self.data = data

	def outlier_check(self):
		pass

	def outlier_visual(self):
		pass

	def outlier_deal(self):
		pass




def outlier_remove(data, limit_value = 10, method = "box", percentile_limit_set = 90, changed_feature_box = []):
	"""
	# limit_value: 最小处理样本个数set,当独立样本大于limit_value,认为非可onehot字段
	"""
	feature_cnt = data.shape[1]
	feature_change = []
	if method == "box":

		"""
		# 离群点盖帽
		"""
		for i in range(feature_cnt):
			if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) >= limit_value:
				q1 = np.percentile(np.array(data.iloc[:, i]), 25)
				q3 = np.percentile(np.array(data.iloc[:, i]), 75)
				top = q3 + 1.5 * (q3 - q1)
				data.iloc[:, i][data.iloc[:, i] > top] = top
				feature_change.append(i)
		return data, feature_change

	if method == "self_def":
		"""
		# 快速截断
		"""
		if len(changed_feature_box) == 0:
			# 当方法选择为自定义,且没有定义changed_feature_box则全量数据全部按照percentile_limit_set的分位点大小进行截断
			for i in range(feature_cnt):
				if len(pd.DataFrame(data.iloc[:, i]).drop_duplicates()) >= limit_value:
					q_limit = np.percentile(np.array(data.iloc[:, i]), percentile_limit_set)
					data.iloc[:, i][data.iloc[:, i]] = q_limit
					feature_change.append(i)
		else:
			# 如果定义了changed_feature_box，则将changed_feature_box里面的按照box方法，changed_feature_box的feature index按照percentile_limit_set的分位点大小进行截断
			for i in range(feature_cnt):
				if len(pd.DataFrame(data.iloc[:, 1]).drop_duplicates()) >= limit_value:
					if i in changed_feature_box:
						q1 = np.percentile(np.array(data.iloc[:, i]), 25)
						q3 = np.percentile(np.array(data.iloc[:, i]), 75)
						top = q3 + 1.5 * (q3 - q1)
						data.iloc[:, i][data.iloc[:, i] > top] = top
						feature_change.append(i)
					else:
						q_limit = np.percentile(np.array(data.iloc[:, i]), percentile_limit_set)
						data.iloc[:, i][data.iloc[:, i]] = q_limit
						feature_change.append(i)
		return data, feature_change