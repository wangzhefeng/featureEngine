#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author:
@date:
"""

import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer



def targetTransfromer():
	trans = QuantileTransformer(output_distribution = "normal")
	return trans

def binarize_label(y, classes_list):
	y = label_binarize(y, classes = classes_list)

	return y

