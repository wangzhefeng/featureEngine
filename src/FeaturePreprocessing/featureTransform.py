#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-13 21:39:03
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import preprocessing

# 标准化
from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import scale

# 特征缩放到一个范围
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler

# 特征稳健缩放(存在异常值特征)
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import robust_scale

# 分箱离散化
from sklearn.preprocessing import Binarizer
# from sklearn.preprocessing import binarize
from sklearn.preprocessing import KBinDiscretizer

# 类别型特征重新编码
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder


from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Quantiletransformer
from sklearn.preprocessing import Powertransformer
from sklearn.preprocessing import label_binarize
from sklearn.compose import transformedTargetRegressor
from scipy.stats import skew



# ================================================
# 数值型变量分布转换
# ================================================
def standard_center(data, is_copy = True, with_mean = True, with_std = True):
    """
    标准化/方差缩放
    """
    ss = StandardScaler(copy = is_copy, with_mean = with_mean, with_std = with_std)
    transformed_data = ss.fit_transform(data)

    return transformed_data


def min_max(data):
    """
    min-max 缩放
    """
    mms = MinMaxScaler()
    transformed_data = mms.fit_transform()

    return transformed_data


def normalizer(data, norm, axis, is_copy = True, return_norm = False):
    """
    正则化:将每个样本或特征正则化为L1, L2范数
    """
    transformed_data = normalize(X = data,
                                 norm = norm,
                                 axis = axis,
                                 copy = is_copy,
                                 return_norm = return_norm)

    return transformed_data


def robust(data):
    """
    稳健缩放
    """
    rs = RobustScaler()
    transformed_data = RobustScaler(data)

    return transformed_data




def log_transform(feat):
    feat_trans = np.log1p(feat)

    return feat_trans

def Box_Cox(feat):
    bc = Powertransformer(method = "box-cox", standardize = False)
    feat_trans = bc.fit_transform(feat)

    return feat_trans

def yeo_johnson():
    yj = Powertransformer(method = "yeo-johnson", standardize = False)
    feat_trans = yj.fit_transform(feat)
    
    return feat_trans





# ================================================
# 类别性特征重编码
# ================================================

def oneHotEncoding(data, limit_value = 10):
    """
    One-Hot Encoding: pandas get_dummies
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
    """
    enc = OneHotEncoder()
    encoded_feats = enc.fit_transform(cate_feats)

    return encoded_feats



# ================================================
# 数值型变量分箱离散化
# ================================================
def binarization(data, threshold = 0.0, is_copy = True):
    """
    二值化
    """
    bined = Binarizer(threshold = threshold, copy = is_copy)
    transformed_data = bined.fit_transform(data)

    return transformed_data


def k_bins(data, n_bins, encoder = "ordinal", strategy = "quantile"):
    """
    分箱离散化
    * encode:
        - "ordinal"
        - "onehot"
        - "onehot-dense"
    * strategy:
        - "uniform"
        - "quantile"
        - "kmeans"
    """
    est = preprocessing.KBinsDiscretizer(n_bins = n_bins, encoder = encoder, strategy = strategy)
    transformed_data = est.fit_transform(data)

    return transformed_data


# ================================================
# 数值型变量分箱离散化
# ================================================



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
    #   - log(1 + x) transform
    #   - Yeo-Johnson transform
    #   - Box-Cox transform
    #   - Quantile transform
    """
    pass






def quantileNorm(feat):
    qt = Quantiletransformer(output_distribution = "normal", random_state = 0)
    feat_trans = qt.fit_transform(feat)

    return feat_trans

def quantileUniform(feat, feat_test = None):
    qu = Quantiletransformer(random_state = 0)
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





def targettransformer():
    trans = Quantiletransformer(output_distribution = "normal")
    return trans


def binarize_label(y, classes_list):
    y = label_binarize(y, classes = classes_list)

    return y



# ################################################

def main():
    pass

if __name__ == "__main__":
    main()