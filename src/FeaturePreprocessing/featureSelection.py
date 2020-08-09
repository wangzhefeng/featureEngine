#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-13 21:39:28
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import SparsePCA

from sklearn.decomposition import FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np
from sklearn import feature_selection
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier





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

def nan_feature_remove(data, rate_base = 0.4):
    """
    # 针对每一列feature统计nan的个数
    # 个数大于全量样本的rate_base的认为是异常feature, 进行剔除
    """
    all_cnt = data.shape[0]
    feature_cnt = data.shape[1]
    available_index = []
    for i in range(feature_cnt):
        rate = np.isnan(np.array(data.iloc[:, i])).sum() / all_cnt
        if rate <= rate_base:
            available_index.append(i)
    data_available = data.iloc[:, available_index]
    return data_available, available_index


def low_variance_feature_remove(data, rate_base = 0.0):
    """
    # 对样本数据集中方差小于某一阈值的特征进行剔除
    :param data:
    :param p:
    :return:
    """
    sel = VarianceThreshold(threshold = rate_base)
    data_available = sel.fit_transform(data)

    return data_available


def model_based_feature_selection(data, target, model = "tree", n_estimators = 50):
    if model == "tree":
        clf = ExtraTreesClassifier(n_estimators = n_estimators).fit(data, target)
        model = SelectFromModel(clf, prefit=True)
        data_available = model.transform(data)
        return data_available
    elif model == "svm":
        clf = LinearSVC(C = 0.01, penalty = "l1", dual = False).fit(data, target)
        model = SelectFromModel(clf, prefit=True)
        data_available = model.transform(data)
        return data_available
    elif model == "lr":
        clf = LogisticRegression(C = 0.01, penalty = "l1", dual = False).fit(data, target)
        model = SelectFromModel(clf, prefit=True)
        data_available = model.transform(data)
        return data_available
    elif model == "lasso":
        clf = ""
    else:
        print("Error model, Please choose one of 'tree', 'svm' or 'lr'!")


