#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-08-13 21:39:56
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$


import numpy as np
import pandas as pd
from collections import Counter


class outlier(object):
    """
    * 异常值检测
    * 异常值处理
    """
    def __init__(self):
        pass

    def box_detect_outlier(df, n, feature):
        """
        * 箱型图异常值检测
        Example:
        df = pd.read.csv("data.csv")
        outliers_to_drop = box_detect_outlier(df, 2, ["feature1", "feature2", ...])
        df = df.drop(outliers_to_drop, axis = 0).reset_index(drop = True)
        """
        outlier_indices = []
        for col in feature: 
            Q1 = np.percentile(df[col], 25)
            Q3 = np.percentile(df[col], 75)
            IQR = Q3 - Q1
            outlier_step = 1.5 * IQR
            outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
            outlier_indices.extend(outlier_list_col)
        outlier_indices = Counter(outlier_indices)
        multiple_outliers = [k for k, v in outlier_indices.items() if v > n]

        return multiple_outliers




 




# ====================================================
def main():
    from featurePreprocessing.outlier import box_detect_outlier
    df = pd.DataFrame()
    features_to_analysis = []
    outliers_to_drop = box_detect_outlier(df, 2, features_to_analysis)
    df = df.drop(outliers_to_drop, axis = 0).reset_index(drop = True)


if __name__ == "__main__":
    main()

