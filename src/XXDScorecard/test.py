#!/usr/bin/env python
# -*- coding: utf-8 -*-

import XXDBinning as binning
import pandas as pd
from sklearn.model_selection import train_test_split

# data
df = pd.read_csv("data.csv")
train_df, test_df = train_test_split(df,
                                     test_size = 0.3,
                                     random_state = 100,
                                     stratify = df.flgGood)

with pd.option_context("display.max_column", None):
    print(df.head())

print("df's shape: ", df.shape)
print("train_df's shape: ", train_df.shape)
print("test_df's shape: ", test_df.shape)



# data preprocessing
nb = binning.XXDNumberBin()
print("-" * 100)

nb.pct_bin(train_df, 'req_inc_ratio', 'flgGood', max_bin = 10)
nb.get_bin_stats()
nb.plot_woe()
nb.trans_to_woe(test_df['req_inc_ratio'])


nb.manual_bin(train_df, 'req_inc_ratio', 'flgGood', [20,30,40])
nb.get_bin_stats()
nb.plot_woe()
nb.trans_to_woe(test_df['req_inc_ratio'])


nb.monotone_bin(train_df, 'req_inc_ratio', 'flgGood', max_bin = 3)
nb.get_bin_stats()
nb.plot_woe()
nb.trans_to_woe(test_df['req_inc_ratio'])








cb = binning.XXDCharBin()


