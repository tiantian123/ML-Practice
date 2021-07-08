#!/usr/bin/env python
# -** coding: utf-8 -*-
# @Time: 2021/7/8 0:09
# @Author: Tian Chen
# @File: main.py
import LGBM
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold


def load_data(target_label):
    train_data = pd.read_csv("final/train_final.csv")
    test_data = pd.read_csv("final/test_final.csv")
    col_drop = ['continuous_annual_inc_joint', 'continuous_dti_joint', 'continuous_mths_since_last_delinq',
                'continuous_mths_since_last_major_derog', 'continuous_mths_since_last_record']
    # x_train = train_data.drop(columns=target_label)
    # x_test = test_data.drop(columns=target_label)
    # y_train = train_data[target_label]
    # y_test = test_data[target_label]
    # print(train_data.info())
    train_data.drop(columns=col_drop, inplace=True)
    test_data.drop(columns=col_drop, inplace=True)
    # print(train_data.info())
    return train_data, test_data


if __name__ == '__main__':
    target_label = 'loan_status'
    train, test = load_data(target_label)
    kfold = KFold(n_splits=5)
    fitter = LGBM.LGBFitter(label=target_label)
    leaves = [8, 16, 32, 48, 64, 127]
    rounds = [2000, 2000, 2000, 1500, 1500, 1000]
    for num_leave, num_r in zip(leaves, rounds):
        params = {'num_thread': 4, 'num_leaves': num_leave, 'metric': 'binary', 'objective': 'binary',
                  'num_round': num_r, 'learning_rate': 0.005, 'feature_fraction': 0.8, 'bagging_fraction': 0.8}
        train_pred, test_pred, acc_result, models = fitter.train_k_fold(kfold, train, test, params=params)
        if fitter.metric != 'auc':
            test_pred = (test_pred > 0.5).astype(int)
        test_acc = fitter.get_loss(test[target_label], test_pred)

        print("validate data error rate:", np.mean(acc_result))
        print("test data error rate:", test_acc)
        print(params)
        print("\n")