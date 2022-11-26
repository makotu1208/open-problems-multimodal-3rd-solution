# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

import catboost as cb
import scipy
from catboost import CatBoostRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold, GroupKFold

from sklearn.decomposition import PCA, TruncatedSVD
import pickle

from tqdm.notebook import tqdm
import gc
import datetime

# %matplotlib inline
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


# -

# ## data load

def correlation_score(y_true, y_pred):
    """Scores the predictions according to the competition rules.

    It is assumed that the predictions are not constant.

    Returns the average of each sample's Pearson correlation coefficient"""
    if type(y_true) == pd.DataFrame: y_true = y_true.values
    if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
    if y_true.shape != y_pred.shape: raise ValueError("Shapes are different.")
    corrsum = 0
    for i in range(len(y_true)):
        corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
    return corrsum / len(y_true)


# +
preprocess_path = '../../../../input/preprocess/multi/'
validation_path = '../../../../input/fold/'

target_path = '../../../../input/target/'
feature_path = '../../../../input/features/multi/'
output_path = '../../../../model/multi/cb/'
# -

Y = pd.read_pickle(target_path + 'multi_train_target_128.pickle')
svd = pickle.load(open(target_path + 'multi_all_target_128.pkl', 'rb'))
Y = np.array(Y)
fold_df = pd.read_pickle(validation_path + 'multi_fold_val_df.pickle')

# ### feature path

# +
feature_dict = {}

feature_dict['lsi_128'] = ['multi_train_okapi_lsi_128.pickle', 'multi_test_okapi_lsi_128.pickle']
feature_dict['lsi_64'] = ['multi_train_okapi_lsi_64.pickle', 'multi_test_okapi_lsi_64.pickle']
feature_dict['lsi_w2v_col_64'] = ['multi_train_lsi_w2v_col_64.pickle', 'multi_test_lsi_w2v_col_64.pickle']
# -

# ## train

save_model = True # If you want to save the model in the output path, set this to True.

for i in feature_dict.keys():

    print(f'start: {i}')

    train_file = feature_dict[i][0]
    test_file = feature_dict[i][1]

    X = pd.read_pickle(feature_path  + train_file)
    X_test = pd.read_pickle(feature_path  + test_file)

    X = np.array(X)
    X_test = np.array(X_test)

    oof = np.zeros([X.shape[0], 128])
    pred = np.zeros([X_test.shape[0], 128])

    train_index = fold_df[fold_df['flg_donor_val'] == 0].index
    test_index = fold_df[fold_df['flg_donor_val'] == 1].index

    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = Y[train_index], Y[test_index]

    for t in tqdm(range(y_train.shape[1])):

        parameters = {
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 2,
            'loss_function': 'RMSE',
            'task_type': 'GPU',
            'iterations': 100000, # 800
            'od_type': 'Iter',
            'boosting_type': 'Plain',
            'bootstrap_type': 'Bayesian',
            'bagging_temperature': 0.3,
            'allow_const_label': True,
            'random_state': 42,
            'verbose': 0
        }

        model = CatBoostRegressor(**parameters)
        model.fit(X_train,
                  y_train[:,t],
                  eval_set=(X_val, y_val[:,t]),
                  metric_period=500,
                  early_stopping_rounds=20,
                  use_best_model=True)

        oof[test_index, t] = model.predict(X_val)

        best_iteration = model.get_best_iteration()
        if best_iteration == 0:
            best_iteration = 1
        print(f'best iter:{best_iteration}')
        parameters['iterations'] = best_iteration

        # train all data
        model = CatBoostRegressor(**parameters)
        model.fit(X, Y[:,t])

        if save_model == True:
            pkl_filename = f"{model_path}/multi_cb_{t}_{i}_flg_donor_val_{best_iteration}.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(model, file)
