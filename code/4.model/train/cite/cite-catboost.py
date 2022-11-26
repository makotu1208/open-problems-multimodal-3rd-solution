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
preprocess_path = '../../../../input/preprocess/cite/'
validation_path = '../../../../input/fold/'

target_path = '../../../../input/target/'
feature_path = '../../../../input/features/cite/'
output_path = '../../../../model/cite/cb/'

# +
df_meta = pd.read_pickle(validation_path + "cite_fold_val_df.pickle")
X = np.load(preprocess_path + 'train_cite_inputs_idxcol.npz', allow_pickle=True)

fold_df = pd.DataFrame(list(X['index']), columns = ['cell_id'])
fold_df = fold_df.merge(df_meta[['cell_id', 'flg_donor_val', 'flg_all_val', 'rank_per_donor']], on = ['cell_id'], how = 'inner')
fold_df.shape

del X
# -

Y = pd.read_hdf(target_path + "train_cite_targets.h5")

# ### feature path

feature_dict = {}
feature_dict['best_128'] = ['X_best_128.pickle', 'X_test_best_128.pickle']
feature_dict['best_64'] = ['X_best_64.pickle', 'X_test_best_64.pickle']

# ## train

save_model = False # If you want to save the model in the output path, set this to True.

for i in feature_dict.keys():

    print(f'start: {i}')

    train_file = feature_dict[i][0]
    test_file = feature_dict[i][1]

    X = pd.read_pickle(feature_path  + train_file)
    X_test = pd.read_pickle(feature_path  + test_file)

    oof = np.zeros([X.shape[0], 140])
    pred = np.zeros([X_test.shape[0], 140])

    train_index = fold_df[fold_df['flg_donor_val'] == 0].index
    test_index = fold_df[fold_df['flg_donor_val'] == 1].index

    X_train, X_val = X.iloc[train_index], X.iloc[test_index]
    y_train, y_val = Y.iloc[train_index], Y.iloc[test_index]


    for t in tqdm(range(y_train.shape[1])):

        # check best round
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

        # get rounds
        model = CatBoostRegressor(**parameters)
        model.fit(X_train,
              y_train.iloc[:,t],
              eval_set=(X_val, y_val.iloc[:,t]),
              metric_period=500,
              early_stopping_rounds=20,
              use_best_model=True)

        oof[test_index, t] += model.predict(X_val)

        best_iteration = int(model.get_best_iteration())
        print(f'rounds:{best_iteration}')

        if best_iteration == 0:
            best_iteration = 1
        parameters['iterations'] = best_iteration

        # train all data
        model = CatBoostRegressor(**parameters)
        model.fit(X, Y.iloc[:,t])

        if save_model == True:
            pkl_filename = f"{output_path}/cite_cb_{t}_{i}_flg_donor_val_{best_iteration}.pkl"
            with open(pkl_filename, 'wb') as file:
                pickle.dump(model, file)
