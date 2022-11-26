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

# make validation sample: 'summary/input/fold/cite_fold_val_df.pickle'

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
raw_path = '../../input/raw/'
preprocess_path = '../../input/preprocess/cite/'
validation_path = '../../input/fold/'

target_path = '../../input/target/'
feature_path = '../../input/features/cite/'
model_path = '../../model/validation/cite/'
# -

train_ids = np.load(preprocess_path + "train_cite_raw_inputs_idxcol.npz", allow_pickle=True)
test_ids = np.load(preprocess_path + "test_cite_raw_inputs_idxcol.npz", allow_pickle=True)
df_meta = pd.read_csv(raw_path + "metadata.csv")
df_meta = df_meta[df_meta['day'] == 7][['cell_id']].reset_index(drop=True)

# ### feature path

feature_dict = {}
feature_dict['best_128'] = ['X_best_128.pickle', 'X_test_best_128.pickle']

# ## train

parameters = {
    'boosting': 'gbdt',
    'objective': 'binary',
    #'metric': 'binary_logloss',
    'metric': 'auc',
    'learning_rate': 0.05,
    'boosting': 'gbdt',
    'max_depth': 7,
    'min_data_in_leaf': 20,
    'verbose': -1
}

save_model = False # If you want to save the model in the output path, set this to True.

for i in feature_dict.keys():

    print(f'start: {i}')

    train_file = feature_dict[i][0]
    test_file = feature_dict[i][1]

    X = pd.read_pickle(feature_path  + train_file)
    X_test = pd.read_pickle(feature_path  + test_file)

    train_num = len(X)

    X.index = list(train_ids['index'])
    X_test.index = list(test_ids['index'])

    X_test = X_test.reset_index().rename(columns = {'index': 'cell_id'})
    X_test = df_meta.merge(X_test, on = 'cell_id', how = 'left')
    X_test = X_test[X_test['base_svd_0'].isnull() == False]

    X_test.drop(['cell_id'], axis = 1, inplace = True)

    X['target'] = 0
    X_test['target'] = 1

    Y = pd.concat([X['target'].reset_index(drop=True),
                   X_test['target'].reset_index(drop=True),
                  ]).reset_index(drop=True)
    X = pd.concat([X.reset_index(drop=True),
                   X_test.reset_index(drop=True),
                  ]).reset_index(drop=True)

    del X['target']

    print(X.shape)

    oof = np.zeros(X.shape[0])
    fold = 5
    kf = KFold(n_splits=fold, shuffle=True, random_state=1)

    for f, (train_index, test_index) in enumerate(kf.split(X)):

        print(f'fold{f}')
        X_train, X_val = X.iloc[train_index], X.iloc[test_index]
        y_train, y_val = Y.iloc[train_index], Y.iloc[test_index]

        X_train = lgb.Dataset(X_train, label=y_train)
        X_val = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(parameters,
                       X_train,
                       valid_sets=X_val,
                       num_boost_round=10000,
                       early_stopping_rounds=300,
                       verbose_eval = 100
                    )

        importance = pd.DataFrame(model.feature_importance(), index=list(X.columns), columns=['importance'])
        display(importance.sort_values('importance', ascending = False).head(20))

        if save_model == True:
            file = model_path + '/'+ f'validation_lgb_fold{f}.pkl'
            pickle.dump(model, open(file, 'wb'))

        oof[test_index] = model.predict(X.iloc[test_index])

X_result = pd.DataFrame(oof, columns = ['pred'])
X_result = pd.concat([X, X_result], axis = 1)

X_train_result = X_result[:70988][['pred']]
X_train_result['cell_id'] = list(train_ids['index'])

df_meta_raw = pd.read_csv(raw_path + "metadata.csv")
X_train_result = X_train_result.merge(df_meta_raw, on = 'cell_id', how = 'left')

# ### make validation
# - Use 2000 people close to pb in each donor as validation.

X_train_result.shape

X_train_result['rank_per_donor'] = X_train_result.groupby(['donor'])['pred'].rank(ascending=False)
X_train_result['rank_all'] = X_train_result['pred'].rank(ascending=False)

X_train_result['flg_donor_val'] = np.where(X_train_result['rank_per_donor'] < 2000, 1, 0)
X_train_result['flg_all_val'] = np.where(X_train_result['rank_all'] < 5000, 1, 0)

print(X_train_result[X_train_result['flg_donor_val'] == 1]['donor'].value_counts())
print(X_train_result[X_train_result['flg_all_val'] == 1]['donor'].value_counts())

X_train_result.to_pickle(validation_path + 'cite_fold_val_df.pickle')

X_train_result.shape
