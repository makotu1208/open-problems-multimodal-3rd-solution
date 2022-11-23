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

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

preprocess_path = '../../../input/base_features/cite/'
output_path = '../../../input/features/cite/'

# ### load base feature

# +
X_128 = pd.read_pickle(preprocess_path + 'train_svd_128_imp84.pickle')
X_test_128 = pd.read_pickle(preprocess_path + 'test_svd_128_imp84.pickle')
X_128 = X_128.add_prefix('base_svd_')
X_test_128 = X_test_128.add_prefix('base_svd_')

X_64 = pd.read_pickle(preprocess_path + 'train_svd_64_imp84.pickle')
X_test_64 = pd.read_pickle(preprocess_path + 'test_svd_64_imp84.pickle')
X_64 = X_64.add_prefix('base_svd_')
X_test_64 = X_test_64.add_prefix('base_svd_')

X_imp = pd.read_pickle(preprocess_path + 'train_imp_84.pickle')
X_test_imp = pd.read_pickle(preprocess_path + 'test_imp_84.pickle')

X_imp_norm = pd.read_pickle(preprocess_path + 'train_imp_merge_norm_84.pickle')
X_test_imp_norm = pd.read_pickle(preprocess_path + 'test_imp_merge_norm_84.pickle')

X_imp_feature = pd.read_pickle(preprocess_path + 'train_imp_feature_imp84.pickle')
X_test_imp_feature = pd.read_pickle(preprocess_path + 'test_imp_feature_imp84.pickle')

X_cluster = pd.read_pickle(preprocess_path + 'train_cite_cluster84_mean.pickle')
X_test_cluster = pd.read_pickle(preprocess_path + 'test_cite_cluster84_mean.pickle')

X_all_cluster = pd.read_pickle(preprocess_path + 'cite_train_all_cluster.pickle')
X_test_all_cluster = pd.read_pickle(preprocess_path + 'cite_test_all_cluster.pickle')

X_col_cluster = pd.read_pickle(preprocess_path + 'cite_train_col_mean.pickle')
X_test_col_cluster = pd.read_pickle(preprocess_path + 'cite_test_col_mean.pickle')

X_cnorm = pd.read_pickle(preprocess_path + 'train_svd_64_c_norm.pickle')
X_test_cnorm = pd.read_pickle(preprocess_path + 'test_svd_64_c_norm.pickle')
X_cnorm = X_cnorm.add_prefix('cnorm_')
X_test_cnorm = X_test_cnorm.add_prefix('cnorm_')

X_imp_cnorm = pd.read_pickle(preprocess_path + 'train_imp_c_norm_84.pickle')
X_test_imp_cnorm = pd.read_pickle(preprocess_path + 'test_imp_c_norm_84.pickle')

X_w2v = pd.read_pickle(preprocess_path + 'train_w2v_top100.pickle')
X_test_w2v = pd.read_pickle(preprocess_path + 'test_w2v_top100.pickle')

X_imp_w2v = pd.read_pickle(preprocess_path + 'train_w2v_imp.pickle')
X_test_imp_w2v = pd.read_pickle(preprocess_path + 'test_w2v_imp.pickle')

X_imp_snorm = pd.read_pickle(preprocess_path + 'train_imp_s_norm_84.pickle')
X_test_imp_snorm = pd.read_pickle(preprocess_path + 'test_imp_s_norm_84.pickle')

X_imp_cell = pd.read_pickle(preprocess_path + 'train_cite_cellt_mean.pickle')
X_test_imp_cell = pd.read_pickle(preprocess_path + 'test_cite_cellt_mean.pickle')

X_c_all_mean = pd.read_pickle(preprocess_path + 'train_cite_cluster_84_all_mean.pickle')
X_test_c_all_mean = pd.read_pickle(preprocess_path + 'test_cite_cluster_84_all_mean.pickle')


# 120 dims -------------------------------------------------------------------------

X_128_120 = pd.read_pickle(preprocess_path + 'train_svd_128_imp120.pickle')
X_test_128_120 = pd.read_pickle(preprocess_path + 'test_svd_128_imp120.pickle')

X_64_120 = pd.read_pickle(preprocess_path + 'train_svd_64_imp120.pickle')
X_test_64_120 = pd.read_pickle(preprocess_path + 'test_svd_64_imp120.pickle')

X_imp_120 = pd.read_pickle(preprocess_path + 'train_imp_120.pickle')
X_test_imp_120 = pd.read_pickle(preprocess_path + 'test_imp_120.pickle')

X_imp_feature_120 = pd.read_pickle(preprocess_path + 'train_imp_feature_imp120.pickle')
X_test_imp_feature_120 = pd.read_pickle(preprocess_path + 'test_imp_feature_imp120.pickle')

X_imp_norm_120 = pd.read_pickle(preprocess_path + 'train_imp_merge_norm_120.pickle')
X_test_imp_norm_120 = pd.read_pickle(preprocess_path + 'test_imp_merge_norm_120.pickle')

X_cluster_120 = pd.read_pickle(preprocess_path + 'train_cite_cluster120_mean.pickle')
X_test_cluster_120 = pd.read_pickle(preprocess_path + 'test_cite_cluster120_mean.pickle')

X_cnorm_120 = pd.read_pickle(preprocess_path + 'train_svd_64_c_norm_120.pickle')
X_test_cnorm_120 = pd.read_pickle(preprocess_path + 'test_svd_64_c_norm_120.pickle')

X_imp_cnorm_120 = pd.read_pickle(preprocess_path + 'train_imp_c_norm_120.pickle')
X_test_imp_cnorm_120 = pd.read_pickle(preprocess_path + 'test_imp_c_norm_120.pickle')

X_imp_cell_120 = pd.read_pickle(preprocess_path + 'train_cite_cellt_mean_120.pickle')
X_test_imp_cell_120 = pd.read_pickle(preprocess_path + 'test_cite_cellt_mean_120.pickle')

X_c_all_mean_120 = pd.read_pickle(preprocess_path + 'train_cite_cluster_120_all_mean.pickle')
X_test_c_all_mean_120 = pd.read_pickle(preprocess_path + 'test_cite_cluster_120_all_mean.pickle')

# other -------------------------------------------------------------------------

X_cell_ratio = pd.read_pickle(preprocess_path + 'train_celltype.pickle')
X_test_cell_ratio = pd.read_pickle(preprocess_path + 'test_celltype.pickle')

X_con_imp = pd.read_pickle(preprocess_path + 'train_cite_imp_confeature.pickle')
X_test_con_imp = pd.read_pickle(preprocess_path + 'test_cite_imp_confeature.pickle')

X_last_svd_v4 = pd.read_pickle(preprocess_path + 'train_cite_last_svd_v4.pickle')
X_test_last_svd_v4  = pd.read_pickle(preprocess_path + 'test_cite_last_svd_v4.pickle')

X_c_cell_ratio= pd.read_pickle(preprocess_path + 'train_cite_cluster_celltype_ratio.pickle')
X_test_c_cell_ratio = pd.read_pickle(preprocess_path + 'test_cite_cluster_celltype_ratio.pickle')
X_test_c_cell_ratio = X_test_c_cell_ratio.fillna(0)
# -

# ### make feature pickle

# +
X_add_con_imp =  pd.concat([
               X_128_120.reset_index(drop=True),
               X_imp_120.reset_index(drop=True),
               X_imp_norm_120.reset_index(drop=True),
               X_imp_feature,
               X_cluster_120.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_col_cluster.reset_index(drop=True),
               X_cnorm_120.reset_index(drop=True),
               X_imp_cnorm_120.reset_index(drop=True),
               X_w2v.reset_index(drop=True),
               X_imp_w2v.reset_index(drop=True),
               #X_mt.reset_index(drop=True),
               X_con_imp.reset_index(drop=True),
               ], axis = 1)

X_test_add_con_imp = pd.concat([
                    X_test_128_120.reset_index(drop=True),
                    X_test_imp_120.reset_index(drop=True),
                    X_test_imp_norm_120.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster_120.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_col_cluster.reset_index(drop=True),
                   X_test_cnorm_120.reset_index(drop=True),
                   X_test_imp_cnorm_120.reset_index(drop=True),
                    X_test_w2v.reset_index(drop=True),
                    X_test_imp_w2v.reset_index(drop=True),
                    #X_test_mt.reset_index(drop=True),
                    X_test_con_imp.reset_index(drop=True),
                    ], axis = 1)

X_add_con_imp = X_add_con_imp.astype(np.float32)
X_test_add_con_imp = X_test_add_con_imp.astype(np.float32)

print(X_add_con_imp.shape, X_test_add_con_imp.shape)

X_add_con_imp.to_pickle(output_path + 'X_add_con_imp.pickle')
X_test_add_con_imp.to_pickle(output_path + 'X_test_add_con_imp.pickle')

# +
# base
X_last_v3 = pd.concat([
               X_last_svd_v4.reset_index(drop=True),
               X_imp_120.reset_index(drop=True),
               X_imp_norm_120.reset_index(drop=True),
               X_imp_feature_120,
               X_cluster_120.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_col_cluster.reset_index(drop=True),
               X_cnorm_120.reset_index(drop=True),
               X_imp_cnorm_120.reset_index(drop=True),
               X_c_all_mean_120.reset_index(drop=True),
               X_c_cell_ratio.reset_index(drop=True),
               X_w2v.reset_index(drop=True),
               X_imp_w2v.reset_index(drop=True),
               ], axis = 1)

X_test_last_v3 = pd.concat([
                   X_test_last_svd_v4.reset_index(drop=True),
                    X_test_imp_120.reset_index(drop=True),
                    X_test_imp_norm_120.reset_index(drop=True),
                    X_test_imp_feature_120,
                    X_test_cluster_120.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_col_cluster.reset_index(drop=True),
                    X_test_cnorm_120.reset_index(drop=True),
                    X_test_imp_cnorm_120.reset_index(drop=True),
                    X_test_c_all_mean_120.reset_index(drop=True),
                    X_test_c_cell_ratio.reset_index(drop=True),
                    X_test_w2v.reset_index(drop=True),
                    X_test_imp_w2v.reset_index(drop=True),
                    ], axis = 1)

X_last_v3 = X_last_v3.astype(np.float32)
X_test_last_v3 = X_test_last_v3.astype(np.float32)

print(X_last_v3.shape, X_test_last_v3.shape)

X_last_v3.to_pickle(output_path + 'X_last_v3.pickle')
X_test_last_v3.to_pickle(output_path + 'X_test_last_v3.pickle')

# +
# base
X_c_add_w2v_v1 = pd.concat([
                X_128_120.reset_index(drop=True),
               X_imp_120.reset_index(drop=True),
               X_imp_norm_120.reset_index(drop=True),
               X_imp_feature,
               X_cluster_120.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_col_cluster.reset_index(drop=True),
               X_cnorm_120.reset_index(drop=True),
               X_imp_cnorm_120.reset_index(drop=True),
               X_c_all_mean_120.reset_index(drop=True),
               X_c_cell_ratio.reset_index(drop=True),
               X_w2v.reset_index(drop=True),
               X_imp_w2v.reset_index(drop=True),
               ], axis = 1)

X_test_c_add_w2v_v1 = pd.concat([
                    X_test_128_120.reset_index(drop=True),
                    X_test_imp_120.reset_index(drop=True),
                    X_test_imp_norm_120.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster_120.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_col_cluster.reset_index(drop=True),
                   X_test_cnorm_120.reset_index(drop=True),
                   X_test_imp_cnorm_120.reset_index(drop=True),
                    X_test_c_all_mean_120.reset_index(drop=True),
                    X_test_c_cell_ratio.reset_index(drop=True),
                    X_test_w2v.reset_index(drop=True),
                    X_test_imp_w2v.reset_index(drop=True),
                    ], axis = 1)

X_c_add_w2v_v1 = X_c_add_w2v_v1.astype(np.float32)
X_test_c_add_w2v_v1 = X_test_c_add_w2v_v1.astype(np.float32)

print(X_c_add_w2v_v1.shape, X_test_c_add_w2v_v1.shape)

X_c_add_w2v_v1.to_pickle(output_path + 'X_c_add_w2v_v1.pickle')
X_test_c_add_w2v_v1.to_pickle(output_path + 'X_test_c_add_w2v_v1.pickle')

# +
# base
X_c_add_84_v1 = pd.concat([
               X_128.reset_index(drop=True),
               X_imp.reset_index(drop=True),
               X_imp_norm.reset_index(drop=True),
               X_imp_feature,
               X_cluster.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_col_cluster.reset_index(drop=True),
               X_cnorm.reset_index(drop=True),
               X_imp_cnorm.reset_index(drop=True),
               X_c_all_mean.reset_index(drop=True),
               X_c_cell_ratio.reset_index(drop=True),
               ], axis = 1)

X_test_c_add_84_v1 = pd.concat([
                    X_test_128.reset_index(drop=True),
                    X_test_imp.reset_index(drop=True),
                    X_test_imp_norm.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_col_cluster.reset_index(drop=True),
                   X_test_cnorm.reset_index(drop=True),
                   X_test_imp_cnorm.reset_index(drop=True),
                    X_test_c_all_mean.reset_index(drop=True),
                    X_test_c_cell_ratio.reset_index(drop=True),
                    ], axis = 1)

X_c_add_84_v1 = X_c_add_84_v1.astype(np.float32)
X_test_c_add_84_v1 = X_test_c_add_84_v1.astype(np.float32)

print(X_c_add_84_v1.shape, X_test_c_add_84_v1.shape)

X_c_add_84_v1.to_pickle(output_path + 'X_c_add_84_v1.pickle')
X_test_c_add_84_v1.to_pickle(output_path + 'X_test_c_add_84_v1.pickle')

# +
# base
X_c_add_v1 = pd.concat([
                X_128_120.reset_index(drop=True),
               X_imp_120.reset_index(drop=True),
               X_imp_norm_120.reset_index(drop=True),
               X_imp_feature,
               X_cluster_120.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_col_cluster.reset_index(drop=True),
               X_cnorm_120.reset_index(drop=True),
               X_imp_cnorm_120.reset_index(drop=True),
               X_c_all_mean_120.reset_index(drop=True),
               X_c_cell_ratio.reset_index(drop=True),
               ], axis = 1)

X_test_c_add_v1 = pd.concat([
                    X_test_128_120.reset_index(drop=True),
                    X_test_imp_120.reset_index(drop=True),
                    X_test_imp_norm_120.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster_120.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_col_cluster.reset_index(drop=True),
                   X_test_cnorm_120.reset_index(drop=True),
                   X_test_imp_cnorm_120.reset_index(drop=True),
                    X_test_c_all_mean_120.reset_index(drop=True),
                    X_test_c_cell_ratio.reset_index(drop=True),
                    ], axis = 1)

X_c_add_v1 = X_c_add_v1.astype(np.float32)
X_test_c_add_v1 = X_test_c_add_v1.astype(np.float32)

print(X_c_add_v1.shape, X_test_c_add_v1.shape)

X_c_add_v1.to_pickle(output_path + 'X_c_add_v1.pickle')
X_test_c_add_v1.to_pickle(output_path + 'X_test_c_add_v1.pickle')

# +
# w2v_cell
X_feature_w2v_cell = pd.concat([X_128.reset_index(drop=True),
               X_imp.reset_index(drop=True),
               X_imp_norm.reset_index(drop=True),
               X_imp_feature,
               X_cluster.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_col_cluster.reset_index(drop=True),
               X_cnorm.reset_index(drop=True),
               X_imp_cnorm.reset_index(drop=True),
               X_w2v.reset_index(drop=True),
               X_imp_w2v.reset_index(drop=True),
               X_imp_cell.reset_index(drop=True),
               X_cell_ratio.reset_index(drop=True),
               ], axis = 1)

X_test_feature_w2v_cell = pd.concat([X_test_128.reset_index(drop=True),
                    X_test_imp.reset_index(drop=True),
                    X_test_imp_norm.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_col_cluster.reset_index(drop=True),
                   X_test_cnorm.reset_index(drop=True),
                   X_test_imp_cnorm.reset_index(drop=True),
                    X_test_w2v.reset_index(drop=True),
                    X_test_imp_w2v.reset_index(drop=True),
                   X_test_imp_cell.reset_index(drop=True),
                   X_test_cell_ratio.reset_index(drop=True),
                    ], axis = 1)

X_feature_w2v_cell = X_feature_w2v_cell.fillna(0).astype(np.float32)
X_test_feature_w2v_cell = X_test_feature_w2v_cell.fillna(0).astype(np.float32)

print(X_feature_w2v_cell.shape, X_test_feature_w2v_cell.shape)

X_feature_w2v_cell.to_pickle(output_path + 'X_feature_w2v_cell.pickle')
X_test_feature_w2v_cell.to_pickle(output_path + 'X_test_feature_w2v_cell.pickle')

# +
# best_128 with cell
X_best_cell_128_120 = pd.concat([X_128_120.reset_index(drop=True),
               X_imp_120.reset_index(drop=True),
               X_imp_norm_120.reset_index(drop=True),
               X_imp_feature_120,
               X_cluster_120.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_col_cluster.reset_index(drop=True),
               X_cnorm_120.reset_index(drop=True),
               X_imp_cnorm_120.reset_index(drop=True),
               X_imp_cell_120.reset_index(drop=True),
               X_cell_ratio.reset_index(drop=True),
               ], axis = 1)

X_test_best_cell_128_120 = pd.concat([X_test_128_120.reset_index(drop=True),
                    X_test_imp_120.reset_index(drop=True),
                    X_test_imp_norm_120.reset_index(drop=True),
                    X_test_imp_feature_120,
                    X_test_cluster_120.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_col_cluster.reset_index(drop=True),
                    X_test_cnorm_120.reset_index(drop=True),
                    X_test_imp_cnorm_120.reset_index(drop=True),
                   X_test_imp_cell_120.reset_index(drop=True),
                   X_test_cell_ratio.reset_index(drop=True),
                    ], axis = 1)

X_best_cell_128_120 = X_best_cell_128_120.fillna(0).astype(np.float32)
X_test_best_cell_128_120 = X_test_best_cell_128_120.fillna(0).astype(np.float32)

print(X_best_cell_128_120.shape, X_test_best_cell_128_120.shape)

X_best_cell_128_120.to_pickle(output_path + 'X_best_cell_128_120.pickle')
X_test_best_cell_128_120.to_pickle(output_path + 'X_test_best_cell_128_120.pickle')

# +
# cluster_cell
X_cluster_cell_128 = pd.concat([X_128.reset_index(drop=True),
               X_imp.reset_index(drop=True),
               X_imp_norm.reset_index(drop=True),
               X_imp_feature,
               X_cluster.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_imp_cell.reset_index(drop=True),
               X_cell_ratio.reset_index(drop=True),
               ], axis = 1)

X_test_cluster_cell_128 = pd.concat([X_test_128.reset_index(drop=True),
                    X_test_imp.reset_index(drop=True),
                    X_test_imp_norm.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_imp_cell.reset_index(drop=True),
                    X_test_cell_ratio.reset_index(drop=True),
                    ], axis = 1)

X_cluster_cell_128 = X_cluster_cell_128.fillna(0).astype(np.float32)
X_test_cluster_cell_128 = X_test_cluster_cell_128.fillna(0).astype(np.float32)

print(X_cluster_cell_128.shape, X_test_cluster_cell_128.shape)

X_cluster_cell_128.to_pickle(output_path + 'X_cluster_cell_128.pickle')
X_test_cluster_cell_128.to_pickle(output_path + 'X_test_cluster_cell_128.pickle')

# +
# w2v

X_feature_w2v = pd.concat([X_128.reset_index(drop=True),
               X_imp.reset_index(drop=True),
               X_imp_norm.reset_index(drop=True),
               X_imp_feature,
               X_cluster.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_col_cluster.reset_index(drop=True),
               X_cnorm.reset_index(drop=True),
               X_imp_cnorm.reset_index(drop=True),
               X_w2v.reset_index(drop=True),
               X_imp_w2v.reset_index(drop=True),
               ], axis = 1)

X_test_feature_w2v = pd.concat([X_test_128.reset_index(drop=True),
                    X_test_imp.reset_index(drop=True),
                    X_test_imp_norm.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_col_cluster.reset_index(drop=True),
                   X_test_cnorm.reset_index(drop=True),
                   X_test_imp_cnorm.reset_index(drop=True),
                    X_test_w2v.reset_index(drop=True),
                    X_test_imp_w2v.reset_index(drop=True),
                    ], axis = 1)

X_feature_w2v = X_feature_w2v.astype(np.float32)
X_test_feature_w2v = X_test_feature_w2v.astype(np.float32)

print(X_feature_w2v.shape, X_test_feature_w2v.shape)

X_feature_w2v.to_pickle(output_path + 'X_feature_w2v.pickle')
X_test_feature_w2v.to_pickle(output_path + 'X_test_feature_w2v.pickle')

# +
# imp_w2v
X_feature_imp_w2v = pd.concat([X_128.reset_index(drop=True),
               X_imp.reset_index(drop=True),
               X_imp_norm.reset_index(drop=True),
               X_imp_feature,
               X_cluster.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_col_cluster.reset_index(drop=True),
               X_cnorm.reset_index(drop=True),
               X_imp_cnorm.reset_index(drop=True),
               #X_w2v.reset_index(drop=True),
               X_imp_w2v.reset_index(drop=True),
               ], axis = 1)

X_test_feature_imp_w2v = pd.concat([X_test_128.reset_index(drop=True),
                    X_test_imp.reset_index(drop=True),
                    X_test_imp_norm.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_col_cluster.reset_index(drop=True),
                    X_test_cnorm.reset_index(drop=True),
                    X_test_imp_cnorm.reset_index(drop=True),
                    #X_test_w2v.reset_index(drop=True),
                    X_test_imp_w2v.reset_index(drop=True),
                    ], axis = 1)

X_feature_imp_w2v = X_feature_imp_w2v.astype(np.float32)
X_test_feature_imp_w2v = X_test_feature_imp_w2v.astype(np.float32)

print(X_feature_imp_w2v.shape, X_test_feature_imp_w2v.shape)

X_feature_imp_w2v.to_pickle(output_path + 'X_feature_imp_w2v.pickle')
X_test_feature_imp_w2v.to_pickle(output_path + 'X_test_feature_imp_w2v.pickle')

# +
# snorm
X_feature_snorm = pd.concat([X_128.reset_index(drop=True),
               X_imp.reset_index(drop=True),
               X_imp_norm.reset_index(drop=True),
               X_imp_feature,
               X_cluster.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_col_cluster.reset_index(drop=True),
               X_cnorm.reset_index(drop=True),
               X_imp_cnorm.reset_index(drop=True),
               X_w2v.reset_index(drop=True),
               X_imp_w2v.reset_index(drop=True),
               X_imp_snorm.reset_index(drop=True),
               ], axis = 1)
X_test_feature_snorm = pd.concat([
                    X_test_128.reset_index(drop=True),
                    X_test_imp.reset_index(drop=True),
                    X_test_imp_norm.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_col_cluster.reset_index(drop=True),
                   X_test_cnorm.reset_index(drop=True),
                   X_test_imp_cnorm.reset_index(drop=True),
                    X_test_w2v.reset_index(drop=True),
                    X_test_imp_w2v.reset_index(drop=True),
                    X_test_imp_snorm.reset_index(drop=True),
                    ], axis = 1)

X_feature_snorm = X_feature_snorm.astype(np.float32)
X_test_feature_snorm = X_test_feature_snorm.astype(np.float32)

print(X_feature_snorm.shape, X_test_feature_snorm.shape)

X_feature_snorm.to_pickle(output_path + 'X_feature_snorm.pickle')
X_test_feature_snorm.to_pickle(output_path + 'X_test_feature_snorm.pickle')

# +
# best_128
X_best_128 = pd.concat([X_128.reset_index(drop=True),
               X_imp.reset_index(drop=True),
               X_imp_norm.reset_index(drop=True),
               X_imp_feature,
               X_cluster.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_col_cluster.reset_index(drop=True),
               X_cnorm.reset_index(drop=True),
               X_imp_cnorm.reset_index(drop=True),
               ], axis = 1)

X_test_best_128 = pd.concat([X_test_128.reset_index(drop=True),
                    X_test_imp.reset_index(drop=True),
                    X_test_imp_norm.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_col_cluster.reset_index(drop=True),
                   X_test_cnorm.reset_index(drop=True),
                   X_test_imp_cnorm.reset_index(drop=True),
                    ], axis = 1)

print(X_best_128.shape, X_test_best_128.shape)

X_best_128.to_pickle(output_path + 'X_best_128.pickle')
X_test_best_128.to_pickle(output_path + 'X_test_best_128.pickle')

# +
# best_64
X_best_64 = pd.concat([X_64.reset_index(drop=True),
               X_imp.reset_index(drop=True),
               X_imp_norm.reset_index(drop=True),
               X_imp_feature,
               X_cluster.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               X_col_cluster.reset_index(drop=True),
               X_cnorm.reset_index(drop=True),
               X_imp_cnorm.reset_index(drop=True),
               ], axis = 1)

X_test_best_64 = pd.concat([X_test_64.reset_index(drop=True),
                    X_test_imp.reset_index(drop=True),
                    X_test_imp_norm.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    X_test_col_cluster.reset_index(drop=True),
                   X_test_cnorm.reset_index(drop=True),
                   X_test_imp_cnorm.reset_index(drop=True),
                    ], axis = 1)

print(X_best_64.shape, X_test_best_64.shape)

X_best_64.to_pickle(output_path + 'X_best_64.pickle')
X_test_best_64.to_pickle(output_path + 'X_test_best_64.pickle')

# +
# only cluster
X_cluster_128 = pd.concat([X_128.reset_index(drop=True),
               X_imp.reset_index(drop=True),
               X_imp_norm.reset_index(drop=True),
               X_imp_feature,
               X_cluster.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               ], axis = 1)

X_test_cluster_128 = pd.concat([X_test_128.reset_index(drop=True),
                    X_test_imp.reset_index(drop=True),
                    X_test_imp_norm.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    ], axis = 1)

print(X_cluster_128.shape, X_test_cluster_128.shape)

X_cluster_128.to_pickle(output_path + 'X_cluster_128.pickle')
X_test_cluster_128.to_pickle(output_path + 'X_test_cluster_128.pickle')

# +
# only cluster
X_cluster_64 = pd.concat([X_64.reset_index(drop=True),
               X_imp.reset_index(drop=True),
               X_imp_norm.reset_index(drop=True),
               X_imp_feature,
               X_cluster.reset_index(drop=True),
               X_all_cluster.reset_index(drop=True),
               ], axis = 1)

X_test_cluster_64 = pd.concat([X_test_64.reset_index(drop=True),
                    X_test_imp.reset_index(drop=True),
                    X_test_imp_norm.reset_index(drop=True),
                    X_test_imp_feature,
                    X_test_cluster.reset_index(drop=True),
                    X_test_all_cluster.reset_index(drop=True),
                    ], axis = 1)

print(X_cluster_64.shape, X_test_cluster_64.shape)

X_cluster_64.to_pickle(output_path + 'X_cluster_64.pickle')
X_test_cluster_64.to_pickle(output_path + 'X_test_cluster_64.pickle')

# +
# only svd & best
X_svd_128 = pd.concat([X_128.reset_index(drop=True),
                       X_imp.reset_index(drop=True),
                      ], axis = 1)

X_test_svd_128 = pd.concat([X_test_128.reset_index(drop=True),
                            X_test_imp.reset_index(drop=True),
                           ], axis = 1)

print(X_svd_128.shape, X_test_svd_128.shape)

X_svd_128.to_pickle(output_path + 'X_svd_128.pickle')
X_test_svd_128.to_pickle(output_path + 'X_test_svd_128.pickle')

# +
# only svd & best
X_svd_64 = pd.concat([X_64.reset_index(drop=True),
                       X_imp.reset_index(drop=True),
                      ], axis = 1)

X_test_svd_64 = pd.concat([X_test_64.reset_index(drop=True),
                            X_test_imp.reset_index(drop=True),
                           ], axis = 1)

print(X_svd_64.shape, X_test_svd_64.shape)

X_svd_64.to_pickle(output_path + 'X_svd_64.pickle')
X_test_svd_64.to_pickle(output_path + 'X_test_svd_64.pickle')
