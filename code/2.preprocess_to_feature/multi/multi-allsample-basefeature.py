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

# https://www.kaggle.com/competitions/open-problems-multimodal/discussion/360384

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import scipy
from tqdm import notebook as tqdm
import pickle
import gensim
from gensim.models import word2vec

# +
# %matplotlib inline
from tqdm.notebook import tqdm
import gc

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# -

import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from muon import atac as ac
from muon import prot as pt
import muon


# ## data load

def save(name, model):
    with open(name, 'wb') as f:
        pickle.dump(model, f)


raw_path_base = '../../../input/raw/'
raw_cite_path = '../../../input/preprocess/cite/'
raw_multi_path = '../../../input/preprocess/multi/'
feature_path = '../../../input/base_features/multi/'
#feature_path = '../../../../summary/input/sample/'

# feat
df_meta = pd.read_csv(raw_path_base + "metadata.csv")
train = scipy.sparse.load_npz(raw_multi_path + "train_multi_inputs_values.sparse.npz")
test = scipy.sparse.load_npz(raw_multi_path + "test_multi_inputs_values.sparse.npz")
train_raw = scipy.sparse.load_npz(raw_multi_path + "train_multi_raw_inputs_values.sparse.npz")
test_raw = scipy.sparse.load_npz(raw_multi_path +  "test_multi_raw_inputs_values.sparse.npz")

# +
train_ids = np.load(raw_multi_path + "train_multi_inputs_idxcol.npz", allow_pickle=True)
test_ids = np.load(raw_multi_path + "test_multi_inputs_idxcol.npz", allow_pickle=True)
train_raw_ids = np.load(raw_multi_path + "train_multi_raw_inputs_idxcol.npz", allow_pickle=True)
test_raw_ids = np.load(raw_multi_path + "test_multi_raw_inputs_idxcol.npz", allow_pickle=True)

train_index = train_ids["index"]
test_index = test_ids["index"]
multi_cols = train_ids["columns"]
train_raw_index = train_raw_ids["index"]
test_raw_index = test_raw_ids["index"]
# -

base_df = pd.DataFrame(train_index, columns = ['cell_id'])
raw_df = pd.DataFrame(train_raw_index, columns = ['cell_id'])
raw_df['flg'] = 1
base_df = base_df.merge(raw_df, on = 'cell_id', how = 'left')
use_only_base_index = list(base_df[base_df['flg'] != 1].index)

train_only_base = train[use_only_base_index]

train_only_base

all_raw = scipy.sparse.vstack([train_raw, test_raw])

all_raw_ids = list(train_raw_index) + list(test_raw_index) + list(train_index[use_only_base_index])
train_ids = list(train_raw_index) + list(train_index[use_only_base_index])
test_ids = list(test_raw_index)

del train_raw, test_raw
gc.collect()

# ## base features

# ### bm25

# +
# coding: UTF-8
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.feature_extraction.text import _document_frequency

class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b : float, optional (default=0.75)

    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b

    def fit(self, X):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
        return self

    def transform(self, X, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        dl = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]
        # In each row, repeat `dl` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(dl), sz)
        # Average document length
        # Scalar value
        avgdl = np.average(dl)
        # Compute BM25 score only for non-zero elements
        data = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        return X


# -

# %%time
bm25 = BM25Transformer()
all_raw = bm25.fit_transform(all_raw)
all_raw = scipy.sparse.vstack([all_raw, train_only_base])

all_raw

# ### simple svd

svd_64 = TruncatedSVD(n_components=64, random_state=1) # 512
embedding_feature = svd_64.fit_transform(all_raw)

train_simple_svd = np.concatenate([embedding_feature[:105933], embedding_feature[105933 + 55935:]])
test_simple_svd = embedding_feature[105933:105933 + 55935]

train_simple_svd = pd.DataFrame(train_simple_svd, index = train_ids).add_prefix('svd_simple_')
test_simple_svd = pd.DataFrame(test_simple_svd, index = test_ids).add_prefix('svd_simple_')

# ### only okapi with col cluster

all_raw_okapi = sc.AnnData(X = all_raw)
ac.tl.lsi(all_raw_okapi)

# +
svd_64 = TruncatedSVD(n_components=64, random_state=1) # 512
feature_okapi_64 = svd_64.fit_transform(all_raw_okapi.X)

train_okapi_64 = np.concatenate([feature_okapi_64[:105933], feature_okapi_64[105933 + 55935:]])
test_okapi_64 = feature_okapi_64[105933:105933 + 55935]

train_okapi_64 = pd.DataFrame(train_okapi_64, index = train_ids).add_prefix('svd_all_')
train_okapi_64.to_pickle(feature_path + 'multi_train_okapi_feature_64.pickle')

test_okapi_64 = pd.DataFrame(test_okapi_64, index = test_ids).add_prefix('svd_all_')
test_okapi_64.to_pickle(feature_path + 'multi_test_okapi_feature_64.pickle')
# -

# ### col cluster

all_raw_okapi = all_raw_okapi.T

col_svd_128 = TruncatedSVD(n_components=128, random_state=1) # 512
feature_col_128 = col_svd_128.fit_transform(all_raw_okapi.X)

feature_col_128 = sc.AnnData(X = feature_col_128)
sc.pp.neighbors(feature_col_128, use_rep = 'X', n_neighbors=32, method='umap')
sc.tl.leiden(feature_col_128)

col_cluster_df = pd.DataFrame(list(feature_col_128.obs['leiden']), columns = ['leiden_col'])
col_cluster_df['gene_id'] = multi_cols
col_cluster_df['leiden_col'] = col_cluster_df['leiden_col'].astype('int')

# +
df_col = pd.DataFrame()

for i in range(len(col_cluster_df['leiden_col'].unique())):
    print(i)
    df_col[f'cluster_{i}'] = np.asarray(all_raw[:,col_cluster_df[col_cluster_df['leiden_col'] == i].index].mean(1)).flatten()
# -

output = np.array(df_col).astype(np.float32)
train_col = pd.DataFrame(np.concatenate([output[:105933], output[105933 + 55935:]]), index = train_ids).add_prefix('cluster_col_mean_')
test_col = pd.DataFrame(np.array(output[105933:105933 + 55935]), index = test_ids).add_prefix('cluster_col_mean_')

del all_raw_okapi, feature_col_128
gc.collect()

# ### lsi

# +
all_raw_lsi = sc.AnnData(X = all_raw)
muon.atac.tl.lsi(all_raw_lsi, n_comps=64)
feature_lsi_64 = all_raw_lsi.obsm['X_lsi']

train_lsi_64 = np.concatenate([feature_lsi_64[:105933], feature_lsi_64[105933 + 55935:]])
test_lsi_64 = feature_lsi_64[105933:105933 + 55935]

train_lsi_64 = pd.DataFrame(train_lsi_64, index = train_ids).add_prefix('svd_lsi_')
train_lsi_64.to_pickle(feature_path + 'multi_train_okapi_lsi_64.pickle')

test_lsi_64 = pd.DataFrame(test_lsi_64, index = test_ids).add_prefix('svd_lsi_')
test_lsi_64.to_pickle(feature_path + 'multi_test_okapi_lsi_64.pickle')

# +
all_raw_lsi = sc.AnnData(X = all_raw)
muon.atac.tl.lsi(all_raw_lsi, n_comps=128)
feature_lsi_128 = all_raw_lsi.obsm['X_lsi']

train_lsi_128 = np.concatenate([feature_lsi_128[:105933], feature_lsi_128[105933 + 55935:]])
test_lsi_128 = feature_lsi_128[105933:105933 + 55935]

train_lsi_128 = pd.DataFrame(train_lsi_128, index = train_ids).add_prefix('svd_lsi_')
train_lsi_128.to_pickle(feature_path + 'multi_train_okapi_lsi_128.pickle')

test_lsi_128 = pd.DataFrame(test_lsi_128, index = test_ids).add_prefix('svd_lsi_')
test_lsi_128.to_pickle(feature_path + 'multi_test_okapi_lsi_128.pickle')
# -

# ### con(connectivities) feature

embedding_feature = sc.AnnData(X = feature_lsi_64)

# %%time
sc.pp.neighbors(embedding_feature, use_rep = 'X', n_neighbors=32, method='umap')

svd_16 = TruncatedSVD(n_components=16, random_state=1)
feature_con_16 = svd_16.fit_transform(embedding_feature.obsp['connectivities'])

svd_32 = TruncatedSVD(n_components=32, random_state=1)
feature_con_32 = svd_32.fit_transform(embedding_feature.obsp['connectivities'])

# +
train_con_16 = pd.DataFrame(np.concatenate([feature_con_16[:105933],
                                            feature_con_16[105933 + 55935:]]),
                                           index = train_ids).add_prefix('con_svd_')

test_con_16 = pd.DataFrame(np.array(feature_con_16[105933:105933 + 55935]),
                           index = test_ids).add_prefix('con_svd_')

# +
train_con_32 = pd.DataFrame(np.concatenate([feature_con_32[:105933],
                                            feature_con_32[105933 + 55935:]]),
                                           index = train_ids).add_prefix('con_svd_')

test_con_32 = pd.DataFrame(np.array(feature_con_32[105933:105933 + 55935]),
                           index = test_ids).add_prefix('con_svd_')
# -

del all_raw_lsi
gc.collect()

# ### lsi cluster(last cluster)

embedding_feature = sc.AnnData(X = feature_lsi_64)

sc.pp.neighbors(embedding_feature, use_rep = 'X', n_neighbors=16, method='umap')
sc.tl.leiden(embedding_feature)

cluster_df = pd.DataFrame(embedding_feature.obs['leiden'].astype(int), columns = ['leiden'])

# +
result_df = pd.DataFrame()

for i in tqdm(range(230)):

    start = i * 1000
    end = (i + 1) * 1000

    if end > all_raw.shape[1]:
        end = all_raw.shape[1]

    row_df = pd.DataFrame(all_raw[:,start:end].toarray())
    chunk_df = pd.concat([cluster_df.reset_index(drop=True), row_df], axis = 1).groupby('leiden').mean().reset_index().iloc[:,1:]
    result_df = pd.concat([result_df, chunk_df], axis = 1)

    if end == all_raw.shape[1]:
        break
# -

svd_16 = TruncatedSVD(n_components=16, random_state=1) # 512
cluster_mean_feature = svd_16.fit_transform(np.array(result_df))

cluster_factor = pd.DataFrame(cluster_mean_feature).reset_index().rename(columns = {'index': 'leiden'})

output = cluster_df.merge(cluster_factor, on = 'leiden', how = 'left')
output.drop(['leiden'], axis = 1, inplace = True)

train_last_cluster = pd.DataFrame(np.concatenate([output[:105933], output[105933 + 55935:]]), index = train_ids).add_prefix('last_c_')
test_last_cluster = pd.DataFrame(np.array(output[105933:105933 + 55935]), index = test_ids).add_prefix('last_c_')

# ## other features

# ### binary

all_raw_binary = all_raw / all_raw.max()
all_raw_binary = all_raw_binary.ceil()

# +
svd_64 = TruncatedSVD(n_components=64, random_state=1) # 512
feature_binary = svd_64.fit_transform(all_raw_binary)

del all_raw_binary
gc.collect()

# +
train_binary = np.concatenate([feature_binary[:105933], feature_binary[105933 + 55935:]])
test_binary = feature_binary[105933:105933 + 55935]

train_binary = pd.DataFrame(train_binary, index = train_ids).add_prefix('binary_')
test_binary = pd.DataFrame(test_binary, index = test_ids).add_prefix('binary_')
# -

# ### w2v

train_num = all_raw.shape[0]

train_num

# +
# %%time
sort_list = []

for i in tqdm(range(162)):
#for i in tqdm(range(2)):

    start = i * 1000
    end = (i + 1) * 1000

    if end > train_num:
        end = train_num

    array = all_raw[start:end,:].toarray()
    sort_array = np.argsort(-array)[:,0:100].astype(str)

    for i in sort_array:
        sort_list.append(list(i))

    if end == train_num:
        print('stop iter')
        break
# -

# %%time
model = word2vec.Word2Vec(sentences=sort_list, size=16, window=5, min_count=1)
word_vectors = model.wv

# +
word_vec_list = []

for l in range(len(sort_list)):

    for i, j in enumerate(sort_list[l]):
        if i == 0:
            word_vec = word_vectors[j] / len(sort_list[l])
        else:
            word_vec += word_vectors[j] / len(sort_list[l])

    word_vec_list.append(word_vec)
# -

w2v_all_df = pd.DataFrame(word_vec_list).add_prefix('w2v_top100_').astype(np.float32)

del word_vec_list, word_vec

train_w2v = pd.concat([w2v_all_df.iloc[:105933,:], w2v_all_df.iloc[105933 + 55935:]]).reset_index(drop=True)
test_w2v = w2v_all_df.iloc[105933:105933 + 55935].reset_index(drop=True)

train_w2v.index = train_ids
test_w2v.index = test_ids

# +
#train_w2v.to_pickle('../../../data/feature/multi/multi_train_w2v_top100.pickle')
#test_w2v.to_pickle('../../../data/feature/multi/multi_test_w2v_top100.pickle')
# -

# ### make features

# +
# base
train_okapi_64.to_pickle(feature_path + 'multi_train_okapi_feature_64.pickle')
test_okapi_64.to_pickle(feature_path + 'multi_test_okapi_feature_64.pickle')

pd.concat([train_okapi_64, train_w2v], axis = 1).to_pickle(feature_path + 'multi_train_okapi_w2v_64.pickle')
pd.concat([test_okapi_64, test_w2v], axis = 1).to_pickle(feature_path + 'multi_test_okapi_w2v_64.pickle')

pd.concat([train_okapi_64, train_w2v, train_col], axis = 1).to_pickle(feature_path + 'multi_train_okapi_w2v_col_64.pickle')
pd.concat([test_okapi_64, test_w2v, test_col], axis = 1).to_pickle(feature_path + 'multi_test_okapi_w2v_col_64.pickle')
# -

print(pd.concat([train_okapi_64, train_w2v, train_col], axis = 1).shape,
      pd.concat([test_okapi_64, test_w2v, test_col], axis = 1).shape)

# +
# lsi 64
train_lsi_64.to_pickle(feature_path + 'multi_train_okapi_lsi_64.pickle')
test_lsi_64.to_pickle(feature_path + 'multi_test_okapi_lsi_64.pickle')

pd.concat([train_lsi_64, train_w2v], axis = 1).to_pickle(feature_path + 'multi_train_lsi_w2v_64.pickle')
pd.concat([test_lsi_64, test_w2v], axis = 1).to_pickle(feature_path + 'multi_test_lsi_w2v_64.pickle')

pd.concat([train_lsi_64, train_w2v, train_col], axis = 1).to_pickle(feature_path + 'multi_train_lsi_w2v_col_64.pickle')
pd.concat([test_lsi_64, test_w2v, test_col], axis = 1).to_pickle(feature_path + 'multi_test_lsi_w2v_col_64.pickle')
# -

print(pd.concat([train_lsi_64, train_w2v, train_col], axis = 1).shape,
      pd.concat([test_lsi_64, test_w2v, test_col], axis = 1).shape)

# +
# lsi 128
train_lsi_128.to_pickle(feature_path + 'multi_train_okapi_lsi_128.pickle')
test_lsi_128.to_pickle(feature_path + 'multi_test_okapi_lsi_128.pickle')

pd.concat([train_lsi_128, train_w2v], axis = 1).to_pickle(feature_path + 'multi_train_lsi_w2v_128.pickle')
pd.concat([test_lsi_128, test_w2v], axis = 1).to_pickle(feature_path + 'multi_test_lsi_w2v_128.pickle')

pd.concat([train_lsi_128, train_w2v, train_col], axis = 1).to_pickle(feature_path + 'multi_train_lsi_w2v_col_128.pickle')
pd.concat([test_lsi_128, test_w2v, test_col], axis = 1).to_pickle(feature_path + 'multi_test_lsi_w2v_col_128.pickle')
# -

print(pd.concat([train_lsi_128, train_w2v, train_col], axis = 1).shape,
      pd.concat([test_lsi_128, test_w2v, test_col], axis = 1).shape)

# multi_train_lc_addsvd_64
pd.concat([train_simple_svd, train_last_cluster], axis = 1).to_pickle(feature_path + 'multi_train_lc_addsvd_64.pickle')
pd.concat([test_simple_svd, test_last_cluster], axis = 1).to_pickle(feature_path + 'multi_test_lc_addsvd_64.pickle')

print(pd.concat([train_simple_svd, train_last_cluster], axis = 1).shape,
      pd.concat([test_simple_svd, test_last_cluster], axis = 1).shape)

# binary
pd.concat([train_lsi_64, train_w2v, train_col, train_binary], axis = 1).to_pickle(feature_path + 'multi_train_binary_16.pickle')
pd.concat([test_lsi_64, test_w2v, test_col, test_binary], axis = 1).to_pickle(feature_path + 'multi_test_binary_16.pickle')

print(pd.concat([train_lsi_64, train_w2v, train_col, train_binary], axis = 1).shape,
      pd.concat([test_lsi_64, test_w2v, test_col, test_binary], axis = 1).shape)

# con16
pd.concat([train_lsi_64, train_w2v, train_col, train_binary, train_con_16], axis = 1).to_pickle(feature_path + 'multi_train_con_16.pickle')
pd.concat([test_lsi_64, test_w2v, test_col, test_binary, test_con_16], axis = 1).to_pickle(feature_path + 'multi_test_con_16.pickle')

print(pd.concat([train_lsi_64, train_w2v, train_col, train_binary, train_con_16], axis = 1).shape,
      pd.concat([test_lsi_64, test_w2v, test_col, test_binary, test_con_16], axis = 1).shape)

# con32
pd.concat([train_lsi_64, train_w2v, train_col, train_binary, train_con_32], axis = 1).to_pickle(feature_path + 'multi_train_con_32.pickle')
pd.concat([test_lsi_64, test_w2v, test_col, test_binary, test_con_32], axis = 1).to_pickle(feature_path + 'multi_test_con_32.pickle')

print(pd.concat([train_lsi_64, train_w2v, train_col, train_binary, train_con_32], axis = 1).shape,
      pd.concat([test_lsi_64, test_w2v, test_col, test_binary, test_con_32], axis = 1).shape)
