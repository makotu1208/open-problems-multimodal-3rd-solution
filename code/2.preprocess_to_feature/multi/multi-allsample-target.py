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

# +
# %matplotlib inline
from tqdm.notebook import tqdm
import gc

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
#from torch.optim import Adam, SGD, AdamW
#from torch.optim.lr_scheduler import CosineAnnealingLR

#from transformers import AdamW
#from transformers import get_cosine_schedule_with_warmup
#from sklearn.preprocessing import RobustScaler, QuantileTransformer

device = torch.device("cuda")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# -

import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from muon import atac as ac
from muon import prot as pt


# ## data load

def save(name, model):
    with open(name, 'wb') as f:
        pickle.dump(model, f)


raw_path_base = '../../../input/raw/'
raw_cite_path = '../../../input/preprocess/cite/'
raw_multi_path = '../../../input/preprocess/multi/'
target_path = '../../../../summary/input/target/multi/'
#target_path = '../../../input/sample/'

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
train_raw_index = train_raw_ids["index"]
test_raw_index = test_raw_ids["index"]
# -

base_df = pd.DataFrame(train_index, columns = ['cell_id'])
raw_df = pd.DataFrame(train_raw_index, columns = ['cell_id'])
raw_df['flg'] = 1
base_df = base_df.merge(raw_df, on = 'cell_id', how = 'left')
use_only_base_index = list(base_df[base_df['flg'] != 1].index)

train_only_base = train[use_only_base_index]

# %%time
train_raw = sc.AnnData(X = train_raw)
sc.pp.normalize_per_cell(train_raw, counts_per_cell_after = 1e6)
sc.pp.log1p(train_raw)

train_raw.X = train_raw.X.astype(np.float32)

all_target_raw = scipy.sparse.vstack([train_raw.X, train_only_base])

all_target_raw

all_raw_ids = list(train_raw_index) + list(train_index[use_only_base_index])

len(all_raw_ids)

# ### target:128dims

# %%time
svd_128 = TruncatedSVD(n_components=128, random_state=1) # 512
target_128 = svd_128.fit_transform(all_target_raw)
save('../../../data/raw_2/multi_all_target_128.pkl', svd_128)

pd.DataFrame(target_128, index = all_raw_ids).add_prefix('target_').to_pickle(target_path + 'multi_train_target_128.pickle')

# ### target:alldims

pd.DataFrame(all_target_raw.toarray().astype(np.float32), index = all_raw_ids).add_prefix('target_').to_pickle(target_path + 'multi_train_target_all.pickle')
