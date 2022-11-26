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
import random
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GroupKFold

# +
# %matplotlib inline
from tqdm.notebook import tqdm
import gc
import pickle

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
# -

# ## data load

# +
preprocess_path = '../../../../input/preprocess/cite/'
validation_path = '../../../../input/fold/'

target_path = '../../../../input/target/'
feature_path = '../../../../input/features/cite/'
output_path = '../../../../model/cite/mlp/'

# +
df_meta = pd.read_pickle(validation_path + "cite_fold_val_df.pickle")
X = np.load(preprocess_path + 'train_cite_inputs_idxcol.npz', allow_pickle=True)

fold_df = pd.DataFrame(list(X['index']), columns = ['cell_id'])
fold_df = fold_df.merge(df_meta[['cell_id', 'flg_donor_val', 'flg_all_val', 'rank_per_donor']], on = ['cell_id'], how = 'inner')
fold_df.shape

del X
# -

Y = pd.read_hdf(target_path + "train_cite_targets.h5")
Y = np.array(Y)

# ### feature path

# +
feature_dict = {}

feature_dict['add_con_imp'] = ['X_add_con_imp.pickle', 'X_test_add_con_imp.pickle']
feature_dict['last_v3'] = ['X_last_v3.pickle', 'X_test_last_v3.pickle']
feature_dict['c_add_w2v_v1_mish'] = ['X_c_add_w2v_v1.pickle', 'X_test_c_add_w2v_v1.pickle']
feature_dict['c_add_w2v_v1'] = ['X_c_add_w2v_v1.pickle', 'X_test_c_add_w2v_v1.pickle']
feature_dict['c_add_84_v1'] = ['X_c_add_84_v1.pickle', 'X_test_c_add_84_v1.pickle']
feature_dict['c_add_120_v1'] = ['X_c_add_v1.pickle', 'X_test_c_add_v1.pickle']

feature_dict['w2v_cell'] = ['X_feature_w2v_cell.pickle', 'X_test_feature_w2v_cell.pickle']
feature_dict['best_cell_120'] = ['X_best_cell_128_120.pickle', 'X_test_best_cell_128_120.pickle']
feature_dict['cluster_cell'] = ['X_cluster_cell_128.pickle', 'X_test_cluster_cell_128.pickle']

feature_dict['w2v_128'] = ['X_feature_w2v.pickle', 'X_test_feature_w2v.pickle']
feature_dict['imp_w2v_128'] = ['X_feature_imp_w2v.pickle', 'X_test_feature_imp_w2v.pickle']
feature_dict['snorm'] = ['X_feature_snorm.pickle', 'X_test_feature_snorm.pickle']
feature_dict['best_128'] = ['X_best_128.pickle', 'X_test_best_128.pickle']
feature_dict['best_64'] = ['X_best_64.pickle', 'X_test_best_64.pickle']
feature_dict['cluster_128'] = ['X_cluster_128.pickle', 'X_test_cluster_128.pickle']
feature_dict['cluster_64'] = ['X_cluster_64.pickle', 'X_test_cluster_64.pickle']
feature_dict['svd_128'] = ['X_svd_128.pickle', 'X_test_svd_128.pickle']
feature_dict['svd_64'] = ['X_svd_64.pickle', 'X_test_svd_64.pickle']


# -

# ### function

class CiteDataset(Dataset):

    def __init__(self, feature, target):

        self.feature = feature
        self.target = target

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):

        d = {
            "X": self.feature[index],
            "y" : self.target[index],
        }
        return d


class CiteDataset_test(Dataset):

    def __init__(self, feature):
        self.feature = feature

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, index):

        d = {
            "X": self.feature[index]
        }
        return d


# +
def partial_correlation_score_torch_faster(y_true, y_pred):
    """Compute the correlation between each rows of the y_true and y_pred tensors.
    Compatible with backpropagation.
    """
    y_true_centered = y_true - torch.mean(y_true, dim=1)[:,None]
    y_pred_centered = y_pred - torch.mean(y_pred, dim=1)[:,None]
    cov_tp = torch.sum(y_true_centered*y_pred_centered, dim=1)/(y_true.shape[1]-1)
    var_t = torch.sum(y_true_centered**2, dim=1)/(y_true.shape[1]-1)
    var_p = torch.sum(y_pred_centered**2, dim=1)/(y_true.shape[1]-1)
    return cov_tp/torch.sqrt(var_t*var_p)

def correl_loss(pred, tgt):
    """Loss for directly optimizing the correlation.
    """
    return -torch.mean(partial_correlation_score_torch_faster(tgt, pred))


# -

class CiteModel(nn.Module):

    def __init__(self, feature_num):
        super(CiteModel, self).__init__()

        self.layer_seq_256 = nn.Sequential(nn.Linear(feature_num, 256),
                                           nn.Linear(256, 128),
                                       nn.LayerNorm(128),
                                       nn.ReLU(),
                                      )
        self.layer_seq_64 = nn.Sequential(nn.Linear(128, 64),
                                       nn.Linear(64, 32),
                                       nn.LayerNorm(32),
                                       nn.ReLU(),
                                      )
        self.layer_seq_8 = nn.Sequential(nn.Linear(32, 16),
                                         nn.Linear(16, 8),
                                       nn.LayerNorm(8),
                                       nn.ReLU(),
                                      )
        self.dropout = nn.Dropout(0.1)
        self.head = nn.Linear(128 + 32 + 8, 140)

    def forward(self, X, y=None):

        X_256 = self.layer_seq_256(X)
        X_64 = self.layer_seq_64(X_256)
        X_8 = self.layer_seq_8(X_64)

        X = torch.cat([X_256, X_64, X_8], axis = 1)
        out = self.head(X)

        return out


class CiteModel_mish(nn.Module):

    def __init__(self, feature_num):
        super(CiteModel_mish, self).__init__()

        self.layer_seq_256 = nn.Sequential(nn.Linear(feature_num, 256),
                                           nn.Linear(256, 128),
                                       nn.LayerNorm(128),
                                       nn.Mish(),
                                      )
        self.layer_seq_64 = nn.Sequential(nn.Linear(128, 64),
                                       nn.Linear(64, 32),
                                       nn.LayerNorm(32),
                                       nn.Mish(),
                                      )
        self.layer_seq_8 = nn.Sequential(nn.Linear(32, 16),
                                         nn.Linear(16, 8),
                                       nn.LayerNorm(8),
                                       nn.Mish(),
                                      )

        self.head = nn.Linear(128 + 32 + 8, 140)

    def forward(self, X, y=None):

        X_256 = self.layer_seq_256(X)
        X_64 = self.layer_seq_64(X_256)
        X_8 = self.layer_seq_8(X_64)

        X = torch.cat([X_256, X_64, X_8], axis = 1)
        out = self.head(X)

        return out


def train_loop(model, optimizer, loader, epoch):

    losses, lrs = [], []
    model.train()
    optimizer.zero_grad()
    #loss_fn = nn.MSELoss()

    with tqdm(total=len(loader),unit="batch") as pbar:
        pbar.set_description(f"Epoch{epoch}")

        for d in loader:
            X = d['X'].to(device)
            y = d['y'].to(device)

            logits = model(X)
            loss = correl_loss(logits, y)
            #loss = torch.sqrt(loss_fn(logits, y))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix({"loss":loss.item()})
            pbar.update(1)

    return model


def valid_loop(model, loader, y_val):

    model.eval()
    partial_correlation_scores = []
    oof_pred = []

    for d in loader:
        with torch.no_grad():
            val_X = d['X'].to(device).float()
            val_y = d['y'].to(device)
            logits = model(val_X)
            oof_pred.append(logits)

    #print(torch.cat(oof_pred).shape, torch.cat(oof_pred).detach().cpu().numpy().shape)
    cor = partial_correlation_score_torch_faster(torch.tensor(y_val).to(device), torch.cat(oof_pred))
    cor = cor.mean().item()
    logits = torch.cat(oof_pred).detach().cpu().numpy()

    return logits, cor


# ### train

es = 30
check_round = 5
epochs = 100000
target_num = Y.shape[1]
save_model = False # If you want to save the model in the output path, set this to True.

for i in feature_dict.keys():

    print(f'start: {i}')

    train_file = feature_dict[i][0]
    test_file = feature_dict[i][1]

    X = pd.read_pickle(feature_path  + train_file)
    X_test = pd.read_pickle(feature_path  + test_file)

    X = np.array(X)
    X_test = np.array(X_test)

    test_ds = CiteDataset_test(X_test)
    test_dataloader = DataLoader(test_ds, batch_size=128, pin_memory=True,
                                 shuffle=False, drop_last=False, num_workers=4)

    # get validation index
    train_index = fold_df[fold_df['flg_donor_val'] == 0].index
    test_index = fold_df[fold_df['flg_donor_val'] == 1].index

    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = Y[train_index], Y[test_index]
    feature_dims = X.shape[1]

    # dataset
    train_ds = CiteDataset(X_train ,y_train)
    train_dataloader = DataLoader(train_ds,
                                  batch_size=128,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=4)

    print(X_train.shape, X_val.shape)

    val_ds = CiteDataset(X_val, y_val)
    val_dataloader = DataLoader(val_ds,
                                batch_size=128,
                                pin_memory=True,
                                shuffle=False,
                                drop_last=False,
                                num_workers=4)

    # to get better epochs, train 5 times, and mean epochs
    train_epochs = 0

    for search in range(check_round):

        print(f'search:{search}')

        if 'mish' in i:
            # Only models with the name 'mish' have the NN activation function as mish.
            model = CiteModel_mish(feature_dims)
        else:
            model = CiteModel(feature_dims)

        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)

        best_corr = 0
        es_counter = 0

        for epoch in range(epochs):

            model = train_loop(model, opt, train_dataloader, epoch)
            logit, val_cor = valid_loop(model, val_dataloader, y_val)
            print(f'val corr:{val_cor}')

            if epoch == 0:
                logit_best = logit

            if val_cor > best_corr:
                best_corr = val_cor
                logit_best = logit
                es_counter = 0
            else:
                es_counter += 1

            if es_counter == es:
                break

        best_epoch = epoch - es
        print(f'best epoch:{best_epoch} all data train start!')

        train_epochs += best_epoch / check_round

    train_epochs = int(train_epochs)
    print(f'train epochs:{train_epochs}')

    # full dataset train
    train_ds = CiteDataset(X, Y)
    train_dataloader = DataLoader(train_ds,
                                  batch_size=128,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=4)

    # train new model
    model = CiteModel(feature_dims)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(train_epochs):
        model = train_loop(model, opt, train_dataloader, epoch)

    if save_model == True:
        torch.save(model.state_dict(), f'{output_path}/cite_mlp_corr_{i}_flg_donor_val_{train_epochs}')
