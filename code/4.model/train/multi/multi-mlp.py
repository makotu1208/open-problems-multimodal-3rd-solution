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
preprocess_path = '../../../../input/preprocess/multi/'
validation_path = '../../../../input/fold/'

target_path = '../../../../input/target/'
feature_path = '../../../../input/features/multi/'
output_path = '../../../../model/multi/mlp/'

# +
Y = pd.read_pickle(target_path + 'multi_train_target_128.pickle')
svd = pickle.load(open(target_path + 'multi_all_target_128.pkl', 'rb'))
Y = np.array(Y)

fold_df = pd.read_pickle(validation_path + 'multi_fold_val_df.pickle')
# -

# ### feature path

# +
corr_feature_dict = {}

corr_feature_dict['con_16'] = ['multi_train_con_16.pickle', 'multi_test_con_16.pickle']
corr_feature_dict['con_32'] = ['multi_train_con_32.pickle', 'multi_test_con_32.pickle']
corr_feature_dict['binary_16'] = ['multi_train_binary_16.pickle', 'multi_test_binary_16.pickle']
corr_feature_dict['lsi_add_lc_svd'] = ['multi_train_lc_addsvd_64.pickle', 'multi_test_lc_addsvd_64.pickle']

corr_feature_dict['lsi_w2v_col_128'] = ['multi_train_lsi_w2v_col_128.pickle', 'multi_test_lsi_w2v_col_128.pickle']
corr_feature_dict['lsi_w2v_128'] = ['multi_train_lsi_w2v_128.pickle', 'multi_test_lsi_w2v_128.pickle']
corr_feature_dict['lsi_128'] = ['multi_train_okapi_lsi_128.pickle', 'multi_test_okapi_lsi_128.pickle']
corr_feature_dict['lsi_w2v_col_64'] = ['multi_train_lsi_w2v_col_64.pickle', 'multi_test_lsi_w2v_col_64.pickle']
corr_feature_dict['lsi_w2v_64'] = ['multi_train_lsi_w2v_64.pickle', 'multi_test_lsi_w2v_64.pickle']
corr_feature_dict['lsi_64'] = ['multi_train_okapi_lsi_64.pickle', 'multi_test_okapi_lsi_64.pickle']

corr_feature_dict['colmean_64'] = ['multi_train_okapi_w2v_col_64.pickle', 'multi_test_okapi_w2v_col_64.pickle']
corr_feature_dict['okapi_w2v_64'] = ['multi_train_okapi_w2v_64.pickle', 'multi_test_okapi_w2v_64.pickle']
corr_feature_dict['okapi_64'] = ['multi_train_okapi_feature_64.pickle', 'multi_test_okapi_feature_64.pickle']

# +
all_feature_dict = {}

all_feature_dict['con_16'] = ['multi_train_con_16.pickle', 'multi_test_con_16.pickle']
all_feature_dict['con_32'] = ['multi_train_con_32.pickle', 'multi_test_con_32.pickle']
all_feature_dict['binary_16'] = ['multi_train_binary_16.pickle', 'multi_test_binary_16.pickle']
all_feature_dict['last_cluster'] = ['multi_train_okapi_64_last_cluster.pickle', 'multi_test_okapi_64_last_cluster.pickle']

all_feature_dict['lsi_w2v_col_128'] = ['multi_train_lsi_w2v_col_128.pickle', 'multi_test_lsi_w2v_col_128.pickle']
all_feature_dict['lsi_w2v_128'] = ['multi_train_lsi_w2v_128.pickle', 'multi_test_lsi_w2v_128.pickle']
all_feature_dict['lsi_128'] = ['multi_train_okapi_lsi_128.pickle', 'multi_test_okapi_lsi_128.pickle']
all_feature_dict['lsi_w2v_col_64'] = ['multi_train_lsi_w2v_col_64.pickle', 'multi_test_lsi_w2v_col_64.pickle']
all_feature_dict['lsi_w2v_64'] = ['multi_train_lsi_w2v_64.pickle', 'multi_test_lsi_w2v_64.pickle']
all_feature_dict['lsi_64'] = ['multi_train_okapi_lsi_64.pickle', 'multi_test_okapi_lsi_64.pickle']

all_feature_dict['okapi_128'] = ['multi_train_okapi_feature_128.pickle', 'multi_test_okapi_feature_128.pickle']
all_feature_dict['okapi_64'] = ['multi_train_okapi_feature_64.pickle', 'multi_test_okapi_feature_64.pickle']
all_feature_dict['colmean_64'] = ['multi_train_okapi_w2v_col_64.pickle', 'multi_test_okapi_w2v_col_64.pickle']


# -

# ### function

class MultiDataset(Dataset):

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


class MultiDataset_test(Dataset):

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


# -

class MultiModel(nn.Module):

    def __init__(self, feature_num):
        super(MultiModel, self).__init__()

        self.layer_seq_128 = nn.Sequential(nn.Linear(feature_num, 128),
                                           nn.LayerNorm(128),
                                           nn.ReLU(),
                                      )

        self.layer_seq_64 = nn.Sequential(nn.Linear(128, 64),
                                           nn.LayerNorm(64),
                                           nn.ReLU(),
                                      )

        self.layer_seq_32 = nn.Sequential(nn.Linear(64, 32),
                                   nn.LayerNorm(32),
                                   nn.ReLU(),
                              )

        self.layer_seq_8 = nn.Sequential(nn.Linear(32, 8),
                                         nn.LayerNorm(8),
                                         nn.ReLU(),
                                      )

        self.head = nn.Linear(128 + 64 + 32 + 8, target_num)

    def forward(self, X, y=None):

        X_128 = self.layer_seq_128(X)
        X_64 = self.layer_seq_64(X_128)
        X_32 = self.layer_seq_32(X_64)
        X_8 = self.layer_seq_8(X_32)
        X = torch.cat([X_128, X_64, X_32, X_8], axis = 1)
        out = self.head(X)

        return out


def train_loop(model, optimizer, loader, epoch):

    losses, lrs = [], []
    model.train()
    optimizer.zero_grad()
    loss_fn = nn.MSELoss()

    with tqdm(total=len(loader),unit="batch") as pbar:
        pbar.set_description(f"Epoch{epoch}")

        for d in loader:
            X = d['X'].to(device).float()
            y = d['y'].to(device)

            logits = model(X)
            #loss = correl_loss(logits, y)
            loss = torch.sqrt(loss_fn(logits, y))

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
    loss_fn = nn.MSELoss()

    for d in loader:
        with torch.no_grad():
            val_X = d['X'].to(device).float()
            val_y = d['y'].to(device)
            logits = model(val_X)
            #oof_pred.append(logits.detach().cpu().numpy())
            oof_pred.append(logits)

    y_val = torch.tensor(y_val).to(device)
    logits = torch.cat(oof_pred)
    #print(logits.shape, y_val.shape)
    loss = torch.sqrt(loss_fn(logits, y_val))
    logits = logits.detach().cpu().numpy()

    return logits, loss


def test_loop(model, loader):

    model.eval()
    predicts=[]

    for d in tqdm(loader):
        with torch.no_grad():
            X = d['X'].to(device).float()
            logits = model(X)
            predicts.append(logits.detach().cpu().numpy())

    return np.concatenate(predicts)


# ### train

es = 30
check_round = 3 #3
epochs = 100000
save_model = False # If you want to save the model in the output path, set this to True.

# #### target:128 dims

for i in corr_feature_dict.keys():

    print(f'start: {i}')
    target_num = Y.shape[1]

    train_file = corr_feature_dict[i][0]
    test_file = corr_feature_dict[i][1]

    X = pd.read_pickle(feature_path  + train_file)
    X_test = pd.read_pickle(feature_path  + test_file)

    X = np.array(X)
    X_test = np.array(X_test)

    test_ds = MultiDataset_test(X_test)
    test_dataloader = DataLoader(test_ds, batch_size=128, pin_memory=True,
                                 shuffle=False, drop_last=False, num_workers=4)

    train_index = fold_df[fold_df['flg_donor_val'] == 0].index
    test_index = fold_df[fold_df['flg_donor_val'] == 1].index

    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = Y[train_index], Y[test_index]
    feature_dims = X.shape[1]

    # dataset
    train_ds = MultiDataset(X_train ,y_train)
    train_dataloader = DataLoader(train_ds,
                                  batch_size=128,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=4)

    print(X_train.shape, X_val.shape)

    val_ds = MultiDataset(X_val, y_val)
    val_dataloader = DataLoader(val_ds,
                                batch_size=128,
                                pin_memory=True,
                                shuffle=False,
                                drop_last=False,
                                num_workers=4)

    del X_train, y_train
    gc.collect()

    train_epochs = 0

    for search in range(check_round):

        print(f'search:{search}')

        model = MultiModel(feature_dims)
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)

        best_corr = 99999
        es_counter = 0

        # make best epoch
        for epoch in range(epochs):

            model = train_loop(model, opt, train_dataloader, epoch)
            logit, val_cor = valid_loop(model, val_dataloader, y_val)
            print(f'val corr:{val_cor}')

            if epoch == 0:
                logit_best = logit

            if val_cor < best_corr:
                best_corr = val_cor
                logit_best = logit
                es_counter = 0
            else:
                es_counter += 1

            if es_counter == es:
                break

        best_epoch = epoch - es
        print(f'done! epoch:{best_epoch}')

        train_epochs += best_epoch / check_round


    train_epochs = int(train_epochs)
    print(f'5 average train epochs:{train_epochs}')

    # full dataset
    train_ds = MultiDataset(X, Y)
    train_dataloader = DataLoader(train_ds,
                                  batch_size=128,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=4)

    # train new model
    model = MultiModel(feature_dims)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(train_epochs):
        model = train_loop(model, opt, train_dataloader, epoch)

    if save_model == True:
        torch.save(model.state_dict(), f'{output_path}/multi_mlp_corr_{i}_flg_donor_val_{train_epochs}')

torch.cuda.empty_cache()

# ### target no svd: all target dims

Y = pd.read_pickle('../../../../../data/target/multi/' + 'multi_train_target_all.pickle')
Y = np.array(Y)


def train_loop(model, optimizer, loader, epoch):

    losses, lrs = [], []
    model.train()
    optimizer.zero_grad()
    #loss_fn = nn.MSELoss()

    with tqdm(total=len(loader),unit="batch") as pbar:
        pbar.set_description(f"Epoch{epoch}")

        for d in loader:
            X = d['X'].to(device).float()
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
            partial_correlation_scores.append(partial_correlation_score_torch_faster(val_y, logits))
            oof_pred.append(logits.detach().cpu().numpy())

    partial_correlation_scores = torch.cat(partial_correlation_scores)
    cor = torch.sum(partial_correlation_scores).cpu().item()/len(partial_correlation_scores)
    logits = np.concatenate(oof_pred)

    return logits, cor


for i in all_feature_dict.keys():

    print(f'start: {i}')
    target_num = Y.shape[1]

    train_file = all_feature_dict[i][0]
    test_file = all_feature_dict[i][1]

    X = pd.read_pickle(feature_path  + train_file)
    X_test = pd.read_pickle(feature_path  + test_file)

    X = np.array(X)
    X_test = np.array(X_test)

    test_ds = MultiDataset_test(X_test)
    test_dataloader = DataLoader(test_ds, batch_size=24, pin_memory=True,
                                 shuffle=False, drop_last=False, num_workers=4)

    train_index = fold_df[fold_df['flg_donor_val'] == 0].index
    test_index = fold_df[fold_df['flg_donor_val'] == 1].index

    X_train, X_val = X[train_index], X[test_index]
    y_train, y_val = Y[train_index], Y[test_index]
    feature_dims = X.shape[1]

    # dataset
    train_ds = MultiDataset(X_train ,y_train)
    train_dataloader = DataLoader(train_ds,
                                  batch_size=24,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=4)

    print(X_train.shape, X_val.shape)

    val_ds = MultiDataset(X_val, y_val)
    val_dataloader = DataLoader(val_ds,
                                batch_size=24,
                                pin_memory=True,
                                shuffle=False,
                                drop_last=False,
                                num_workers=4)

    del X_train, y_train
    gc.collect()

    train_epochs = 0

    for search in range(check_round):

        print(f'search:{search}')

        model = MultiModel(feature_dims)
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)

        best_corr = 0
        es_counter = 0

        # make best epoch
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
        print(f'done! epoch:{best_epoch}')

        train_epochs += best_epoch / check_round

    train_epochs = int(train_epochs)
    print(f'5 average train epochs:{train_epochs}')

    # full dataset
    train_ds = MultiDataset(X, Y)
    train_dataloader = DataLoader(train_ds,
                                  batch_size=24,
                                  pin_memory=True,
                                  shuffle=True,
                                  drop_last=False,
                                  num_workers=4)

    # train new model
    model = MultiModel(feature_dims)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch in range(train_epochs):
        model = train_loop(model, opt, train_dataloader, epoch)

    if save_model == True:
        torch.save(model.state_dict(), f'{output_path}/multi_mlp_all_{i}_flg_donor_val_{train_epochs}')
