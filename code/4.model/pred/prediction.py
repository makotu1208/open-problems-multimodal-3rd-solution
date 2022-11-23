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
raw_path =  '../../../input/raw/'

cite_target_path = '../../../input/target/cite/'
cite_feature_path = '../../../input/features/cite/'
cite_mlp_path = '../../../model/cite/mlp/'
cite_cb_path = '../../../model/cite/cb/'

multi_target_path = '../../../input/target/multi/'
multi_feature_path = '../../../input/features/multi/'
multi_mlp_path = '../../../model/multi/mlp/'
multi_cb_path = '../../../model/multi/cb/'

output_path = '../../../output/'
# -

# ## Cite

# +
# get model name
#mlp_model_path = os.listdir(cite_mlp_path)
# -

mlp_model_name = [
    'corr_add_con_imp',
    'corr_last_v3', 
    'corr_c_add_w2v_v1_mish_flg',
    'corr_c_add_w2v_v1_flg',
    'corr_c_add_84_v1',
    'corr_c_add_120_v1',
    'corr_w2v_cell_flg',
    'corr_best_cell_120',
    'corr_cluster_cell',
    'corr_w2v_128',
    'corr_imp_w2v_128',
    'corr_snorm',
    'corr_best_128',
    'corr_best_64',
    'corr_cluster_128',
    'corr_cluster_64',
    'corr_svd_128',
    'corr_svd_64',
             ]

# +
model_name_list = []

for i in mlp_model_name:
    for num, j in enumerate(os.listdir(cite_mlp_path)):
        if i in j:
            model_name_list.append(j)

len(model_name_list)
model_name_list

# +
weight = [1, 0.3, 1, 1, 1, 1, 1, 1, 1, 0.8, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 1, 1, 2, 2]
weight_sum = np.array(weight).sum()
weight_sum

model_feat_dict = {model_name_list[0]:['X_test_add_con_imp.pickle', 1],
                   model_name_list[1]:['X_test_last_v3.pickle', 0.3],
                   model_name_list[2]:['X_test_c_add_w2v_v1.pickle', 1],
                   model_name_list[3]:['X_test_c_add_w2v_v1.pickle', 1],
                   model_name_list[4]:['X_test_c_add_84_v1.pickle', 1],
                   model_name_list[5]:['X_test_c_add_v1.pickle', 1],
                   
                   model_name_list[6]:['X_test_feature_w2v_cell.pickle', 1],
                   model_name_list[7]:['X_test_best_cell_128_120.pickle', 1],
                   model_name_list[8]:['X_test_cluster_cell_128.pickle', 1],
                   
                   model_name_list[9]:['X_test_feature_w2v.pickle', 0.8],
                   model_name_list[10]:['X_test_feature_imp_w2v.pickle',0.8],
                   model_name_list[11]:['X_test_feature_snorm.pickle', 0.8],
                   model_name_list[12]:['X_test_best_128.pickle', 0.8],
                   model_name_list[13]:['X_test_best_64.pickle', 0.5],
                   model_name_list[14]:['X_test_cluster_128.pickle', 0.5],
                   model_name_list[15]:['X_test_cluster_64.pickle', 0.5],
                   model_name_list[16]:['X_test_svd_128.pickle', 1],
                   model_name_list[17]:['X_test_svd_64.pickle', 1],
                   
                   'best_128':['X_test_best_128.pickle', 2],
                   'best_64':['X_test_best_64.pickle', 2],
                  }


# -

# ### cite model

def std(x):
    x = np.array(x)
    return (x - x.mean(1).reshape(-1, 1)) / x.std(1).reshape(-1, 1)


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


def test_loop(model, loader):
    
    model.eval()
    predicts=[]

    for d in tqdm(loader):
        with torch.no_grad():
            X = d['X'].to(device)
            logits = model(X)
            predicts.append(logits.detach().cpu().numpy())
            
    return np.concatenate(predicts)


# ### pred

# +
pred = np.zeros([48203, 140])

for num, i in enumerate(model_feat_dict.keys()):
    
    print(i)
    
    if 'mlp' in i:
        
        test_file = model_feat_dict[i][0]
        test_weight = model_feat_dict[i][1]
        X_test = pd.read_pickle(cite_feature_path  + test_file)    
        X_test = np.array(X_test)
        feature_dims = X_test.shape[1]

        test_ds = CiteDataset_test(X_test)
        test_dataloader = DataLoader(test_ds, batch_size=128, pin_memory=True, 
                                     shuffle=False, drop_last=False, num_workers=4)
        
        if 'mish' in i:
            model = CiteModel_mish(feature_dims)
        else:
            model = CiteModel(feature_dims)
            
        model = model.to(device)
        model.load_state_dict(torch.load(f'{cite_mlp_path}/{i}'))
        
        result = test_loop(model, test_dataloader).astype(np.float32)
        result = std(result) * test_weight / weight_sum
        pred += result

        torch.cuda.empty_cache()
        
    else:
        test_file = model_feat_dict[i][0]
        test_weight = model_feat_dict[i][1]
        X_test = pd.read_pickle(cite_feature_path  + test_file)
        
        cb_pred = np.zeros([48203, 140])
        
        for t in tqdm(range(140)): 
            cb_model_path = [j for j in os.listdir(cite_cb_path) if f'cb_{t}_{i}' in j][0]
            cb = pickle.load(open(cite_cb_path + cb_model_path, 'rb'))
            cb_pred[:,t] = cb.predict(X_test)
            
        cb_pred = cb_pred.astype(np.float32)
        pred += std(cb_pred) * test_weight / weight_sum
        
        #del cb_pred
# -

cite_sub = pd.DataFrame(pred.round(6))

# +
#cite_sub.to_csv('../../../../../summary/output/submit/cite_submit.csv')
# -

# ## Multi

mlp_model_name = [
    'multi_mlp_all_con_16',
    'multi_mlp_all_con_32', 
    'multi_mlp_all_binary_16',
    'multi_mlp_all_last_cluster',
    'multi_mlp_all_lsi_w2v_col_128_flg',
    'multi_mlp_all_lsi_w2v_128_flg',
    'multi_mlp_all_lsi_128_flg',
    'multi_mlp_all_lsi_w2v_col_64_flg',
    'multi_mlp_all_lsi_w2v_64_flg',
    'multi_mlp_all_lsi_64_flg',
    'multi_mlp_all_okapi_128_flg',
    'multi_mlp_all_okapi_64_flg',
    'multi_mlp_all_colmean_64_flg',
    'multi_mlp_corr_con_16_flg',
    'multi_mlp_corr_con_32_flg',
    'multi_mlp_corr_binary_16',
    'multi_mlp_corr_lsi_add_lc_svd_flg',
    
    'multi_mlp_corr_lsi_w2v_col_128_flg',
    'multi_mlp_corr_lsi_w2v_col_64_flg',
    'multi_mlp_corr_lsi_w2v_128_flg',
    'multi_mlp_corr_lsi_w2v_64_flg',
    
    'multi_mlp_corr_lsi_128_flg',
    'multi_mlp_corr_lsi_64_flg',
    
    'multi_mlp_corr_colmean_64_flg',
    'multi_mlp_corr_okapi_w2v_64_flg',
    'multi_mlp_corr_okapi_64_flg',
    
             ]

# +
model_name_list = []

for i in mlp_model_name:
    for num, j in enumerate(os.listdir(multi_mlp_path)):
        if i in j:
            model_name_list.append(j)

print(len(model_name_list))
model_name_list

# +
weight = [2.5, 2.5, 2.5, 1.2, 1.2, 1.2, 1, 
          1.5, 1.5, 2.5, 0.5, 0.5, 0.5, 
          2.5, 2.5, 1.8, 0.8, 1, 0.8, 1 ,0.8, 1, 0.3, 
          0.3, 0.3, 0.3, 0.2, 0.2, 0.2]
weight_sum = np.array(weight).sum()
weight_sum

model_feat_dict = {model_name_list[0]:['multi_test_con_16.pickle', 2.5],
                   model_name_list[1]:['multi_test_con_32.pickle', 2.5],
                   model_name_list[2]:['multi_test_binary_16.pickle', 2.5],
                   
                   model_name_list[3]:['multi_test_okapi_64_last_cluster.pickle', 1.2],
                   model_name_list[4]:['multi_test_lsi_w2v_col_128.pickle', 1.2],
                   model_name_list[5]:['multi_test_lsi_w2v_128.pickle', 1.2],
                   model_name_list[6]:['multi_test_okapi_lsi_128.pickle', 1],
                   
                   model_name_list[7]:['multi_test_lsi_w2v_col_64.pickle', 1.5],
                   model_name_list[8]:['multi_test_lsi_w2v_64.pickle', 1.5],
                   model_name_list[9]:['multi_test_okapi_lsi_64.pickle', 2.5],
                   
                   model_name_list[10]:['multi_test_okapi_feature_128.pickle', 0.5],
                   model_name_list[11]:['multi_test_okapi_feature_64.pickle', 0.5],
                   model_name_list[12]:['multi_test_okapi_w2v_col_64.pickle', 0.5],
                   
                   model_name_list[13]:['multi_test_con_16.pickle', 2.5],
                   model_name_list[14]:['multi_test_con_32.pickle', 2.5],
                   model_name_list[15]:['multi_test_binary_16.pickle', 1.8],
                   model_name_list[16]:['multi_test_lc_addsvd_64.pickle', 0.8],
                   
                   model_name_list[17]:['multi_test_lsi_w2v_col_128.pickle', 1],
                   model_name_list[18]:['multi_test_lsi_w2v_col_64.pickle', 0.8],
                   model_name_list[19]:['multi_test_lsi_w2v_128.pickle', 1],
                   model_name_list[20]:['multi_test_lsi_w2v_64.pickle', 0.8],
                   model_name_list[21]:['multi_test_okapi_lsi_128.pickle', 1],
                   model_name_list[22]:['multi_test_okapi_lsi_64.pickle', 0.3],
                   
                   model_name_list[23]:['multi_test_okapi_w2v_col_64.pickle', 0.3],
                   model_name_list[24]:['multi_test_okapi_w2v_64.pickle', 0.3],
                   model_name_list[25]:['multi_test_okapi_feature_64.pickle', 0.3],
                   
                   'lsi_128':['multi_test_okapi_lsi_128.pickle', 0.2],
                   'lsi_64':['multi_test_okapi_lsi_64.pickle', 0.2],
                   'lsi_w2v_col_64':['multi_test_lsi_w2v_col_64.pickle', 0.2],
                  }

# -

# ### multi model

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


# +
pred = np.zeros([55935, 23418])
svd = pickle.load(open(multi_target_path + 'multi_all_target_128.pkl', 'rb'))

for num, i in enumerate(model_feat_dict.keys()):
    
    print(i)
    
    if 'mlp' in i:
        
        test_file = model_feat_dict[i][0]
        test_weight = model_feat_dict[i][1]
        X_test = pd.read_pickle(multi_feature_path  + test_file)    
        X_test = np.array(X_test)
        feature_dims = X_test.shape[1]

        test_ds = MultiDataset_test(X_test)
        test_dataloader = DataLoader(test_ds, batch_size=128, pin_memory=True, 
                                     shuffle=False, drop_last=False, num_workers=4)
        
        if 'all' in i:
            target_num = 23418
        else:
            target_num = 128
        
        model = MultiModel(feature_dims)    
        model = model.to(device)
        model.load_state_dict(torch.load(f'{multi_mlp_path}/{i}'))
        
        result = test_loop(model, test_dataloader).astype(np.float32)
        
        if 'all' not in i:
            result = result@svd.components_
                
        result = result * test_weight / weight_sum
        pred += result

        torch.cuda.empty_cache()
        
    else:
        test_file = model_feat_dict[i][0]
        test_weight = model_feat_dict[i][1]
        X_test = pd.read_pickle(multi_feature_path  + test_file)
        
        cb_pred = np.zeros([55935, 128])
        
        for t in tqdm(range(128)): 
            cb_model_path = [j for j in os.listdir(multi_cb_path) if f'cb_{t}_{i}' in j][0]
            cb = pickle.load(open(multi_cb_path + cb_model_path, 'rb'))
            cb_pred[:,t] = cb.predict(X_test)
            
        cb_pred = cb_pred.astype(np.float32)
        cb_pred = cb_pred@svd.components_
        pred += cb_pred * test_weight / weight_sum
        
        #del cb_pred
# -

multi_sub = pd.DataFrame(pred.round(6)).astype(np.float32)

del pred
gc.collect()

# ## Postprocess

preprocess_path = '../../../../summary/input/preprocess/'

# #### first: fix cite output

test_sub_ids = np.load(preprocess_path + "cite/test_cite_inputs_idxcol.npz", allow_pickle=True)
test_sub_ids = test_sub_ids["index"]
test_raw_ids = np.load(preprocess_path + "cite/test_cite_raw_inputs_idxcol.npz", allow_pickle=True)
test_raw_ids = test_raw_ids["index"]

# +
test_cite_df = pd.DataFrame(test_sub_ids, columns = ['cell_id'])
cite_sub['cell_id'] = test_raw_ids
test_cite_df = test_cite_df.merge(cite_sub, on = 'cell_id', how = 'left')
test_cite_df.fillna(0, inplace = True)
test_cite_df.drop(['cell_id'], axis = 1, inplace = True)

cite_sub = test_cite_df.copy()
# -

# ### preprocess

# +
sub = pd.read_csv(raw_path + "sample_submission.csv")  
eval_ids = pd.read_csv(raw_path + "evaluation_ids.csv") 

cite_cols = pd.read_csv(preprocess_path + "cite/cite_test_cols.csv") 
cite_index = pd.read_csv(preprocess_path + "cite/cite_test_indexs.csv") 
cite_index.columns = ['cell_id']

multi_cols = pd.read_csv(preprocess_path + "multi/multi_test_cols.csv") 
multi_index = pd.read_csv(preprocess_path + "multi/multi_test_indexs.csv") 
multi_index.columns = ['cell_id']

submission = pd.Series(name='target',index=pd.MultiIndex.from_frame(eval_ids), dtype=np.float32)
# -

# ### multi

multi_sub = np.array(multi_sub)

# +
cell_dict = dict((k,v) for v,k in enumerate(np.array(multi_index['cell_id'])))
assert len(cell_dict)  == len(multi_index['cell_id'])

gene_dict = dict((k,v) for v,k in enumerate(np.array(multi_cols['gene_id']))) 
assert len(gene_dict)  == len(multi_cols['gene_id'])

eval_ids_cell_num = eval_ids.cell_id.apply(lambda x:cell_dict.get(x, -1))
eval_ids_gene_num = eval_ids.gene_id.apply(lambda x:gene_dict.get(x, -1))

valid_multi_rows = (eval_ids_gene_num !=-1) & (eval_ids_cell_num!=-1)
submission.iloc[valid_multi_rows] = multi_sub[eval_ids_cell_num[valid_multi_rows].to_numpy(),
                                                 eval_ids_gene_num[valid_multi_rows].to_numpy()]
# -

# ### cite

cite_sub = np.array(cite_sub)

# +
cell_dict = dict((k,v) for v,k in enumerate(np.array(cite_index['cell_id'])))
assert len(cell_dict)  == len(cite_index['cell_id'])

gene_dict = dict((k,v) for v,k in enumerate(np.array(cite_cols['gene_id']))) 
assert len(gene_dict)  == len(cite_cols['gene_id'])

eval_ids_cell_num = eval_ids.cell_id.apply(lambda x:cell_dict.get(x, -1))
eval_ids_gene_num = eval_ids.gene_id.apply(lambda x:gene_dict.get(x, -1))

valid_multi_rows = (eval_ids_gene_num !=-1) & (eval_ids_cell_num!=-1)
# -

submission.iloc[valid_multi_rows] = cite_sub[eval_ids_cell_num[valid_multi_rows].to_numpy(),
                                                 eval_ids_gene_num[valid_multi_rows].to_numpy()]

# ### make submission

submission = submission.round(6)
submission = pd.DataFrame(submission, columns = ['target'])
submission = submission.reset_index()

submission[['row_id', 'target']].to_csv(output_path + 'submission.csv', index = False)

# +
# #!kaggle competitions submit -c open-problems-multimodal -f $sub_name_csv -m $message
