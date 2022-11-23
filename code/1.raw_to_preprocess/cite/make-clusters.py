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

# - make leiden/SpectralClustering/adjacency matrix
#     - cite_cluster.pickle / train(test)_spec_cluster_128 / cite_train(test)_connect.pickle

# +
# %matplotlib inline

import os
import gc
import torch
import muon
import numpy as np
import scanpy as sc
import pandas as pd
import scipy
import dask.dataframe as dd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import SpectralClustering

# +
# %%time

raw_path_base = '../../../input/raw/'
raw_path = '../../../input/preprocess/cite/'
raw_multi_path = '../../../input/preprocess/multi/'
feature_path = '../../../input/base_features/cite/'
#feature_path = '../../../../summary/input/sample/'
# -

train_inputs = scipy.sparse.load_npz(raw_path + "train_cite_raw_inputs_values.sparse.npz")
test_inputs = scipy.sparse.load_npz(raw_path + "test_cite_raw_inputs_values.sparse.npz")

# +
train_ids = np.load(raw_path + "train_cite_raw_inputs_idxcol.npz", allow_pickle=True)
test_ids = np.load(raw_path + "test_cite_raw_inputs_idxcol.npz", allow_pickle=True)

train_index = train_ids["index"]
train_column = train_ids["columns"]
test_index = test_ids["index"]
# -

train_num = train_inputs.shape[0]

all_inputs = scipy.sparse.vstack([train_inputs, test_inputs])
del train_inputs, test_inputs
gc.collect()

svd_dims = 128
svd = TruncatedSVD(n_components=svd_dims, random_state=1) # 512
result_svd = svd.fit_transform(all_inputs)

X_all = sc.AnnData(X = result_svd)

sc.pp.neighbors(X_all, use_rep = 'X', n_neighbors=64, method='umap')
sc.tl.leiden(X_all)

# save cluster
X_clus = X_all.obs['leiden']
pd.DataFrame(X_clus).to_pickle(feature_path + 'cite_cluster.pickle')

# ### s_cluster

s_cluster = SpectralClustering(n_clusters=128,
                               affinity = 'precomputed',
                               assign_labels='discretize',
                               random_state=0)

# %%time
result_cluster = s_cluster.fit_predict(X_all.obsp['distances'])

train_cite_cluster = pd.DataFrame(result_cluster[:len(train_index)], index = train_index, columns = ['s_cluster'])
test_cite_cluster = pd.DataFrame(result_cluster[len(train_index):], index = test_index, columns = ['s_cluster'])

train_cite_cluster.to_pickle(feature_path + 'train_spec_cluster_128.pickle')
test_cite_cluster.to_pickle(feature_path + 'test_spec_cluster_128.pickle')

# ### make con(connectivity) feature
# Weighted average the expression levels of important proteins in similar cells based on the similarity of each cell

adj_mat = scipy.sparse.csr_matrix(X_all.obsp['connectivities'])

del X_all
gc.collect()

con_df = pd.DataFrame(np.array(adj_mat.sum(1)), columns = ['connect_sum'])

rows, cols = adj_mat.nonzero()

row_df = pd.DataFrame(rows, columns = ['row'])
row_df['col'] = cols
weight = np.array(con_df['connect_sum'])

# +
fix_vec = np.zeros([adj_mat.shape[0], 22085]).astype(np.float32)

for i in tqdm(row_df.groupby(['row'])):

    index_num = i[0]
    near_index_list = list(i[1]['col'])
    near_vec = np.zeros([1,22085])

    for n_index in near_index_list:
        near_vec += (all_inputs[n_index,:] * adj_mat[index_num, n_index]) / weight[index_num]

    fix_vec[index_num,:] = near_vec

# +
important_cols = ['ENSG00000135218_CD36',
 'ENSG00000010278_CD9',
 'ENSG00000204287_HLA-DRA',
 'ENSG00000117091_CD48',
 'ENSG00000004468_CD38',
 'ENSG00000173762_CD7',
 'ENSG00000137101_CD72',
 'ENSG00000019582_CD74',
 'ENSG00000169442_CD52',
 'ENSG00000170458_CD14',
 'ENSG00000272398_CD24',
 'ENSG00000026508_CD44',
 'ENSG00000114013_CD86',
 'ENSG00000174059_CD34',
 'ENSG00000139193_CD27',
 'ENSG00000105383_CD33',
 'ENSG00000085117_CD82',
 'ENSG00000177455_CD19',
 'ENSG00000002586_CD99',
 'ENSG00000196126_HLA-DRB1',
 'ENSG00000135404_CD63',
 'ENSG00000012124_CD22',
 'ENSG00000134061_CD180',
 'ENSG00000105369_CD79A',
 'ENSG00000116824_CD2',
 'ENSG00000010610_CD4',
 'ENSG00000139187_KLRG1',
 'ENSG00000204592_HLA-E',
 'ENSG00000090470_PDCD7',
 'ENSG00000206531_CD200R1L',
'ENSG00000166710_B2M',
 'ENSG00000198034_RPS4X',
 'ENSG00000188404_SELL',
 'ENSG00000130303_BST2',
 'ENSG00000128040_SPINK2',
 'ENSG00000206503_HLA-A',
 'ENSG00000108107_RPL28',
 'ENSG00000143226_FCGR2A',
 'ENSG00000133112_TPT1',
 'ENSG00000166091_CMTM5',
 'ENSG00000026025_VIM',
 'ENSG00000205542_TMSB4X',
 'ENSG00000109099_PMP22',
 'ENSG00000145425_RPS3A',
 'ENSG00000172247_C1QTNF4',
 'ENSG00000072274_TFRC',
 'ENSG00000234745_HLA-B',
 'ENSG00000075340_ADD2',
 'ENSG00000119865_CNRIP1',
 'ENSG00000198938_MT-CO3',
 'ENSG00000135046_ANXA1',
 'ENSG00000235169_SMIM1',
 'ENSG00000101200_AVP',
 'ENSG00000167996_FTH1',
 'ENSG00000163565_IFI16',
 'ENSG00000117450_PRDX1',
 'ENSG00000124570_SERPINB6',
 'ENSG00000112077_RHAG',
 'ENSG00000051523_CYBA',
 'ENSG00000107130_NCS1',
 'ENSG00000055118_KCNH2',
 'ENSG00000029534_ANK1',
 'ENSG00000169567_HINT1',
 'ENSG00000142089_IFITM3',
 'ENSG00000139278_GLIPR1',
 'ENSG00000142227_EMP3',
 'ENSG00000076662_ICAM3',
 'ENSG00000143627_PKLR',
 'ENSG00000130755_GMFG',
 'ENSG00000160593_JAML',
 'ENSG00000095932_SMIM24',
 'ENSG00000197956_S100A6',
 'ENSG00000171476_HOPX',
 'ENSG00000116675_DNAJC6',
 'ENSG00000100448_CTSG',
 'ENSG00000100368_CSF2RB',
 'ENSG00000047648_ARHGAP6',
 'ENSG00000198918_RPL39',
 'ENSG00000196154_S100A4',
 'ENSG00000233968_AL157895.1',
 'ENSG00000137642_SORL1',
 'ENSG00000133816_MICAL2',
 'ENSG00000130208_APOC1',
 'ENSG00000105610_KLF1']
print('important columns ',len(important_cols))

next_important_cols = ['ENSG00000211899_IGHM',
 'ENSG00000160883_HK3',
 'ENSG00000137818_RPLP1',
 'ENSG00000183087_GAS6',
 'ENSG00000198520_ARMH1',
 'ENSG00000175449_RFESD',
 'ENSG00000106443_PHF14',
 'ENSG00000164929_BAALC',
 'ENSG00000133142_TCEAL4',
 'ENSG00000198336_MYL4',
 'ENSG00000103490_PYCARD',
 'ENSG00000223609_HBD',
 'ENSG00000204257_HLA-DMA',
 'ENSG00000204472_AIF1',
 'ENSG00000136942_RPL35',
 'ENSG00000204525_HLA-C',
 'ENSG00000184500_PROS1',
 'ENSG00000133985_TTC9',
 'ENSG00000198727_MT-CYB',
 'ENSG00000231389_HLA-DPA1',
 'ENSG00000198502_HLA-DRB5',
 'ENSG00000112339_HBS1L',
 'ENSG00000149806_FAU',
 'ENSG00000110852_CLEC2B',
 'ENSG00000104432_IL7',
 'ENSG00000100911_PSME2',
 'ENSG00000160789_LMNA',
 'ENSG00000140022_STON2',
 'ENSG00000118579_MED28',
 'ENSG00000138326_RPS24',
 'ENSG00000133134_BEX2',
 'ENSG00000171388_APLN',
 'ENSG00000198899_MT-ATP6',
 'ENSG00000223865_HLA-DPB1',
 'ENSG00000198804_MT-CO1',
 'ENSG00000101608_MYL12A']

print('next important columns ',len(next_important_cols))

important_cols = important_cols + next_important_cols
print(len(important_cols))
use_imp_cols =  [i for i, j in enumerate(train_column) if j in important_cols]
# -

train_con_imp = fix_vec[:train_num, use_imp_cols]
test_con_imp = fix_vec[train_num:, use_imp_cols]

pd.DataFrame(train_con_imp, index = train_index).add_prefix('con_').to_pickle(feature_path + 'train_cite_imp_confeature.pickle')
pd.DataFrame(test_con_imp, index = test_index).add_prefix('con_').to_pickle(feature_path + 'test_cite_imp_confeature.pickle')
