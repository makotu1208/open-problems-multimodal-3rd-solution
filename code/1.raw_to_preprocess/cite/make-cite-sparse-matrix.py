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

# https://www.kaggle.com/code/fabiencrom/multimodal-single-cell-creating-sparse-data/

import pandas as pd
import numpy as np
import scipy
import scipy.sparse


def convert_h5_to_sparse_csr(filename, out_filename, chunksize=2500):
    start = 0
    total_rows = 0

    sparse_chunks_data_list = []
    chunks_index_list = []
    columns_name = None
    while True:
        df_chunk = pd.read_hdf(filename, start=start, stop=start+chunksize)
        if len(df_chunk) == 0:
            break
        chunk_data_as_sparse = scipy.sparse.csr_matrix(df_chunk.to_numpy())
        sparse_chunks_data_list.append(chunk_data_as_sparse)
        chunks_index_list.append(df_chunk.index.to_numpy())

        if columns_name is None:
            columns_name = df_chunk.columns.to_numpy()
        else:
            assert np.all(columns_name == df_chunk.columns.to_numpy())

        total_rows += len(df_chunk)
        print(total_rows)
        if len(df_chunk) < chunksize:
            del df_chunk
            break
        del df_chunk
        start += chunksize

    all_data_sparse = scipy.sparse.vstack(sparse_chunks_data_list)
    del sparse_chunks_data_list

    all_indices = np.hstack(chunks_index_list)

    scipy.sparse.save_npz(out_filename+"_values.sparse", all_data_sparse)
    np.savez(out_filename+"_idxcol.npz", index=all_indices, columns =columns_name)


raw_path_base = '../../../input/raw/'
raw_cite_path = '../../../input/preprocess/cite/'
#feature_path = '../../../../summary/input/base_features/cite/'
#feature_path = '../../../../summary/input/sample/'

convert_h5_to_sparse_csr(raw_path_base + "train_cite_targets.h5", \
                         raw_cite_path + "train_cite_targets")

convert_h5_to_sparse_csr(raw_path_base + "train_cite_targets_raw.h5", \
                         raw_cite_path + "train_cite_raw_targets")

convert_h5_to_sparse_csr(raw_path_base + "train_cite_inputs.h5", \
                         raw_cite_path + "train_cite_inputs")

convert_h5_to_sparse_csr(raw_path_base + "train_cite_inputs_raw.h5", \
                         raw_cite_path + "train_cite_raw_inputs")

convert_h5_to_sparse_csr(raw_path_base + "test_cite_inputs.h5", \
                         raw_cite_path + "test_cite_inputs")

convert_h5_to_sparse_csr(raw_path_base + "test_cite_inputs_raw.h5", \
                         raw_cite_path + "test_cite_raw_inputs")
