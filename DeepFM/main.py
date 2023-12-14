import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset


def load_preprocessed_data(file_path='./preprocessed_data/'):
    train = pd.read_csv(file_path+'train_set.csv')
    test = pd.read_csv(file_path+'test_set.csv')
    val = pd.read_csv(file_path+'val_set.csv')

    trn_x, trn_y = train.drop(columns=['Label']).values, train['Label'].values
    val_x, val_y = val.drop(columns=['Label']).values, val['Label'].values
    test_x = test.values

    fea_cols = np.load(file_path+'fea_col.npy', allow_pickle=True)
    # print(fea_cols)

    return fea_cols, (trn_x, trn_y), (val_x, val_y), test_x


if __name__ == '__main__':
    fea_cols, (trn_x, trn_y), (val_x, val_y), test_x = load_preprocessed_data()

    train_dataset = TensorDataset(torch.tensor(trn_x).float(), torch.tensor(trn_y).float())
    val_dataset = TensorDataset(torch.tensor(val_x).float(), torch.tensor(val_y).float())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    



    