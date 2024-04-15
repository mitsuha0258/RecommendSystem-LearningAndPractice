import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


def feature2field():
    feature_columns = np.load('dataset/cretio/preprocessed/feat_col.npy', allow_pickle=True)
    # print(feature_columns)
    dense_indices = [1] * len(feature_columns[0])
    sparse_indices = [sparse_col['feat_num'] for sparse_col in feature_columns[1]]
    indices = dense_indices + sparse_indices
    feat2field = {}
    num = 0
    for i, indice in enumerate(indices):
        for j in range(indice):
            feat2field[num + j] = i
        num += indice
    return len(indices), feat2field


class CretioDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('dataset/cretio/preprocessed/cretio_data.csv', header=0)
        self.labels = self.data['Label']
        del self.data['Label']
        self.feats_num = self.data.shape[1]
        self.fields_num, self.feat2field = feature2field()

        self.data = torch.from_numpy(self.data.values).float()
        self.labels = torch.LongTensor(np.array(self.labels))

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


class EarlyStopper(object):
    def __init__(self, patience, save_path):
        self.patience = patience
        self.counter = 1
        self.best_acc = 0.0
        self.save_path = save_path

    def is_continuable(self, model, acc):
        if acc > self.best_acc:
            self.best_acc = acc
            self.counter = 1
            torch.save(model, self.save_path)
            return True
        elif self.counter < self.patience:
            self.counter += 1
            return True
        else:
            return False


if __name__ == '__main__':
   pass



