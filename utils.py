import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class CretioDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('dataset/cretio/preprocessed/cretio_data.csv', header=0)
        self.labels = self.data['Label']
        del self.data['Label']
        self.feats_num = self.data.shape[1]

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