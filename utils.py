import pandas as pd
from torch.utils.data import Dataset


class CretioDataset(Dataset):
    def __init__(self):
        self.data = pd.read_csv('dataset/cretio/preprocessed/cretio_data.csv', header=0)
        self.labels = self.data['Label']
        del self.data['Label']

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

