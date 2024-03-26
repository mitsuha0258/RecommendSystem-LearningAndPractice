import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from utils import CretioDataset


class FM(nn.Module):
    def __init__(self, feats_num, hidden_dim):
        super(FM, self).__init__()
        self.feats_num = feats_num
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(self.feats_num, 1)
        self.v = nn.Parameter(torch.rand(self.feats_num, self.hidden_dim))
        nn.init.xavier_uniform_(self.v.data.weight)

    def forward(self, x):
        first_order = self.linear(x)

        square_of_sum = torch.mm(x, self.v) ** 2
        sum_of_square = torch.mm(x ** 2, self.v ** 2)
        second_order = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        return torch.sigmoid(first_order + second_order)


if __name__ == '__main__':
    dataset = CretioDataset()
    train_lens, val_lens = int(len(dataset) * 0.6), int(len(dataset) * 0.2)
    test_lens = len(dataset) - train_lens - val_lens
    train_set, val_set, test_set = random_split(dataset=dataset, lengths=[train_lens, val_lens, test_lens])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)
