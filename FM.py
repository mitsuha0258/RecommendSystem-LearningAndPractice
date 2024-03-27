import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import random_split, DataLoader

from utils import CretioDataset, EarlyStopper


class FM(nn.Module):
    def __init__(self, feats_num, hidden_dim):
        super(FM, self).__init__()
        self.feats_num = feats_num
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(self.feats_num, 1)
        self.v = nn.Parameter(torch.rand(self.feats_num, self.hidden_dim))
        nn.init.xavier_uniform_(self.v)

    def forward(self, x):
        first_order = self.linear(x)

        square_of_sum = torch.mm(x, self.v) ** 2
        sum_of_square = torch.mm(x ** 2, self.v ** 2)
        second_order = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        return torch.sigmoid(first_order + second_order)


def train(model, optimizer, criterion, train_loader, device):
    model.train()
    loop = tqdm(train_loader)
    for i, (x, label) in enumerate(loop):
        optimizer.zero_grad()

        x, label = x.to(device), label.to(device)
        output = model(x)
        loss = criterion(output.squeeze(), label.float())
        loss.backward()

        optimizer.step()

        loop.set_postfix(loss=loss.item())


def test(model, data_loader, device):
    model.eval()
    correct, total_num = 0, 0
    with torch.no_grad():
        for x, label in tqdm(data_loader):
            x, label = x.to(device), label.to(device)
            output = model(x)
            output = torch.where(output >= 0.5, 1, 0).squeeze()
            correct += (output == label).cpu().sum().item()
            total_num += len(label)
    return correct / total_num


if __name__ == '__main__':
    save_path = 'saved_model/FM.pt'
    device = 'cuda:0'
    batch_size = 64
    hidden_dim = 16
    learning_rate = 1e-3
    weight_decay = 1e-4
    epochs = 100

    dataset = CretioDataset()
    train_lens, val_lens = int(len(dataset) * 0.6), int(len(dataset) * 0.2)
    test_lens = len(dataset) - train_lens - val_lens
    train_set, val_set, test_set = random_split(dataset=dataset, lengths=[train_lens, val_lens, test_lens])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    model = FM(dataset.feats_num, hidden_dim=16).cuda()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    early_stopper = EarlyStopper(patience=10, save_path=save_path)

    print("Start Training...")
    for epoch in range(1, epochs + 1):
        train(model, optimizer, criterion, train_loader, device)
        acc = test(model, val_loader, device)
        print(f'epoch: {epoch}, val acc: {acc: .4f}')
        if not early_stopper.is_continuable(model, acc):
            print(f'val best acc: {early_stopper.best_acc: .4f}')
            break
    acc = test(model, test_loader, device)
    print(f'test acc: {acc}')
