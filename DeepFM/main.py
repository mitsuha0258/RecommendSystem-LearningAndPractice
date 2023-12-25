import datetime

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch import optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from DeepFM.deepFM import DeepFM
from DeepFM.deep_crossing import DeepCrossing
from DeepFM.wide_deep import WideDeep


hidden_units = [64, 32, 16]
dropout = 0.2
epochs = 10
log_step_freq = 10


def load_preprocessed_data(file_path='./preprocessed_data/'):
    train = pd.read_csv(file_path + 'train_set.csv')
    test = pd.read_csv(file_path + 'test_set.csv')
    val = pd.read_csv(file_path + 'val_set.csv')

    trn_x, trn_y = train.drop(columns=['Label']).values, train['Label'].values
    val_x, val_y = val.drop(columns=['Label']).values, val['Label'].values
    test_x = test.values

    fea_cols = np.load(file_path + 'fea_col.npy', allow_pickle=True)
    print(fea_cols)

    return fea_cols, (trn_x, trn_y), (val_x, val_y), test_x


def auc(y_pred, y_true):
    pred = y_pred.data
    y = y_true.data
    return roc_auc_score(y, pred)


def train(model):
    loss_func = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    metric_func = auc
    metric_name = 'auc'

    print('start_training.........')
    nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('========' * 8 + '%s' % nowtime)

    for epoch in range(1, epochs + 1):
        # train
        model.train()
        loss_sum = 0.0
        metric_sum = 0.0
        step = 1

        for (features, labels) in train_loader:
            optimizer.zero_grad()

            pred = model(features)
            pred = pred.squeeze()
            # print(pred.shape, labels.shape)
            loss = loss_func(pred, labels)
            try:
                metric = metric_func(pred, labels)
            except ValueError:
                pass

            loss.backward()
            optimizer.step()

            loss_sum += loss.item()
            metric_sum += metric.item()
            if step % log_step_freq == 0:
                print(("[step=%d] loss: %.3f, " + metric_name + ": %.3f") % (step, loss_sum / step, metric_sum / step))
            step += 1

        # validation
        model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_step = 1

        for (features, labels) in val_loader:
            with torch.no_grad():
                predictions = model(features)
                predictions = predictions.squeeze()
                val_loss = loss_func(predictions, labels)
                try:
                    val_metric = metric_func(predictions, labels)
                except ValueError:
                    pass

            val_loss_sum += val_loss.item()
            val_metric_sum += val_metric.item()
            val_step += 1

        # 记录日志
        info = (epoch, loss_sum / step, metric_sum / step, val_loss_sum / val_step, val_metric_sum / val_step)
        # dfhistory.loc[epoch - 1] = info

        # 打印日志
        print((
                      "\nEPOCH=%d, loss=%.3f, " + metric_name + " = %.3f, val_loss=%.3f, " + "val_" + metric_name + " = %.3f") % info)
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('\n' + '==========' * 8 + '%s' % nowtime)

    print('Finished Training')


if __name__ == '__main__':
    fea_cols, (trn_x, trn_y), (val_x, val_y), test_x = load_preprocessed_data()

    train_dataset = TensorDataset(torch.tensor(trn_x).float(), torch.tensor(trn_y).float())
    val_dataset = TensorDataset(torch.tensor(val_x).float(), torch.tensor(val_y).float())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

    # model = DeepFM(fea_cols, hidden_units, dropout=dropout)
    # model = WideDeep(fea_cols, hidden_units, dropout=dropout)
    model = DeepCrossing(fea_cols, hidden_units, dropout=dropout)
    train(model)
    # dfhistory = pd.DataFrame(columns=['epoch', 'loss', metric_name, 'val_loss', 'val_' + metric_name])

    # 模型的保存与使用
    # torch.save(model, './model/DeepFM.pkl')
    # torch.save(model, './model/WideDeep.pkl')
    torch.save(model, './model/DeepCrossing.pkl')
