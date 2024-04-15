import torch
import torch.nn as nn


class FM(nn.Module):
    def __init__(self, feats_num, hidden_dim):
        """
        :param feats_num: 输入特征的维度
        :param hidden_dim: 隐变量v的维度
        """
        super(FM, self).__init__()
        self.feats_num = feats_num
        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(self.feats_num, 1)
        self.v = nn.Parameter(torch.rand(self.feats_num, self.hidden_dim))
        nn.init.xavier_uniform_(self.v)

    def forward(self, x):
        """
        :param x: ``(batch_size, feats_num)``
        :return:
        """
        # 注意：输入特征x在实际过程中应该全为离散的onehot值。
        # 为了方便，在这里并未将连续值离散化
        first_order = self.linear(x)

        square_of_sum = torch.mm(x, self.v) ** 2
        sum_of_square = torch.mm(x ** 2, self.v ** 2)
        second_order = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        return torch.sigmoid(first_order + second_order)
