import torch


class FFM(torch.nn.Module):
    def __init__(self, feats_num, hidden_dim, fields_num, feature2field):
        """
        :param feats_num: 输入特征的维度
        :param hidden_dim: 隐变量v的维度
        :param fields_num: 特征域的个数
        """
        super(FFM, self).__init__()
        self.feats_num = feats_num
        self.hidden_dim = hidden_dim
        self.fields_num = fields_num
        self.feature2field = feature2field

        self.linear = torch.nn.Linear(self.feats_num, 1)
        self.v = torch.nn.Parameter(torch.rand(self.feats_num, self.fields_num, self.hidden_dim))
        torch.nn.init.xavier_uniform_(self.v)

    def forward(self, x):
        """
        :param x: ``(batch_size, feats_num)``
        :return:
        """
        first_order = self.linear(x)
        second_order = torch.zeros_like(first_order)

        for i in range(self.feats_num):
            for j in range(i + 1, self.feats_num):
                # 同一个field下的特征不做交叉
                if self.feature2field[i] == self.feature2field[j]:
                    continue

                vi_vj = torch.sum(self.v[i, self.feature2field[j], :] * self.v[j, self.feature2field[i], :])  # 内积
                xi_xj = x[:, i] * x[:, j]
                second_order += (vi_vj * xi_xj).unsqueeze(1)

        return torch.sigmoid(first_order + second_order)


if __name__ == '__main__':
    pass
