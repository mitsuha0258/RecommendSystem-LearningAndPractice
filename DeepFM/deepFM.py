import torch
import torch.nn as nn
import torch.nn.functional as F
from torchkeras import summary


class FM(nn.Module):
    def __init__(self, latent_dim, fea_num):
        """
        :param latent_dim: 离散特征的隐向量维度
        :param fea_num: 特征个数
        """
        super(FM, self).__init__()
        self.latent_dim = latent_dim
        self.w0 = nn.Parameter(torch.zeros(1,))
        self.w1 = nn.Parameter(torch.rand([fea_num, 1]))
        self.w2 = nn.Parameter(torch.rand([fea_num, latent_dim]))

    def forward(self, x):
        # 一阶特征交叉
        first_order = self.w0 + torch.mm(x, self.w1)  # (batch_size, 1)
        # 二阶特征交叉，采用FM最终简化形式
        second_order = 0.5 * torch.sum(
            torch.pow(torch.mm(x, self.w2), 2) - torch.mm(torch.pow(x, 2), torch.pow(self.w2, 2)),
            dim=1,
            keepdim=True
        )  # (batch_size, 1)

        return first_order + second_order


class DNN(nn.Module):
    def __init__(self, hidden_units, dropout=0.0):
        """
        :param hidden_units: 记录隐藏层神经元个数列表
        :param dropout: 丢弃率
        """
        super(DNN, self).__init__()

        self.dnn_net = nn.ModuleList([nn.Linear(layer[0], layer[1]) for layer in list(zip(hidden_units[:-1], hidden_units[1:]))])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer in self.dnn_net:
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        return x


class DeepFM(nn.Module):
    def __init__(self, fea_cols, hidden_units, dropout=0.0):
        super(DeepFM, self).__init__()

        self.dense_fea_cols, self.sparse_fea_cols = fea_cols

        self.embed_layers = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(num_embeddings=fea['feat_num'], embedding_dim=fea['embed_dim'])
            for i, fea in enumerate(self.sparse_fea_cols)
        })

        self.fea_num = len(self.dense_fea_cols) + len(self.sparse_fea_cols)*self.sparse_fea_cols[0]['embed_dim']
        hidden_units.insert(0, self.fea_num)

        self.fm = FM(self.sparse_fea_cols[0]['embed_dim'], self.fea_num)
        self.dnn = DNN(hidden_units, dropout=dropout)
        self.fc = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        dense_x, sparse_x = x[:, :len(self.dense_fea_cols)], x[:, len(self.dense_fea_cols):]
        sparse_x = sparse_x.long()
        sparse_embeds = [self.embed_layers['embed_'+str(i)](sparse_x[:, i]) for i in range(sparse_x.shape[1])]
        sparse_embeds = torch.cat(sparse_embeds, dim=-1)

        x = torch.cat([sparse_embeds, dense_x], dim=-1)
        wide_output = self.fm(x)
        deep_output = self.fc(self.dnn(x))

        outputs = torch.sigmoid(torch.add(wide_output, deep_output))
        return outputs


if __name__ == '__main__':
    from DeepFM.main import load_preprocessed_data
    fea_cols, (trn_x, trn_y), (val_x, val_y), test_x = load_preprocessed_data()

    hidden_units = [128, 64, 32]
    dropout = 0.2
    model = DeepFM(fea_cols, hidden_units, dropout=dropout)

    summary(model, input_shape=(trn_x.shape[1],))
