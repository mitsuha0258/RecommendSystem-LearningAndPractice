import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.relu(out+residual)
        return out


class DeepCrossing(nn.Module):
    def __init__(self, features_cols, hidden_units, dropout=0.0):
        super(DeepCrossing, self).__init__()

        self.dense_fea_cols, self.sparse_fea_cols = features_cols
        self.embed_layers = nn.ModuleDict({
            'embed_'+str(i): nn.Embedding(fea['feat_num'], fea['embed_dim'])
            for i, fea in enumerate(self.sparse_fea_cols)
        })

        self.fea_num = len(self.dense_fea_cols) + len(self.sparse_fea_cols)*self.sparse_fea_cols[0]['embed_dim']
        hidden_units.insert(0, self.fea_num)

        self.res_layers = nn.ModuleList([
            ResidualBlock(self.fea_num, unit)
            for unit in hidden_units
        ])
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(self.fea_num, 1)

    def forward(self, x):
        dense_x, sparse_x = x[:, :len(self.dense_fea_cols)], x[:, len(self.dense_fea_cols):]
        sparse_x = sparse_x.long()
        sparse_embeds = [self.embed_layers['embed_'+str(i)](sparse_x[:, i])
                         for i in range(len(self.sparse_fea_cols))]
        sparse_embeds = torch.cat(sparse_embeds, dim=-1)

        x = torch.cat([sparse_embeds, dense_x], dim=-1)
        for layer in self.res_layers:
            x = layer(x)
        x = self.dropout(x)
        out = torch.sigmoid(self.linear(x))
        return out
