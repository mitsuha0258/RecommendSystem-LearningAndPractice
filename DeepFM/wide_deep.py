import torch
import torch.nn as nn
import torch.nn.functional as F


class WideNet(nn.Module):
    def __init__(self, features_num):
        super(WideNet, self).__init__()

        self.linear = nn.Linear(features_num, 1)

    def forward(self, x):
        return self.linear(x)


class DeepNet(nn.Module):
    def __init__(self, hidden_units, dropout=0.0):
        super(DeepNet, self).__init__()

        self.dnn = nn.ModuleList([
            nn.Linear(hidden_units[i], hidden_units[i + 1])
            for i in range(len(hidden_units) - 1)
        ])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        for layer in self.dnn:
            x = layer(x)
            x = F.relu(x)
            x = self.dropout(x)
        return x


class WideDeep(nn.Module):
    def __init__(self, features_columns, hidden_units, dropout=0.0):
        super(WideDeep, self).__init__()

        self.dense_features_cols, self.sparse_features_cols = features_columns

        self.embedding = nn.ModuleDict({
            'embed_' + str(i): nn.Embedding(feat['feat_num'], feat['embed_dim'])
            for i, feat in enumerate(self.sparse_features_cols)
        })
        hidden_units.insert(0, len(self.dense_features_cols) +
                            len(self.sparse_features_cols) * self.sparse_features_cols[0]['embed_dim'])
        self.wide = WideNet(len(self.dense_features_cols))
        self.deep = DeepNet(hidden_units, dropout=dropout)
        self.fc = nn.Linear(hidden_units[-1], 1)

    def forward(self, x):
        dense_x, sparse_x = x[:, :len(self.dense_features_cols)], x[:, len(self.dense_features_cols):]
        sparse_x = sparse_x.long()
        sparse_embeds = [self.embedding['embed_' + str(i)](sparse_x[:, i])
                         for i in range(len(self.sparse_features_cols))]
        sparse_embeds = torch.cat(sparse_embeds, dim=-1)

        wide_output = self.wide(dense_x)

        deep_input = torch.cat([dense_x, sparse_embeds], dim=-1)
        deep_output = self.deep(deep_input)
        deep_output = self.fc(deep_output)

        output = torch.sigmoid(0.5 * (wide_output + deep_output))

        return output
