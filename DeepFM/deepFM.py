import torch
import torch.nn as nn
import torch.nn.functional as F


class FM(nn.Module):
    def __init__(self, latent_dim, fea_num):
        super(FM, self).__init__()
        self.latent_dim = latent_dim
        self.w0 = nn.Parameter(torch.zeros(1,))
        self.w1 = nn.Parameter(torch.rand([fea_num, 1]))
        self.w2 = nn.Parameter(torch.rand([fea_num,]))
