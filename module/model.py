from module.layer import *
from torch import nn
from module.sync_bn import SyncBatchNorm
from helper import context as ctx


class GNNBase(nn.Module):

    def __init__(self, layer_size, activation, dropout=0.5, norm='layer', n_linear=0):
        super(GNNBase, self).__init__()
        self.n_layers = len(layer_size) - 1
        self.layers = nn.ModuleList()
        self.activation = activation
        self.n_linear = n_linear

        if norm is None:
            self.use_norm = False
        else:
            self.use_norm = True
            self.norm = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

class GraphSAGE(GNNBase):

    def __init__(self, layer_size, activation, dropout=0.5, norm='layer', train_size=None, n_linear=0):
        super(GraphSAGE, self).__init__(layer_size, activation, dropout, norm, n_linear)
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:# 前n_layers - n_linear层: GraphSAGELayer层
                self.layers.append(GraphSAGELayer(layer_size[i], layer_size[i + 1]))
            else:#后n_linear层:Linear层
                self.layers.append(nn.Linear(layer_size[i], layer_size[i + 1]))
            if i < self.n_layers - 1 and self.use_norm:
                if norm == 'layer':
                    self.norm.append(nn.LayerNorm(layer_size[i + 1], elementwise_affine=True))
                elif norm == 'batch':
                    self.norm.append(SyncBatchNorm(layer_size[i + 1], train_size))

    def forward(self, g, feat, in_deg=None):
        h = feat
        for i in range(self.n_layers):
            if i < self.n_layers - self.n_linear:
                if self.training:
                    h = ctx.buffer.update(i, h)
                h = self.dropout(h)
                h = self.layers[i](g, h, in_deg)
            else:
                h = self.dropout(h)
                h = self.layers[i](h)

            if i < self.n_layers - 1:
                if self.use_norm:
                    h = self.norm[i](h)
                h = self.activation(h)

        return h
    