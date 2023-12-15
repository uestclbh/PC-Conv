
from torch_geometric.utils import degree
from typing import Optional
from torch_geometric.typing import OptTensor
import math
import torch

from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, contains_self_loops
from torch_geometric.utils import get_laplacian
from scipy.special import comb
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np


def C_n(n, x, lamda):
    if (n == 0):
        return 1
    if (n == 1):
        return x - lamda
    else:
        return (x - n - lamda + 1) * C_n(n - 1, x, lamda) - (n - 1) * lamda * C_n(n - 2, x, lamda)


class PC_prop_gen(MessagePassing):

    def reset_parameters(self):
        self.temp.data.fill_(self.alpha)

    def __init__(self, conv_layer, n_poly, alpha, a, b, c, **kwargs):
        super(PC_prop_gen, self).__init__(aggr='add', **kwargs)
        self.alpha = alpha
        self.K = conv_layer
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()
        self.a = a
        self.b = b
        self.c = c
        self.n_poly = n_poly

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = self.temp

        edge_index, edge_weight = add_self_loops(edge_index,edge_weight, num_nodes=x.size(self.node_dim))
        row, col = edge_index
        deg1 = degree(row, x.size(self.node_dim), dtype=x.dtype)
        deg2 = degree(col, x.size(self.node_dim), dtype=x.dtype)
        deg_inv_sqrt1 = deg1.pow(-self.c)
        deg_inv_sqrt2 = deg2.pow(-self.c)
        norm = deg_inv_sqrt1[row] * deg_inv_sqrt2[col]
        edge_index1, norm1 = add_self_loops(edge_index, -norm, fill_value=self.b, num_nodes=x.size(self.node_dim))

        x_num = range(1, self.K + 1)
        x_num = list(x_num)

        lamda_num = self.a
        out_total = x * TEMP[0]
        tmp1 = []
        tmp1.append(x)
        for k in range(self.n_poly):
            x = self.propagate(edge_index1, x=-x, norm=norm1, size=None)
            tmp1.append(x)
        # CH = 1
        for i in range(len(x_num)):
            out1 = 0
            # CH = CH * TEMP[i + 1]
            for j in range(self.n_poly):
                out1 = out1 + C_n(j, x_num[i], lamda_num) / math.factorial(j) * tmp1[j]
            out_total = out_total + TEMP[i + 1] * out1
        return out_total

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
class PC_prop_ablation(MessagePassing):

    def reset_parameters(self):
        self.temp.data.fill_(self.alpha)

    def __init__(self, conv_layer, n_poly, alpha, a, b, **kwargs):
        super(PC_prop_ablation, self).__init__(aggr='add', **kwargs)
        self.alpha = alpha
        self.K = conv_layer
        self.temp = Parameter(torch.Tensor(self.K + 1))
        self.reset_parameters()
        self.a = a
        self.b = b
        self.n_poly = n_poly

    def forward(self, x, edge_index, edge_weight=None):
        TEMP = self.temp

        if not contains_self_loops(edge_index):
            edge_index, edge_weight = add_self_loops(edge_index, num_nodes=x.size(self.node_dim))
        # L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight, normalization='sym', dtype=x.dtype,
                                           num_nodes=x.size(self.node_dim))
        edge_index1, norm1 = add_self_loops(edge_index1, norm1, fill_value=-self.b,
                                            num_nodes=x.size(self.node_dim))
        x_num = range(1, self.K + 1)
        x_num = list(x_num)

        lamda_num = self.a
        out_total = x * TEMP[0]
        tmp1 = []
        tmp1.append(x)
        for k in range(self.n_poly):
            x = self.propagate(edge_index1, x=x, norm=norm1, size=None)
            tmp1.append(-x)
        for i in range(len(x_num)):
            out1 = 0
            for j in range(self.n_poly):
                out1 = out1 + C_n(j, x_num[i], lamda_num) / math.factorial(j) * tmp1[j]
            out_total = out_total + TEMP[i + 1] * out1
        return out_total

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)