import torch
import random
import math
import torch.nn.functional as F
import os.path as osp
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch.autograd import Variable
from torch.nn import Parameter
from torch.nn import Linear
from torch_geometric.nn import GATConv, GCNConv, ChebConv
from torch_geometric.nn import MessagePassing, APPNP
from torch_geometric.utils import to_scipy_sparse_matrix
import scipy as sp
from scipy.special import comb
from PCpro import *


class GPR_prop(MessagePassing):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, K, alpha, Init, Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0 * np.ones(K + 1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
            TEMP[-1] = (1 - alpha) ** K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha) ** np.arange(K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3 / (K + 1))
            TEMP = np.random.uniform(-bound, bound, K + 1)
            TEMP = TEMP / np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)


class GPRGNN(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GPRGNN, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)

        self.prop1 = GPR_prop(args.K, args.alpha, args.Init)

        self.Init = args.Init
        self.dprate = args.dprate
        self.dropout = args.dropout

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin1(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
            return F.log_softmax(x, dim=1)


class PCNet1(torch.nn.Module):
    def __init__(self, dataset, conv_layer, n_poly,
                 aggr,
                 alpha,
                 dpb1,
                 dpb2,
                 dpt1,
                 a,
                 b, c, large):
        super(PCNet1, self).__init__()
        self.hidden = 64
        self.lin1 = Linear(dataset.x.shape[1], self.hidden)
        self.lin2 = Linear(self.hidden, dataset.num_classes)
        self.prop1 = PC_prop_gen(conv_layer, n_poly, alpha, a, b, c)
        self.dropout1 = dpb1
        self.dprate = dpt1
        self.dropout2 = dpb2
        self.large = large

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        if self.large == 1 :
            x, edge_index, = data.x, data.edge_index
            x = F.dropout(x, p=self.dropout1, training=self.training)
            x = self.lin1(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout2, training=self.training)
            x = self.lin2(x)
        else:
            x, edge_index, = data.x, data.edge_index
            x = F.dropout(x, p=self.dropout1, training=self.training)
            x = self.lin1(x)
            x = F.dropout(x, p=self.dropout2, training=self.training)
            x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class PCNet2(torch.nn.Module):
    def __init__(self, dataset, conv_layer, n_poly,
                 aggr,
                 alpha,
                 dpb1,
                 dpb2,
                 dpt1,
                 a,
                 b, c):
        super(PCNet2, self).__init__()
        self.hidden = 64
        self.lin1 = Linear(dataset.x.shape[1], self.hidden)
        self.lin2 = Linear(self.hidden, dataset.num_classes)
        self.prop1 = PC_prop_ablation(conv_layer, n_poly, alpha, a, b)
        self.dropout1 = dpb1
        self.dprate = dpt1
        self.dropout2 = dpb2

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index, = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout1, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout2, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class PCNet_large(torch.nn.Module):
    def __init__(self, dataset, conv_layer, n_poly,
                 aggr,
                 alpha,
                 dpb1,
                 dpb2,
                 dpt1,
                 a,
                 b, c, d):
        super(PCNet_large, self).__init__()
        self.hidden = d
        # print(dataset.x)
        # print(dataset.num_classes)
        self.lin1 = Linear(dataset.x.shape[1], self.hidden)
        self.lin2 = Linear(self.hidden, dataset.num_classes)
        self.prop1 = PC_prop_gen(conv_layer, n_poly, alpha, a, b, c)
        self.dropout1 = dpb1
        self.dprate = dpt1
        self.dropout2 = dpb2

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x, edge_index, = data.x, data.edge_index

        x = F.dropout(x, p=self.dropout1, training=self.training)
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout2, training=self.training)
        x = self.lin2(x)

        if self.dprate == 0.0:
            x = self.prop1(x, edge_index)
        else:
            x = F.dropout(x, p=self.dprate, training=self.training)
            x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class GCN_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GCN_Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, 32, K=2)
        self.conv2 = ChebConv(32, dataset.num_classes, K=2)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(GAT_Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class APPNP_Net(torch.nn.Module):
    def __init__(self, dataset, args):
        super(APPNP_Net, self).__init__()
        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.prop1 = APPNP(args.K, args.alpha)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        x = self.prop1(x, edge_index)
        return F.log_softmax(x, dim=1)


class MLP(torch.nn.Module):
    def __init__(self, dataset, args):
        super(MLP, self).__init__()

        self.lin1 = Linear(dataset.num_features, args.hidden)
        self.lin2 = Linear(args.hidden, dataset.num_classes)
        self.dropout = args.dropout

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        return F.log_softmax(x, dim=1)
