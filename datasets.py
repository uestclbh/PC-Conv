import torch
from torch_geometric.utils import is_undirected, to_undirected
import os
from torch import Tensor, LongTensor
import dataset_loader

class BaseGraph:
    def __init__(self, x: Tensor, edge_index: LongTensor, edge_weight: Tensor,
                 y: Tensor,train_mask=None,val_mask = None,test_mask = None):
        self.x = x
        self.edge_index = edge_index
        print(edge_weight)
        self.edge_weight = edge_weight
        self.y = y
        self.num_classes = torch.unique(y).shape[0]
        self.num_nodes = x.shape[0]
        self.train_mask = None
        self.val_mask = None
        self.test_mask = None
        self.train_mask_total = train_mask
        self.val_mask_total = val_mask
        self.test_mask_total = test_mask
        self.to_undirected()

    def to_undirected(self):
        if not is_undirected(self.edge_index):
            print(self.edge_index.shape)
            self.edge_index, self.edge_weight = to_undirected(
                self.edge_index, self.edge_weight)

    def to(self, device):
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_weight = self.edge_weight.to(device)
        self.y = self.y.to(device)
        return self



def load_dataset(name: str):
    '''
    load dataset into a base graph format.
    '''
    savepath = f"./data/{name}.pt"
    if name in [ 'chameleon', 'film', 'squirrel']:
        if os.path.exists(savepath):
            bg = torch.load(savepath, map_location="cpu")
            return bg
        ds = dataset_loader.DataLoader(name)
        data = ds[0]
        data.num_classes = ds.num_classes
        x = data.x  # torch.empty((data.x.shape[0], 0))
        ei = data.edge_index
        ea = torch.ones(ei.shape[1])
        y = data.y
        bg = BaseGraph(x, ei, ea, y)
        bg.num_classes = data.num_classes
        bg.y = bg.y.to(torch.int64)
        torch.save(bg, savepath)
        return bg

import numpy as np
def load_dataset_filtered(name: str):
    '''
    load dataset into a base graph format.
    '''
    savepath = f"./data/{name}.pt"
    if name in [ 'chameleon_filtered','squirrel_filtered']:
        data = np.load(os.path.join('data', f'{name.replace("-", "_")}.npz'))
        x = torch.tensor(data['node_features'])
        y = torch.tensor(data['node_labels'])
        print(y.shape)
        edges = torch.tensor(data['edges'])
        ei = edges
        ei = torch.transpose(ei, 0, 1)
        print(ei)
        print(ei.shape)
        ea = torch.ones(ei.shape[1])
        train_masks = torch.tensor(data['train_masks'])
        val_masks = torch.tensor(data['val_masks'])
        test_masks = torch.tensor(data['test_masks'])
        # if os.path.exists(savepath):
        #     bg = torch.load(savepath, map_location="cpu")
        #     return bg
        # ei = torch.transpose(ei,0,1)
        # y = data.y
        bg = BaseGraph(x, ei, ea, y,train_masks,val_masks,test_masks)
        bg.num_classes = 5
        bg.y = bg.y.to(torch.int64)
        torch.save(bg, savepath)

        return bg