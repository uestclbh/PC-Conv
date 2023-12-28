# PC-Conv: Unifying Homophily and Heterophily with Two-fold Filtering
This repository contains a PyTorch implementation of (["PC-Conv: Unifying Homophily and Heterophily with Two-fold Filtering"](https://arxiv.org/abs/2312.14438)).

## Requirements
Tested combination: Python 3.9.6 + [PyTorch 1.9.0](https://pytorch.org/get-started/previous-versions/) + [PyTorch_Geometric 2.0.3](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) + [PyTorch Sparse 0.6.12](https://github.com/rusty1s/pytorch_sparse)

Other required python libraries include: numpy, scikit-learn, optuna, seaborn etc.

# Run
We list the details of PCNet performance in Tables 1, 2 and 3 of the paper, 
including the corresponding optimal parameters, 
in the comments section of "bestHyperparams.py". 
Since the experimental settings for 5.1, 5.2 and 5.3 are different, 
we set "--split, --gnn_type" to determine the running state.
All .pys have detailed annotations.
### Experimental setup for 5.1
    # --gnn_type 2  --split 2 --net PCNet 
### Experimental setup for 5.2
    # --gnn_type 0  --split 0 --net PCNet 
### Experimental setup for 5.3
    # --gnn_type 0  --split 1 --net PCNet 
### Two ways to reproduce performance for a specific data set in a specific table
#### 1. Hyperparameter search using optuna, which is also the method we use. 
    --dataset $dataset --gnn_type $gnn_type  --split $split --net $net
e.g. for dataset cora of table 1 (Experimental 5.1)
    

```
--dataset Cora --gnn_type 2  --split 2 --net PCNet --reproduce 1
```

#### 2.Straightforward method, but may deviate from the results in the paper due to random seed. (roughly  same)
    --dataset $dataset --gnn_type $gnn_type  --split $split --net $net --test --reproduce $reproduce
e.g. for dataset cora of table 1 (Experimental 5.1)

```
--dataset Cora --gnn_type 2  --split 2 --net PCNet --test --reproduce 1
```

e.g. for dataset Pubmed of table 2 (Experimental 5.2)
```
--dataset Pubmed --gnn_type 0  --split 0 --net PCNet --test --reproduce 2
```

e.g. for dataset Citeseer of table 3 (Experimental 5.3)
```
--dataset Citeseer --gnn_type 0  --split 1 --net PCNet --test --reproduce 3
```

e.g. for dataset Penn94 of table 3 (Experimental 5.3)
```
--dataset Penn94 --gnn_type 0  --split 3 --net PCNet --test --reproduce 3
```



