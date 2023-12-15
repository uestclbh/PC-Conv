import torch
import math
import numpy as np


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask
    #split 0 random 60% 20% 20%     #split  sparse 2.5% 2.5% 95%
    #split 1 fixed  60% 20% 20%

    #split 2 20,50,1000
    #split 3 big data fixed

def random_planetoid_splits0(data, num_classes, percls_trn, val_lb, seed):
    print("random,the split : {},{}".format(percls_trn,val_lb))
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)

    return data


def random_planetoid_splits1(data, runs, dataset_str):
    print("fixed")
    if dataset_str == 'Actor':
        dataset_str = 'film'
    if dataset_str == 'Squirrel_filtered' or dataset_str == 'Chameleon_filtered':
        if dataset_str == 'Squirrel_filtered':
            dataset_str = 'squirrel'
        if dataset_str == 'Chameleon_filtered':
            dataset_str = 'chameleon'

        data.train_mask = data.train_mask_total[runs].bool()
        data.val_mask = data.val_mask_total[runs].bool()
        data.test_mask = data.test_mask_total[runs].bool()
        return data
    else:
        splits_file_path = 'splits/' + dataset_str.lower() + \
                           '_split_0.6_0.2_' + str(runs) + '.npz'
        with np.load(splits_file_path) as splits_file:
            train_mask = splits_file['train_mask']
            val_mask = splits_file['val_mask']
            test_mask = splits_file['test_mask']
        data.train_mask = torch.from_numpy(train_mask).bool()
        data.val_mask = torch.from_numpy(val_mask).bool()
        data.test_mask = torch.from_numpy(test_mask).bool()
        return data


def random_planetoid_splits2(data, num_classes, seed):
    percls_trn = 20
    val_lb = 500
    test_num = 1000
    index = [i for i in range(0, data.y.shape[0])]
    train_idx = []
    rnd_state = np.random.RandomState(seed)
    for c in range(num_classes):
        class_idx = np.where(data.y.cpu() == c)[0]
        if len(class_idx) < percls_trn:
            train_idx.extend(class_idx)
        else:
            train_idx.extend(rnd_state.choice(class_idx, percls_trn, replace=False))
    rest_index = [i for i in index if i not in train_idx]
    val_idx = rnd_state.choice(rest_index, val_lb, replace=False)
    test_idx = [i for i in rest_index if i not in val_idx]
    test_idx = rnd_state.choice(test_idx, test_num, replace=False)

    data.train_mask = index_to_mask(train_idx, size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx, size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx, size=data.num_nodes)

    return data


def random_planetoid_splits3(data,dataset,runs):
    name = dataset
    # if name == 'Penn94':
    name = 'fb100-Penn94'
    # if not os.path.exists(f'./data/splits/{name}-splits.npy'):
    #     assert dataset in splits_drive_url.keys()
    #     gdown.download(
    #         id=splits_drive_url[dataset], \
    #         output=f'./data/splits/{name}-splits.npy', quiet=False)

    splits_lst = np.load(f'./splits/{name}-splits.npy', allow_pickle=True)
    for i in range(len(splits_lst)):
        for key in splits_lst[i]:
            if not torch.is_tensor(splits_lst[i][key]):
                splits_lst[i][key] = torch.as_tensor(splits_lst[i][key])

    train_idx = splits_lst[runs]['train']
    val_idx=splits_lst[runs]['valid']
    test_idx=splits_lst[runs]['test']

    data.train_mask = index_to_mask(train_idx,size=data.num_nodes)
    data.val_mask = index_to_mask(val_idx,size=data.num_nodes)
    data.test_mask = index_to_mask(test_idx,size=data.num_nodes)
    return data


def random_planetoid_splits(args,data):
    split_case = args.split
    num_classes = data.num_classes
    percls_trn = int(round(args.train_rate * len(data.y) / data.num_classes))
    val_lb = int(round(args.val_rate * len(data.y)))
    seed =  args.seed
    runs = args.runs
    dataset = args.dataset
    if split_case == 0 :
        data1 = random_planetoid_splits0(data, num_classes, percls_trn, val_lb, seed)
    if split_case == 1:
        data1 = random_planetoid_splits1(data, runs, dataset)
    if split_case == 2:
        data1 = random_planetoid_splits2(data, num_classes, seed)
    if split_case == 3:
        data1 = random_planetoid_splits3(data,dataset,runs)
    return data1
