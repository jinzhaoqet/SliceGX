import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from dig.xgraph.dataset import SynGraphDataset
from sklearn.model_selection import train_test_split
from torch import default_generator
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset, Planetoid, PPI, Reddit, Coauthor, AmazonProducts, FacebookPagePage
from torch_geometric.loader import DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
import os


def get_dataloader(dataset, batch_size, data_split_ratio, seed=3407):
    """

    Args:
        dataset: which dataset you want
        batch_size: int
        data_split_ratio: list [train, valid, test]
        seed: random seed to split the dataset randomly

    Returns:
        a dictionary of training, validation, and testing dataLoader
    """

    num_train = int(data_split_ratio[0] * len(dataset))
    num_eval = int(data_split_ratio[1] * len(dataset))
    num_test = len(dataset) - num_train - num_eval

    train, eval, test = random_split(dataset,
                                     lengths=[num_train, num_eval, num_test],
                                     generator=default_generator.manual_seed(seed))

    dataloader = dict()
    dataloader['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloader['eval'] = DataLoader(eval, batch_size=batch_size, shuffle=True)
    dataloader['test'] = DataLoader(test, batch_size=batch_size, shuffle=True)
    return dataloader

def load_syn_data(dataset_dir, dataset_name):
    """ The synthetic dataset """
    dataset = SynGraphDataset(root=dataset_dir, name=dataset_name)
    dataset.node_type_dict = {k: v for k, v in enumerate(range(dataset.num_classes))}
    dataset.node_color = None
    return dataset


def split_dataset(dataset):
    splits = [7, 2, 1]
    num_nodes = dataset[0].num_nodes
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask =torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    ratios = [np.sum(splits[:i]) / np.sum(splits) for i in range(len(splits) + 1)]
    i = 0
    train_nids = torch.tensor(range(int(ratios[i] * num_nodes), int(ratios[i + 1] * num_nodes)))
    train_mask[train_nids]=True
    i += 1
    valid_nids = torch.tensor(range(int(ratios[i] * num_nodes), int(ratios[i + 1] * num_nodes)))
    valid_mask[valid_nids]=True
    i += 1
    test_nids = torch.tensor(range(int(ratios[i] * num_nodes), int(ratios[i + 1] * num_nodes)))
    test_mask[test_nids]=True
    return dataset, train_mask, valid_mask, test_mask

def get_dataset(dataset_root, dataset_name):
    sync_dataset_dict = {
        'BA_Shapes'.lower(): 'ba_shapes',
        'Tree_Cycle'.lower(): 'tree_cycle',
        'Tree_Grid'.lower(): 'tree_grid',
    }
    if dataset_name.lower() in sync_dataset_dict.keys():
        sync_dataset_filename = sync_dataset_dict[dataset_name.lower()]
        return load_syn_data(dataset_root, sync_dataset_filename)
    elif dataset_name.lower() in ['cora']:
        return Planetoid(root=dataset_root, name=dataset_name, split="public")
    elif dataset_name.lower() in ['cs']:
        return split_dataset(Coauthor(root=dataset_root,name=dataset_name))
    elif dataset_name.lower() in ['products']:
        return AmazonProducts(root=os.path.join(dataset_root, dataset_name))
    elif dataset_name.lower() in ['facebook']:
        return split_dataset(FacebookPagePage(root=os.path.join(dataset_root, dataset_name)))
    else:
        raise ValueError(f"{dataset_name} is not defined.")