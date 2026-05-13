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
from torch_geometric.utils import from_scipy_sparse_matrix, subgraph
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

def load_yelpchi_data(dataset_dir, seed=2):
    import scipy.io as sio
    from torch_geometric.data import Data

    mat_path = os.path.join(dataset_dir, 'YelpChi', 'raw', 'YelpChi.mat')
    mat = sio.loadmat(mat_path)

    x = torch.FloatTensor(mat['features'].toarray())
    y = torch.LongTensor(mat['label'].flatten())
    edge_index, _ = from_scipy_sparse_matrix(mat['homo'])

    data = Data(x=x, edge_index=edge_index, y=y)

    num_nodes = x.shape[0]
    rng = np.random.default_rng(seed)
    perm = torch.from_numpy(rng.permutation(num_nodes))

    n_train = int(0.7 * num_nodes)
    n_val   = int(0.1 * num_nodes)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask  = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[perm[:n_train]]               = True
    valid_mask[perm[n_train:n_train + n_val]] = True
    test_mask [perm[n_train + n_val:]]        = True

    class _YelpChiDataset:
        def __init__(self, data):
            self._data = data
            self.num_node_features = data.x.shape[1]
            self.num_classes = int(data.y.max().item()) + 1

        def __getitem__(self, idx):
            return self._data

    return _YelpChiDataset(data), train_mask, valid_mask, test_mask


def sample_subgraph(data, sample_ratio, seed=2):
    """Randomly sample a fraction of nodes and extract the induced subgraph."""
    from torch_geometric.data import Data
    num_nodes = data.num_nodes
    rng = np.random.default_rng(seed)
    n_sample = int(num_nodes * sample_ratio)
    perm = torch.from_numpy(rng.permutation(num_nodes))
    subset, _ = perm[:n_sample].sort()

    edge_index, _ = subgraph(subset, data.edge_index, relabel_nodes=True, num_nodes=num_nodes)
    sampled = Data(
        x=data.x[subset],
        edge_index=edge_index,
        y=data.y[subset],
        train_mask=data.train_mask[subset],
        val_mask=data.val_mask[subset],
        test_mask=data.test_mask[subset],
    )

    class _SampledDataset:
        def __init__(self, d):
            self._data = d
            self.num_node_features = d.x.shape[1]
            # multi-label one-hot: y is 2-D; single-label: y is 1-D
            self.num_classes = d.y.shape[1] if d.y.dim() == 2 else int(d.y.max().item()) + 1

        def __getitem__(self, idx):
            return self._data

    return _SampledDataset(sampled)


def get_dataset(dataset_root, dataset_name, sample_ratio=None):
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
    elif dataset_name.lower() in ['yelpchi']:
        return load_yelpchi_data(dataset_root)
    else:
        raise ValueError(f"{dataset_name} is not defined.")