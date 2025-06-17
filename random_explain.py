import copy
import os
import random
from warnings import simplefilter
import torch
import hydra
from omegaconf import OmegaConf
from dataset import get_dataset
from utils import get_logger, check_dirs
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def node_induced_edges(dataset, subset):
    subset_indices = torch.tensor(subset, dtype=torch.long).to(device)
    mask_0 = dataset.data.edge_index[0].unsqueeze(1) == subset_indices
    mask_1 = dataset.data.edge_index[1].unsqueeze(1) == subset_indices
    mask = mask_0.any(dim=1) & mask_1.any(dim=1)
    return dataset.data.edge_index[:, mask]


def pipeline(cut_layer,config,dataset,node_indices,logger,size_run):
    config.models.gnn_saving_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    config.explainers.explanation_result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if config.models.param.graph_classification:
        pass
    else:
        data = dataset.data
        data.to(device)
        explanation_saving_dir = os.path.join('',
                                              config.datasets.dataset_name,
                                              config.models.gnn_name)
        check_dirs(explanation_saving_dir)
        for node_idx in node_indices:
            logger.info('node_idx:{}'.format(node_idx))
            sample_nodes = list(random.sample(range(dataset.data.num_nodes), size_run))
            if node_idx not in sample_nodes:
                sample_nodes[-1] = node_idx
            other_nodes = [node for node in range(dataset.data.num_nodes) if node not in sample_nodes]
            if node_idx not in other_nodes:
                other_nodes.append(node_idx)
            edge_index = node_induced_edges(dataset, sample_nodes)
            torch.save(sample_nodes, os.path.join(explanation_saving_dir,
                                                                 f'{len(config.models.param.gnn_latent_dim)}_example_{node_idx}.pt'))
            other_edge_index = node_induced_edges(dataset, other_nodes)


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):
    config.models.param = config.models.param[config.datasets.dataset_name]
    dataset = get_dataset(dataset_root='datasets', dataset_name=config.datasets.dataset_name)
    if dataset.data.x is not None:
        dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()

    log_file = (
        f"random_{config.datasets.dataset_name}_{len(config.models.param.gnn_latent_dim)}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.info(OmegaConf.to_yaml(config))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    logger.info(f'Using data: {dataset.data}')

    layer_nums = len(config.models.param.gnn_latent_dim)
    config.models.gnn_savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    state_dict = torch.load(os.path.join(config.models.gnn_savedir,
                                         config.datasets.dataset_name,
                                         f'{config.models.gnn_name}_'
                                         f'{len(config.models.param.gnn_latent_dim)}l_best.pth'))['net']

    if config.datasets.dataset_name in ['tree_grid', 'tree_cycle', 'tree_grid']:
        test_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
    else:
        test_indices = torch.where(dataset.data.test_mask)[0].tolist()
    logger.info(f'test nodes : {test_indices}')
    for size_run in config.explainers.size:
        logger.info(f'size:{size_run}')
        for i in range(layer_nums - 1, -1, -1):
            logger.info(f'test layer: {len(config.models.param.gnn_latent_dim) - i}')
            pipeline(len(config.models.param.gnn_latent_dim) - i, config, dataset, test_indices, logger, size_run)


if __name__ == '__main__':
    import sys
    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
