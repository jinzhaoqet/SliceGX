import copy
import os
from warnings import simplefilter
import hydra
from torch_geometric.utils import add_remaining_self_loops
import torch
from baseline_utils.utils import k_hop_subgraph
from dataset import get_dataset
from explain import Explainer
from gnnNets import get_gnnNets
from explainers import GraphMaskExplainer
from utils import get_logger, check_dirs
device = torch.device("cuda" if torch.cuda.is_available() else "")
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import time
from explain.config import MaskType, ModelMode, ModelTaskLevel, ExplainerConfig, ExplanationType
from explain.algorithm.base import ModelConfig 
import torch.nn.functional as F

from torch_geometric.utils import k_hop_subgraph
import gc

def pipeline(config, dataset, test_indices, model, device, logger):
    """
    为给定的节点执行GraphMask解释，并收集 Fidelity+ 和 Fidelity- 指标。
    （已修改为使用子图来节省显存）
    """
    model.eval()
    model.to(device)

    data = dataset.data
    
    if data.edge_index is not None:
        data.edge_index, _ = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)

    explainer = GraphMaskExplainer(
        num_layers=len(config.models.param.gnn_latent_dim)
    )
    explainer_config = ExplainerConfig(
        explanation_type=ExplanationType.model,
        node_mask_type=MaskType.object,
        edge_mask_type=MaskType.object
    )
    model_config = ModelConfig(
        mode='multiclass_classification',
        task_level='node',
        return_type='raw'
    )
    explainer.connect(explainer_config, model_config)
    fidelity_plus_scores = []
    fidelity_minus_scores = []

    # [新增] 确定子图的跳数，通常等于GNN的层数
    num_hops = len(config.models.param.gnn_latent_dim)
    
    logger.info("=" * 40)
    start_time = time.time()

    # [修改] 循环逻辑，每次处理一个节点的子图
    for node_idx in tqdm(test_indices, desc):
        subset, subgraph_edge_index, mapping, _ = k_hop_subgraph(
            node_idx, num_hops, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)

        subgraph_x = data.x[subset]
        
        new_node_idx = mapping.item()

        subgraph_x = subgraph_x.to(device)
        subgraph_edge_index = subgraph_edge_index.to(device)

        _, _, fidelity_plus, fidelity_minus = explainer.explain_node(
            model, new_node_idx, subgraph_x, subgraph_edge_index # 使用子图数据
        )
        
        fidelity_plus_scores.append(fidelity_plus)
        fidelity_minus_scores.append(fidelity_minus)

        logger.info(f"节点 {node_idx:<4d} | Fidelity+: {fidelity_plus:8.4f} | Fidelity-: {fidelity_minus:8.4f}")

        del subgraph_x, subgraph_edge_index, subset, mapping
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    end_time = time.time()
    
    avg_fidelity_plus = np.mean(fidelity_plus_scores)
    std_fidelity_plus = np.std(fidelity_plus_scores)
    avg_fidelity_minus = np.mean(fidelity_minus_scores)
    std_fidelity_minus = np.std(fidelity_minus_scores)



@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):
    config.models.param = config.models.param[config.datasets.dataset_name]
    if config.datasets.dataset_name in ['CS', 'Physics', 'Facebook']:
        dataset, train_mask, valid_mask, test_mask = get_dataset(config.datasets.dataset_root,
                                                                 config.datasets.dataset_name)
    else:
        dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
    if dataset.data.x is not None:
        dataset.data.x = dataset.data.x.float()
    if config.datasets.dataset_name in ['products']:
        dataset.data.y = torch.argmax(dataset.data.y, dim=1)
    dataset.data.y = dataset.data.y.squeeze().long()

    log_file = (
        f"Gnnexplainer_{config.datasets.dataset_name}_{len(config.models.param.gnn_latent_dim)}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.info(OmegaConf.to_yaml(config))
    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')
    logger.info(f'Using device: {device}')
    logger.info(f'Using data: {dataset.data}')

    config.models.gnn_savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    state_dict = torch.load(os.path.join(config.models.gnn_savedir,
                                         config.datasets.dataset_name,
                                         f'{config.models.gnn_name}_'
                                         f'{len(config.models.param.gnn_latent_dim)}l_best.pth'))['net']
    model = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models)
    print(model)
    model.load_state_dict(state_dict)
    if config.datasets.dataset_name in ['tree_grid', 'tree_cycle']:
        test_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
    elif config.datasets.dataset_name in ['CS', 'Physics', 'Facebook']:
        test_indices = torch.where(test_mask)[0].tolist()
    else:
        test_indices = torch.where(dataset.data.test_mask)[0].tolist()

    
    num_samples = 100
    import random
    if len(test_indices) > num_samples:
        test_indices = random.sample(test_indices, num_samples)
    else:
        test_indices = test_indices
    # test_indices=[2295, 2532, 1798, 2093, 1871, 2480, 2015, 2234, 1747, 2499, 2321, 2309, 2231, 2308, 2122, 2457, 2414, 2263, 2279, 2182, 2620, 1731, 2187, 2369, 2317, 2201, 1906, 1710, 1882, 2257, 2316, 2422, 2149, 2320, 1890, 2012, 1736, 2088, 2515, 2469, 2046, 1768, 1934, 2199, 1840, 2516, 1907, 2296, 1962, 2222, 2513, 1827, 2096, 2569, 2634, 2072, 1765, 1769, 2119, 2393, 2238, 2522, 2200, 2584, 2275, 1983, 2024, 2586, 1759, 1857, 2484, 2097, 2546, 2397, 2405, 2461, 1810, 2519, 2420, 1804, 2259, 1921, 2636, 2349, 2674, 2158, 1734, 2488, 1974, 2311, 2386, 2551, 2326, 1806, 1969, 2392, 2107, 2623, 2235, 2033]
    
    logger.info(f'test nodes : {test_indices}')
    pipeline(config, dataset, test_indices, model, device, logger)

if __name__ == '__main__':
    import sys
    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append(f"models.gnn_savedir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()