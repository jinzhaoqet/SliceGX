import time
import multiprocessing
import torch
import os
import hydra
import networkx as nx
from omegaconf import OmegaConf
from SliceGX import Declarative, layerwise_run, Slicedmodel, Subfunction, GreedyAlgorithm
from dataset import get_dataset
from warnings import simplefilter
from utils import get_logger, check_dirs
torch.cuda.empty_cache()

def layerwise_parallel(cut_layer, test_indices, device, logger, dec, state_dict):
    explanation_saving_dir = os.path.join('SliceGX',
                                          dec.config.datasets.dataset_name,
                                          dec.config.models.gnn_name,
                                          str(len(dec.config.models.param.gnn_latent_dim) - cut_layer),
                                          str(dec.K), str(dec.h), str(dec.theta))
    check_dirs(explanation_saving_dir)
    modelslice = Slicedmodel(config=dec.config, device=device,
                             total_num_hop=len(dec.config.models.param.gnn_latent_dim),
                             num_hop=len(dec.config.models.param.gnn_latent_dim) - cut_layer, logger=logger,
                             dataset=dec.dataset, state_dict=state_dict, explain=True)

    quality_saving_file = os.path.join('SliceGX',
                                       dec.config.datasets.dataset_name,
                                       dec.config.models.gnn_name,
                                       str(len(dec.config.models.param.gnn_latent_dim) - cut_layer),
                                       str(dec.config.datasets.K[0]), str(dec.h), str(dec.theta))
    print(quality_saving_file)
    if os.path.exists(os.path.join(quality_saving_file, f'quality.pt')):
        quality = torch.load(os.path.join(quality_saving_file, f'quality.pt'))
    else:
        quality = Subfunction(test_indices, dec, modelslice, logger, device)
        torch.save(quality, os.path.join(quality_saving_file, f'quality.pt'))
    fidelity = []
    fidelity_inv = []
    for test_node in test_indices:
        if os.path.exists(os.path.join(explanation_saving_dir, f'{test_node}.pt')):
            continue
        logger.info(f'test node: {test_node}')
        dec.dataset.data.to(device)
        algorithm = GreedyAlgorithm(dec, modelslice, test_node, logger, quality)
        optimal = algorithm.get_solution()
        if optimal != None:
            fidelity.append(optimal['Fid+'])
            fidelity_inv.append(optimal['Fid-'])
        torch.save(optimal['nodes'], os.path.join(explanation_saving_dir, f'{test_node}.pt'))

def split_list(data, n_parts):
    avg = len(data) // n_parts
    remainder = len(data) % n_parts
    result = []
    start = 0
    for i in range(n_parts):
        end = start + avg + (1 if i < remainder else 0)
        result.append(data[start:end])
        start = end
    return result

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):
    K = config.datasets.K
    h = config.datasets.h
    theta = config.datasets.theta
    gamma = config.datasets.gamma
    config.models.param = config.models.param[config.datasets.dataset_name]
    print(config.models)
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
    print(dataset.data)

    log_file = (
        f"{config.datasets.dataset_name}_{len(config.models.param.gnn_latent_dim)}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.info(OmegaConf.to_yaml(config))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    logger.info(f'Using data: {dataset.data}')

    layer_nums = len(config.models.param.gnn_latent_dim)
    state_dict = torch.load(os.path.join(config.models.gnn_savedir,
                                         config.datasets.dataset_name,
                                         f'{config.models.gnn_name}_'
                                         f'{len(config.models.param.gnn_latent_dim)}l_best.pth'))['net']

    test_indices = config.datasets.test_nodes
    logger.info(f'test nodes : {test_indices}')

    n_cpu = multiprocessing.cpu_count()
    logger.info(f'number of cpu cores: {n_cpu}')
    partition_cpu = 4
    partition_num = len(test_indices)*layer_nums // partition_cpu
    pool = multiprocessing.Pool(processes=partition_cpu)
    split_data = split_list(test_indices, partition_num)

    start_time = time.time()
    for h_mini in h:
        for th in theta:
            for size_run in K:
                logger.info(f'h:{h_mini}')
                logger.info(f'theta: {th}')
                logger.info(f'size: {size_run}')
                dec = Declarative(config, dataset, size_run, th, h_mini, gamma)
                for i in range(partition_num):
                    pool.apply_async(func=layerwise_parallel,
                                     args=(layer_nums, split_data[i], device, logger, dec, state_dict))
    pool.close()
    pool.join()
    end_time = time.time()
    logger.info(f'Execution time: {end_time - start_time:.6f}')


if __name__ == '__main__':
    import sys

    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append(f"models.gnn_savedir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
