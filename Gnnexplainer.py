import copy
import os
from warnings import simplefilter
from dig.xgraph.method import GNNExplainer
from dataset import get_dataset
from utils import check_dirs, get_logger
import torch
torch.cuda.empty_cache()
import hydra
from omegaconf import OmegaConf
import time
from gnnNets import get_gnnNets
from torch_geometric.utils import add_remaining_self_loops, k_hop_subgraph
IS_FRESH = False

def pipeline(config, dataset, test_indices, model, device, logger):
    model.eval()
    model.to(device)
    test_indices = torch.tensor(test_indices)
    config.models.param.add_self_loop = False

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    gnn_explainer = GNNExplainer(model,
                                 epochs=config.explainers.param[config.datasets.dataset_name].epochs,
                                 lr=config.explainers.param[config.datasets.dataset_name].lr,
                                 coff_size=config.explainers.param[config.datasets.dataset_name].coff_size,
                                 coff_ent=config.explainers.param[config.datasets.dataset_name].coff_ent,
                                 explain_graph=config.models.param.graph_classification)
    gnn_explainer.device = device

    if config.models.param.graph_classification:
        pass

    else:
        data = copy.deepcopy(dataset.data)
        data.edge_index = add_remaining_self_loops(data.edge_index)[0]
        data.to(device)
        prediction = model(data).argmax(-1)
        for node_idx in test_indices:
            print(data.y[node_idx])
            subset, edge_index, _, edge_mask = k_hop_subgraph(node_idx.item(), len(config.models.param.gnn_latent_dim),
                                                              dataset.data.edge_index)

            sparsity=config.explainers.sparsity

            logger.info(f'test node: {node_idx}')
            logger.info(f'sparsity: {sparsity}')
            start_time = time.time()

            edge_masks, hard_edge_masks, related_preds = \
                gnn_explainer(data.x, data.edge_index,
                              node_idx=node_idx,
                              sparsity=sparsity,
                              num_classes=dataset.num_classes)

            edge_masks = [edge_mask.to('cpu') for edge_mask in edge_masks]
            explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                                  config.datasets.dataset_name,
                                                  config.models.gnn_name)
            check_dirs(explanation_saving_dir)
            torch.save(edge_masks, os.path.join(explanation_saving_dir,
                                                f'{len(config.models.param.gnn_latent_dim)}_example_{node_idx}.pt'))

            end_time = time.time()
            logger.info(f'Execution time for node_run: {end_time - start_time:.6f}')



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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
    if config.datasets.dataset_name in ['tree_grid', 'tree_cycle', 'tree_grid']:
        test_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
    else:
        test_indices = torch.where(dataset.data.test_mask)[0].tolist()
    logger.info(f'test nodes : {test_indices}')
    pipeline(config, dataset, test_indices, model, device, logger)

if __name__ == '__main__':
    import sys
    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append('explainers=gnn_explainer')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()

