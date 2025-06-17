import os
import time
from warnings import simplefilter
import torch
import hydra
from omegaconf import OmegaConf
from torch_geometric.utils import add_remaining_self_loops
from dig.xgraph.method.subgraphx import PlotUtils, SubgraphX
from dataset import get_dataset
from gnnNets import get_gnnNets
from utils import check_dirs, get_logger
IS_FRESH = False


def pipeline(config, dataset, test_indices, model, device, logger):
    config.explainers.param = config.explainers.param[config.datasets.dataset_name]
    model.eval()
    model.to(device)
    config.models.param.add_self_loop = False
    if not os.path.isdir(config.record_filename):
        os.makedirs(config.record_filename)

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    if config.models.param.graph_classification:
        pass


    explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                          config.datasets.dataset_name,
                                          config.models.gnn_name,
                                          config.explainers.param.reward_method)
    check_dirs(explanation_saving_dir)
    plot_utils = PlotUtils(dataset_name=config.datasets.dataset_name, is_show=False)

    if config.models.param.graph_classification:
        pass

    else:
        data = dataset.data
        data = data.to(device)
        data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]

        predictions = model(data).argmax(-1)
        print(config.explainers.param)

        subgraphx = SubgraphX(model,
                              dataset.num_classes,
                              device,
                              explain_graph=config.models.param.graph_classification,
                              verbose=config.explainers.param.verbose,
                              c_puct=config.explainers.param.c_puct,
                              rollout=config.explainers.param.rollout,
                              high2low=config.explainers.param.high2low,
                              min_atoms=config.explainers.param.min_atoms,
                              expand_atoms=config.explainers.param.expand_atoms,
                              reward_method=config.explainers.param.reward_method,
                              subgraph_building_method=config.explainers.param.subgraph_building_method,
                              save_dir=explanation_saving_dir)

        for node_idx in test_indices:
            logger.info(f'test node: {node_idx}')
            start_time = time.time()

            data.to(device)
            saved_MCTSInfo_list = None

            explain_result, related_preds = \
                subgraphx.explain(data.x, data.edge_index,
                                  node_idx=node_idx,
                                  max_nodes=config.explainers.max_ex_size,
                                  label=predictions[node_idx].item(),
                                  saved_MCTSInfo_list=saved_MCTSInfo_list)

            torch.save(explain_result, os.path.join(explanation_saving_dir, f'{len(config.models.param.gnn_latent_dim)}_example_{node_idx}.pt'))

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
        f"subgraphx_{config.datasets.dataset_name}_{len(config.models.param.gnn_latent_dim)}.log"
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
    sys.argv.append('explainers=subgraphx')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
