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
    dataset.data.y = dataset.data.y.squeeze().long()
    if config.datasets.dataset_name in ['products']:
        dataset.data.y = torch.argmax(dataset.data.y, dim=1)
    data = dataset[0]
    dataset.data.to(device)
    gnn = get_gnnNets(dataset.num_node_features, dataset.num_classes, config.models)

    state_dict = torch.load(os.path.join(config.models.gnn_savedir,
                                         config.datasets.dataset_name,
                                         f'{config.models.gnn_name}_'
                                         f'{len(config.models.param.gnn_latent_dim)}l_best.pth'))['net']

    gnn.load_state_dict(state_dict)
    gnn.eval()

    log_file = (
        f"graphmask{config.datasets.dataset_name}_{len(config.models.param.gnn_latent_dim)}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    gnn = gnn.to(device)
    data = data.to(device)

    explainer_name='graphmask'
    explainer = Explainer(
        model=gnn,
        algorithm=GraphMaskExplainer(len(config.models.param.gnn_latent_dim), epochs=10, lr=1e-3).to(device),
        explanation_type='model',
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node',
            return_type='raw',
        )
    )
    explainer.algorithm.fidelity_plus = False
    explanation_saving_dir = os.path.join('',
                                          config.datasets.dataset_name,
                                          config.models.gnn_name)
    check_dirs(explanation_saving_dir)
    if config.datasets.dataset_name in ['tree_grid', 'tree_cycle', 'tree_grid']:
        node_ids = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
    else:
        node_ids = torch.where(dataset.data.test_mask)[0].tolist()
    logger.info(f'test nodes : {node_ids}')
    logger.info(len(node_ids))
    for node_index in node_ids:
        data_with_loops=copy.deepcopy(data)
        data_with_loops.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        indices, edge_index, mapping, _ = k_hop_subgraph(node_index,
                                                            len(config.models.param.gnn_latent_dim),
                                                            data.edge_index,
                                                            relabel_nodes=True,
                                                            num_nodes=data.num_nodes)
        _, edge_index_true, _, _ = k_hop_subgraph(node_index,
                                                  len(config.models.param.gnn_latent_dim),
                                                  data.edge_index,
                                                  relabel_nodes=False,
                                                  num_nodes=data.num_nodes)
        out = gnn(x=data.x, edge_index=edge_index)
        pred = out.argmax(dim=-1)
        if edge_index.shape[1] == 1:
            continue
        torch.cuda.empty_cache()
        explanation = explainer(data.x[indices], edge_index,
                                target=pred[indices] if explainer_name == 'pgexplainer' else None, index=mapping[0].item())


        _, _, _, mask = k_hop_subgraph(node_index,
                                       len(config.models.param.gnn_latent_dim),
                                       data_with_loops.edge_index,
                                       relabel_nodes=True,
                                       num_nodes=data.num_nodes)

        torch.save(explanation, os.path.join(explanation_saving_dir,
                                                f'{len(config.models.param.gnn_latent_dim)}_example_{node_index}.pt'))


if __name__ == '__main__':
    import sys
    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append(f"models.gnn_savedir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()