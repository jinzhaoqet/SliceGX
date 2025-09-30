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

    explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                          config.datasets.dataset_name,
                                          config.models.gnn_name,
                                          config.explainers.param.reward_method)
    check_dirs(explanation_saving_dir)

    if config.models.param.graph_classification:
        pass
    else:
        data = dataset.data
        data = data.to(device)
        total_fidelity_plus = 0.0
        total_fidelity_minus = 0.0
        total_time = 0.0
        with torch.no_grad():
            original_logits = model(data)
        original_probs = torch.softmax(original_logits, dim=-1)
        predictions = original_logits.argmax(-1)
        
        print("SubgraphX Explainer Parameters:")
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
            logger.info(f'Explaining node: {node_idx}')
            start_time = time.time()

            explain_result, related_preds = \
                subgraphx.explain(data.x, data.edge_index,
                                  node_idx=node_idx,
                                  max_nodes=config.explainers.max_ex_size,
                                  label=predictions[node_idx].item())
            
            explanation_nodes = explain_result[0]['coalition']

            original_pred_class = predictions[node_idx]
            original_prob_for_node = original_probs[node_idx, original_pred_class]

            explanation_nodes_tensor = torch.tensor(explanation_nodes, device=device)

            src, dst = data.edge_index

            nodes_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
            nodes_mask[explanation_nodes_tensor] = True

            edge_mask = nodes_mask[src] & nodes_mask[dst]
            
            subgraph_edge_index = data.edge_index[:, edge_mask]
            with torch.no_grad():
                subgraph_data = data.clone()
                subgraph_data.edge_index = subgraph_edge_index
                subgraph_logits = model(subgraph_data)
            subgraph_probs = torch.softmax(subgraph_logits, dim=-1)
            masked_prob_for_node = subgraph_probs[node_idx, original_pred_class]
            fidelity_plus = original_prob_for_node.item() - masked_prob_for_node.item()

            complement_edge_index = data.edge_index[:, ~edge_mask]
            with torch.no_grad():
                compl_data = data.clone()
                compl_data.edge_index = complement_edge_index
                compl_logits = model(compl_data)
            compl_probs = torch.softmax(compl_logits, dim=-1)
            compl_prob_for_node = compl_probs[node_idx, original_pred_class]
            fidelity_minus = original_prob_for_node.item() - compl_prob_for_node.item()

            end_time = time.time()
            total_fidelity_plus += fidelity_plus
            total_fidelity_minus += fidelity_minus
            total_time += end_time - start_time
            logger.info(f'Execution time: {end_time - start_time:.4f}s')
            logger.info(f'Fidelity+: {fidelity_plus:.4f}')
            logger.info(f'Fidelity-: {fidelity_minus:.4f}')

            save_data = {
                'explanation_nodes': explanation_nodes,
                'related_predictions': related_preds,
                'fidelity_plus': fidelity_plus,
                'fidelity_minus': fidelity_minus
            }
            save_path = os.path.join(explanation_saving_dir, f'{len(config.models.param.gnn_latent_dim)}_example_{node_idx}.pt')
            torch.save(save_data, save_path)
            logger.info(f"Results for node {node_idx} saved to {save_path}")
        num_test_nodes = len(test_indices)
        if num_test_nodes > 0:
            avg_fidelity_plus = total_fidelity_plus / num_test_nodes
            avg_fidelity_minus = total_fidelity_minus / num_test_nodes

            logger.info("=" * 30)
            logger.info(f"Final Average Fidelity+ over {num_test_nodes} nodes: {avg_fidelity_plus:.4f}")
            logger.info(f"Final Average Fidelity- over {num_test_nodes} nodes: {avg_fidelity_minus:.4f}")
            logger.info(f"total_time over {num_test_nodes} nodes: {total_time:.4f}")
            logger.info("=" * 30)
        else:
            logger.info("No test nodes were processed, cannot compute average fidelity.")

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):
    config.models.param = config.models.param[config.datasets.dataset_name]
    if config.datasets.dataset_name in ['CS', 'Physics', 'Facebook']:
        dataset, train_mask, valid_mask, test_mask = get_dataset(config.datasets.dataset_root,
                                                                 config.datasets.dataset_name)
        dataset.data.train_mask = train_mask
        dataset.data.valid_mask = valid_mask
        dataset.data.test_mask = test_mask
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
    if config.datasets.dataset_name in ['tree_grid', 'tree_cycle', 'tree_grid']:
        test_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
    else:
        test_indices = torch.where(dataset.data.test_mask)[0].tolist()

    test_indices=[2295, 2532, 1798, 2093, 1871, 2480, 2015, 2234, 1747, 2499, 2321, 2309, 2231, 2308, 2122, 2457, 2414, 2263, 2279, 2182, 2620, 1731, 2187, 2369, 2317, 2201, 1906, 1710, 1882, 2257, 2316, 2422, 2149, 2320, 1890, 2012, 1736, 2088, 2515, 2469, 2046, 1768, 1934, 2199, 1840, 2516, 1907, 2296, 1962, 2222, 2513, 1827, 2096, 2569, 2634, 2072, 1765, 1769, 2119, 2393, 2238, 2522, 2200, 2584, 2275, 1983, 2024, 2586, 1759, 1857, 2484, 2097, 2546, 2397, 2405, 2461, 1810, 2519, 2420, 1804, 2259, 1921, 2636, 2349, 2674, 2158, 1734, 2488, 1974, 2311, 2386, 2551, 2326, 1806, 1969, 2392, 2107, 2623, 2235, 2033]
    logger.info(f'test nodes : {test_indices}')
    pipeline(config, dataset, test_indices, model, device, logger)

if __name__ == '__main__':
    import sys
    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append('explainers=subgraphx')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
