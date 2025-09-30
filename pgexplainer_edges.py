import copy
import os
from warnings import simplefilter
import torch
import hydra
from dig.xgraph.method import PGExplainer
from omegaconf import OmegaConf
from dig.xgraph.method.base_explainer import ExplainerBase
from dig.xgraph.utils.compatibility import compatible_state_dict
from torch import Tensor
from typing import List, Dict, Tuple
from torch_geometric.utils import add_remaining_self_loops, k_hop_subgraph
from dataset import get_dataset
from gnnNets import get_gnnNets
from utils import get_logger, check_dirs
import random
import numpy as np

def fix_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class PGExplainer_edges(ExplainerBase):
    def __init__(self, pgexplainer, model, molecule: bool):
        super().__init__(model=model,
                         explain_graph=pgexplainer.explain_graph,
                         molecule=molecule)
        self.explainer = pgexplainer

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                **kwargs)\
            -> Tuple[List, List, List[Dict]]:
        # set default subgraph with 10 edges

        pred_label = kwargs.get('pred_label')
        num_classes = 107
        self.model.eval()
        self.explainer.__clear_masks__()

        x = x.to(self.device)
        # edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
        edge_index = edge_index.to(self.device)

        if self.explain_graph:
            embed = self.model.get_emb(x, edge_index)
            _, edge_mask = self.explainer.explain(x,
                                                  edge_index,
                                                  embed=embed,
                                                  tmp=1.0,
                                                  training=False)
            # edge_masks
            edge_masks = [edge_mask for _ in range(num_classes)]
            # Calculate mask
            hard_edge_masks = [self.control_sparsity(edge_mask, sparsity=kwargs.get('sparsity')).sigmoid()
                               for _ in range(num_classes)]

            self.__clear_masks__()
            self.__set_masks__(x, edge_index)
            with torch.no_grad():
                if self.explain_graph:
                    related_preds = self.eval_related_pred(x, edge_index, hard_edge_masks)

            self.__clear_masks__()

        else:
            node_idx = kwargs.get('node_idx')
            sparsity = kwargs.get('sparsity')
            assert kwargs.get('node_idx') is not None, "please input the node_idx"
            select_edge_index = torch.arange(0, edge_index.shape[1])
            subgraph_x, subgraph_edge_index, _, subset, kwargs = \
                self.explainer.get_subgraph(node_idx, x, edge_index, select_edge_index=select_edge_index)
            select_edge_index = kwargs['select_edge_index']
            self.select_edge_mask = edge_index.new_empty(edge_index.size(1),
                                                         device=self.device,
                                                         dtype=torch.bool)
            self.select_edge_mask.fill_(False)
            self.select_edge_mask[select_edge_index] = True
            self.hard_edge_mask = edge_index.new_empty(subgraph_edge_index.size(1),
                                                       device=self.device,
                                                       dtype=torch.bool)
            self.hard_edge_mask.fill_(True)
            self.subset = subset
            self.new_node_idx = torch.where(subset == node_idx)[0]

            subgraph_embed = self.model.get_emb(subgraph_x, subgraph_edge_index)
            _, subgraph_edge_mask = self.explainer.explain(subgraph_x,
                                                           subgraph_edge_index,
                                                           embed=subgraph_embed,
                                                           tmp=1.0,
                                                           training=False,
                                                           node_idx=self.new_node_idx)

            # edge_masks
            edge_masks = [subgraph_edge_mask for _ in range(num_classes)]
            # Calculate mask
            hard_edge_masks = [
                self.control_sparsity(subgraph_edge_mask, sparsity=sparsity).sigmoid()
                for _ in range(num_classes)]

            self.__clear_masks__()
            self.__set_masks__(subgraph_x, subgraph_edge_index)
            with torch.no_grad():
                related_preds = self.eval_related_pred(
                    subgraph_x, subgraph_edge_index, hard_edge_masks, node_idx=self.new_node_idx)

            self.__clear_masks__()

        return edge_masks, hard_edge_masks, related_preds

import time

def pipeline(config, model, dataset, test_indices, device, logger):
    config.models.param.add_self_loop = False
    fix_random_seed(config.random_seed)

    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    if config.models.param.graph_classification:
        pass # Graph classification logic remains the same
    else:
        eval_model = copy.deepcopy(model)
        model.to(device)
        eval_model.to(device)
        total_fidelity_plus = 0.0
        total_fidelity_minus = 0.0
        
        if config.models.param.concate:
            input_dim = sum(config.models.param.gnn_latent_dim) * 3
        else:
            input_dim = config.models.param.gnn_latent_dim[-1] * 3
        pgexplainer = PGExplainer(model, in_channels=input_dim, device=device,
                                  explain_graph=config.models.param.graph_classification,
                                  epochs=config.explainers.param[config.datasets.dataset_name].ex_epochs,
                                  lr=config.explainers.param[config.datasets.dataset_name].ex_learning_rate,
                                  coff_size=config.explainers.param[config.datasets.dataset_name].coff_size,
                                  coff_ent=config.explainers.param[config.datasets.dataset_name].coff_ent,
                                  sample_bias=config.explainers.param[config.datasets.dataset_name].sample_bias,
                                  t0=config.explainers.param[config.datasets.dataset_name].t0,
                                  t1=config.explainers.param[config.datasets.dataset_name].t1)
        pgexplainer_saving_path = os.path.join(config.explainers.explainer_saving_dir,
                                               config.datasets.dataset_name,
                                               f'{config.explainers.explainer_saving_name}_{len(config.models.param.gnn_latent_dim)}.pth')
        
        
        logger.info("Training new PGExplainer...")
        pgexplainer.train_explanation_network(dataset)
        torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
        state_dict = pgexplainer.state_dict()

        state_dict = compatible_state_dict(state_dict)
        pgexplainer.load_state_dict(state_dict)
        pgexplainer_edges = PGExplainer_edges(pgexplainer=pgexplainer, model=eval_model, molecule=True)
        pgexplainer_edges.device = pgexplainer.device

        data = dataset.data
        # data.edge_index = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        model_directory = os.path.join(config.models.gnn_savedir,
                               config.datasets.dataset_name)

        logits_filename = (f'{config.models.gnn_name}_'
                        f'{len(config.models.param.gnn_latent_dim)}l_best_logits.pt')

        logits_checkpoint_path = os.path.join(model_directory, logits_filename)

        try:
            original_logits = torch.load(logits_checkpoint_path, map_location=device)
        except FileNotFoundError:
            exit() 
        except Exception as e:
            exit()

        original_logits = original_logits.to(device)
        original_probs = torch.softmax(original_logits, dim=-1)
        predictions = original_probs.argmax(dim=-1)
        

        for i, node_idx in enumerate(test_indices):
            logger.info(f"Explaining node {node_idx} ({i+1}/{len(test_indices)})...")
            
            num_hops = len(config.models.param.gnn_latent_dim)
            subset, local_edge_index, mapping, _ = k_hop_subgraph(
                node_idx, num_hops, data.edge_index, relabel_nodes=True, num_nodes=data.num_nodes)
            
            local_node_idx = (subset == node_idx).nonzero().item()
            subgraph_x = data.x[subset]
            local_edge_index, _ = add_remaining_self_loops(local_edge_index, num_nodes=subgraph_x.size(0))
            sparsity = config.explainers.sparsity
            with torch.no_grad():
                _, local_hard_edge_masks, _ = \
                    pgexplainer_edges(subgraph_x, local_edge_index,
                                      node_idx=local_node_idx,
                                      num_classes=dataset.num_classes,
                                      sparsity=sparsity,
                                      pred_label=predictions[node_idx].item())
            
            local_hard_mask = local_hard_edge_masks[0].bool() 

            original_pred_class = predictions[node_idx]
            original_prob_for_node = original_probs[node_idx, original_pred_class]
            
            explanation_edge_index = local_edge_index[:, local_hard_mask]
            with torch.no_grad():
                subgraph_logits = eval_model(subgraph_x, explanation_edge_index)
                subgraph_probs = torch.softmax(subgraph_logits, dim=-1)
                masked_prob_for_node = subgraph_probs[local_node_idx, original_pred_class]
            
            fidelity_plus = original_prob_for_node.item() - masked_prob_for_node.item()
            complement_edge_index = local_edge_index[:, ~local_hard_mask]
            with torch.no_grad():
                compl_logits = eval_model(subgraph_x, complement_edge_index)
                compl_probs = torch.softmax(compl_logits, dim=-1)
                compl_prob_for_node = compl_probs[local_node_idx, original_pred_class]
            
            fidelity_minus = original_prob_for_node.item() - compl_prob_for_node.item()

            logger.info(f'Fidelity+: {fidelity_plus:.4f}')
            logger.info(f'Fidelity-: {fidelity_minus:.4f}')
            
            total_fidelity_plus += fidelity_plus
            total_fidelity_minus += fidelity_minus

        
        end_time = time.time() 
        num_test_nodes = len(test_indices)
        if num_test_nodes > 0:
            avg_fidelity_plus = total_fidelity_plus / num_test_nodes
            avg_fidelity_minus = total_fidelity_minus / num_test_nodes
            total_time = end_time - start_time
            logger.info("=" * 30)
            logger.info(f"Final Average Fidelity+ over {num_test_nodes} nodes: {avg_fidelity_plus:.4f}")
            logger.info(f"Final Average Fidelity- over {num_test_nodes} nodes: {avg_fidelity_minus:.4f}")
            logger.info(f"Total time over {num_test_nodes} nodes: {total_time:.4f} seconds")
            logger.info(f"Average time per node: {total_time / num_test_nodes:.4f} seconds")
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
        dataset.data.val_mask = valid_mask   #
        dataset.data.test_mask = test_mask
    else:
        dataset = get_dataset(config.datasets.dataset_root, config.datasets.dataset_name)
    if dataset.data.x is not None:
        dataset.data.x = dataset.data.x.float()
    print(dataset[0])
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
    import random

    test_indices=[16609, 17940, 17817, 17603, 16635, 17283, 18004, 17599, 17241, 16916, 16503, 17178, 17692, 18284, 17340, 18299, 17435, 16613, 17685, 18141, 16825, 17298, 18100, 17869, 17250, 17661, 16719, 17533, 17610, 17646, 17729, 17393, 18134, 18156, 18222, 17011, 17691, 17448, 16920, 17094, 17918, 17720, 17829, 17710, 17406, 16594, 18323, 18115, 17181, 18244, 18043, 18113, 17871, 17046, 17385, 18086, 17765, 17423, 17135, 17405, 17001, 17780, 16670, 16815, 18236, 17163, 16658, 17028, 16821, 17998, 17771, 16732, 17513, 17988, 18212, 17807, 18324, 16589, 17897, 17952, 17407, 16746, 17329, 17644, 17821, 17300, 17659, 17682, 18285, 16531, 18199, 16671, 17457, 17239, 16850, 18198, 16792, 17561, 17828, 17961]

    logger.info(f'test nodes : {test_indices}')
    pipeline(config, model, dataset, test_indices, device, logger)

if __name__ == '__main__':
    import sys
    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append('explainers=pgexplainer')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explainer_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
