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
        num_classes = kwargs.get('num_classes')
        self.model.eval()
        self.explainer.__clear_masks__()

        x = x.to(self.device)
        edge_index, _ = add_remaining_self_loops(edge_index, num_nodes=x.size(0))
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



def pipeline(config, model, dataset, test_indices, device, logger):
    config.models.param.add_self_loop = False
    fix_random_seed(config.random_seed)


    if torch.cuda.is_available():
        device = torch.device('cuda', index=config.device_id)
    else:
        device = torch.device('cpu')

    if config.models.param.graph_classification:
        pass
    else:
        train_indices = test_indices
    eval_model = copy.deepcopy(model)

    model.to(device)
    eval_model.to(device)

    if config.models.param.graph_classification:
        pass
    else:
        if config.models.param.concate:
            input_dim = sum(config.models.param.gnn_latent_dim) * 3
        else:
            input_dim = config.models.param.gnn_latent_dim[-1] * 3

    pgexplainer = PGExplainer(model,
                              in_channels=input_dim,
                              device=device,
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

    pgexplainer.train_explanation_network(dataset,test_indices)
    torch.save(pgexplainer.state_dict(), pgexplainer_saving_path)
    state_dict = torch.load(pgexplainer_saving_path)
    state_dict = compatible_state_dict(state_dict)
    pgexplainer.load_state_dict(state_dict)

    index = 0
    pgexplainer_edges = PGExplainer_edges(pgexplainer=pgexplainer,
                                          model=eval_model,
                                          molecule=True)
    pgexplainer_edges.device = pgexplainer.device

    if config.models.param.graph_classification:
        pass

    else:
        data = dataset.data
        data.edge_index = add_remaining_self_loops(data.edge_index)[0]
        data.to(device)

        predictions = model(data).softmax(dim=-1).argmax(dim=-1)
        for node_idx in test_indices:
            index += 1
            subset, edge_index, _, edge_mask = k_hop_subgraph(node_idx,
                                                              len(config.models.param.gnn_latent_dim),
                                                              dataset.data.edge_index)
            sparsity = config.explainers.sparsity
            logger.info(f'sparsity: {sparsity}')
            with torch.no_grad():
                edge_masks, hard_edge_masks, related_preds = \
                    pgexplainer_edges(data.x, data.edge_index,
                                      node_idx=node_idx,
                                      num_classes=dataset.num_classes,
                                      sparsity=sparsity,
                                      pred_label=predictions[node_idx].item())
                edge_masks = [mask.detach() for mask in edge_masks]

                explanation_saving_dir = os.path.join(config.explainers.explanation_result_dir,
                                                      config.datasets.dataset_name,
                                                      config.models.gnn_name)
                check_dirs(explanation_saving_dir)
                torch.save(edge_masks, os.path.join(explanation_saving_dir,f'{len(config.models.param.gnn_latent_dim)}_example_{node_idx}.pt'))


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
        f"pgexplainer_{config.datasets.dataset_name}_{len(config.models.param.gnn_latent_dim)}.log"
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
    pipeline(config, model, dataset, test_indices, device, logger)

if __name__ == '__main__':
    import sys
    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append('explainers=pgexplainer')
    sys.argv.append(f"datasets.dataset_root={os.path.join(os.path.dirname(__file__), 'datasets')}")
    sys.argv.append(f"models.gnn_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    sys.argv.append(f"explainers.explainer_saving_dir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
