import copy
import time
from collections import defaultdict, deque
import scipy.sparse as sp
import torch
import os
import hydra
import networkx as nx
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, add_remaining_self_loops, to_scipy_sparse_matrix, \
    to_networkx
from dataset import get_dataset
from warnings import simplefilter
from gnnNets import get_gnnNets_explain
from utils import get_logger
import numpy as np

MIN_VALUE = -2 ** 31
torch.cuda.empty_cache()


def convert_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_to_dict(v) for k, v in d.items()}
    elif isinstance(d, set):
        d = list(d)
    return d


def aug_normalized_adj(adj_matrix):
    row_sum = np.array(adj_matrix.sum(1))
    d_inv_sqrt = np.power(row_sum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
    d_matrix_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_matrix_inv_sqrt.dot(adj_matrix).dot(d_matrix_inv_sqrt).tocoo()


class Slicedmodel:
    def __init__(self, config, device, total_num_hop, num_hop, logger, dataset, state_dict):
        self.config = config
        self.device = device
        self.dataset = dataset
        self.total_num_hop = total_num_hop
        self.num_hop = num_hop
        self.logger = logger
        self.state_dict = state_dict
        self.submodel = self.createsub()
        self.model = get_gnnNets_explain(self.dataset.num_node_features, self.dataset.num_classes, self.config.models)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def createsub(self):
        self.logger.info(f'test layer: {self.num_hop}')
        layer_config = copy.deepcopy(self.config.models)
        if self.num_hop >= 1:
            layer_config.param.gnn_latent_dim = layer_config.param.gnn_latent_dim[:self.num_hop]
        else:
            self.logger.info(f'layer out of range')
        submodel = get_gnnNets_explain(self.dataset.num_node_features, self.dataset.num_classes, layer_config)
        new_state_dict = copy.deepcopy(self.state_dict)
        for key in list(new_state_dict.keys()):
            if 'convs' in key:
                conv_idx = int(key.split('.')[1])
                if conv_idx >= len(layer_config.param.gnn_latent_dim):
                    del new_state_dict[key]
        submodel.load_state_dict(new_state_dict)
        submodel.eval()
        submodel.to(self.device)
        return submodel

class Subfunction:
    def __init__(self, test_indices, dec, modelslice, logger, device):
        self.config = dec.config
        self.test_indices = test_indices
        self.dataset_name = dec.config.datasets.dataset_name
        self.dataset = dec.dataset
        self.K = dec.K
        self.theta = dec.theta
        self.h = dec.h
        self.model = modelslice.submodel
        self.logger = logger
        self.num_hop = modelslice.num_hop
        self.device = device
        self.global_emb, self.logit = self.update_global()
        self.influence_set = self.influence()
        self.diversity_set = self.diversity()

    def diversity(self):
        diversity_set = defaultdict(lambda: defaultdict(set))
        for i in self.test_indices:
            subset, _, _, _ = k_hop_subgraph(i, self.num_hop, self.dataset.data.edge_index)
            subset = subset.cpu().tolist()
            for j in subset:
                subset2, _, _, _ = k_hop_subgraph(j, self.num_hop, self.dataset.data.edge_index)
                subset2 = subset2.cpu().tolist()
                for k in subset2:
                    distance = torch.norm(self.global_emb[k].detach() - self.global_emb[j].detach(), p=2).item()
                    if distance <= self.theta:
                        diversity_set[i][j].add(k)
        diversity_set = convert_to_dict(diversity_set)
        return diversity_set

    def update_global(self):
        self.dataset.data.to(self.device)
        emb_list, logits = self.model(self.dataset.data)
        return emb_list[-1], logits

    def influence(self):
        influence_set = defaultdict(lambda: defaultdict(set))
        for i in self.test_indices:
            subset, edge_index, _, edge_mask = k_hop_subgraph(i, self.num_hop, self.dataset.data.edge_index)
            sub_data = copy.deepcopy(self.dataset.data)
            sub_data.edge_index = edge_index
            adj_matrix = add_remaining_self_loops(sub_data.edge_index, num_nodes=sub_data.num_nodes)[0]
            coo_adj_matrix = to_scipy_sparse_matrix(adj_matrix)
            aug_normalized_adj_matrix = aug_normalized_adj(coo_adj_matrix)
            influence_matrix = torch.FloatTensor(aug_normalized_adj_matrix.todense()).to(self.device)
            influence_matrix_power = influence_matrix
            for _ in range(self.num_hop - 1):
                influence_matrix_power = torch.mm(influence_matrix_power, influence_matrix)
            influence_matrix3 = influence_matrix_power.cpu()
            subset = subset.cpu().tolist()
            for j in subset:
                subset2, _, _, _ = k_hop_subgraph(j, self.num_hop, self.dataset.data.edge_index)
                subset2 = subset2.cpu().tolist()
                influence_set[i][j].add(j)
                for k in subset2:
                    if influence_matrix3[j][k].item() > self.h:
                        influence_set[i][j].add(k)
        influence_set = convert_to_dict(influence_set)
        return influence_set

class Declarative:
    def __init__(self, config, dataset, K, theta, h, gamma):
        self.K = K
        self.config = config
        self.dataset = dataset
        self.theta = theta
        self.h = h
        self.gamma = gamma


class GreedyAlgorithm:
    def __init__(self, dec, modelslice, node, logger, quality):
        self.dec = dec
        self.dataset_name = dec.config.datasets.dataset_name
        self.dataset = dec.dataset
        self.num_hop = modelslice.num_hop
        self.K = dec.K
        self.node = node
        self.device = modelslice.device
        self.model = modelslice.submodel
        self.ori_model = modelslice.model
        self.optimal = {'factual': False, 'counterfactual': False, 'both': False, 'score': MIN_VALUE, 'nodes': [],
                        'Fid+': 0, 'Fid-': 0}
        self.influence_set = quality.influence_set
        self.diversity_set = quality.diversity_set
        self.explanatory = [node]
        self.connector = []
        self.global_logits = None
        self.prediction = None
        self.subset = None
        self.ori_logits = None
        self.initial_status()
        self.logger = logger

    def initial_status(self):
        self.subset, edge_index, _, _ = k_hop_subgraph(self.node, self.num_hop, self.dataset.data.edge_index)
        self.subset = self.subset.cpu().tolist()
        _, self.ori_logits = self.ori_model(self.dataset.data)
        _, self.global_logits = self.model(self.dataset.data.x, edge_index)
        self.prediction = self.global_logits.argmax(-1)[self.node].item()
        emb_gs, logits_gs = self.model(self.dataset.data.x, self.dataset.data.edge_index[:, []])
        label_gs = logits_gs[self.node].argmax(-1).item()
        self.optimal['factual'] = (label_gs == self.prediction and self.prediction == self.ori_logits[self.node].argmax(-1).item())
        self.optimal['score'] = self.evaluate(self.explanatory)
        self.optimal['Fid+'] = 0
        self.optimal['Fid-'] = self.global_logits.softmax(dim=-1)[self.node][self.prediction].item() - \
                               logits_gs.softmax(dim=-1)[self.node][self.prediction].item()

    def evaluate(self, nodeset):
        all_influence = set()
        all_diversity = set()
        for v in nodeset:
            all_influence.update(self.influence_set[self.node][v])
            all_diversity.update(self.diversity_set[self.node][v])
        f1=len(all_influence)
        f2=len(all_diversity)
        return self.dec.gamma*f1+(1-self.dec.gamma)*f2

    def get_solution(self):
        while len(self.explanatory) < self.K and len(self.explanatory) < len(self.subset):
            self.add_new_node()
        self.verify_connectivity()
        factual, counterfactual, both, sub_score, counter_score = self.verify()
        if factual == True or counterfactual == True:
            self.update_optimal(factual, counterfactual, both, sub_score, counter_score)
        else:
            self.postprocessing()
        print(self.optimal)
        return self.optimal

    def update_optimal(self, factual, counterfactual, both, sub_score, counter_score):
        sub_nodes = list(set(self.explanatory) | set(self.connector))
        self.optimal['factual'] = factual
        self.optimal['counterfactual'] = counterfactual
        self.optimal['both'] = both
        self.optimal['score'] = self.evaluate(sub_nodes)
        self.optimal['nodes'] = sub_nodes
        self.optimal['Fid+'] = sub_score
        self.optimal['Fid-'] = counter_score

    def node_induced_edges(self, subset):
        subset_indices = torch.tensor(subset, dtype=torch.long).to(self.device)
        mask_0 = self.dataset.data.edge_index[0].unsqueeze(1) == subset_indices
        mask_1 = self.dataset.data.edge_index[1].unsqueeze(1) == subset_indices
        mask = mask_0.any(dim=1) & mask_1.any(dim=1)
        return self.dataset.data.edge_index[:, mask]

    def verify(self):
        sub_nodes = list(set(self.explanatory) | set(self.connector))
        emb_gs, logits_gs = self.model(self.dataset.data.x, self.node_induced_edges(sub_nodes))
        counter_nodes = [node for node in self.subset if node not in sub_nodes]
        counter_nodes.append(self.node)
        emb_gsr, logits_gsr = self.model(self.dataset.data.x, self.node_induced_edges(counter_nodes))
        factual = (logits_gs[self.node].argmax(-1).item() == self.prediction and self.prediction == self.ori_logits[self.node].argmax(-1).item())
        counterfactual = (logits_gsr[self.node].argmax(-1).item() != self.ori_logits[self.node].argmax(-1).item())
        both = (factual and counterfactual)
        counter_score = self.global_logits.softmax(dim=-1)[self.node][self.prediction].item() - \
                        logits_gs.softmax(dim=-1)[self.node][self.prediction].item()
        sub_score = self.global_logits.softmax(dim=-1)[self.node][self.prediction].item() - \
                    logits_gsr.softmax(dim=-1)[self.node][self.prediction].item()
        return factual, counterfactual, both, sub_score, counter_score

    def add_new_node(self):
        left_set = [node for node in self.subset if node not in self.explanatory]
        evaluated_scores = list(map(lambda node: (node, self.evaluate(self.explanatory + [node])), left_set))
        sorted_node = sorted(evaluated_scores, key=lambda x: x, reverse=True)
        sorted_node = [node for node, score in sorted_node]
        if self.K - len(self.explanatory) == 1 or len(sorted_node) == 1:
            self.explanatory.append(sorted_node[0])
        else:
            self.explanatory.append(sorted_node[0])
            self.explanatory.append(sorted_node[1])

    def verify_connectivity(self):
        trans_data = Data(x=self.dataset.data.x, edge_index=self.dataset.data.edge_index)
        G = to_networkx(trans_data, to_undirected=True)
        subgraph = G.subgraph(self.explanatory)
        if nx.is_connected(subgraph) == False:
            self.bfs_connect_disconnected_nodes(G, subgraph)
        print(nx.is_connected(G.subgraph(list(set(self.explanatory) | set(self.connector)))))

    def bfs_connect_disconnected_nodes(self, graph, subgraph):
        disconnected_nodes = [node for node in self.explanatory if not nx.has_path(subgraph, self.node, node)]
        shortest_paths = {}
        for node in disconnected_nodes:
            path = nx.shortest_path(graph, source=node, target=self.node)
            shortest_paths[node] = path
            for node in path:
                if node not in self.explanatory and node not in self.connector:
                    self.connector.append(node)
        return shortest_paths

    def postprocessing(self):
        print("postprocessing")
        print(self.explanatory, self.connector, self.subset)
        sub_nodes = list(set(self.explanatory) | set(self.connector))
        sub_evaluated_scores = list(
            map(lambda node: (node, self.evaluate(list(filter(lambda x: x != node, sub_nodes)))), sub_nodes))
        sub_sorted_node = sorted(sub_evaluated_scores, key=lambda x: x, reverse=False)
        sub_sorted_node = [node for node, score in sub_sorted_node]
        sub_sorted_node.remove(self.node)
        if len(sub_sorted_node) == 0:
            return
        node_old = sub_sorted_node[0]
        removed_set = list(filter(lambda x: x != node_old, sub_nodes))

        left_set = [node for node in self.subset if node not in sub_nodes]
        left_evaluated_scores = list(
            map(lambda node: (node, self.evaluate(removed_set + [node])), left_set))
        left_sorted_node = sorted(left_evaluated_scores, key=lambda x: x, reverse=True)
        left_sorted_node = [node for node, score in left_sorted_node]
        for each_node in left_sorted_node:
            if each_node != self.node:
                trans_data = Data(x=self.dataset.data.x, edge_index=self.dataset.data.edge_index)
                G = to_networkx(trans_data, to_undirected=True)
                subgraph = G.subgraph(removed_set + [each_node])
                if nx.is_connected(subgraph) == True:
                    # copy the ori result
                    ori_explanatory_set = copy.deepcopy(self.explanatory)
                    ori_connector = copy.deepcopy(self.connector)
                    print(node_old, self.explanatory, self.connector)
                    if node_old in self.explanatory:
                        self.explanatory.remove(node_old)
                        self.explanatory.append(each_node)
                    else:
                        self.connector.remove(node_old)
                        self.connector.append(each_node)
                    factual, counterfactual, both, sub_score, counter_score = self.verify()
                    if factual == True or counterfactual == True:
                        self.update_optimal(factual, counterfactual, both, sub_score, counter_score)
                        return
                    self.explanatory = ori_explanatory_set
                    self.connector = ori_connector
        factual, counterfactual, both, sub_score, counter_score = self.verify()
        self.update_optimal(factual, counterfactual, both, sub_score, counter_score)


def layerwise_run(cut_layer, test_indices, device, logger, dec, state_dict):
    modelslice = Slicedmodel(dec.config, device, len(dec.config.models.param.gnn_latent_dim),
                             len(dec.config.models.param.gnn_latent_dim) - cut_layer, logger,
                             dec.dataset, state_dict)
    quality = Subfunction(test_indices, dec, modelslice, logger, device)

    fidelity = []
    fidelity_inv = []
    for test_node in test_indices:
        logger.info(f'test node: {test_node}')
        dec.dataset.data.to(device)
        algorithm = GreedyAlgorithm(dec, modelslice, test_node, logger, quality)
        optimal = algorithm.get_solution()
        if optimal != None:
            fidelity.append(optimal['Fid+'])
            fidelity_inv.append(optimal['Fid-'])
    logger.info(f'number: {len(fidelity)}')
    logger.info(f'Fidelity: {sum(fidelity) / len(fidelity):.4f}\n'
                f'Fidelity_inv: {sum(fidelity_inv) / len(fidelity_inv): .4f}\n')
    

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

    if config.datasets.dataset_name in ['tree_grid','tree_cycle','tree_grid']:
        test_indices = torch.where(dataset[0].test_mask * dataset[0].y != 0)[0].tolist()
    else:
        test_indices = torch.where(dataset.data.test_mask)[0].tolist()
    logger.info(f'test nodes : {test_indices}')
    start_time = time.time()
    for h_mini in h:
        for th in theta:
            for size_run in K:
                logger.info(f'h:{h_mini}')
                logger.info(f'theta: {th}')
                logger.info(f'size: {size_run}')
                for i in range(layer_nums - 1, -1, -1):
                    dec = Declarative(config, dataset, size_run, th, h_mini, gamma)
                    layerwise_run(i, test_indices, device, logger, dec, state_dict)
    end_time = time.time()
    logger.info(f'Execution time: {end_time - start_time:.6f}')


if __name__ == '__main__':
    import sys
    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append(f"models.gnn_savedir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()
