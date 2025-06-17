import copy
import time
import logging
from collections import defaultdict, deque
import scipy.sparse as sp
import torch
import os
import hydra
import networkx as nx
from omegaconf import OmegaConf
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch_geometric.utils import k_hop_subgraph, add_remaining_self_loops, to_scipy_sparse_matrix,  \
    to_networkx
import torch.nn.functional as F
from dataset import get_dataset
from warnings import simplefilter
from gnnNets import get_gnnNets_explain
from utils import get_logger, check_dirs
import numpy as np
MIN_VALUE = -2 ** 31
MAX_VALUE = 2 ** 31 - 1
torch.cuda.empty_cache()
def convert_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: convert_to_dict(v) for k, v in d.items()}
    elif isinstance(d, set):
        d = list(d)
    return d

def aug_normalized_adj(adj_matrix):
        """
        Args:
            adj_matrix: input adj_matrix
        Returns:
            a normalized_adj which follows influence spread idea
        """
        row_sum = np.array(adj_matrix.sum(1))
        d_inv_sqrt = np.power(row_sum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0
        d_matrix_inv_sqrt = sp.diags(d_inv_sqrt)
        return d_matrix_inv_sqrt.dot(adj_matrix).dot(d_matrix_inv_sqrt).tocoo()
class Slicedmodel:
    def __init__(self, config, device,num_hop,logger,dataset,state_dict):
        self.config=config
        self.device=device
        self.dataset=dataset
        self.num_hop=num_hop
        self.logger=logger
        self.state_dict=state_dict
        self.submodel=self.createsub()

    def createsub(self):
        self.logger.info(f'test layer: {self.num_hop}')
        layer_config = copy.deepcopy(self.config.models)
        if self.num_hop >= 1:
            layer_config.param.gnn_latent_dim = layer_config.param.gnn_latent_dim[:self.num_hop]
        else:
            self.logger.info(f'layer out of range')
        if self.config.datasets.dataset_name in ['products']:
            submodel = get_gnnNets_explain(self.dataset.num_node_features, 107, layer_config)
        else:
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
        self.test_indices=test_indices
        self.dataset_name = dec.config.datasets.dataset_name
        self.dataset = dec.dataset
        self.K = dec.K
        self.theta=dec.theta
        self.h=dec.h
        self.model = modelslice.submodel
        self.logger=logger
        self.num_hop=modelslice.num_hop
        self.device = device
        self.global_emb, self.logit = self.update_global()
        self.influence_set=self.influence()
        self.diversity_set=self.diversity()

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
            sub_data=copy.deepcopy(self.dataset.data)
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
    def __init__(self,config,dataset,K,theta,h,gamma):
        self.K=K
        self.config=config
        self.dataset=dataset
        self.theta=theta
        self.h=h
        self.gamma=gamma

class Elements:
    def __init__(self):
        self.nodes = []
        self.B = []
        self.Use = []
        self.node_index_map = {}

    def add(self, node, test_node):
        if node not in self.nodes:
            self.nodes.append(node)
            self.B.append([test_node])
            self.Use.append(False)
            self.node_index_map[node] = len(self.nodes) - 1
        else:
            index = self.node_index_map[node]
            self.B[index].append(test_node)

class GreedyAlgorithm:
    def __init__(self, dec, modelslice, test_indices, logger, quality):
        self.dec=dec
        self.dataset_name = dec.config.datasets.dataset_name
        self.dataset = dec.dataset
        self.num_hop = modelslice.num_hop
        self.K = dec.K
        self.test_indices = test_indices
        self.node_index_map = {}
        self.optimal = [{'ID':None, 'Finish': False, 'Fill':False, 'subset':[], 'current_influence': set(), 'current_diversity': set(), 'factual': False,'counterfactual': False, 'both': False, 'prediction': None, 'score': MIN_VALUE, 'nodes': [], 'Fid+':0, 'Fid-': 0} for _ in range(len(test_indices))]
        self.device = modelslice.device
        self.model = modelslice.submodel
        self.candidate = Elements()
        self.influence_set = quality.influence_set
        self.diversity_set = quality.diversity_set
        self.global_logits = None
        self.initial_nodes()
        self.logger = logger

    def initial_nodes(self):
        _, self.global_logits = self.model(self.dataset.data.x, self.dataset.data.edge_index)
        emb_gs, logits_gs = self.model(self.dataset.data.x, self.dataset.data.edge_index[:, []])
        for i,vt in enumerate(self.test_indices):
            self.node_index_map[vt] = i
            subset, edge_index, _, _ = k_hop_subgraph(vt, self.num_hop, self.dataset.data.edge_index)
            subset = subset.cpu().tolist()
            self.optimal[i]['subset'] = subset
            self.optimal[i]['ID'] = vt
            self.optimal[i]['nodes'].append(vt)
            self.optimal[i]['current_influence'] = self.optimal[i]['current_influence'].union(self.influence_set[vt][vt])
            self.optimal[i]['current_diversity'] = self.optimal[i]['current_diversity'].union(self.diversity_set[vt][vt])
            prediction = self.global_logits.argmax(-1)[vt].item()
            factual = (logits_gs[vt].argmax(-1).item() == prediction)
            self.optimal[i]['prediction'] = prediction
            self.optimal[i]['factual'] = factual
            self.optimal[i]['score'] = self.evaluate(i,[-1], 'None')
            self.optimal[i]['Fid+'] = 0
            self.optimal[i]['Fid-'] = self.global_logits.softmax(dim=-1)[vt][prediction].item() - logits_gs.softmax(dim=-1)[vt][prediction].item()
            for node in subset:
                self.candidate.add(node,vt)

    def evaluate(self, index, nodeset, operator):
        if operator == 'None':
            f1 = len(self.optimal[index]['current_influence'])
            f2 = len(self.optimal[index]['current_diversity'])
        elif operator == '+':
            all_influence = self.optimal[index]['current_influence']
            all_diversity = self.optimal[index]['current_diversity']
            for v in nodeset:
                all_influence.update(self.influence_set[self.test_indices[index]][v])
                all_diversity.update(self.diversity_set[self.test_indices[index]][v])
            f1 = len(all_influence)
            f2 = len(all_diversity)
        elif operator == '-':
            all_influence = set()
            all_diversity = set()
            for v in nodeset:
                all_influence.update(self.influence_set[self.test_indices[index]][v])
                all_diversity.update(self.diversity_set[self.test_indices[index]][v])
            f1 = len(all_influence)
            f2 = len(all_diversity)
        return self.dec.gamma * f1 + (1 - self.dec.gamma) * f2

    def get_solution(self):
        filtered = [item for item in self.optimal if item['Fill'] == False]
        while(len(filtered)>=1):
            self.add_new_node()
            filtered = [item for item in self.optimal if item['Fill'] == False]
        self.connect_and_verify()
        self.postprocessing()
        print(self.optimal)
        return self.optimal

    def postprocessing(self):
        print("postprocessing")
        for i,vt in enumerate(self.test_indices):
            if self.optimal[i]['Finish'] == True:
                continue
            explanatory = self.optimal[i]['nodes']
            connector = []
            trans_data = Data(x=self.dataset.data.x, edge_index=self.dataset.data.edge_index)
            G = to_networkx(trans_data, to_undirected=True)
            subgraph = G.subgraph(explanatory)
            if nx.is_connected(subgraph) == False:
                connector = self.bfs_connect_disconnected_nodes(G, subgraph, explanatory, vt)
            sub_nodes = list(set(explanatory) | set(connector))
            sub_evaluated_scores = list(map(lambda node: (node, self.evaluate(i, list(filter(lambda x: x != node, sub_nodes)), '-')), sub_nodes))
            sub_sorted_node = sorted(sub_evaluated_scores, key=lambda x: x, reverse=False)
            sub_sorted_node = [node for node, score in sub_sorted_node]
            sub_sorted_node.remove(vt)
            if len(sub_sorted_node) == 0:
                return
            node_old = sub_sorted_node[0]
            removed_set = list(filter(lambda x: x != node_old, sub_nodes))

            left_set = [node for node in self.optimal[i]['subset'] if node not in sub_nodes]
            left_evaluated_scores = list(map(lambda node: (node, self.evaluate(i, self.evaluate(removed_set+[node]), '-')), left_set))
            left_sorted_node = sorted(left_evaluated_scores, key=lambda x: x, reverse=True)
            left_sorted_node = [node for node, score in left_sorted_node]
            for each_node in left_sorted_node:
                if each_node != vt:
                    trans_data = Data(x=self.dataset.data.x, edge_index=self.dataset.data.edge_index)
                    G = to_networkx(trans_data, to_undirected=True)
                    subgraph = G.subgraph(removed_set+[each_node])
                    if nx.is_connected(subgraph) == True:
                        # copy the ori result
                        ori_explanatory_set = copy.deepcopy(explanatory)
                        ori_connector = copy.deepcopy(connector)
                        print(node_old, explanatory, connector)
                        if node_old in explanatory:
                            explanatory.remove(node_old)
                            explanatory.append(each_node)
                        else:
                            connector.remove(node_old)
                            connector.append(each_node)
                        self.optimal[i]['nodes'] = list(set(explanatory) | set(connector))
                        self.connect_and_verify()
                        if self.optimal[i]['factual']==True or self.optimal[i]['counterfactual']==True:
                            return
                        self.optimal[i]['nodes'] = ori_explanatory_set
                        explanatory = ori_explanatory_set
                        connector = ori_connector


    def connect_and_verify(self):
        all_explanatory = set()
        for vt in self.test_indices:
            if self.optimal[self.node_index_map[vt]]['Finish'] == True:
                continue
            trans_data = Data(x=self.dataset.data.x, edge_index=self.dataset.data.edge_index)
            G = to_networkx(trans_data, to_undirected=True)
            explanatory = self.optimal[self.node_index_map[vt]]['nodes']
            all_explanatory.update(explanatory)
            subgraph = G.subgraph(explanatory)
            if nx.is_connected(subgraph) == False:
                all_explanatory.update(self.bfs_connect_disconnected_nodes(G, subgraph, explanatory, vt))
        self.verify(all_explanatory)


    def bfs_connect_disconnected_nodes(self, graph, subgraph, explanatory, test_node):
        disconnected_nodes = [node for node in explanatory if not nx.has_path(subgraph, test_node, node)]
        shortest_paths = {}
        new_connector = []
        for node in disconnected_nodes:
            path = nx.shortest_path(graph, source=node, target=test_node)
            shortest_paths[node] = path
            for node in path:
                if node not in explanatory:
                    new_connector.append(node)
        return new_connector

    def verify(self, all_explanatory):
        emb_gs, logits_gs = self.model(self.dataset.data.x, self.node_induced_edges(all_explanatory))
        counter_nodes = [node for node in self.candidate.nodes if node not in all_explanatory]
        for vt in self.test_indices:
            if self.optimal[self.node_index_map[vt]]['Finish'] == True:
                continue
            prediction = self.optimal[self.node_index_map[vt]]['prediction']
            emb_gsr, logits_gsr = self.model(self.dataset.data.x, self.node_induced_edges(counter_nodes+[vt]))
            factual = (logits_gs[vt].argmax(-1).item() == prediction)
            counterfactual = (logits_gsr[vt].argmax(-1).item() != prediction)
            both = (factual and counterfactual)
            counter_score = self.global_logits.softmax(dim=-1)[vt][prediction].item() - logits_gs.softmax(dim=-1)[vt][prediction].item()
            sub_score = self.global_logits.softmax(dim=-1)[vt][prediction].item() - logits_gsr.softmax(dim=-1)[vt][prediction].item()
            if factual == True or counterfactual == True:
                index = self.node_index_map[vt]
                self.optimal[index]['Finish'] = True
                self.optimal[index]['Fill'] = True
                self.optimal[index]['factual'] = factual
                self.optimal[index]['counterfactual'] = counterfactual
                self.optimal[index]['both'] = both
                self.optimal[index]['score'] = self.evaluate(index,[-1], 'None')
                self.optimal[index]['Fid+'] = sub_score
                self.optimal[index]['Fid-'] = counter_score

    def node_induced_edges(self, subset):
        subset_indices = torch.tensor(list(subset), dtype=torch.long).to(self.device)
        mask_0 = self.dataset.data.edge_index[0].unsqueeze(1) == subset_indices
        mask_1 = self.dataset.data.edge_index[1].unsqueeze(1) == subset_indices
        mask = mask_0.any(dim=1) & mask_1.any(dim=1)
        return self.dataset.data.edge_index[:, mask]

    def add_new_node(self):
        marginal_gain = [{'choices': [], 'gain': []} for _ in range(len(self.test_indices))]
        for i,vt in enumerate(self.test_indices):
            if self.optimal[i]['Fill'] == True:
                continue
            for node in self.optimal[i]['subset']:
                if node in self.optimal[i]['nodes'] or self.candidate.Use[self.candidate.node_index_map[node]] == True:
                    continue
                mg = self.count_marginal_gain(i, node)
                if len(marginal_gain[i]['gain']) < 2:
                    marginal_gain[i]['choices'].append(node)
                    marginal_gain[i]['gain'].append(mg)
                else:
                    min_mg = min(marginal_gain[i]['gain'])
                    if mg > min_mg:
                        min_index = marginal_gain[i]['gain'].index(min_mg)
                        marginal_gain[i]['choices'][min_index] = node
                        marginal_gain[i]['gain'][min_index] = mg
            self.update_Vs(i, marginal_gain[i])


    def count_marginal_gain(self, index, node):
        ori_score = self.optimal[index]['score']
        result = self.evaluate(index, [node], '+') - ori_score
        return result

    def update_Vs(self, index, gain_dict):
        add_node = []
        if len(gain_dict['gain']) == 1:
            add_node.append(gain_dict['choices'][0])
        elif len(gain_dict['gain']) == 2:
            sort_status = (gain_dict['gain'][0] > gain_dict['gain'][1])
            if sort_status:
                add_node.append(gain_dict['choices'][0])
                add_node.append(gain_dict['choices'][1])
            else:
                add_node.append(gain_dict['choices'][1])
                add_node.append(gain_dict['choices'][0])
        for node in add_node:
            related_test_nodes = self.candidate.B[self.candidate.node_index_map[node]]
            self.candidate.Use[self.candidate.node_index_map[node]] = True
            for each_vt in related_test_nodes:
                if self.optimal[self.node_index_map[each_vt]]['Fill'] == False and node not in self.optimal[self.node_index_map[each_vt]]['nodes']:
                    self.optimal[self.node_index_map[each_vt]]['nodes'].append(node)
                    self.optimal[self.node_index_map[each_vt]]['current_influence'] = self.optimal[self.node_index_map[each_vt]]['current_influence'].union(self.influence_set[each_vt][node])
                    self.optimal[self.node_index_map[each_vt]]['current_diversity'] = self.optimal[self.node_index_map[each_vt]]['current_diversity'].union(self.diversity_set[each_vt][node])
                    if len(self.optimal[self.node_index_map[each_vt]]['nodes']) >= self.K or len(self.optimal[self.node_index_map[each_vt]]['nodes']) == len(self.optimal[self.node_index_map[each_vt]]['subset']):
                        self.optimal[self.node_index_map[each_vt]]['Fill'] = True

def layerwise_run(cut_layer, test_indices, device,logger, dec,state_dict):
    modelslice=Slicedmodel(dec.config, device,len(dec.config.models.param.gnn_latent_dim) - cut_layer,logger,dec.dataset, state_dict)
    quality = Subfunction(test_indices, dec, modelslice, logger, device)

    cover_file=os.path.join('Greedy',dec.config.datasets.dataset_name,'diversity_file')
    label_cover_file = os.path.join('Greedy', dec.config.datasets.dataset_name, 'relative_influence_file')
    check_dirs(cover_file)
    check_dirs(label_cover_file)

    fidelity = []
    fidelity_inv = []
    start_time = time.time()
    dec.dataset.data.to(device)
    algorithm = GreedyAlgorithm(dec, modelslice, test_indices, logger, quality)
    optimal = algorithm.get_solution()
    for i in range(len(test_indices)):
        fidelity.append(optimal[i]['Fid+'])
        fidelity_inv.append(optimal[i]['Fid-'])
    logger.info(f'Fidelity: {sum(fidelity) / len(fidelity):.4f}\n'
                f'Fidelity_inv: {sum(fidelity_inv) / len(fidelity_inv): .4f}')
    end_time = time.time()
    logger.info(f'Execution time: {end_time - start_time:.6f}')




@hydra.main(version_base=None, config_path='config', config_name='config')
def main(config):
    K = [10]
    h=[0.3]
    theta=[0.2]
    gamma=0.5
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

    # if config.datasets.dataset_name in ['products']:
    #     sampled_nodes = torch.load('/home/Ztt277/LWE/checkpoints/products/product_nodes.pt')
    #     # 从原始图中抽取子图
    #     edge_index, _ = subgraph(sampled_nodes, dataset.data.edge_index, relabel_nodes=True)
    #
    #     dataset.data.x = dataset.data.x[sampled_nodes]
    #     dataset.data.edge_index = edge_index
    #     dataset.data.train_mask = dataset.data.train_mask[sampled_nodes]
    #     dataset.data.y = dataset.data.y[sampled_nodes]
    #
    #     dataset.data.val_mask = dataset.data.val_mask[sampled_nodes]
    #     dataset.data.test_mask = dataset.data.test_mask[sampled_nodes]

    log_file = (
        f"{config.datasets.dataset_name}_{len(config.models.param.gnn_latent_dim)}.log"
    )
    logger = get_logger(config.log_path, log_file, config.console_log, config.log_level)
    logger.info(OmegaConf.to_yaml(config))
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    logger.info(f'Using data: {dataset.data}')

    layer_nums = len(config.models.param.gnn_latent_dim)
    state_dict = torch.load(os.path.join(config.models.gnn_savedir,
                                         config.datasets.dataset_name,
                                         f'{config.models.gnn_name}_'
                                         f'{len(config.models.param.gnn_latent_dim)}l_best.pth'))['net']

    test_indices = config.datasets.test_indices
    # test_indices = torch.where(dataset.data.test_mask)[0].tolist()
    logger.info(f'test nodes : {test_indices}')
    for h_mini in h:
        for th in theta:
            for size_run in K:
                logger.info(f'h:{h_mini}')
                logger.info(f'theta: {th}')
                logger.info(f'size: {size_run}')
                # for i in range(layer_nums - 1, -1, -1):
                for i in [0,1,2]:
                    dec=Declarative(config,dataset,size_run,th,h_mini,gamma)
                    layerwise_run(i, test_indices, device, logger, dec, state_dict)


if __name__ == '__main__':
    import sys
    simplefilter(action="ignore", category=FutureWarning)
    sys.argv.append(f"models.gnn_savedir={os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    main()

