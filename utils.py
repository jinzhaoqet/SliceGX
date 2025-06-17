import pytz
import logging
import networkx as nx
from datetime import datetime
from torch_geometric.data import Data, Batch, Dataset, DataLoader
from typing import Union, List
from textwrap import wrap
import matplotlib.pyplot as plt
import json
import os
import torch
import random
import numpy as np
from abc import ABC


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_dirs(save_dirs):
    print(save_dirs)
    if save_dirs:
        if os.path.isdir(save_dirs):
            pass
        else:
            os.makedirs(save_dirs)

def timetz(*args):
    tz = pytz.timezone("US/Pacific")
    return datetime.now(tz).timetuple()

def perturb_input(data, hard_edge_mask, subset):
    """add 2 additional empty node into the motif graph"""
    num_add_node = 2
    num_perturb_graph = 10
    subgraph_x = data.x[subset]
    subgraph_edge_index = data.edge_index[:, hard_edge_mask]
    row, col = data.edge_index

    mappings = row.new_full((data.num_nodes,), -1)
    mappings[subset] = torch.arange(subset.size(0), device=row.device)
    subgraph_edge_index = mappings[subgraph_edge_index]

    subgraph_y = data.y[subset]

    num_node_subgraph = subgraph_x.shape[0]

    # add two nodes to the subgraph, the node features are all 0.1
    subgraph_x = torch.cat([subgraph_x,
                            torch.ones(2, subgraph_x.shape[1]).to(subgraph_x.device)],
                           dim=0)
    subgraph_y = torch.cat([subgraph_y,
                            torch.zeros(num_add_node).type(torch.long).to(subgraph_y.device)], dim=0)

    perturb_input_list = []
    for _ in range(num_perturb_graph):
        to_node = torch.randint(0, num_node_subgraph, (num_add_node,))
        frm_node = torch.arange(num_node_subgraph, num_node_subgraph + num_add_node, 1)
        add_edges = torch.cat([torch.stack([to_node, frm_node], dim=0),
                               torch.stack([frm_node, to_node], dim=0),
                               torch.stack([frm_node, frm_node], dim=0)], dim=1)
        perturb_subgraph_edge_index = torch.cat([subgraph_edge_index,
                                                 add_edges.to(subgraph_edge_index.device)], dim=1)
        perturb_input_list.append(Data(x=subgraph_x, edge_index=perturb_subgraph_edge_index, y=subgraph_y))

    return perturb_input_list

class Recorder(ABC):
    def __init__(self, recorder_filename):
        # init the recorder
        self.recorder_filename = recorder_filename
        if os.path.isfile(recorder_filename):
            with open(recorder_filename, 'r') as f:
                self.recorder = json.load(f)
        else:
            self.recorder = {}
            check_dirs(os.path.dirname(recorder_filename))

    @classmethod
    def load_and_change_dict(cls, ori_dict, experiment_settings, experiment_data):
            key = experiment_settings[0]
            if key not in ori_dict.keys():
                ori_dict[key] = {}
            if len(experiment_settings) == 1:
                ori_dict[key] = experiment_data
            else:
                ori_dict[key] = cls.load_and_change_dict(ori_dict[key],
                                                         experiment_settings[1:],
                                                         experiment_data)
            return ori_dict

    def append(self, experiment_settings, experiment_data):
        ex_dict = self.recorder

        self.recorder = self.load_and_change_dict(ori_dict=ex_dict,
                                                  experiment_settings=experiment_settings,
                                                  experiment_data=experiment_data)

    def save(self):
        with open(self.recorder_filename, 'w') as f:
            json.dump(self.recorder, f, indent=2)


def get_logger(log_path, log_file, console_log=False, log_level=logging.INFO):
    check_dirs(log_path)

    tz = pytz.timezone("US/Pacific")
    logger = logging.getLogger(__name__)
    logger.propagate = False  # avoid duplicate logging
    logger.setLevel(log_level)

    # Clean logger first to avoid duplicated handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)

    file_handler = logging.FileHandler(os.path.join(log_path, log_file))
    formatter = logging.Formatter("%(asctime)s: %(message)s", datefmt="%b%d %H-%M-%S")
    formatter.converter = timetz
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console_log:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger

class PlotUtils(object):
    def __init__(self, dataset_name, is_show=True):
        self.dataset_name = dataset_name
        self.is_show = is_show

    def plot_subgraph(self,
                      graph,
                      nodelist,
                      colors: Union[None, str, List[str]] = '#FFA500',
                      labels=None,
                      edge_color='gray',
                      edgelist=None,
                      subgraph_edge_color='black',
                      title_sentence=None,
                      figname=None):

        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]
        pos = nx.kamada_kawai_layout(graph)
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}
        plt.figure(figsize=(12, 8))
        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=6,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))
        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_subgraph_with_nodes(self,
                                 graph,
                                 nodelist,
                                 node_idx,
                                 colors='#FFA500',
                                 labels=None,
                                 edge_color='gray',
                                 edgelist=None,
                                 subgraph_edge_color='black',
                                 title_sentence=None,
                                 figname=None):
        node_idx = int(node_idx)
        if edgelist is None:
            edgelist = [(n_frm, n_to) for (n_frm, n_to) in graph.edges()
                        if n_frm in nodelist and n_to in nodelist]

        pos = nx.kamada_kawai_layout(graph)  # calculate according to graph.nodes()
        pos_nodelist = {k: v for k, v in pos.items() if k in nodelist}

        nx.draw_networkx_nodes(graph, pos,
                               nodelist=list(graph.nodes()),
                               node_color=colors,
                               node_size=300)
        if isinstance(colors, list):
            list_indices = int(np.where(np.array(graph.nodes()) == node_idx)[0])
            node_idx_color = colors[list_indices]
        else:
            node_idx_color = colors
        nx.draw_networkx_nodes(graph, pos=pos,
                               nodelist=[node_idx],
                               node_color=node_idx_color,
                               node_size=600)

        nx.draw_networkx_edges(graph, pos, width=3, edge_color=edge_color, arrows=False)

        nx.draw_networkx_edges(graph, pos=pos_nodelist,
                               edgelist=edgelist, width=3,
                               edge_color=subgraph_edge_color,
                               arrows=False)

        if labels is not None:
            nx.draw_networkx_labels(graph, pos, labels)

        plt.axis('off')
        if title_sentence is not None:
            plt.title('\n'.join(wrap(title_sentence, width=60)))

        if figname is not None:
            plt.savefig(figname)
        if self.is_show:
            plt.show()
        if figname is not None:
            plt.close()

    def plot_bashapes(self,
                      graph,
                      nodelist,
                      y,
                      node_idx,
                      edgelist=None,
                      title_sentence=None,
                      figname=None):
        node_idxs = {k: int(v) for k, v in enumerate(y.reshape(-1).tolist())}
        node_color = ['#FFA500', '#4970C6', '#FE0000', 'green']
        colors = [node_color[v % len(node_color)] for k, v in node_idxs.items()]
        self.plot_subgraph_with_nodes(graph,
                                      nodelist,
                                      node_idx,
                                      colors,
                                      edgelist=edgelist,
                                      title_sentence=title_sentence,
                                      figname=figname,
                                      subgraph_edge_color='black')

    def drawcora(self, graph, dataset, select_nodes):
        G = nx.Graph()
        for i in select_nodes:
            G.add_node(i, feature=dataset.data.x[i])

        for i in range(graph.edge_index.size(1)):
            src, dst = graph.edge_index[:, i]
            G.add_edge(src.item(), dst.item())
        color_map = {
            0: 'red',
            1: 'blue',
            2: 'pink',
            3: 'green',
            4: 'yellow',
            5: 'orange',
            6: 'purple',
        }
        for i in G.nodes:
            print(i, dataset.data.y[i].item())
        self.plot_subgraph(G, nodelist=G.nodes,
                                edgelist=None,
                                edge_color="gray",
                                subgraph_edge_color="black",
                                title_sentence=None,
                                labels={n: n for n in G.nodes},
                                figname=None,
                                colors=[color_map[dataset.data.y[n].item()] for n in G.nodes],
                                )

    def draw(self, graph, dataset, select_nodes):
        G = nx.Graph()
        print(G.nodes)
        for i in select_nodes:
            G.add_node(i, feature=dataset.data.x[i])

        for i in range(graph.edge_index.size(1)):
            src, dst = graph.edge_index[:, i]
            G.add_edge(src.item(), dst.item())
        self.plot_subgraph(G, nodelist=G.nodes,
                                edgelist=None,
                                edge_color="gray",
                                subgraph_edge_color="black",
                                title_sentence=None,
                                labels={n: n for n in G.nodes},
                                figname=None)

    def draw_subgraph_edges(self, graph, dataset, select_nodes, main_index):
        G = nx.Graph()
        print(G.nodes)
        for i in select_nodes:
            G.add_node(i, feature=dataset.data.x[i])
        for i in range(graph.edge_index.size(1)):
            src, dst = graph.edge_index[:, i]
            G.add_edge(src.item(), dst.item())

        color = []
        for u, v in G.edges:
            flag = 0
            for j in main_index:
                if (dataset.data.edge_index[0][j] == u and dataset.data.edge_index[1][j] == v) or (
                        dataset.data.edge_index[0][j] == v and dataset.data.edge_index[1][j] == u):
                    color.append("green")
                    flag = 1
                    break
            if flag == 0:
                color.append("black")
        self.plot_subgraph(G, nodelist=G.nodes,
                           edgelist=G.edges,
                           subgraph_edge_color=color,
                           title_sentence=None,
                           labels={n: n for n in G.nodes},
                           figname=None,
                           )


