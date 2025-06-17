"""
FileName: utils.py
Description: The utils we may use for GNN model or Explainable model construction
Time: 2020/7/31 11:29
Project: GNN_benchmark
Author: Shurui Gui
"""
import torch


class ReadOut(torch.nn.Module):

    def __init__(self):
        super().__init__()

    @staticmethod
    def divided_graph(x, batch_index):
        graph = []
        for i in range(batch_index[-1] + 1):
            graph.append(x[batch_index == i])

        return graph

    def forward(self, x: torch.tensor, batch_index) -> torch.tensor:
        graph = ReadOut.divided_graph(x, batch_index)

        for i in range(len(graph)):
            graph[i] = graph[i].mean(dim=0).unsqueeze(0)

        out_readoout = torch.cat(graph, dim=0)

        return out_readoout


def normalize(x: torch.Tensor):
    x -= x.min()  # This operation -= may lead to mem leak without detach() before this assignment. (x = x - x.min() won't lead to such a problem.)
    if x.max() == 0:
        return torch.zeros(x.size(), device=x.device)
    x /= x.max()
    return x


def gumbel_softmax(log_alpha: torch.Tensor, beta: float = 1.0, training: bool = True):
    r""" Sample from the instantiation of concrete distribution when training
    Args:
        log_alpha: input probabilities
        beta: temperature for softmax
    """
    if training:
        random_noise = torch.rand(log_alpha.shape)
        random_noise = torch.log(random_noise) - torch.log(1.0 - random_noise)
        gate_inputs = (random_noise.to(log_alpha.device) + log_alpha) / beta
        gate_inputs = gate_inputs.sigmoid()
    else:
        gate_inputs = log_alpha.sigmoid()

    return gate_inputs
