from typing import Optional, Union
import torch
from torch import Tensor
import torch.nn.functional as F
from explain import Explanation, GraphMaskExplainer
from explain.config import ModelMode


def forward_(
        self,
        model: torch.nn.Module,
        x: Tensor,
        edge_index: Tensor,
        *,
        target: Tensor,
        index: Optional[Union[int, Tensor]] = None,
        **kwargs,
) -> Explanation:
    explanation = super(type(self), self).forward(model, x, edge_index, target=target, index=index, **kwargs)
    hard_node_mask, hard_edge_mask = self._get_hard_masks(model, index, edge_index, x.shape[0])
    if self.fidelity_plus:
        if explanation.get('edge_mask') is not None:
            explanation.edge_mask = 1 - explanation.edge_mask
            if hard_edge_mask is not None:
                explanation.edge_mask[~hard_edge_mask] = 0
        if explanation.get('node_mask') is not None:
            explanation.node_mask = 1 - explanation.node_mask
            if hard_node_mask is not None and explanation.node_mask.shape == hard_node_mask.shape:
                explanation.node_mask[~hard_node_mask] = 0
    return explanation




class GraphMaskExplainer(GraphMaskExplainer):
    fidelity_plus = False
    forward = forward_

    def _loss(self, y_hat: Tensor, y: Tensor, penalty: float) -> Tensor:
        if self.model_config.mode == ModelMode.binary_classification:
            loss = self._loss_binary_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.multiclass_classification:
            loss = self._loss_multiclass_classification(y_hat, y)
        elif self.model_config.mode == ModelMode.regression:
            loss = self._loss_regression(y_hat, y)
        else:
            assert False

        if self.fidelity_plus:
            g = -loss
        else:
            g = torch.relu(loss - self.allowance).mean()
        f = penalty * self.penalty_scaling

        loss = f + F.softplus(self.lambda_op) * g

        m = self.node_feat_mask.sigmoid()
        node_feat_reduce = getattr(torch, self.coeffs['node_feat_reduction'])
        loss = loss + self.coeffs['node_feat_size'] * node_feat_reduce(m)
        ent = -m * torch.log(m + self.coeffs['EPS']) - (
                1 - m) * torch.log(1 - m + self.coeffs['EPS'])
        loss = loss + self.coeffs['node_feat_ent'] * ent.mean()

        return loss
