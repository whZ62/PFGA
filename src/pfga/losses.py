"""Paper-aligned prediction, utility, cost, and fairness objectives."""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor


def gini_coefficient(income: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Differentiable sorted-form Gini coefficient in O(m log m)."""

    if income.ndim != 1:
        raise ValueError("income must be one-dimensional")
    if income.numel() == 0:
        raise ValueError("income must not be empty")
    if (income < 0).any():
        raise ValueError("income must be non-negative")
    total = income.sum()
    if total.detach().item() <= epsilon:
        return total * 0.0
    sorted_income = torch.sort(income).values
    n = sorted_income.numel()
    rank = torch.arange(1, n + 1, dtype=sorted_income.dtype, device=sorted_income.device)
    numerator = torch.sum((2.0 * rank - n - 1.0) * sorted_income)
    return (numerator / (n * total + n * n * epsilon)).clamp(0.0, 1.0)


@dataclass
class LossBreakdown:
    total: Tensor
    prediction: Tensor
    utility: Tensor
    cost: Tensor
    fairness: Tensor
    positive_weight: Tensor

    def scalars(self) -> Dict[str, float]:
        return {
            "loss": float(self.total.detach().item()),
            "prediction_loss": float(self.prediction.detach().item()),
            "utility_loss": float(self.utility.detach().item()),
            "cost_loss": float(self.cost.detach().item()),
            "fairness_loss": float(self.fairness.detach().item()),
            "positive_weight": float(self.positive_weight.detach().item()),
        }


def soft_allocation(logits, batch, edges, threshold):
    """Eq17. Hard threshold mask has stop-gradient; retained weights are differentiable."""
    p = torch.sigmoid(logits)
    p = p * (p >= threshold).to(p.dtype)
    task_sum = torch.zeros(batch.num_tasks, device=p.device).index_add(0, edges[1], p)
    a = p / (1 + task_sum[edges[1]])
    worker_sum = torch.zeros(batch.num_workers, device=p.device).index_add(0, edges[0], a)
    scale = (batch.worker_capacity / (worker_sum + 1e-8)).clamp(max=1)
    return a * scale[edges[0]]


def composite_loss(
    logits,
    labels,
    batch,
    edge_index,
    config,
    historical_utility=None,
    threshold=0.4,
    prediction_logits=None,
    prediction_labels=None,
):
    """Eq16-22; supervision edges are independent of policy candidate pruning."""
    pred = logits if prediction_logits is None else prediction_logits
    y = labels if prediction_labels is None else prediction_labels
    y = y.to(pred.dtype)
    pos, neg = y.sum(), (1 - y).sum()
    # Single-class/empty batches have a finite unweighted fallback.
    weight = neg / pos if pos.item() > 0 and neg.item() > 0 else pred.new_tensor(1.0)
    if pred.numel():
        prediction = F.binary_cross_entropy_with_logits(
            pred, y, pos_weight=weight, reduction="sum"
        ) / (weight * pos + neg).clamp_min(1)
    else:
        prediction = pred.sum() * 0
    mass = soft_allocation(logits, batch, edge_index, threshold)
    utility = -(mass * batch.edge_benefit(edge_index)).sum() / batch.num_tasks
    cost = (mass * batch.edge_cost(edge_index) / batch.distance_scale_km).sum() / batch.num_tasks
    if historical_utility is None:
        history = torch.zeros(batch.num_workers, device=logits.device)
    else:
        history = historical_utility.detach().clone()
    # Global worker population, including inactive / zero-allocation workers.
    ids = batch.worker_ids[edge_index[0]] if historical_utility is not None else edge_index[0]
    income = history.index_add(0, ids, mass * batch.edge_benefit(edge_index))
    fairness = gini_coefficient(income)
    total = config.lambda_p * prediction + (1 - config.lambda_p) * (
        config.utility_weight * utility
        + config.cost_weight * cost
        + config.fairness_weight * fairness
    )
    return LossBreakdown(total, prediction, utility, cost, fairness, weight)
