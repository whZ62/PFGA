"""Paper-aligned prediction, utility, cost, and fairness objectives."""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

from pfga.config import LossSection
from pfga.data import MonthlyBatch


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
    rank = torch.arange(
        1, n + 1, dtype=sorted_income.dtype, device=sorted_income.device
    )
    numerator = torch.sum((2.0 * rank - n - 1.0) * sorted_income)
    return (numerator / (n * total + epsilon)).clamp(0.0, 1.0)


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


def _reduce(values: Tensor, reduction: str) -> Tensor:
    return values.mean() if reduction == "mean" else values.sum()


def composite_loss(
    logits: Tensor,
    labels: Tensor,
    batch: MonthlyBatch,
    edge_index: Tensor,
    config: LossSection,
) -> LossBreakdown:
    """Compute Eq. (12)-(16) on sparse candidate edges."""

    labels = labels.to(dtype=logits.dtype)
    positive_count = labels.sum()
    negative_count = labels.numel() - positive_count
    if positive_count.detach().item() > 0:
        positive_weight = (negative_count / positive_count).clamp_min(1.0)
    else:
        positive_weight = torch.ones((), dtype=logits.dtype, device=logits.device)
    prediction = F.binary_cross_entropy_with_logits(
        logits,
        labels,
        pos_weight=positive_weight,
        reduction="mean",
    )

    probabilities = torch.sigmoid(logits)
    raw_utility = batch.edge_benefit(edge_index)
    raw_cost = batch.edge_cost(edge_index)
    utility = raw_utility
    cost = raw_cost
    if config.normalize_utility:
        utility = utility / utility.detach().max().clamp_min(1e-8)
    if config.normalize_cost:
        cost = cost / cost.detach().max().clamp_min(1e-8)

    utility_loss = -_reduce(probabilities * utility, config.reduction)
    cost_loss = _reduce(probabilities * cost, config.reduction)

    worker_income = torch.zeros(
        batch.num_workers,
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    worker_income.index_add_(0, edge_index[0], probabilities * raw_utility)
    fairness_loss = gini_coefficient(worker_income)

    allocation_objectives = (
        config.utility_weight * utility_loss
        + config.cost_weight * cost_loss
        + config.fairness_weight * fairness_loss
    )
    total = config.lambda_p * prediction + (1.0 - config.lambda_p) * allocation_objectives
    return LossBreakdown(
        total=total,
        prediction=prediction,
        utility=utility_loss,
        cost=cost_loss,
        fairness=fairness_loss,
        positive_weight=positive_weight,
    )
