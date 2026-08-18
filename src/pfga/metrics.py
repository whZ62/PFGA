"""Evaluation metrics used by the PFGA paper."""

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor

from pfga.assignment import AssignmentResult
from pfga.data import MonthlyBatch
from pfga.losses import gini_coefficient


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size > 1 else 0.5


def evaluate_assignment(
    batch: MonthlyBatch,
    candidate_edge_index: Tensor,
    logits: Tensor,
    assignment: AssignmentResult,
    candidate_recall: float,
    inference_seconds: float,
    cost_unit_km: float,
    dense_auc_pair_limit: int,
) -> Dict[str, object]:
    """Compute AUC, pair accuracy, utility, Gini (`fair`), cost, and time."""

    n = batch.num_tasks
    total_pairs = batch.num_workers * n
    positive_linear = torch.unique(
        batch.positive_edge_index[0] * n + batch.positive_edge_index[1]
    )
    candidate_linear = candidate_edge_index[0] * n + candidate_edge_index[1]
    candidate_probabilities = torch.sigmoid(logits)

    if total_pairs <= dense_auc_pair_limit:
        dense_labels = torch.zeros(total_pairs, dtype=torch.float32, device=batch.device)
        dense_scores = torch.zeros(total_pairs, dtype=torch.float32, device=batch.device)
        dense_labels[positive_linear] = 1.0
        dense_scores[candidate_linear] = candidate_probabilities
        auc = _safe_auc(
            dense_labels.detach().cpu().numpy(), dense_scores.detach().cpu().numpy()
        )
        auc_scope = "all_pairs"
    else:
        candidate_labels = batch.labels_for_edges(candidate_edge_index)
        auc = _safe_auc(
            candidate_labels.detach().cpu().numpy(),
            candidate_probabilities.detach().cpu().numpy(),
        )
        auc_scope = "candidate_edges"

    assigned_linear = torch.unique(
        assignment.edge_index[0] * n + assignment.edge_index[1]
    )
    true_positive = int(torch.isin(assigned_linear, positive_linear).sum().item())
    predicted_positive = int(assigned_linear.numel())
    positive_count = int(positive_linear.numel())
    false_positive = predicted_positive - true_positive
    false_negative = positive_count - true_positive
    true_negative = total_pairs - true_positive - false_positive - false_negative
    accuracy = float((true_positive + true_negative) / max(total_pairs, 1))

    assigned_count = int(assignment.edge_index.shape[1])
    if assigned_count:
        assigned_utility = batch.edge_benefit(assignment.edge_index)
        assigned_cost = batch.edge_cost(assignment.edge_index)
        total_utility = float(assigned_utility.sum().item())
        average_utility = float(assigned_utility.mean().item())
        total_distance_km = float(assigned_cost.sum().item())
    else:
        total_utility = 0.0
        average_utility = 0.0
        total_distance_km = 0.0

    worker_income = torch.zeros(
        batch.num_workers, dtype=batch.benefit_matrix.dtype, device=batch.device
    )
    if assigned_count:
        worker_income.index_add_(
            0,
            assignment.edge_index[0],
            batch.edge_benefit(assignment.edge_index),
        )
    fair = float(gini_coefficient(worker_income).item())

    return {
        "period": batch.period,
        "auc": auc,
        "auc_scope": auc_scope,
        "acc": accuracy,
        "utility": average_utility,
        "fair": fair,
        "cost": total_distance_km / cost_unit_km,
        "time": inference_seconds / max(batch.num_tasks, 1),
        "candidate_recall": candidate_recall,
        "assigned_tasks": assigned_count,
        "unassigned_tasks": int(assignment.unassigned_tasks.numel()),
        "assignment_rate": assigned_count / max(batch.num_tasks, 1),
        "total_utility": total_utility,
        "total_distance_km": total_distance_km,
        "threshold": assignment.threshold,
    }
