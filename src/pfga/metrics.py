"""Cumulative replay metrics, with explicit counts and units."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from pfga.losses import gini_coefficient


class ReplayMetrics:
    def __init__(self, num_workers):
        self.utility = torch.zeros(num_workers, dtype=torch.float64)
        self.rewards = torch.zeros_like(self.utility)
        self.opportunities = torch.zeros_like(self.utility)
        self.released = 0
        self.assigned = 0
        self.reward_denominator = 0.0
        self.distance = 0.0
        self.seconds = 0.0
        self.labels = []
        self.scores = []

    def add(self, batch, assignment, labels, scores, new_rewards, seconds):
        self.released += len(new_rewards)
        self.reward_denominator += float(sum(new_rewards))
        e = assignment.edge_index.cpu()
        ids = batch.worker_ids.cpu()[e[0]]
        self.assigned += e.shape[1]
        self.utility.index_add_(
            0, ids, batch.edge_benefit(assignment.edge_index).detach().cpu().double()
        )
        self.rewards.index_add_(0, ids, batch.task_rewards[assignment.edge_index[1]].cpu().double())
        self.distance += float(batch.edge_cost(assignment.edge_index).sum())
        self.seconds += seconds
        if batch.reference_opportunities is not None:
            self.opportunities.index_add_(
                0, batch.worker_ids.cpu(), batch.reference_opportunities.cpu().double()
            )
        self.labels.extend(labels.detach().cpu().tolist())
        self.scores.extend(scores.detach().cpu().tolist())

    def result(self, threshold, with_predictions=True):
        y = np.asarray(self.labels)
        p = np.asarray(self.scores)
        auc = float(roc_auc_score(y, p)) if with_predictions and len(np.unique(y)) > 1 else None
        acc = float(((p >= threshold) == y).mean()) if len(y) else None
        mask = self.opportunities > 0
        opp = (
            float(gini_coefficient(self.utility[mask] / self.opportunities[mask]))
            if mask.any()
            else None
        )
        return {
            "auc": auc,
            "acc": acc,
            "auc_scope": "fixed_reference_feasible_1_to_2_sample",
            "prediction_pairs": len(y),
            "positive_pairs": int(y.sum()),
            "assignment_rate": self.assigned / max(self.released, 1),
            "utility": float(self.utility.sum()) / max(self.reward_denominator, 1e-8),
            "fair": float(gini_coefficient(self.utility)),
            "reward_gini": float(gini_coefficient(self.rewards)),
            "opportunity_gini": opp,
            "zero_opportunity_workers": int((~mask).sum()),
            "cost": self.distance,
            "cost_unit": "km_total",
            "average_distance_km": self.distance / max(self.assigned, 1),
            "time": self.seconds / max(self.released, 1),
            "time_unit": "seconds_per_released_task",
            "released_tasks": self.released,
            "assigned_tasks": self.assigned,
            "unassigned_tasks": self.released - self.assigned,
            "total_utility": float(self.utility.sum()),
            "released_reward": self.reward_denominator,
            "total_reward": float(self.rewards.sum()),
            "threshold": threshold,
        }
