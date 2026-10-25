"""Sparse feasible candidate graph materialization (paper Eq. 23-28)."""

from dataclasses import dataclass

import torch
from torch import Tensor

from pfga.config import CandidateSection
from pfga.data import MonthlyBatch


@dataclass
class CandidateGraph:
    edge_index: Tensor
    scores: Tensor
    feasible_per_task: Tensor
    retained_per_task: Tensor
    positive_recall: float


class CandidateGraphMaterializer:
    """Rank feasible edges by preference, distance, fairness deficit, and slack."""

    def __init__(self, config: CandidateSection):
        self.config = config

    def __call__(self, batch: MonthlyBatch, historical_utility: Tensor) -> CandidateGraph:
        edges = batch.feasible_edge_index
        if edges.shape[1] == 0:
            z = torch.zeros(batch.num_tasks, dtype=torch.long, device=batch.device)
            return CandidateGraph(edges, torch.empty(0, device=batch.device), z, z.clone(), 0.0)

        worker_index, task_index = edges
        edge_cost = batch.edge_cost(edges)
        maximum_distance = batch.distance_scale_km
        normalized_distance = edge_cost / maximum_distance
        preference = batch.preference_matrix[worker_index, task_index]
        slack = batch.slack_matrix[worker_index, task_index]

        mean_utility = historical_utility.mean()
        x = (mean_utility - historical_utility[batch.worker_ids]) / (mean_utility + 1e-8)
        deficit = (x / (1 + x.abs())).clamp_min(0)

        scores = (
            self.config.preference_weight * preference
            - self.config.distance_weight * normalized_distance
            + self.config.fairness_weight * deficit[worker_index]
            + self.config.slack_weight * slack
        )

        if self.config.enabled:
            order = torch.argsort(task_index, stable=True)
            sorted_tasks = task_index[order]
            _, counts = torch.unique_consecutive(sorted_tasks, return_counts=True)
            selected_parts = []
            offset = 0
            for count in counts.tolist():
                segment = order[offset : offset + count]
                keep = min(self.config.top_k, count)
                local_top = torch.topk(scores[segment], k=keep, sorted=True).indices
                selected_parts.append(segment[local_top])
                offset += count
            selected = torch.cat(selected_parts)
        else:
            selected = torch.arange(edges.shape[1], device=edges.device)
        candidate_edges = edges[:, selected]
        candidate_scores = scores[selected]

        feasible_per_task = torch.bincount(task_index, minlength=batch.num_tasks)
        retained_per_task = torch.bincount(candidate_edges[1], minlength=batch.num_tasks)
        positive_recall = self._positive_recall(batch, candidate_edges)

        return CandidateGraph(
            edge_index=candidate_edges,
            scores=candidate_scores,
            feasible_per_task=feasible_per_task,
            retained_per_task=retained_per_task,
            positive_recall=positive_recall,
        )

    @staticmethod
    def _positive_recall(batch: MonthlyBatch, candidate_edges: Tensor) -> float:
        if batch.positive_edge_index.shape[1] == 0:
            return 1.0
        n = batch.num_tasks
        positives = torch.unique(batch.positive_edge_index[0] * n + batch.positive_edge_index[1])
        retained = torch.unique(candidate_edges[0] * n + candidate_edges[1])
        hit = torch.isin(positives, retained).sum().item()
        return float(hit / positives.numel())
