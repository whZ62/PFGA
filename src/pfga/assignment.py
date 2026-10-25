"""Fixed-threshold greedy allocation. Replay validates updated travel schedules."""

import math
from dataclasses import dataclass
from typing import List, Set

import torch
from torch import Tensor

from pfga.config import AssignmentSection
from pfga.data import MonthlyBatch


def adaptive_threshold(
    num_workers: int,
    num_tasks: int,
    base: float = 0.4,
    scale: float = 0.1,
) -> float:
    """Legacy helper only; not used by the current manuscript allocation."""

    if num_workers <= 0:
        raise ValueError("num_workers must be positive")
    return base + scale * math.tanh(1.0 - num_tasks / num_workers)


@dataclass
class AssignmentResult:
    edge_index: Tensor
    probabilities: Tensor
    threshold: float
    unassigned_tasks: Tensor


def constrained_greedy_assignment(
    batch: MonthlyBatch,
    candidate_edge_index: Tensor,
    logits: Tensor,
    config: AssignmentSection,
    accept=None,
) -> AssignmentResult:
    """Fixed validation threshold; callback rechecks current mobility constraints."""
    if config.force_assign or config.threshold_scale != 0:
        raise ValueError("Use fixed threshold without forced assignment for Eq14")
    probabilities = torch.sigmoid(logits)
    threshold = config.threshold_base
    worker_cpu = candidate_edge_index[0].detach().cpu().tolist()
    task_cpu = candidate_edge_index[1].detach().cpu().tolist()
    probability_cpu = probabilities.detach().cpu().tolist()
    capacity = batch.worker_capacity.detach().cpu().tolist()

    ranked = sorted(
        range(len(probability_cpu)),
        key=lambda index: (-probability_cpu[index], task_cpu[index], worker_cpu[index]),
    )
    selected: List[int] = []
    assigned_tasks: Set[int] = set()
    used = [0 for _ in range(batch.num_workers)]

    for index in ranked:
        worker = worker_cpu[index]
        task = task_cpu[index]
        if probability_cpu[index] < threshold:
            continue
        if task in assigned_tasks or used[worker] >= capacity[worker]:
            continue
        if accept is not None and not accept(worker, task):
            continue
        selected.append(index)
        assigned_tasks.add(task)
        used[worker] += 1

    if selected:
        selected_tensor = torch.tensor(
            selected, dtype=torch.long, device=candidate_edge_index.device
        )
        assigned_edges = candidate_edge_index[:, selected_tensor]
        assigned_probabilities = probabilities[selected_tensor]
    else:
        assigned_edges = torch.empty((2, 0), dtype=torch.long, device=candidate_edge_index.device)
        assigned_probabilities = torch.empty(
            (0,), dtype=probabilities.dtype, device=probabilities.device
        )
    unassigned = sorted(set(range(batch.num_tasks)) - assigned_tasks)
    unassigned_tensor = torch.tensor(
        unassigned, dtype=torch.long, device=candidate_edge_index.device
    )
    return AssignmentResult(
        edge_index=assigned_edges,
        probabilities=assigned_probabilities,
        threshold=threshold,
        unassigned_tasks=unassigned_tensor,
    )
