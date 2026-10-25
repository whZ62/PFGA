import math

import torch

from pfga.assignment import adaptive_threshold, constrained_greedy_assignment
from pfga.config import AssignmentSection
from tests.helpers import make_batch


def test_adaptive_threshold_matches_equation() -> None:
    expected = 0.4 + 0.1 * math.tanh(1.0 - 2.0 / 3.0)
    assert adaptive_threshold(3, 2) == expected


def test_assignment_respects_task_and_capacity_constraints() -> None:
    batch = make_batch()
    edges = torch.tensor([[0, 0, 1, 2], [0, 1, 0, 1]], dtype=torch.long)
    logits = torch.tensor([4.0, 3.0, 2.0, 1.0])
    result = constrained_greedy_assignment(
        batch, edges, logits, AssignmentSection(force_assign=False)
    )
    assert torch.unique(result.edge_index[1]).numel() == result.edge_index.shape[1]
    worker_counts = torch.bincount(result.edge_index[0], minlength=batch.num_workers)
    assert torch.all(worker_counts <= batch.worker_capacity)
    assert result.unassigned_tasks.numel() == 0


def test_below_threshold_candidates_remain_unassigned() -> None:
    batch = make_batch()
    edges = torch.tensor([[0, 2], [0, 1]], dtype=torch.long)
    logits = torch.tensor([-10.0, -9.0])
    result = constrained_greedy_assignment(
        batch, edges, logits, AssignmentSection(force_assign=False)
    )
    assert result.edge_index.shape[1] == 0
