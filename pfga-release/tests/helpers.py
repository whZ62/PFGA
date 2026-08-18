import torch

from pfga.data import MonthlyBatch


def make_batch() -> MonthlyBatch:
    batch = MonthlyBatch(
        period="2020-01",
        worker_features=torch.tensor(
            [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]], dtype=torch.float32
        ),
        task_features=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        positive_edge_index=torch.tensor([[0, 2], [0, 1]], dtype=torch.long),
        social_edge_index=torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        ),
        feasible_edge_index=torch.tensor(
            [[0, 1, 2, 0, 1, 2], [0, 0, 0, 1, 1, 1]], dtype=torch.long
        ),
        benefit_matrix=torch.tensor(
            [[1.0, 0.2], [0.7, 0.7], [0.2, 1.0]], dtype=torch.float32
        ),
        cost_matrix=torch.tensor(
            [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]], dtype=torch.float32
        ),
        preference_matrix=torch.tensor(
            [[0.9, 0.1], [0.8, 0.8], [0.1, 0.9]], dtype=torch.float32
        ),
        slack_matrix=torch.ones((3, 2), dtype=torch.float32),
        worker_ids=torch.arange(3, dtype=torch.long),
        worker_capacity=torch.ones(3, dtype=torch.long),
    )
    batch.validate()
    return batch

