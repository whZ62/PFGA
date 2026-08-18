import pytest

from pfga.data import adapt_monthly_batch
from tests.helpers import make_batch


def test_labels_for_edges() -> None:
    batch = make_batch()
    labels = batch.labels_for_edges(batch.feasible_edge_index)
    assert int(labels.sum().item()) == 2


def test_strict_adapter_rejects_missing_global_ids() -> None:
    batch = make_batch()
    raw = {
        "worker_features": batch.worker_features,
        "task_features": batch.task_features,
        "positive_edge_index": batch.positive_edge_index,
        "social_edge_index": batch.social_edge_index,
        "feasible_edge_index": batch.feasible_edge_index,
        "benefit_matrix": batch.benefit_matrix,
        "cost_matrix": batch.cost_matrix,
        "preference_matrix": batch.preference_matrix,
        "slack_matrix": batch.slack_matrix,
    }
    with pytest.raises(ValueError, match="worker_ids"):
        adapt_monthly_batch(raw, "2020-01", strict=True)
