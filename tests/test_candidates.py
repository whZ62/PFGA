import torch

from pfga.candidates import CandidateGraphMaterializer
from pfga.config import CandidateSection
from tests.helpers import make_batch


def test_top_k_preserves_expected_candidates() -> None:
    batch = make_batch()
    config = CandidateSection(
        top_k=1,
        preference_weight=1.0,
        distance_weight=0.0,
        fairness_weight=0.0,
        slack_weight=0.0,
    )
    candidate = CandidateGraphMaterializer(config)(batch, torch.zeros(3))
    selected = set(map(tuple, candidate.edge_index.T.tolist()))
    assert selected == {(0, 0), (2, 1)}
    assert candidate.positive_recall == 1.0
    assert candidate.edge_index.shape[1] <= batch.num_tasks * config.top_k


def test_fairness_deficit_changes_ranking() -> None:
    batch = make_batch()
    batch.preference_matrix.zero_()
    config = CandidateSection(
        top_k=1,
        preference_weight=0.0,
        distance_weight=0.0,
        fairness_weight=1.0,
        slack_weight=0.0,
    )
    history = torch.tensor([0.0, 2.0, 3.0])
    candidate = CandidateGraphMaterializer(config)(batch, history)
    assert set(candidate.edge_index[0].tolist()) == {0}
