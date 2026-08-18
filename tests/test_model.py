import torch

from pfga.config import ModelSection
from pfga.model import FairnessAntibodyMemory, PFGA
from tests.helpers import make_batch


def test_fairness_memory_marks_disadvantaged_worker_positive() -> None:
    memory = FairnessAntibodyMemory(num_workers=3, gamma=0.0)
    memory.update(torch.arange(3), torch.tensor([0.0, 2.0, 1.0]))
    assert memory.state[0] > 0
    assert memory.state[1] < 0


def test_pfga_scores_only_requested_edges() -> None:
    batch = make_batch()
    config = ModelSection(
        hidden_dim=8,
        attention_heads=2,
        dropout=0.0,
        antibody_hidden_dim=4,
        fairness_gamma=0.9,
    )
    model = PFGA(2, 2, 3, config)
    edges = torch.tensor([[0, 2], [0, 1]], dtype=torch.long)
    logits = model(batch, edges)
    assert logits.shape == (2,)
    model.update_fairness_memory(batch, edges, torch.sigmoid(logits))
    assert model.fairness_memory.cumulative_utility.sum() > 0

