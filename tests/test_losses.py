import torch

from pfga.config import LossSection
from pfga.losses import composite_loss, gini_coefficient
from tests.helpers import make_batch


def test_gini_known_values() -> None:
    assert torch.isclose(gini_coefficient(torch.ones(3)), torch.tensor(0.0))
    expected = torch.tensor(2.0 / 3.0)
    assert torch.isclose(gini_coefficient(torch.tensor([0.0, 0.0, 1.0])), expected, atol=1e-6)


def test_composite_loss_is_finite_and_differentiable() -> None:
    batch = make_batch()
    edges = batch.feasible_edge_index
    logits = torch.zeros(edges.shape[1], requires_grad=True)
    labels = batch.labels_for_edges(edges)
    loss = composite_loss(logits, labels, batch, edges, LossSection())
    assert torch.isfinite(loss.total)
    loss.total.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
