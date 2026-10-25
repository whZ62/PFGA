import pytest
import torch
from pfga.config import PFGAConfig, ModelSection, LossSection, AssignmentSection
from pfga.losses import soft_allocation, composite_loss, gini_coefficient
from pfga.model import PFGA, FairnessAntibodyMemory
from pfga.candidates import CandidateGraphMaterializer
from pfga.assignment import constrained_greedy_assignment
from pfga.metrics import ReplayMetrics
from pfga.replay import StreamData
from tests.helpers import make_batch


def test_equation8_and9_exact_values():
    memory = FairnessAntibodyMemory(3, 0.9)
    memory.update(torch.arange(3), torch.tensor([0.0, 1.0, 2.0]))
    assert torch.equal(memory.state, torch.zeros(3))
    memory.begin_step(torch.arange(3))
    assert torch.allclose(memory.deficit(), torch.tensor([0.5, 0.0, -0.5]))
    assert torch.allclose(memory.state, torch.tensor([0.05, 0.0, -0.05]))
    memory.begin_step(torch.arange(3))
    assert torch.allclose(memory.state, torch.tensor([0.095, 0.0, -0.095]))
    memory.reset()
    assert memory.cumulative_utility.sum() == 0


def test_disabling_memory_preserves_cumulative_deficit():
    memory = FairnessAntibodyMemory(3, 0.9, False)
    memory.update(torch.arange(3), torch.tensor([0.0, 1.0, 2.0]))
    memory.begin_step(torch.arange(3))
    assert memory.state.sum() == 0 and memory.deficit()[0] > 0.0


def test_real_allocations_only_update_memory():
    b = make_batch()
    model = PFGA(2, 2, 3, ModelSection(hidden_dim=8, attention_heads=2))
    e = torch.tensor([[1], [0]])
    model.update_fairness_memory(b, e)
    assert torch.allclose(model.fairness_memory.cumulative_utility, torch.tensor([0.0, 0.7, 0.0]))


def test_soft_constraints_and_threshold_gradient():
    b = make_batch()
    b.worker_capacity = torch.tensor([0, 1, 1])
    e = b.feasible_edge_index
    logit_values = torch.tensor([5.0, 5.0, -10.0, 5.0, 5.0, 5.0], requires_grad=True)
    mass = soft_allocation(logit_values, b, e, 0.4)
    assert (torch.zeros(2).index_add(0, e[1], mass) <= 1 + 1e-6).all()
    assert (torch.zeros(3).index_add(0, e[0], mass) <= b.worker_capacity + 1e-6).all()
    assert mass[2] == 0
    mass.sum().backward()
    assert torch.isfinite(logit_values.grad).all()


def test_weighted_bce_normalization():
    b = make_batch()
    e = b.feasible_edge_index
    logit_values = torch.zeros(6, requires_grad=True)
    loss = composite_loss(logit_values, b.labels_for_edges(e), b, e, LossSection())
    assert loss.positive_weight == 2
    assert loss.prediction.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())


def test_fairness_loss_contains_history():
    b = make_batch()
    e = b.feasible_edge_index
    logit_values = torch.full((6,), -20.0, requires_grad=True)
    h = torch.tensor([0.0, 0.0, 2.0])
    loss = composite_loss(logit_values, b.labels_for_edges(e), b, e, LossSection(), h)
    assert torch.isclose(loss.fairness, gini_coefficient(h))


def test_empty_candidates_finite_backward():
    b = make_batch()
    b.feasible_edge_index = torch.empty((2, 0), dtype=torch.long)
    c = CandidateGraphMaterializer(PFGAConfig().candidates)(b, torch.zeros(3))
    model = PFGA(2, 2, 3, ModelSection(hidden_dim=8, attention_heads=2))
    logit_values = model(b, c.edge_index)
    loss = composite_loss(logit_values, logit_values.detach(), b, c.edge_index, LossSection())
    assert torch.isfinite(loss.total)
    loss.total.backward()


def test_cumulative_metric_denominator_and_reward_gini():
    b = make_batch()
    e = torch.tensor([[0], [0]])
    a = constrained_greedy_assignment(b, e, torch.tensor([10.0]), AssignmentSection())
    metrics = ReplayMetrics(3)
    metrics.add(b, a, torch.tensor([1.0, 0.0]), torch.tensor([0.9, 0.1]), [1.0, 1.0], 0.2)
    x = metrics.result(0.4)
    assert x["utility"] == 0.5 and x["assignment_rate"] == 0.5 and x["acc"] == 1
    metrics.add(b, a, torch.tensor([]), torch.tensor([]), [1.0, 1.0], 0.2)
    y = metrics.result(0.4)
    assert y["utility"] == 0.5 and y["total_utility"] == 2 and y["cost"] == 2


def test_invalid_old_allocation_config_rejected():
    c = PFGAConfig()
    c.assignment.force_assign = True
    with pytest.raises(ValueError, match="fixed"):
        c.validate()


def stream_fixture(tmp_path):
    c = PFGAConfig()
    c.experiment.device = "cpu"
    common = {
        "worker_features": torch.zeros((2, 6)),
        "initial_locations": torch.tensor([[0.0, 0.0], [0.0, 0.001]], dtype=torch.float64),
        "centers": torch.tensor([12.0, 12.0]),
        "capacities": torch.tensor([2, 2]),
        "affinity": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "retrieval_profile": torch.eye(2),
        "social_edge_index": torch.empty((2, 0), dtype=torch.long),
        "num_categories": 2,
        "speed_kmh": 5.0,
        "window_hours": 12.0,
        "deadline_hours": 12.0,
    }
    tasks = [
        {"id": 0, "time": 12.0, "worker": 0, "category": 0, "location": [0.0, 0.01], "reward": 1.0},
        {
            "id": 1,
            "time": 12.0,
            "worker": 1,
            "category": 1,
            "location": [0.0, 0.015],
            "reward": 1.0,
        },
        {"id": 2, "time": 36.0, "worker": 0, "category": 0, "location": [0.0, 0.01], "reward": 1.0},
    ]
    path = tmp_path / "stream.pt"
    torch.save({"common": common, "tasks": tasks, "split": "test"}, path)
    return StreamData(path, c)


def test_replay_busy_worker_window_reset_and_no_future_task(tmp_path):
    data = stream_fixture(tmp_path)
    r = data.replay()
    b = next(r)
    assert b.num_tasks == 2 and len(r.new_tasks) == 2
    assert r.accept(0, 0)
    assert not r.accept(0, 1)  # busy until first travel completes
    assert r.remaining[0] == 1
    assert r.accept(1, 1)
    b2 = next(r)
    assert r.time == 36.0 and b2.num_tasks == 1
    assert r.remaining.tolist() == [2, 2]


def test_replay_does_not_force_infeasible_label_into_candidates(tmp_path):
    data = stream_fixture(tmp_path)
    data.tasks[0]["location"] = [80.0, 150.0]
    r = data.replay()
    b = next(r)
    assert not (b.feasible_edge_index[1] == 0).any()


def test_supervision_fixed_before_top_k(tmp_path):
    data = stream_fixture(tmp_path)
    r = data.replay()
    b = next(r)
    before = b.supervision_edge_index.clone()
    conf = PFGAConfig().candidates
    conf.top_k = 1
    _ = CandidateGraphMaterializer(conf)(b, torch.tensor([0.0, 10.0]))
    assert torch.equal(before, b.supervision_edge_index)


def test_backprop_all_compensation_variants():
    b = make_batch()
    for mode in ["representation", "deficit_score", "memory_score", "joint_score", "concat"]:
        conf = ModelSection(hidden_dim=8, attention_heads=2, compensation=mode)
        model = PFGA(2, 2, 3, conf)
        model.fairness_memory.update(torch.arange(3), torch.tensor([0.0, 1.0, 2.0]))
        model.fairness_memory.begin_step(b.worker_ids)
        logits = model(b, b.feasible_edge_index)
        logits.sum().backward()
        assert any(p.grad is not None for p in model.parameters())


def test_prepare_equation33_uses_train_only(tmp_path):
    import pandas as pd
    from pfga.prepare import prepare
    from pfga.config import DateRange

    config = PFGAConfig()
    config.data.directory = str(tmp_path / "prepared")
    config.data.train = DateRange("2020-01", "2020-01")
    config.data.validation = DateRange("2020-02", "2020-02")
    config.data.test = DateRange("2020-03", "2020-03")
    rows = [
        ("a", "x", "2020-01-01"),
        ("a", "x", "2020-01-02"),
        ("b", "y", "2020-01-02"),
        ("a", "x", "2020-02-01"),
        ("b", "y", "2020-02-01"),
        ("a", "y", "2020-03-01"),
        ("b", "x", "2020-03-01"),
    ]
    df = pd.DataFrame(
        [
            {
                "worker_id": w,
                "poi_id": k,
                "category": k,
                "timestamp": t + "T12:00:00Z",
                "latitude": 0.0,
                "longitude": 0.001,
            }
            for w, k, t in rows
        ]
    )
    path = tmp_path / "input.csv"
    df.to_csv(path, index=False)
    prepare(config, path)
    from pfga.data import _torch_load

    common = _torch_load(tmp_path / "prepared" / "train.pt")["common"]
    assert torch.allclose(common["affinity"][:, :2], torch.tensor([[1.0, 0.125], [0.5, 1.0]]))
    df.loc[df.timestamp.str.contains("2020-03"), "category"] = "new_test_only_category"
    df.to_csv(path, index=False)
    prepare(config, path)
    changed = _torch_load(tmp_path / "prepared" / "train.pt")["common"]
    assert torch.equal(common["affinity"], changed["affinity"])
    assert torch.equal(common["worker_features"], changed["worker_features"])


def test_end_to_end_checkpoint_and_temporal_consistency(tmp_path):
    from pfga.trainer import PFGATrainer

    data = stream_fixture(tmp_path)
    config = data.config
    config.model.hidden_dim = 8
    config.model.attention_heads = 2
    config.training.epochs = 1
    config.assignment.threshold_grid = (0.2, 0.4)
    config.training.output_directory = str(tmp_path / "out")
    trainer = PFGATrainer(config, data, data, data)
    result = trainer.train()
    assert (tmp_path / "out" / "best.pt").exists()
    for key in ["utility", "fair", "cost", "assignment_rate"]:
        assert result["test"][key] == result["test_periods"][-1][key]
    first, _, _ = trainer.replay(data)
    second, _, _ = trainer.replay(data)
    for key in ["utility", "fair", "auc", "acc", "assigned_tasks"]:
        assert first[key] == second[key]  # isolated replay resets all runtime state
