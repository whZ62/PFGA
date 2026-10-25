"""Monthly PFGA data schema, legacy adaptation, and chronological loading."""

import re
import warnings
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


PERIOD_PATTERN = re.compile(r"^\d{4}-\d{2}$")


def _read(raw: Any, names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(raw, Mapping) and name in raw:
            return raw[name]
        if hasattr(raw, name):
            return getattr(raw, name)
    return default


def _empty_edges(device: Optional[torch.device] = None) -> Tensor:
    return torch.empty((2, 0), dtype=torch.long, device=device)


def _all_pair_edges(num_workers: int, num_tasks: int, device: torch.device) -> Tensor:
    workers = torch.arange(num_workers, device=device).repeat_interleave(num_tasks)
    tasks = torch.arange(num_tasks, device=device).repeat(num_workers)
    return torch.stack((workers, tasks), dim=0)


@dataclass
class MonthlyBatch:
    """One chronological worker-task batch.

    Worker and task indices in all edge tensors are local to this batch. `worker_ids`
    maps local worker rows to stable global IDs used by longitudinal fairness memory.
    """

    period: str
    worker_features: Tensor
    task_features: Tensor
    positive_edge_index: Tensor
    social_edge_index: Tensor
    feasible_edge_index: Tensor
    benefit_matrix: Tensor
    cost_matrix: Tensor
    preference_matrix: Tensor
    slack_matrix: Tensor
    worker_ids: Tensor
    worker_capacity: Tensor
    task_rewards: Optional[Tensor] = None
    supervision_edge_index: Optional[Tensor] = None
    reference_opportunities: Optional[Tensor] = None
    distance_scale_km: float = 60.0

    @property
    def num_workers(self) -> int:
        return int(self.worker_features.shape[0])

    @property
    def num_tasks(self) -> int:
        return int(self.task_features.shape[0])

    @property
    def device(self) -> torch.device:
        return self.worker_features.device

    def to(self, device: torch.device) -> "MonthlyBatch":
        values: Dict[str, Any] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if isinstance(value, Tensor) else value
        return MonthlyBatch(**values)

    def labels_for_edges(self, edge_index: Tensor) -> Tensor:
        """Return binary historical labels for local worker-task edges."""

        num_tasks = self.num_tasks
        positive_linear = self.positive_edge_index[0] * num_tasks + self.positive_edge_index[1]
        query_linear = edge_index[0] * num_tasks + edge_index[1]
        if positive_linear.numel() == 0:
            return torch.zeros(query_linear.shape[0], device=edge_index.device)
        positive_linear = torch.unique(positive_linear)
        return torch.isin(query_linear, positive_linear).to(self.worker_features.dtype)

    def edge_benefit(self, edge_index: Tensor) -> Tensor:
        return self.benefit_matrix[edge_index[0], edge_index[1]]

    def edge_cost(self, edge_index: Tensor) -> Tensor:
        return self.cost_matrix[edge_index[0], edge_index[1]]

    def validate(self) -> None:
        m, n = self.num_workers, self.num_tasks
        if m == 0 or n == 0:
            raise ValueError(f"{self.period}: workers and tasks must both be non-empty")
        if self.worker_features.ndim != 2 or self.task_features.ndim != 2:
            raise ValueError(f"{self.period}: feature tensors must be two-dimensional")
        for name in ("benefit_matrix", "cost_matrix", "preference_matrix", "slack_matrix"):
            value = getattr(self, name)
            if tuple(value.shape) != (m, n):
                raise ValueError(f"{self.period}: {name} must have shape {(m, n)}")
            if not torch.isfinite(value).all():
                raise ValueError(f"{self.period}: {name} contains non-finite values")
        if (self.benefit_matrix < 0).any() or (self.cost_matrix < 0).any():
            raise ValueError(f"{self.period}: benefits and costs must be non-negative")
        for name in ("positive_edge_index", "social_edge_index", "feasible_edge_index"):
            edge_index = getattr(self, name)
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                raise ValueError(f"{self.period}: {name} must have shape [2, E]")
            if edge_index.dtype != torch.long:
                raise TypeError(f"{self.period}: {name} must use torch.long indices")
        for name in ("positive_edge_index", "feasible_edge_index"):
            edge_index = getattr(self, name)
            if edge_index.numel() and (
                edge_index[0].min() < 0
                or edge_index[0].max() >= m
                or edge_index[1].min() < 0
                or edge_index[1].max() >= n
            ):
                raise IndexError(f"{self.period}: {name} contains out-of-range indices")
        social = self.social_edge_index
        if social.numel() and (social.min() < 0 or social.max() >= m):
            raise IndexError(f"{self.period}: social_edge_index contains out-of-range indices")
        if tuple(self.worker_ids.shape) != (m,) or self.worker_ids.dtype != torch.long:
            raise ValueError(f"{self.period}: worker_ids must be a torch.long vector of length {m}")
        if self.worker_ids.min() < 0 or torch.unique(self.worker_ids).numel() != m:
            raise ValueError(f"{self.period}: worker_ids must be unique and non-negative")
        if tuple(self.worker_capacity.shape) != (m,):
            raise ValueError(f"{self.period}: worker_capacity must have length {m}")
        if self.task_rewards is not None:
            if (
                self.task_rewards.shape != (n,)
                or not torch.isfinite(self.task_rewards).all()
                or (self.task_rewards <= 0).any()
            ):
                raise ValueError("task_rewards must be a finite positive vector")
            if (self.benefit_matrix > self.task_rewards[None, :] + 1e-6).any():
                raise ValueError("preference utility must not exceed task reward")
        if (self.worker_capacity < 0).any():
            raise ValueError(f"{self.period}: worker_capacity must be non-negative")


def _derive_preference_matrix(raw: Any, m: int, n: int, device: torch.device) -> Tensor:
    explicit = _read(raw, ("preference_matrix", "preference_similarity"))
    if explicit is not None:
        return torch.as_tensor(explicit, dtype=torch.float32, device=device)
    worker_preferences = _read(raw, ("worker_preferences", "worker_preference"))
    task_categories = _read(raw, ("task_categories", "task_category"))
    if worker_preferences is None or task_categories is None:
        return torch.zeros((m, n), dtype=torch.float32, device=device)
    workers = F.normalize(
        torch.as_tensor(worker_preferences, dtype=torch.float32, device=device), dim=1
    )
    tasks = F.normalize(torch.as_tensor(task_categories, dtype=torch.float32, device=device), dim=1)
    if workers.shape[1] != tasks.shape[1]:
        raise ValueError("worker preference and task category dimensions must match")
    return workers @ tasks.T


def _derive_feasibility_and_slack(
    raw: Any,
    cost_matrix: Tensor,
    strict: bool,
) -> tuple:
    m, n = cost_matrix.shape
    device = cost_matrix.device
    explicit_edges = _read(raw, ("feasible_edge_index", "candidate_feasible_edge_index"))
    explicit_mask = _read(raw, ("feasible_matrix", "feasible_mask"))
    explicit_slack = _read(raw, ("slack_matrix", "temporal_slack"))

    if explicit_edges is not None:
        feasible_edges = torch.as_tensor(explicit_edges, dtype=torch.long, device=device)
    elif explicit_mask is not None:
        mask = torch.as_tensor(explicit_mask, dtype=torch.bool, device=device)
        if tuple(mask.shape) != (m, n):
            raise ValueError(f"feasible_mask must have shape {(m, n)}")
        feasible_edges = mask.nonzero(as_tuple=False).T.contiguous()
    else:
        available = _read(raw, ("worker_available_time", "worker_available_times"))
        task_time = _read(raw, ("task_time", "task_times"))
        speed = _read(raw, ("worker_speed", "worker_speeds"))
        if available is not None and task_time is not None and speed is not None:
            available_t = torch.as_tensor(available, dtype=torch.float32, device=device).view(m, 1)
            task_t = torch.as_tensor(task_time, dtype=torch.float32, device=device).view(1, n)
            speed_t = torch.as_tensor(speed, dtype=torch.float32, device=device).view(m, 1)
            time_budget = (task_t - available_t).clamp_min(0.0)
            mask = (available_t <= task_t) & (cost_matrix <= speed_t * time_budget)
            feasible_edges = mask.nonzero(as_tuple=False).T.contiguous()
        elif strict:
            raise ValueError(
                "paper-schema data require feasible_edge_index/feasible_mask or "
                "worker_available_time, task_time, and worker_speed"
            )
        else:
            warnings.warn(
                "No feasibility fields found; treating all worker-task pairs as feasible.",
                RuntimeWarning,
                stacklevel=2,
            )
            feasible_edges = _all_pair_edges(m, n, device)

    if explicit_slack is not None:
        slack = torch.as_tensor(explicit_slack, dtype=torch.float32, device=device)
    else:
        available = _read(raw, ("worker_available_time", "worker_available_times"))
        task_time = _read(raw, ("task_time", "task_times"))
        speed = _read(raw, ("worker_speed", "worker_speeds"))
        if available is not None and task_time is not None and speed is not None:
            available_t = torch.as_tensor(available, dtype=torch.float32, device=device).view(m, 1)
            task_t = torch.as_tensor(task_time, dtype=torch.float32, device=device).view(1, n)
            speed_t = torch.as_tensor(speed, dtype=torch.float32, device=device).view(m, 1)
            reach = speed_t * (task_t - available_t).clamp_min(0.0)
            slack = (1.0 - cost_matrix / (reach + 1e-8)).clamp(0.0, 1.0)
        else:
            slack = torch.zeros((m, n), dtype=torch.float32, device=device)
    return feasible_edges, slack


def adapt_monthly_batch(raw: Any, period: str, strict: bool = True) -> MonthlyBatch:
    """Adapt a tensor mapping or legacy PyG-like object to the release schema."""

    worker_features = _read(raw, ("worker_features", "worker_feat"))
    task_features = _read(raw, ("task_features", "task_feat"))
    benefit_matrix = _read(raw, ("benefit_matrix", "utility_matrix"))
    cost_matrix = _read(raw, ("cost_matrix", "distance_matrix"))
    required = (worker_features, task_features, benefit_matrix, cost_matrix)
    if any(value is None for value in required):
        raise ValueError(f"{period}: worker/task features and benefit/cost matrices are required")

    workers = torch.as_tensor(worker_features, dtype=torch.float32)
    tasks = torch.as_tensor(task_features, dtype=torch.float32, device=workers.device)
    benefits = torch.as_tensor(benefit_matrix, dtype=torch.float32, device=workers.device)
    costs = torch.as_tensor(cost_matrix, dtype=torch.float32, device=workers.device)
    m, n = workers.shape[0], tasks.shape[0]

    positive = _read(raw, ("positive_edge_index", "worker_task_edge_index"))
    if positive is None:
        dense_labels = _read(raw, ("labels", "y"))
        if dense_labels is None:
            raise ValueError(f"{period}: positive_edge_index or dense labels are required")
        labels = torch.as_tensor(dense_labels, device=workers.device).reshape(m, n)
        positive_edges = (labels > 0).nonzero(as_tuple=False).T.contiguous()
    else:
        positive_edges = torch.as_tensor(positive, dtype=torch.long, device=workers.device)

    social = _read(raw, ("social_edge_index", "worker_worker_edge_index"))
    social_edges = (
        _empty_edges(workers.device)
        if social is None
        else torch.as_tensor(social, dtype=torch.long, device=workers.device)
    )

    global_ids = _read(raw, ("worker_ids", "global_worker_ids"))
    if global_ids is None:
        if strict:
            raise ValueError(
                f"{period}: stable worker_ids are required for longitudinal fairness memory"
            )
        warnings.warn(
            f"{period}: worker_ids missing; assuming stable row order across months.",
            RuntimeWarning,
            stacklevel=2,
        )
        worker_ids = torch.arange(m, dtype=torch.long, device=workers.device)
    else:
        worker_ids = torch.as_tensor(global_ids, dtype=torch.long, device=workers.device)

    capacity = _read(raw, ("worker_capacity", "worker_capacities", "q"))
    worker_capacity = (
        torch.ones(m, dtype=torch.long, device=workers.device)
        if capacity is None
        else torch.as_tensor(capacity, dtype=torch.long, device=workers.device)
    )
    preference = _derive_preference_matrix(raw, m, n, workers.device)
    feasible_edges, slack = _derive_feasibility_and_slack(raw, costs, strict)

    task_rewards = _read(raw, ("task_rewards", "task_reward"))
    if task_rewards is None and strict:
        raise ValueError(
            "New Utility metric requires task_rewards; cannot infer rewards from benefit_matrix"
        )
    batch = MonthlyBatch(
        period=period,
        worker_features=workers,
        task_features=tasks,
        positive_edge_index=positive_edges,
        social_edge_index=social_edges,
        feasible_edge_index=feasible_edges,
        benefit_matrix=benefits,
        cost_matrix=costs,
        preference_matrix=preference,
        slack_matrix=slack,
        worker_ids=worker_ids,
        worker_capacity=worker_capacity,
        task_rewards=None
        if task_rewards is None
        else torch.as_tensor(task_rewards, dtype=torch.float32),
    )
    batch.validate()
    return batch


def available_periods(directory: str) -> List[str]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    return sorted(
        child.name
        for child in root.iterdir()
        if child.is_dir() and PERIOD_PATTERN.fullmatch(child.name)
    )


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_periods(
    directory: str,
    start: str,
    end: str,
    strict: bool = True,
) -> List[MonthlyBatch]:
    periods = [period for period in available_periods(directory) if start <= period <= end]
    batches: List[MonthlyBatch] = []
    for period in periods:
        path = Path(directory) / period / "graph_data.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing monthly file: {path}")
        batches.append(adapt_monthly_batch(_torch_load(path), period=period, strict=strict))
    if not batches:
        raise ValueError(f"No monthly batches found in {directory} for {start} to {end}")
    return batches


def infer_global_num_workers(splits: Iterable[Iterable[MonthlyBatch]]) -> int:
    largest = -1
    for batches in splits:
        for batch in batches:
            largest = max(largest, int(batch.worker_ids.max().item()))
    if largest < 0:
        raise ValueError("Cannot infer worker count from empty splits")
    return largest + 1


def validate_feature_dimensions(splits: Iterable[Iterable[MonthlyBatch]]) -> tuple:
    worker_dim: Optional[int] = None
    task_dim: Optional[int] = None
    for batches in splits:
        for batch in batches:
            current_worker = int(batch.worker_features.shape[1])
            current_task = int(batch.task_features.shape[1])
            worker_dim = current_worker if worker_dim is None else worker_dim
            task_dim = current_task if task_dim is None else task_dim
            if current_worker != worker_dim or current_task != task_dim:
                raise ValueError("Feature dimensions must remain constant across monthly batches")
    if worker_dim is None or task_dim is None:
        raise ValueError("No batches supplied")
    return worker_dim, task_dim
