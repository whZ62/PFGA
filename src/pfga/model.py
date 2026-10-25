"""PFGA model components corresponding to manuscript Equations (4)-(15)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pfga.config import ModelSection


class FairnessAntibodyMemory(nn.Module):
    """Eq8-9: global-population deficit; EMA advances once per decision."""

    def __init__(self, num_workers: int, gamma: float, longitudinal: bool = True):
        super().__init__()
        self.gamma, self.longitudinal = gamma, longitudinal
        self.register_buffer("state", torch.zeros(num_workers))
        self.register_buffer("cumulative_utility", torch.zeros(num_workers))
        self.register_buffer("seen", torch.zeros(num_workers, dtype=torch.bool))

    def deficit(self, worker_ids=None):
        u = self.cumulative_utility
        x = (u.mean() - u) / (u.mean() + 1e-8)
        d = x / (1 + x.abs())
        return d if worker_ids is None else d[worker_ids]

    @torch.no_grad()
    def begin_step(self, worker_ids):
        d = self.deficit(worker_ids)
        if self.longitudinal:
            self.state[worker_ids] = self.gamma * self.state[worker_ids] + (1 - self.gamma) * d
        else:
            self.state[worker_ids] = 0
        self.seen[worker_ids] = True

    @torch.no_grad()
    def update(self, worker_ids, allocated_utility):
        self.cumulative_utility.index_add_(0, worker_ids, allocated_utility.detach())

    @torch.no_grad()
    def reset(self):
        self.state.zero_()
        self.cumulative_utility.zero_()
        self.seen.zero_()

    def snapshot(self):
        return {k: getattr(self, k).clone() for k in ("state", "cumulative_utility", "seen")}

    @torch.no_grad()
    def restore(self, snapshot):
        for k, v in snapshot.items():
            getattr(self, k).copy_(v)


class FairnessAntibodyGenerator(nn.Module):
    """Eq10: [current deficit, EMA memory] and bounded dimension adapter."""

    def __init__(self, hidden_dim, antibody_hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, antibody_hidden_dim),
            nn.ReLU(),
            nn.Linear(antibody_hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.adapter = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, deficit, state):
        return self.network(torch.stack((deficit, state), -1)) * torch.sigmoid(self.adapter)


class TaskEncoder(nn.Module):
    """Two-layer feature encoder (Eq. 4-5)."""

    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.encoder(features)


class SocialPersonalizationAttention(nn.Module):
    """Residual multi-head graph attention without a PyG dependency (Eq. 6-7)."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.attention_source = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attention_target = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.normalization = nn.LayerNorm(hidden_dim)
        nn.init.xavier_uniform_(self.attention_source)
        nn.init.xavier_uniform_(self.attention_target)

    def forward(self, worker_embedding: Tensor, edge_index: Tensor) -> Tensor:
        num_workers, hidden_dim = worker_embedding.shape
        if edge_index.shape[1] == 0:
            return self.normalization(worker_embedding)

        source, target = edge_index
        projected = self.projection(worker_embedding).view(num_workers, self.heads, self.head_dim)
        logits = (projected[source] * self.attention_source).sum(dim=-1) + (
            projected[target] * self.attention_target
        ).sum(dim=-1)
        logits = F.leaky_relu(logits, negative_slope=0.2)

        scatter_index = target.unsqueeze(-1).expand(-1, self.heads)
        maximum = torch.full(
            (num_workers, self.heads),
            -torch.inf,
            dtype=logits.dtype,
            device=logits.device,
        )
        maximum.scatter_reduce_(0, scatter_index, logits, reduce="amax", include_self=True)
        exponent = torch.exp(logits - maximum[target])
        denominator = torch.zeros_like(maximum)
        denominator.index_add_(0, target, exponent)
        attention = exponent / denominator[target].clamp_min(1e-8)

        messages = (
            attention.unsqueeze(-1)
            * self.value_projection(worker_embedding).view(num_workers, self.heads, self.head_dim)[
                source
            ]
        )
        aggregate = torch.zeros(
            (num_workers, self.heads, self.head_dim),
            dtype=worker_embedding.dtype,
            device=worker_embedding.device,
        )
        aggregate.index_add_(0, target, messages)
        social = torch.sigmoid(aggregate.reshape(num_workers, hidden_dim) + self.bias)
        return self.normalization(worker_embedding + self.dropout(social))


class PFGA(nn.Module):
    """Personalized Fair Graph Assigner."""

    def __init__(
        self,
        worker_feature_dim: int,
        task_feature_dim: int,
        num_global_workers: int,
        config: ModelSection,
    ):
        super().__init__()
        self.config = config
        self.fairness_memory = FairnessAntibodyMemory(
            num_workers=num_global_workers,
            gamma=config.fairness_gamma,
            longitudinal=config.use_fairness_memory,
        )
        self.antibody_generator = FairnessAntibodyGenerator(
            hidden_dim=config.hidden_dim,
            antibody_hidden_dim=config.antibody_hidden_dim,
        )
        self.worker_encoder = TaskEncoder(worker_feature_dim, config.hidden_dim, config.dropout)
        self.task_encoder = TaskEncoder(task_feature_dim, config.hidden_dim, config.dropout)
        self.social_aggregation = SocialPersonalizationAttention(
            config.hidden_dim, config.attention_heads, config.dropout
        )
        self.gate = nn.Linear(2 * config.hidden_dim, config.hidden_dim)
        extra = 2 if config.compensation == "concat" else 0
        self.assignment_predictor = nn.Sequential(
            nn.Linear(3 * config.hidden_dim + extra, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def encode(self, batch):
        worker = self.worker_encoder(batch.worker_features)
        if self.config.use_social_aggregation:
            worker = self.social_aggregation(worker, batch.social_edge_index)
        if self.config.use_antibody_injection and self.config.compensation == "representation":
            antibody = self.antibody_generator(
                self.fairness_memory.deficit(batch.worker_ids),
                self.fairness_memory.state[batch.worker_ids],
            )
            gate = torch.sigmoid(self.gate(torch.cat((worker, antibody), -1)))
            worker = (1 - gate) * worker + gate * antibody
        return worker, self.task_encoder(batch.task_features)

    def forward(self, batch, candidate_edge_index):
        worker, task = self.encode(batch)
        source, target = candidate_edge_index
        parts = [worker[source], task[target], worker[source] * task[target]]
        deficit = self.fairness_memory.deficit(batch.worker_ids)[source]
        state = self.fairness_memory.state[batch.worker_ids][source]
        if self.config.compensation == "concat":
            parts.append(torch.stack((deficit, state), -1))
        score = self.assignment_predictor(torch.cat(parts, -1)).squeeze(-1)
        if self.config.compensation in {"deficit_score", "joint_score"}:
            score = score + self.config.deficit_score_weight * deficit
        if self.config.compensation in {"memory_score", "joint_score"}:
            score = score + self.config.memory_score_weight * state
        return score

    @torch.no_grad()
    def update_fairness_memory(self, batch, assigned_edge_index):
        # Eq15: realized allocation utility, NEVER sigmoid-weighted candidate utility.
        income = torch.zeros(batch.num_workers, device=batch.device)
        income.index_add_(0, assigned_edge_index[0], batch.edge_benefit(assigned_edge_index))
        self.fairness_memory.update(batch.worker_ids, income)
