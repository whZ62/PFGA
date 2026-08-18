"""PFGA model components corresponding to paper Equations (1)-(10)."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from pfga.config import ModelSection
from pfga.data import MonthlyBatch


class FairnessAntibodyMemory(nn.Module):
    """Longitudinal utility memory and smoothed pairwise fairness state.

    The paper's Eq. (1)-(3) has incompatible published tensor shapes. This executable
    form retains the stated behavior: compare every active worker with peers, apply a
    bounded signed transform, Frobenius-normalize, and exponentially smooth with gamma.
    See docs/PAPER_ALIGNMENT.md for the exact resolution.
    """

    def __init__(self, num_workers: int, gamma: float, longitudinal: bool = True):
        super().__init__()
        self.gamma = gamma
        self.longitudinal = longitudinal
        self.register_buffer("state", torch.zeros(num_workers))
        self.register_buffer("cumulative_utility", torch.zeros(num_workers))
        self.register_buffer("seen", torch.zeros(num_workers, dtype=torch.bool))

    @torch.no_grad()
    def update(self, worker_ids: Tensor, current_utility: Tensor) -> None:
        current_utility = current_utility.detach().to(self.cumulative_utility.dtype)
        self.cumulative_utility[worker_ids] += current_utility
        utility = (
            self.cumulative_utility[worker_ids]
            if self.longitudinal
            else current_utility
        )
        if utility.numel() <= 1:
            raw_state = torch.zeros_like(utility)
        else:
            scale = utility.abs().mean().clamp_min(1e-8)
            # Positive values indicate that peers receive more utility than worker i.
            pairwise_gap = (utility.unsqueeze(0) - utility.unsqueeze(1)) / scale
            bounded_gap = pairwise_gap / (1.0 + pairwise_gap.abs())
            raw_state = bounded_gap.sum(dim=1) / (utility.numel() - 1)
        normalized = raw_state / raw_state.norm(p=2).clamp_min(1e-8)
        gamma = self.gamma if self.longitudinal else 0.0
        self.state[worker_ids] = gamma * self.state[worker_ids] + (1.0 - gamma) * normalized
        self.seen[worker_ids] = True

    def snapshot(self) -> Dict[str, Tensor]:
        return {
            "state": self.state.clone(),
            "cumulative_utility": self.cumulative_utility.clone(),
            "seen": self.seen.clone(),
        }

    @torch.no_grad()
    def restore(self, snapshot: Dict[str, Tensor]) -> None:
        self.state.copy_(snapshot["state"])
        self.cumulative_utility.copy_(snapshot["cumulative_utility"])
        self.seen.copy_(snapshot["seen"])


class FairnessAntibodyGenerator(nn.Module):
    """Map a smoothed worker fairness state to a hidden-dimensional antibody."""

    def __init__(self, hidden_dim: int, antibody_hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, antibody_hidden_dim),
            nn.ReLU(),
            nn.Linear(antibody_hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.adapter = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, fairness_state: Tensor) -> Tensor:
        return self.network(fairness_state.unsqueeze(-1)) * self.adapter


class AntibodyEnhancedEncoder(nn.Module):
    """Two-layer worker encoder with scalar adaptive antibody gate (Eq. 4-6)."""

    def __init__(
        self, input_dim: int, hidden_dim: int, dropout: float, use_antibody: bool = True
    ):
        super().__init__()
        self.use_antibody = use_antibody
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gate = nn.Linear(2 * hidden_dim, 1)

    def forward(self, features: Tensor, antibody: Tensor) -> Tensor:
        base = self.encoder(features)
        if not self.use_antibody:
            return base
        gate = torch.sigmoid(self.gate(torch.cat((base, antibody), dim=-1)))
        return (1.0 - gate) * base + gate * antibody


class TaskEncoder(nn.Module):
    """Independent two-layer task feature encoder (Eq. 7)."""

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
    """Residual multi-head graph attention without a PyG dependency (Eq. 8-9)."""

    def __init__(self, hidden_dim: int, heads: int, dropout: float):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
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
        projected = self.projection(worker_embedding).view(
            num_workers, self.heads, self.head_dim
        )
        logits = (
            (projected[source] * self.attention_source).sum(dim=-1)
            + (projected[target] * self.attention_target).sum(dim=-1)
        )
        logits = F.leaky_relu(logits, negative_slope=0.2)

        scatter_index = target.unsqueeze(-1).expand(-1, self.heads)
        maximum = torch.full(
            (num_workers, self.heads),
            -torch.inf,
            dtype=logits.dtype,
            device=logits.device,
        )
        maximum.scatter_reduce_(
            0, scatter_index, logits, reduce="amax", include_self=True
        )
        exponent = torch.exp(logits - maximum[target])
        denominator = torch.zeros_like(maximum)
        denominator.index_add_(0, target, exponent)
        attention = exponent / denominator[target].clamp_min(1e-8)

        messages = attention.unsqueeze(-1) * projected[source]
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
        self.worker_encoder = AntibodyEnhancedEncoder(
            worker_feature_dim,
            config.hidden_dim,
            config.dropout,
            use_antibody=config.use_antibody_injection,
        )
        self.task_encoder = TaskEncoder(task_feature_dim, config.hidden_dim, config.dropout)
        self.social_aggregation = SocialPersonalizationAttention(
            config.hidden_dim, config.attention_heads, config.dropout
        )
        # Eq. (10): theta^T [h_i^a || h_j^t]. Sigmoid is applied by loss/inference.
        self.assignment_predictor = nn.Linear(2 * config.hidden_dim, 1)

    def encode(self, batch: MonthlyBatch) -> Tuple[Tensor, Tensor]:
        fairness_state = self.fairness_memory.state[batch.worker_ids]
        antibody = self.antibody_generator(fairness_state)
        worker = self.worker_encoder(batch.worker_features, antibody)
        if self.config.use_social_aggregation:
            worker = self.social_aggregation(worker, batch.social_edge_index)
        task = self.task_encoder(batch.task_features)
        return worker, task

    def forward(self, batch: MonthlyBatch, candidate_edge_index: Tensor) -> Tensor:
        worker, task = self.encode(batch)
        source, target = candidate_edge_index
        pair_embedding = torch.cat((worker[source], task[target]), dim=-1)
        return self.assignment_predictor(pair_embedding).squeeze(-1)

    @staticmethod
    def expected_worker_utility(
        batch: MonthlyBatch,
        edge_index: Tensor,
        probabilities: Tensor,
    ) -> Tensor:
        income = torch.zeros(
            batch.num_workers,
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
        income.index_add_(0, edge_index[0], probabilities * batch.edge_benefit(edge_index))
        return income

    @torch.no_grad()
    def update_fairness_memory(
        self,
        batch: MonthlyBatch,
        edge_index: Tensor,
        probabilities: Tensor,
    ) -> None:
        income = self.expected_worker_utility(batch, edge_index, probabilities)
        self.fairness_memory.update(batch.worker_ids, income)
