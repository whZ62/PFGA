import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class FairnessAntibodyGenerator(nn.Module):
    """Fairness Antibody Generator"""

    def __init__(self, feature_dim):
        super().__init__()
        # Antibody generation network
        self.antibody_net = nn.Sequential(
            nn.Linear(1, 32),  # Input historical fairness score
            nn.LeakyReLU(0.2),
            nn.Linear(32, feature_dim),
            nn.Tanh()
        )

        # Antibody adapter
        self.adapter = nn.Parameter(torch.randn(feature_dim))

    def forward(self, historical_fairness):
        """
        Generate fairness antibody vectors
        historical_fairness: historical fairness score ∈ [0,1] (low score indicates disadvantage)
        """
        base_antibody = self.antibody_net(historical_fairness.unsqueeze(1))
        return base_antibody * self.adapter


class AntibodyInjection(nn.Module):
    """Antibody Injection Mechanism"""

    def __init__(self, feature_dim):
        super().__init__()
        # Gate controller
        self.gate_controller = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, features, antibody):
        """
        Antibody injection: features = [1 - gate] * features + gate * antibody
        """
        gate = self.gate_controller(torch.cat([features, antibody], dim=1))
        return (1 - gate) * features + gate * antibody


class FairnessMemory(nn.Module):
    """Fairness Memory Network"""

    def __init__(self, num_workers):
        super().__init__()
        # Fairness state memory
        self.register_buffer('fairness_state', torch.ones(num_workers))

    def update(self, worker_gini):
        """Update fairness state"""
        # Calculate current Gini coefficient
        current_gini = worker_gini

        # Exponential decay update
        self.fairness_state = 0.9 * self.fairness_state + 0.1 * current_gini

    def get_state(self):
        """Get historical fairness state"""
        return self.fairness_state.clone()


class FairGAT(nn.Module):
    """Enhanced Fairness-Aware Graph Attention Network (Integrated Antibody Mechanism)"""

    def __init__(self, worker_feat_dim, task_feat_dim, hidden_dim, num_workers, num_heads=4, dropout=0.2):
        super(FairGAT, self).__init__()
        self.worker_feat_dim = worker_feat_dim
        self.task_feat_dim = task_feat_dim

        # Fairness components
        self.fairness_memory = FairnessMemory(num_workers)
        self.antibody_gen = FairnessAntibodyGenerator(hidden_dim)
        self.antibody_injector = AntibodyInjection(hidden_dim)

        # Replace original worker encoder
        self.worker_encoder = nn.Sequential(
            nn.Linear(worker_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Task feature encoder
        self.task_encoder = nn.Sequential(
            nn.Linear(task_feat_dim, hidden_dim * 6),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 6, hidden_dim * num_heads),
            nn.ReLU()
        )

        # Graph Attention Layer
        self.gat = GATConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            dropout=dropout,
            concat=True
        )

        # Task assignment prediction layer
        self.allocation_layer = nn.Sequential(
            nn.Linear(hidden_dim * num_heads * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, data, current_util=None):
        # Update fairness memory
        if current_util is not None:
            self.fairness_memory.update(current_util)

        # Get historical fairness state
        hist_fairness = self.fairness_memory.get_state()

        # Basic worker feature encoding
        base_worker_emb = self.worker_encoder(data.worker_feat)

        # Generate antibodies
        antibodies = self.antibody_gen(hist_fairness)

        # Inject antibodies
        worker_emb = self.antibody_injector(base_worker_emb, antibodies)

        # Encode task features
        task_emb = self.task_encoder(data.task_feat)

        # Graph attention propagation
        worker_emb_gat = self.gat(worker_emb, data.worker_worker_edge_index)

        # Task assignment prediction
        combined_emb = expand_and_concat(worker_emb_gat, task_emb)
        allocation_scores = self.allocation_layer(combined_emb).squeeze()

        return allocation_scores

    # def adversarial_training_step(self, data, optimizer, criterion):
    #     """Antibody adversarial training strategy"""
    #     # Standard forward pass
    #     outputs = self(data)
    #     loss_standard = criterion(outputs, data.y)
    #
    #     # Generate adversarial antibodies
    #     with torch.no_grad():
    #         # Invert fairness state (1 - fairness)
    #         adv_fairness = 1 - self.fairness_memory.get_state()[data.worker_ids]
    #         adv_antibodies = self.antibody_gen(adv_fairness)
    #
    #     # Temporarily save original antibody generator
    #     original_antibody_gen = self.antibody_gen
    #
    #     # Adversarial forward pass
    #     self.antibody_gen = lambda x: adv_antibodies
    #     adv_outputs = self(data)
    #     loss_adv = criterion(adv_outputs, data.y)
    #
    #     # Restore original antibody generator
    #     self.antibody_gen = original_antibody_gen
    #
    #     # Adversarial training loss
    #     total_loss = loss_standard + 0.3 * loss_adv
    #
    #     # Optimization
    #     optimizer.zero_grad()
    #     total_loss.backward()
    #     optimizer.step()
    #
    #     return total_loss.item()


class FocalLoss(nn.Module):
    """Focal Loss to address class imbalance"""

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevent gradient explosion
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


def expand_and_concat(worker_emb, task_emb):
    """
    Efficiently expand and concatenate features
    worker_emb: [num_workers, hidden_dim]
    task_emb: [num_tasks, hidden_dim]
    Return: [num_workers * num_tasks, 2 * hidden_dim]
    """
    num_workers = worker_emb.size(0)
    num_tasks = task_emb.size(0)
    hidden_dim = worker_emb.size(1)

    # Expand worker features (no actual data copy)
    worker_expanded = worker_emb.unsqueeze(1)  # [num_workers, 1, hidden_dim]
    worker_expanded = worker_expanded.expand(-1, num_tasks, -1)  # [num_workers, num_tasks, hidden_dim]
    worker_expanded = worker_expanded.reshape(num_workers * num_tasks,
                                              hidden_dim)  # [num_workers*num_tasks, hidden_dim]

    # Expand task features (no actual data copy)
    task_expanded = task_emb.unsqueeze(0)  # [1, num_tasks, hidden_dim]
    task_expanded = task_expanded.expand(num_workers, -1, -1)  # [num_workers, num_tasks, hidden_dim]
    task_expanded = task_expanded.reshape(num_workers * num_tasks, hidden_dim)  # [num_workers*num_tasks, hidden_dim]

    # Concatenate features
    combined_features = torch.cat([worker_expanded, task_expanded], dim=1)  # [num_workers*num_tasks, 2*hidden_dim]

    return combined_features


def calculate_fairness_loss(allocation_scores, data, criterion):
    """
    Calculate comprehensive loss combining benefit, cost, and fairness (with normalization factors)
    """
    # Reshape allocation probabilities into matrix form
    allocation_matrix = allocation_scores.view(data.worker_feat.shape[0], data.task_feat.shape[0])
    prob_matrix = torch.sigmoid(allocation_matrix)

    # 1. Allocation loss (prediction accuracy)
    allocation_loss = criterion(allocation_scores, data.y)
    # allocation_loss = F.binary_cross_entropy_with_logits(
    #     allocation_scores, data.y
    # )

    # 2. Benefit maximization loss (normalized)
    max_benefit = torch.max(data.benefit_matrix).item()
    total_benefit = prob_matrix * (data.benefit_matrix / max(max_benefit, 1e-8))
    benefit_loss = -torch.mean(total_benefit)  # Normalized

    # 3. Cost minimization loss (normalized)
    max_cost = torch.max(data.cost_matrix).item()
    total_cost = prob_matrix * (data.cost_matrix / max(max_cost, 1e-8))
    cost_loss = torch.mean(total_cost)   # Normalized

    # 4. Fairness loss (Income-based Gini coefficient)
    worker_income = torch.sum(prob_matrix * data.benefit_matrix, dim=1)
    gini = calculate_gini_coefficient(worker_income)

    return allocation_loss, benefit_loss, cost_loss, gini

# def calculate_gini_coefficient(assignment_probs):
#     """Calculate Gini coefficient (Fairness metric)"""
#     # Calculate expected task load for each worker
#     worker_load = assignment_probs.sum(dim=1)
#
#     # Sort task loads
#     sorted_load, _ = torch.sort(worker_load)
#
#     # Calculate Gini coefficient
#     n = sorted_load.size(0)
#     index = torch.arange(1, n + 1, device=sorted_load.device)
#     gini = (torch.sum((2 * index - n - 1) * sorted_load)) / (n * torch.sum(sorted_load))
#
#     return gini

# def calculate_gini_coefficient(assignment_probs):
#     """Corrected Gini coefficient calculation"""
#     # Ensure non-negative
#     assignment_probs = torch.clamp(assignment_probs, min=0)
#
#     # Calculate expected task load for each worker
#     worker_load = assignment_probs.sum(dim=1)
#
#     # If total task load is 0, return 0 (avoid division by zero)
#     if worker_load.sum() < 1e-7:
#         return torch.tensor(0.0, device=worker_load.device)
#
#     # Sort task loads
#     sorted_load, _ = torch.sort(worker_load)
#
#     # Calculate Gini coefficient
#     n = sorted_load.size(0)
#     index = torch.arange(1, n + 1, device=sorted_load.device, dtype=torch.float)
#     numerator = torch.sum((2 * index - n - 1) * sorted_load)
#     denominator = n * torch.sum(sorted_load)
#     gini = numerator / denominator
#
#     # Ensure within 0-1 range
#     return torch.clamp(gini, min=0, max=1)

def calculate_gini_coefficient(income):
    """Calculate Gini coefficient"""
    # Sort income
    sorted_income, _ = torch.sort(income)
    n = income.size(0)

    # Calculate cumulative sum
    cum_income = torch.cumsum(sorted_income, dim=0)
    total_income = cum_income[-1]

    # Avoid division by zero
    if total_income == 0:
        return torch.tensor(0.0)

    # Calculate Gini coefficient
    gini = (n + 1 - 2 * torch.sum(cum_income) / total_income) / n
    return gini