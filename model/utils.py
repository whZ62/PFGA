import torch
from torch_geometric.data import Data

def move_data_to_device(data, device):
    data.worker_feat = data.worker_feat.to(device)
    data.task_feat = data.task_feat.to(device)
    data.worker_task_edge_index = data.worker_task_edge_index.to(device)
    data.worker_worker_edge_index = data.worker_worker_edge_index.to(device)
    # data.worker_nodes = data.worker_nodes.to(device)
    data.y = data.y.to(device)
    data.benefit_matrix = data.benefit_matrix.to(device)
    data.cost_matrix = data.cost_matrix.to(device)

    return data


def edge_index_to_labels(data):
    """
    Convert worker_task_edge_index to label vector y
    y: A 1D vector of length num_workers * num_tasks
        1 indicates a connection exists, 0 indicates no connection
    """
    # Get node count information
    num_workers = data.worker_feat.shape[0]
    num_tasks = data.task_feat.shape[0]

    # Create initial label vector (all zeros)
    y = torch.zeros(num_workers * num_tasks, dtype=torch.float)

    # If there are no edges, return the all-zero vector directly
    if data.worker_task_edge_index.size(1) == 0:
        return y

    # Extract edge indices
    worker_indices = data.worker_task_edge_index[0]
    task_indices = data.worker_task_edge_index[1]

    # Convert task indices to local indices (starting from 0)
    task_indices_local = task_indices

    # Calculate 1D index positions
    positions = worker_indices * num_tasks + task_indices_local

    # Set labels to 1
    y[positions] = 1

    return y


def calculate_pos_weight(train_data):
    """Calculate positive sample weights"""
    total_pos = 0
    total_samples = 0

    for data in train_data:
        total_pos += data.y.sum().item()
        total_samples += len(data.y)

    # Positive sample weight = number of negative samples / number of positive samples
    pos_weight = (total_samples - total_pos) / max(total_pos, 1)
    return torch.tensor([pos_weight], device=train_data.device)


def adaptive_loss_weight(allocation_loss, fairness_loss):
    """Adaptively adjust loss weights"""
    # Calculate loss ratios
    alloc_ratio = allocation_loss / (allocation_loss + fairness_loss)
    fair_ratio = 1 - alloc_ratio

    # Dynamically adjust lambda
    lambda_fair = max(0.01, min(1.0, fair_ratio / alloc_ratio))
    return lambda_fair


def oversample_positive(data):
    """Oversample positive samples (adapted for graph data structure)"""
    # Ensure data contains labels
    if not hasattr(data, 'y'):
        raise ValueError("Data object missing label attribute 'y'")

    # Separate positive and negative sample indices
    pos_indices = (data.y == 1).nonzero(as_tuple=True)[0]
    neg_indices = (data.y == 0).nonzero(as_tuple=True)[0]

    # If there are no positive samples, return original data directly
    if len(pos_indices) == 0:
        return data

    # Calculate oversampling ratio
    oversample_ratio = len(neg_indices) // len(pos_indices)

    # Oversample positive sample indices
    oversampled_pos = pos_indices.repeat(oversample_ratio)

    # Merge all sample indices
    all_indices = torch.cat([neg_indices, oversampled_pos])

    # ====================== Key: Create new data object ======================
    # Create new data object
    new_data = Data()

    # Copy all original attributes
    for key in data.keys:
        if key not in ['x', 'y', 'worker_task_edge_index', 'worker_worker_edge_index']:
            setattr(new_data, key, getattr(data, key))

    # Process node features
    new_data.x = data.x.clone()

    # Process labels
    new_data.y = data.y[all_indices].clone()

    # ====================== Process edge indices ======================
    # Get original edge indices
    worker_task_edges = data.worker_task_edge_index.clone()
    worker_worker_edges = data.worker_worker_edge_index.clone()

    # Create mapping table: original index -> new index
    index_mapping = {}
    for new_idx, orig_idx in enumerate(all_indices.tolist()):
        index_mapping[orig_idx] = new_idx

    # Process worker-task edges
    new_worker_task_edges = []
    for i in range(worker_task_edges.size(1)):
        src, dst = worker_task_edges[:, i]
        if src.item() in index_mapping and dst.item() in index_mapping:
            new_src = index_mapping[src.item()]
            new_dst = index_mapping[dst.item()]
            new_worker_task_edges.append([new_src, new_dst])

    # Process worker-worker edges
    new_worker_worker_edges = []
    for i in range(worker_worker_edges.size(1)):
        src, dst = worker_worker_edges[:, i]
        if src.item() in index_mapping and dst.item() in index_mapping:
            new_src = index_mapping[src.item()]
            new_dst = index_mapping[dst.item()]
            new_worker_worker_edges.append([new_src, new_dst])

    # Convert to tensor
    if new_worker_task_edges:
        new_data.worker_task_edge_index = torch.tensor(
            new_worker_task_edges, dtype=torch.long
        ).t().contiguous()
    else:
        new_data.worker_task_edge_index = torch.empty((2, 0), dtype=torch.long)

    if new_worker_worker_edges:
        new_data.worker_worker_edge_index = torch.tensor(
            new_worker_worker_edges, dtype=torch.long
        ).t().contiguous()
    else:
        new_data.worker_worker_edge_index = torch.empty((2, 0), dtype=torch.long)

    # Update node counts
    new_data.num_nodes = data.num_nodes.clone()

    # Update worker and task counts
    if hasattr(data, 'num_workers') and hasattr(data, 'num_tasks'):
        # Calculate new worker and task counts
        worker_mask = (data.node_type == 0)
        task_mask = (data.node_type == 1)

        new_worker_count = (data.node_type[worker_mask] == 0).sum().item()
        new_task_count = (data.node_type[task_mask] == 1).sum().item()

        new_data.num_workers = new_worker_count
        new_data.num_tasks = new_task_count

    return new_data


def balance_labels(allocation_scores, y):
    """
    Balance positive and negative sample labels
    Input:
        allocation_scores: Allocation score tensor (n,)
        y: Label tensor (n,)
    Output:
        balanced_scores: Balanced allocation scores
        balanced_y: Balanced labels
    """
    # Count number of positive and negative samples
    pos_indices = (y == 1).nonzero(as_tuple=True)[0]
    neg_indices = (y == 0).nonzero(as_tuple=True)[0]
    n_pos = len(pos_indices)
    n_neg = len(neg_indices)

    # If there are no positive samples, return original data directly
    if n_pos == 0:
        return allocation_scores, y

    # Calculate the replication factor needed
    replicate_factor = n_neg // n_pos

    # Replicate positive samples
    replicated_pos_indices = pos_indices.repeat(replicate_factor)

    # Merge all indices
    all_indices = torch.cat([neg_indices, pos_indices, replicated_pos_indices])

    # Create balanced scores and labels
    balanced_scores = allocation_scores[all_indices]
    balanced_y = y[all_indices]

    return balanced_scores, balanced_y