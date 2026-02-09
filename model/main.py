import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import GATConv
import numpy as np
import pandas as pd
import os
import json
import time
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
import argparse
from model import FairGAT, calculate_gini_coefficient, calculate_fairness_loss
from utils import move_data_to_device, edge_index_to_labels, balance_labels, adaptive_loss_weight, oversample_positive


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Training for Fairness-Aware Spatial Crowdsourcing Task Assignment Model")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--lambda_fair", type=float, default=0.1, help="Fairness weight")

    # New: Learning rate scheduler parameters
    parser.add_argument("--lr_patience", type=int, default=10, help="Learning rate scheduler patience")
    parser.add_argument("--lr_factor", type=float, default=0.5, help="Learning rate decay factor")
    parser.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate")

    # New: Checkpoint parameters
    parser.add_argument("--resume", type=str, default="", help="Resume training from checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint save directory")
    parser.add_argument("--checkpoint_freq", type=int, default=10, help="Checkpoint save frequency (epoch)")

    # Time range parameters
    parser.add_argument("--train_start", type=str, default='2019-01', help="Training start month (YYYY-MM)")
    parser.add_argument("--train_end", type=str, default='2019-12', help="Training end month (YYYY-MM)")
    parser.add_argument("--val_start", type=str, default='2020-01', help="Validation start month (YYYY-MM)")
    parser.add_argument("--val_end", type=str, default='2020-06', help="Validation end month (YYYY-MM)")
    parser.add_argument("--test_start", type=str, default='2020-07', help="Test start month (YYYY-MM)")
    parser.add_argument("--test_end", type=str, default='2020-12', help="Test end month (YYYY-MM)")


    return parser.parse_args()


def train(model, data, optimizer, criterion, lambda_p=0.6, current_util=None):
    """Train on a single month of data (with loss normalization)"""
    model.train()
    optimizer.zero_grad()

    # Forward pass
    allocation_scores = model(data, current_util)

    # Calculate various losses
    alloc_loss, benefit_loss, cost_loss, gini = calculate_fairness_loss(
        allocation_scores, data, criterion
    )

    # # Normalization processing
    # max_benefit = data.benefit_matrix.max().item()
    # max_cost = data.cost_matrix.max().item()
    #
    # # Normalization factors (prevent division by zero)
    # norm_benefit = max(max_benefit, 1e-8)
    # norm_cost = max(max_cost, 1e-8)
    #
    # # Normalized losses
    # norm_benefit_loss = benefit_loss / norm_benefit
    # norm_cost_loss = cost_loss / norm_cost

    # Total loss
    total_loss = (
            lambda_p * alloc_loss + (benefit_loss + cost_loss + gini)
    )

    # Backward pass
    total_loss.backward()
    optimizer.step()

    return {
        "total_loss": total_loss.item(),
        "alloc_loss": alloc_loss.item(),
        "benefit_loss": benefit_loss.item(),
        "cost_loss": cost_loss.item(),
        "gini": gini.item()
    }


def evaluate(model, data, criterion):
    """Evaluate on a single month of data (enhanced version)"""
    model.eval()
    with torch.no_grad():
        # Forward pass
        allocation_scores = model(data)

        # Calculate loss
        alloc_loss, benefit_loss, cost_loss, gini = calculate_fairness_loss(
            allocation_scores, data, criterion
        )

        # Calculate predicted probabilities
        y_pred_prob = torch.sigmoid(allocation_scores).cpu().numpy()
        y_true = data.y.cpu().numpy()

        # Calculate AUC
        auc = roc_auc_score(y_true, y_pred_prob) if len(np.unique(y_true)) > 1 else 0.5

        # Binarize prediction results
        y_pred_binary = (y_pred_prob >= 0.5).astype(int)

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred_binary) if len(y_true) > 0 else 0.0

        # ====================== Positive Sample Hit Rate ======================
        # Calculate number of predicted positives
        predicted_positives = np.sum(y_pred_binary)
        # Calculate number of actual positives
        actual_positives = np.sum(y_true)
        # Calculate true positives among predicted positives
        true_positives = np.sum((y_pred_binary == 1) & (y_true == 1))
        # Calculate hit rate
        hit_rate = true_positives / actual_positives if actual_positives > 0 else 0.0

        # ====================== Calculate Average Utility and Travel Distance ======================
        # Get utility matrix and distance matrix (assuming they are stored in the data object)
        utility_matrix = data.benefit_matrix.cpu().numpy()
        distance_matrix = data.cost_matrix.cpu().numpy()

        # Get number of workers and tasks
        n_workers, n_tasks = utility_matrix.shape

        # Reshape prediction results into a 2D matrix (workers x tasks)
        y_pred_matrix = y_pred_binary.reshape(n_workers, n_tasks)

        # Initialize cumulative values
        total_utility = 0.0
        total_distance = 0.0
        assigned_count = 0

        # Iterate through all workers and tasks
        for i in range(n_workers):
            for j in range(n_tasks):
                if y_pred_matrix[i, j] == 1:  # If the task is assigned
                    total_utility += utility_matrix[i, j]
                    total_distance += distance_matrix[i, j]
                    assigned_count += 1

        # Calculate averages
        avg_utility = total_utility / true_positives if true_positives > 0 else 0.0
        avg_distance = total_distance / assigned_count if assigned_count > 0 else 0.0

        # ====================== Calculate Fairness Metrics ======================
        # Calculate total utility for each worker
        worker_utilities = np.zeros(n_workers)
        for i in range(n_workers):
            for j in range(n_tasks):
                if y_pred_matrix[i, j] == 1:
                    worker_utilities[i] += utility_matrix[i, j]

        # Calculate Gini coefficient
        if np.sum(worker_utilities) > 0:
            gini_coef = gini_coefficient(worker_utilities)
        else:
            gini_coef = 0.0
        fairness = 1 - gini_coef  # Fairness = 1 - Gini coefficient

        return {
            "alloc_loss": alloc_loss.item(),
            "gini": gini_coef,
            "auc": auc,
            "accuracy": accuracy,
            "hit_rate": hit_rate,
            "avg_utility": avg_utility,
            "avg_distance": avg_distance,
            "fairness": fairness,
            "assigned_count": assigned_count,
            "total_utility": total_utility,
            "total_distance": total_distance
        }


def gini_coefficient(values):
    """Calculate Gini coefficient"""
    values = np.sort(values)
    n = len(values)
    if n == 0:
        return 0.0

    cum_values = np.cumsum(values)
    lorentz_area = np.sum(cum_values[:-1] + cum_values[1:]) / 2
    lorentz_area = lorentz_area / cum_values[-1] / n
    equality_area = n / 2
    gini = (equality_area - lorentz_area) / equality_area
    return gini


def load_monthly_data(period_dir):
    """Load monthly graph data"""
    data_path = os.path.join(period_dir, "graph_data.pt")
    # stats_path = os.path.join(period_dir, "stats.json")

    if not os.path.exists(data_path):
        return None

    # Load graph data
    data = torch.load(data_path)

    # Load statistics
    # stats = pd.read_json(stats_path, typ='series').to_dict()

    # # Add statistics to data object
    # data.num_workers = stats['num_workers']
    # data.num_tasks = stats['num_tasks']

    return data


def get_available_periods(data_dir):
    """Get all available time periods"""
    period_dirs = sorted([
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d)) and len(d) == 7
    ])
    return period_dirs


def filter_periods_by_date(periods, start_date, end_date):
    """Filter time periods by date range"""
    if start_date is None and end_date is None:
        return periods

    filtered = []
    for period in periods:
        try:
            period_date = datetime.strptime(period, "%Y-%m")
            if start_date and period_date < start_date:
                continue
            if end_date and period_date > end_date:
                continue
            filtered.append(period)
        except:
            continue

    return filtered


def save_checkpoint(state, filename):
    """Save training checkpoint"""
    torch.save(state, filename)
    print(f"Checkpoint saved to: {filename}")


def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    """Load training checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file does not exist: {checkpoint_path}")
        return 0, float('inf')

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']

    print(f"Resuming training from checkpoint: epoch={checkpoint['epoch']}, best val loss={best_val_loss:.4f}")
    return start_epoch, best_val_loss


def main(args):
    # Configuration parameters
    data_dir = "../dataset/Yelp-JSON/select_graph_data/monthly_1000_300_plan1_reduce"
    num_epochs = args.epochs
    lr = args.lr
    hidden_dim = args.hidden_dim
    num_heads = args.num_heads
    lambda_fair = args.lambda_fair
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Get all available time periods
    all_periods = get_available_periods(data_dir)

    if not all_periods:
        print("Error: No available data periods found")
        return

    # Parse date ranges
    train_start = datetime.strptime(args.train_start, "%Y-%m") if args.train_start else None
    train_end = datetime.strptime(args.train_end, "%Y-%m") if args.train_end else None
    val_start = datetime.strptime(args.val_start, "%Y-%m") if args.val_start else None
    val_end = datetime.strptime(args.val_end, "%Y-%m") if args.val_end else None
    test_start = datetime.strptime(args.test_start, "%Y-%m") if args.test_start else None
    test_end = datetime.strptime(args.test_end, "%Y-%m") if args.test_end else None

    # Filter time periods
    train_periods = filter_periods_by_date(all_periods, train_start, train_end)
    val_periods = filter_periods_by_date(all_periods, val_start, val_end)
    test_periods = filter_periods_by_date(all_periods, test_start, test_end)

    num_pos = 0
    num_neg = 0
    # Load training data
    train_data = []
    for period in train_periods:
        data = load_monthly_data(os.path.join(data_dir, period))
        if data:
            train_data.append(data)
            num_pos += data.worker_task_edge_index.shape[1]
            num_neg += data.worker_feat.shape[0] * data.task_feat.shape[0]

    # Load validation data
    val_data = []
    for period in val_periods:
        data = load_monthly_data(os.path.join(data_dir, period))
        if data:
            val_data.append(data)

    # Load test data
    test_data = []
    for period in test_periods:
        data = load_monthly_data(os.path.join(data_dir, period))
        if data:
            test_data.append(data)

    if not train_data:
        print("Error: No available training data found")
        return

    # Initialize model (using feature dimensions from the first training data period)
    worker_feat_dim = train_data[0].worker_feat.shape[1]
    task_feat_dim = train_data[0].task_feat.shape[1]

    model = FairGAT(worker_feat_dim, task_feat_dim, hidden_dim, 1000, num_heads).to(device)
    # model = FairGAT(worker_feat_dim, task_feat_dim, hidden_dim, 511, num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
    # criterion = FocalLoss(alpha=0.25, gamma=2.0).to(device)

    # ====================== New: Learning Rate Scheduler ======================
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.lr_patience,
        verbose=True,
        min_lr=args.min_lr
    )

    # ====================== New: Checkpoint Resume ======================
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.resume)
        start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)

    # Training log
    train_log = {
        'epoch': [],
        'month': [],
        'total_loss': [],
        'alloc_loss': [],
        'gini': [],
        'learning_rate': []  # New: Record learning rate
    }

    # Initialize validation log (stored separately)
    val_log = {
        'epoch': [],
        'val_loss': [],
        'val_gini': [],
        'val_auc': [],
        'val_hit_rate': []  # New: Record hit rate
    }

    # Best model tracking
    best_val_auc = 0.0
    best_epoch = -1
    best_model_path = ""

    current_util = None
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        epoch_start = time.time()
        epoch_total_loss = 0
        epoch_alloc_loss = 0
        epoch_gini = 0

        # Train by month
        for month_idx, data in enumerate(train_data):
            # Train current month
            data.y = edge_index_to_labels(data)
            # oversampled_data = oversample_positive(data)
            data = move_data_to_device(data, device)

            loss = train(
                model, data, optimizer, criterion, lambda_fair=lambda_fair, current_util=current_util
            )
            current_util = loss["benefit_loss"]
            epoch_total_loss += loss['total_loss']
            epoch_alloc_loss += loss['alloc_loss']
            epoch_gini += loss['gini']

            # Print monthly training info
            print(f"Epoch {epoch + 1}/{num_epochs} | Month {month_idx + 1}/{len(train_data)} | "
                  f"Loss: {loss['total_loss']:.4f} (Alloc: {loss['alloc_loss']:.4f}, Fair: {loss['gini']:.4f})")

            # Log records
            train_log['epoch'].append(epoch + 1)
            # train_log['month'].append(data.period)
            train_log['total_loss'].append(loss['total_loss'])
            train_log['alloc_loss'].append(loss['alloc_loss'])
            train_log['gini'].append(loss['gini'])
            train_log['learning_rate'].append(optimizer.param_groups[0]['lr'])  # Record current learning rate

            # Validation (last month)
            if month_idx == len(train_data) - 1:
                val_losses = []
                val_ginis = []
                val_aucs = []
                val_utility = []
                val_distance = []
                val_hit_rates = []

                for val_data_item in val_data:
                    val_data_item.y = edge_index_to_labels(val_data_item)
                    val_data_item = move_data_to_device(val_data_item, device)

                    # Use new evaluation function
                    eval_results = evaluate(model, val_data_item, criterion)

                    val_losses.append(eval_results['alloc_loss'])
                    val_ginis.append(eval_results["fairness"])
                    val_aucs.append(eval_results['auc'])
                    val_utility.append(eval_results["avg_utility"])
                    val_distance.append(eval_results["avg_distance"])
                    val_hit_rates.append(eval_results['hit_rate'])

                if val_losses:
                    avg_val_loss = np.mean(val_losses)
                    avg_val_gini = np.mean(val_ginis)
                    avg_val_utility = np.mean(val_utility)
                    avg_val_distance = np.mean(val_distance)
                    avg_val_auc = np.mean(val_aucs)
                    avg_val_hit_rate = np.mean(val_hit_rates)

                    print(f"Validation Results | Loss: {avg_val_loss:.4f} | Gini: {avg_val_gini:.4f} | "
                          f"AUC: {avg_val_auc:.4f} | Acc: {avg_val_hit_rate:.4f} | Utility: {avg_val_utility:.4f} | Distance: {avg_val_distance:.4f}")

                    # ====================== New: Update Learning Rate Scheduler ======================
                    scheduler.step(avg_val_loss)

                    # Record validation log
                    val_log['epoch'].append(epoch + 1)
                    val_log['val_loss'].append(avg_val_loss)
                    val_log['val_gini'].append(avg_val_gini)
                    val_log['val_auc'].append(avg_val_auc)
                    val_log['val_hit_rate'].append(avg_val_hit_rate)

                    # Check if it is the best model
                    if avg_val_auc > best_val_auc:
                        best_val_auc = avg_val_auc
                        best_epoch = epoch

                        # Save best model
                        # os.makedirs("models", exist_ok=True)
                        # best_model_path = f"models/best_model_epoch{epoch + 1}_auc{avg_val_auc:.4f}.pth"
                        # torch.save(model.state_dict(), best_model_path)
                        # print(f"New best model found! AUC: {avg_val_auc:.4f}, saved to: {best_model_path}")
                else:
                    print("Warning: No validation data")

                # Save log for current epoch
                train_log_df = pd.DataFrame(train_log)
                val_log_df = pd.DataFrame(val_log)

                train_log_path = f"logs/training_log_train_{args.train_start}_{args.train_end}_epoch{epoch + 1}.csv"
                val_log_path = f"logs/training_log_val_{args.train_start}_{args.train_end}_epoch{epoch + 1}.csv"

                os.makedirs("logs", exist_ok=True)
                train_log_df.to_csv(train_log_path, index=False)
                val_log_df.to_csv(val_log_path, index=False)

        # Calculate average loss
        avg_total_loss = epoch_total_loss / len(train_data)
        avg_alloc_loss = epoch_alloc_loss / len(train_data)
        avg_gini = epoch_gini / len(train_data)

        print(f"Epoch {epoch + 1} Average Loss | Total: {avg_total_loss:.4f} | "
              f"Alloc: {avg_alloc_loss:.4f} | Fair: {avg_gini:.4f}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch + 1} completed, duration: {epoch_time:.2f}s")

        # ====================== New: Save Checkpoint ======================
        # if (epoch + 1) % args.checkpoint_freq == 0 or epoch == num_epochs - 1:
        #     checkpoint_path = os.path.join(
        #         args.checkpoint_dir,
        #         f"checkpoint_epoch{epoch + 1}.pth"
        #     )
        #
        #     save_checkpoint({
        #         'epoch': epoch,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'scheduler_state_dict': scheduler.state_dict(),
        #         'best_val_loss': best_val_loss,
        #         'train_log': train_log,
        #         'val_log': val_log
        #     }, checkpoint_path)
        #
        #     # Update best validation loss
        #     if avg_val_loss < best_val_loss:
        #         best_val_loss = avg_val_loss
        #         best_checkpoint_path = os.path.join(
        #             args.checkpoint_dir,
        #             "best_checkpoint.pth"
        #         )
        #         save_checkpoint({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'scheduler_state_dict': scheduler.state_dict(),
        #             'best_val_loss': best_val_loss,
        #             'train_log': train_log,
        #             'val_log': val_log
        #         }, best_checkpoint_path)

    # Final test
    if test_data:
        test_losses = []
        test_ginis = []
        test_aucs = []
        test_hit_rates = []

        for test_data_item in test_data:
            test_data_item.y = edge_index_to_labels(test_data_item)
            test_data_item = move_data_to_device(test_data_item, device)

            # Use new evaluation function
            eval_results = evaluate(model, test_data_item, criterion)

            test_losses.append(eval_results['alloc_loss'])
            test_ginis.append(eval_results['gini'])
            test_aucs.append(eval_results['auc'])
            test_hit_rates.append(eval_results['hit_rate'])

        avg_test_loss = np.mean(test_losses)
        avg_test_gini = np.mean(test_ginis)
        avg_test_auc = np.mean(test_aucs)
        avg_test_hit_rate = np.mean(test_hit_rates)

        print("\nFinal Test Results:")
        print(f"Test Loss: {avg_test_loss:.4f}")
        print(f"Gini Coefficient: {avg_test_gini:.4f}")
        print(f"AUC: {avg_test_auc:.4f}")
        print(f"ACC: {avg_test_hit_rate:.4f}")

    # Save final model and logs
    os.makedirs("models", exist_ok=True)
    final_model_path = f"models/final_model_epoch{num_epochs}.pth"
    torch.save(model.state_dict(), final_model_path)

    # Save complete logs
    os.makedirs("logs", exist_ok=True)
    final_train_log_path = f"logs/training_log_train_{args.train_start}_{args.train_end}_final.csv"
    final_val_log_path = f"logs/training_log_val_{args.train_start}_{args.train_end}_final.csv"

    pd.DataFrame(train_log).to_csv(final_train_log_path, index=False)
    pd.DataFrame(val_log).to_csv(final_val_log_path, index=False)

    print("\nTraining completed!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Training log saved to: {final_train_log_path}")
    print(f"Validation log saved to: {final_val_log_path}")

    if best_epoch >= 0:
        print(f"Best model at Epoch {best_epoch + 1}, AUC: {best_val_auc:.4f}, Path: {best_model_path}")
    else:
        print("Warning: Best model not found")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)