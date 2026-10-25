"""Chronological optimization and frozen-parameter validation/test replay."""

import json
import logging
import math
import time
from pathlib import Path
import torch
from pfga.assignment import constrained_greedy_assignment
from pfga.candidates import CandidateGraphMaterializer
from pfga.data import load_periods, _torch_load
from pfga.losses import composite_loss
from pfga.metrics import ReplayMetrics
from pfga.model import PFGA
from pfga.replay import StreamData
from pfga.reproducibility import environment_metadata, resolve_device, seed_everything

LOGGER = logging.getLogger(__name__)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


class LegacyData:
    """Explicit compatibility adapter. No claim of chronological mobility replay."""

    def __init__(self, batches, config):
        self.batches = batches
        self.config = config
        self.num_workers = max(int(x.worker_ids.max()) for x in batches) + 1
        self.worker_dim = batches[0].worker_features.shape[1]
        self.task_dim = batches[0].task_features.shape[1]

    def replay(self):
        return LegacyReplay(self)


class LegacyReplay:
    def __init__(self, data):
        self.data = data
        self.it = iter(data.batches)
        self.batch = None

    def __iter__(self):
        return self

    def __next__(self):
        b = next(self.it)
        gen = torch.Generator().manual_seed(self.data.config.data.sampling_seed)
        e = b.feasible_edge_index
        y = b.labels_for_edges(e)
        pos = e[:, y == 1]
        neg = e[:, y == 0]
        neg = neg[
            :,
            torch.randperm(neg.shape[1], generator=gen)[
                : self.data.config.data.negative_ratio * pos.shape[1]
            ],
        ]
        b.supervision_edge_index = torch.cat((pos, neg), 1)
        b.reference_opportunities = torch.bincount(e[0], minlength=b.num_workers).float()
        b.distance_scale_km = self.data.config.evaluation.distance_scale_km
        if b.task_rewards is None:
            raise ValueError(
                "Legacy upgrade needs explicit task_rewards; do not infer from utility"
            )
        self.batch = b
        self.new_tasks = [{"reward": float(x)} for x in b.task_rewards]
        return b

    def prediction_batch(self):
        return self.batch

    def accept(self, i, j):
        return True


class PFGATrainer:
    def __init__(self, config, train_batches, validation_batches, test_batches):
        self.config = config
        sources = [train_batches, validation_batches, test_batches]
        self.train_data, self.val_data, self.test_data = [
            LegacyData(x, config) if isinstance(x, list) else x for x in sources
        ]
        if (
            len(
                {
                    (x.worker_dim, x.task_dim)
                    for x in (self.train_data, self.val_data, self.test_data)
                }
            )
            != 1
        ):
            raise ValueError("Feature dimensions differ across splits")
        self.num_workers = max(
            x.num_workers for x in (self.train_data, self.val_data, self.test_data)
        )
        self.device = resolve_device(config.experiment.device)
        self.model = PFGA(
            self.train_data.worker_dim, self.train_data.task_dim, self.num_workers, config.model
        ).to(self.device)
        self.materializer = CandidateGraphMaterializer(config.candidates)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.training.scheduler_factor,
            patience=config.training.scheduler_patience,
            min_lr=config.training.minimum_learning_rate,
        )
        self.output = Path(config.training.output_directory)
        self.output.mkdir(parents=True, exist_ok=True)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def replay(self, data, training=False):
        # Defined split protocol: fresh zero fairness states and common train-based locations.
        self.model.fairness_memory.reset()
        self.model.train(training)
        tracker = ReplayMetrics(self.num_workers)
        rows = []
        loss_sum = 0.0
        steps = 0
        stream = data.replay()
        for cpu in stream:
            b = cpu.to(self.device)
            ref = stream.prediction_batch().to(self.device)
            self.model.fairness_memory.begin_step(b.worker_ids)
            if training:
                self.optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                self.sync()
                start = time.perf_counter()
                candidate = self.materializer(b, self.model.fairness_memory.cumulative_utility)
                logits = self.model(b, candidate.edge_index)
                with torch.no_grad():
                    assignment = constrained_greedy_assignment(
                        b,
                        candidate.edge_index,
                        logits,
                        self.config.assignment,
                        accept=stream.accept,
                    )
                self.sync()
                elapsed = time.perf_counter() - start
                # Fixed reference samples are scored even when policy-pruned or busy.
                pred = self.model(ref, ref.supervision_edge_index)
                y = ref.labels_for_edges(ref.supervision_edge_index)
                losses = composite_loss(
                    logits,
                    b.labels_for_edges(candidate.edge_index),
                    b,
                    candidate.edge_index,
                    self.config.loss,
                    historical_utility=self.model.fairness_memory.cumulative_utility,
                    threshold=self.config.assignment.threshold_base,
                    prediction_logits=pred,
                    prediction_labels=y,
                )
                if not torch.isfinite(losses.total):
                    raise FloatingPointError("Nonfinite loss")
                if training:
                    losses.total.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clip_norm
                    )
                    self.optimizer.step()
            self.model.update_fairness_memory(b, assignment.edge_index)
            tracker.add(
                b,
                assignment,
                y,
                torch.sigmoid(pred),
                [x["reward"] for x in stream.new_tasks],
                elapsed,
            )
            loss_sum += float(losses.total.detach())
            steps += 1
            if not training:
                row = tracker.result(self.config.assignment.threshold_base, with_predictions=False)
                row.update(
                    {
                        "decision_step": steps,
                        "timestamp_hours": getattr(stream, "time", None),
                        "candidate_edges": candidate.edge_index.shape[1],
                        "candidate_positive_recall": candidate.positive_recall,
                    }
                )
                rows.append(row)
            if steps % 100 == 0:
                LOGGER.info(
                    "%s step=%d released=%d assigned=%d",
                    "train" if training else "eval",
                    steps,
                    tracker.released,
                    tracker.assigned,
                )
        summary = tracker.result(self.config.assignment.threshold_base)
        summary["loss"] = loss_sum / max(steps, 1)
        summary["decision_steps"] = steps
        if self.config.data.mode == "legacy_monthly":
            summary["protocol"] = "legacy_static_months_NOT_paper_event_replay"
        return summary, rows, tracker

    def checkpoint(self, path, epoch, best_auc):
        # Store parameters with zero runtime state, so inference cannot inherit a previous split.
        snapshot = self.model.fairness_memory.snapshot()
        self.model.fairness_memory.reset()
        torch.save(
            {
                "epoch": epoch,
                "best_validation_auc": best_auc,
                "model": self.model.state_dict(),
                "config": self.config.to_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )
        self.model.fairness_memory.restore(snapshot)

    def train(self):
        dump(self.output / "config.resolved.json", self.config.to_dict())
        dump(self.output / "environment.json", environment_metadata())
        best = -math.inf
        best_epoch = 0
        with (self.output / "history.jsonl").open("w", encoding="utf-8") as history:
            for epoch in range(1, self.config.training.epochs + 1):
                tr, _, _ = self.replay(self.train_data, True)
                val, _, _ = self.replay(self.val_data)
                self.scheduler.step(val["loss"])
                # Undefined AUC must not masquerade as .5; fall back explicitly to validation loss.
                value = val["auc"] if val["auc"] is not None else -val["loss"]
                if value > best:
                    best = value
                    best_epoch = epoch
                    self.checkpoint(self.output / "best.pt", epoch, val["auc"])
                self.checkpoint(self.output / "last.pt", epoch, val["auc"])
                history.write(json.dumps({"epoch": epoch, "train": tr, "validation": val}) + "\n")
                history.flush()
                LOGGER.info(
                    "epoch=%d/%d train_loss=%.5f val_auc=%s val_utility=%.4f val_fair=%.4f",
                    epoch,
                    self.config.training.epochs,
                    tr["loss"],
                    val["auc"],
                    val["utility"],
                    val["fair"],
                )
        best_checkpoint = _torch_load(self.output / "best.pt")
        self.model.load_state_dict(best_checkpoint["model"])
        sweep = []
        for threshold in self.config.assignment.threshold_grid:
            self.config.assignment.threshold_base = float(threshold)
            result, _, _ = self.replay(self.val_data)
            sweep.append(result)
        if sweep:
            chosen = max(
                sweep,
                key=lambda x: (
                    x["acc"] if x["acc"] is not None else -math.inf,
                    x["utility"],
                    -x["fair"],
                ),
            )
            self.config.assignment.threshold_base = chosen["threshold"]
        dump(self.output / "validation_thresholds.json", sweep)
        self.checkpoint(self.output / "best.pt", best_epoch, best_checkpoint["best_validation_auc"])
        test, rows, tracker = self.replay(self.test_data)
        dump(self.output / "test_trajectory.json", rows)
        dump(
            self.output / "worker_outcomes.json",
            {
                "utility": tracker.utility.tolist(),
                "reward": tracker.rewards.tolist(),
                "reference_opportunities": tracker.opportunities.tolist(),
            },
        )
        summary = {
            "experiment": self.config.experiment.name,
            "best_epoch": best_epoch,
            "selected_threshold": self.config.assignment.threshold_base,
            "test": test,
            "test_periods": rows,
            "selection_rule": "validation AUC (loss fallback), threshold by validation ACC",
            "protocol": "each split starts with zero utility/memory; test parameters frozen",
            "timing_scope": (
                "retrieval, model inference and greedy; "
                "excludes event/feature construction, reference evaluation, loss and backprop"
            ),
        }
        dump(self.output / "summary.json", summary)
        dump(self.output / "config.resolved.json", self.config.to_dict())
        return summary


def load_experiment_data(config):
    root = Path(config.data.directory)
    if config.data.mode == "stream":
        return tuple(StreamData(root / f"{s}.pt", config) for s in ("train", "validation", "test"))
    return tuple(
        load_periods(
            str(root),
            getattr(config.data, s).start,
            getattr(config.data, s).end,
            config.data.strict_paper_schema,
        )
        for s in ("train", "validation", "test")
    )


def run_experiment(config):
    config.validate()
    seed_everything(config.experiment.seed, config.experiment.deterministic)
    return PFGATrainer(config, *load_experiment_data(config)).train()
