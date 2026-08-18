"""Chronological training, isolated validation, checkpointing, and test evaluation."""

import json
import logging
import math
import time
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Tuple

import torch

from pfga.assignment import constrained_greedy_assignment
from pfga.candidates import CandidateGraphMaterializer
from pfga.config import PFGAConfig
from pfga.data import (
    MonthlyBatch,
    infer_global_num_workers,
    load_periods,
    validate_feature_dimensions,
)
from pfga.losses import composite_loss
from pfga.metrics import evaluate_assignment
from pfga.model import PFGA
from pfga.reproducibility import environment_metadata, resolve_device, seed_everything


LOGGER = logging.getLogger(__name__)


def _json_dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _torch_load(path: Path, device: torch.device) -> object:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _average_numeric(records: Iterable[Dict[str, object]]) -> Dict[str, object]:
    records = list(records)
    if not records:
        return {}
    result: Dict[str, object] = {}
    keys = set().union(*(record.keys() for record in records))
    for key in sorted(keys):
        values = [record[key] for record in records if key in record]
        numeric = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if len(numeric) == len(values):
            result[key] = mean(numeric)
        elif key == "auc_scope":
            result[key] = sorted(set(str(value) for value in values))
    return result


class PFGATrainer:
    def __init__(
        self,
        config: PFGAConfig,
        train_batches: List[MonthlyBatch],
        validation_batches: List[MonthlyBatch],
        test_batches: List[MonthlyBatch],
    ):
        self.config = config
        self.train_batches = train_batches
        self.validation_batches = validation_batches
        self.test_batches = test_batches
        self.device = resolve_device(config.experiment.device)
        worker_dim, task_dim = validate_feature_dimensions(
            (train_batches, validation_batches, test_batches)
        )
        num_workers = infer_global_num_workers(
            (train_batches, validation_batches, test_batches)
        )
        self.model = PFGA(worker_dim, task_dim, num_workers, config.model).to(self.device)
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
        self.history_path = self.output / "history.jsonl"
        self.history_path.write_text("", encoding="utf-8")

    def _train_batch(self, cpu_batch: MonthlyBatch) -> Dict[str, float]:
        batch = cpu_batch.to(self.device)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        candidate = self.materializer(
            batch, self.model.fairness_memory.cumulative_utility
        )
        logits = self.model(batch, candidate.edge_index)
        labels = batch.labels_for_edges(candidate.edge_index)
        losses = composite_loss(
            logits, labels, batch, candidate.edge_index, self.config.loss
        )
        if not torch.isfinite(losses.total):
            raise FloatingPointError(f"{batch.period}: non-finite training loss")
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.training.gradient_clip_norm
        )
        self.optimizer.step()
        with torch.no_grad():
            self.model.update_fairness_memory(
                batch, candidate.edge_index, torch.sigmoid(logits)
            )
        record = losses.scalars()
        record.update(
            {
                "candidate_recall": candidate.positive_recall,
                "candidate_edges": float(candidate.edge_index.shape[1]),
            }
        )
        return record

    @torch.no_grad()
    def _evaluate_split(
        self,
        batches: List[MonthlyBatch],
        isolate_memory: bool,
    ) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
        snapshot = self.model.fairness_memory.snapshot() if isolate_memory else None
        self.model.eval()
        period_records: List[Dict[str, object]] = []
        try:
            for cpu_batch in batches:
                batch = cpu_batch.to(self.device)
                start = time.perf_counter()
                candidate = self.materializer(
                    batch, self.model.fairness_memory.cumulative_utility
                )
                logits = self.model(batch, candidate.edge_index)
                assignment = constrained_greedy_assignment(
                    batch, candidate.edge_index, logits, self.config.assignment
                )
                elapsed = time.perf_counter() - start
                labels = batch.labels_for_edges(candidate.edge_index)
                losses = composite_loss(
                    logits, labels, batch, candidate.edge_index, self.config.loss
                )
                metrics = evaluate_assignment(
                    batch=batch,
                    candidate_edge_index=candidate.edge_index,
                    logits=logits,
                    assignment=assignment,
                    candidate_recall=candidate.positive_recall,
                    inference_seconds=elapsed,
                    cost_unit_km=self.config.evaluation.cost_unit_km,
                    dense_auc_pair_limit=self.config.evaluation.dense_auc_pair_limit,
                )
                metrics.update(losses.scalars())
                period_records.append(metrics)
                self.model.update_fairness_memory(
                    batch, candidate.edge_index, torch.sigmoid(logits)
                )
        finally:
            if snapshot is not None:
                self.model.fairness_memory.restore(snapshot)
        return _average_numeric(period_records), period_records

    def _save_checkpoint(self, path: Path, epoch: int, best_auc: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "best_validation_auc": best_auc,
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "config": self.config.to_dict(),
            },
            path,
        )

    def _append_history(self, record: Dict[str, object]) -> None:
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    def train(self) -> Dict[str, object]:
        _json_dump(self.output / "config.resolved.json", self.config.to_dict())
        _json_dump(self.output / "environment.json", environment_metadata())
        best_auc = -math.inf
        best_epoch = 0

        for epoch in range(1, self.config.training.epochs + 1):
            batch_records = [self._train_batch(batch) for batch in self.train_batches]
            train_summary = _average_numeric(batch_records)
            validation_summary, validation_periods = self._evaluate_split(
                self.validation_batches, isolate_memory=True
            )
            validation_loss = float(validation_summary["loss"])
            validation_auc = float(validation_summary["auc"])
            self.scheduler.step(validation_loss)

            history_record: Dict[str, object] = {
                "epoch": epoch,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "train": train_summary,
                "validation": validation_summary,
                "validation_periods": validation_periods,
            }
            self._append_history(history_record)
            self._save_checkpoint(self.output / "last.pt", epoch, best_auc)
            if validation_auc > best_auc:
                best_auc = validation_auc
                best_epoch = epoch
                self._save_checkpoint(self.output / "best.pt", epoch, best_auc)

            LOGGER.info(
                "epoch=%d train_loss=%.5f val_loss=%.5f val_auc=%.5f",
                epoch,
                float(train_summary["loss"]),
                validation_loss,
                validation_auc,
            )

        best_checkpoint = _torch_load(self.output / "best.pt", self.device)
        if not isinstance(best_checkpoint, dict) or "model" not in best_checkpoint:
            raise ValueError("Invalid best checkpoint")
        self.model.load_state_dict(best_checkpoint["model"])
        test_summary, test_periods = self._evaluate_split(
            self.test_batches, isolate_memory=False
        )
        summary: Dict[str, object] = {
            "experiment": self.config.experiment.name,
            "best_epoch": best_epoch,
            "best_validation_auc": best_auc,
            "test": test_summary,
            "test_periods": test_periods,
        }
        _json_dump(self.output / "summary.json", summary)
        return summary


def load_experiment_data(
    config: PFGAConfig,
) -> Tuple[List[MonthlyBatch], List[MonthlyBatch], List[MonthlyBatch]]:
    strict = config.data.strict_paper_schema
    directory = config.data.directory
    train = load_periods(
        directory, config.data.train.start, config.data.train.end, strict=strict
    )
    validation = load_periods(
        directory,
        config.data.validation.start,
        config.data.validation.end,
        strict=strict,
    )
    test = load_periods(
        directory, config.data.test.start, config.data.test.end, strict=strict
    )
    return train, validation, test


def run_experiment(config: PFGAConfig) -> Dict[str, object]:
    seed_everything(config.experiment.seed, config.experiment.deterministic)
    train, validation, test = load_experiment_data(config)
    trainer = PFGATrainer(config, train, validation, test)
    return trainer.train()
