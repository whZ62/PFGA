#!/usr/bin/env python3
"""Run the ablations described in the paper from one base configuration."""

import argparse
import copy
import json
from pathlib import Path
from typing import Callable, Dict

from pfga.config import PFGAConfig, load_config
from pfga.trainer import run_experiment


def _without_antibody(config: PFGAConfig) -> None:
    config.model.use_antibody_injection = False


def _without_memory(config: PFGAConfig) -> None:
    config.model.use_fairness_memory = False


def _without_social(config: PFGAConfig) -> None:
    config.model.use_social_aggregation = False


def _without_materialization(config: PFGAConfig) -> None:
    config.candidates.enabled = False


def _without_prediction_loss(config: PFGAConfig) -> None:
    config.loss.lambda_p = 0.0


def _without_utility_loss(config: PFGAConfig) -> None:
    config.loss.utility_weight = 0.0


def _without_cost_loss(config: PFGAConfig) -> None:
    config.loss.cost_weight = 0.0


def _without_fairness_loss(config: PFGAConfig) -> None:
    config.loss.fairness_weight = 0.0


def _score(config, mode):
    config.model.compensation = mode
    config.model.use_antibody_injection = False


VARIANTS: Dict[str, Callable[[PFGAConfig], None]] = {
    "full": lambda c: None,
    "deficit_score": lambda c: _score(c, "deficit_score"),
    "memory_score": lambda c: _score(c, "memory_score"),
    "joint_score": lambda c: _score(c, "joint_score"),
    "state_concat": lambda c: _score(c, "concat"),
    "without_antibody_injection": _without_antibody,
    "without_fairness_memory": _without_memory,
    "without_social_aggregation": _without_social,
    "without_graph_materialization": _without_materialization,
    "without_prediction_loss": _without_prediction_loss,
    "without_utility_loss": _without_utility_loss,
    "without_cost_loss": _without_cost_loss,
    "without_fairness_loss": _without_fairness_loss,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--variants", nargs="*", choices=tuple(VARIANTS), default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    root = Path(args.output_root or f"{base.training.output_directory.rstrip('/')}_ablations")
    chosen = args.variants or list(VARIANTS)
    summaries = {}
    for name in chosen:
        config = copy.deepcopy(base)
        VARIANTS[name](config)
        config.experiment.name = f"{base.experiment.name}_{name}"
        config.training.output_directory = str(root / name)
        if args.epochs is not None:
            config.training.epochs = args.epochs
        config.validate()
        summaries[name] = run_experiment(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / "ablation_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
