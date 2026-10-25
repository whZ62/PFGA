#!/usr/bin/env python3
"""Run repeated seeds and aggregate test metrics as mean and sample std (ddof=1)."""

import argparse
import copy
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

from pfga.config import load_config
from pfga.trainer import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    return parser.parse_args()


def _aggregate(summaries: Dict[str, object]) -> Dict[str, Dict[str, float]]:
    tests = [summary["test"] for summary in summaries.values()]
    keys = (
        "auc",
        "acc",
        "utility",
        "fair",
        "cost",
        "time",
        "assignment_rate",
        "reward_gini",
        "opportunity_gini",
    )
    result: Dict[str, Dict[str, float]] = {}
    for key in keys:
        values: List[float] = [float(test[key]) for test in tests if test.get(key) is not None]
        if not values:
            continue
        result[key] = {
            "mean": mean(values),
            "std": stdev(values) if len(values) > 1 else 0.0,
            "n": len(values),
        }
    return result


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    root = Path(args.output_root or f"{base.training.output_directory.rstrip('/')}_seeds")
    summaries = {}
    for seed in args.seeds:
        config = copy.deepcopy(base)
        config.experiment.seed = seed
        config.experiment.name = f"{base.experiment.name}_seed_{seed}"
        config.training.output_directory = str(root / f"seed_{seed}")
        if args.epochs is not None:
            config.training.epochs = args.epochs
        summaries[str(seed)] = run_experiment(config)
    payload = {"runs": summaries, "aggregate": _aggregate(summaries)}
    root.mkdir(parents=True, exist_ok=True)
    (root / "seed_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
