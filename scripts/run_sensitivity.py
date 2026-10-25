#!/usr/bin/env python3
"""Run one-dimensional PFGA hyperparameter sensitivity experiments."""

import argparse
import copy
import json
from pathlib import Path
from typing import List, Union

from pfga.config import PFGAConfig, load_config
from pfga.trainer import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--parameter",
        required=True,
        choices=(
            "learning_rate",
            "hidden_dim",
            "lambda_p",
            "epochs",
            "candidate_k",
            "fairness_weight",
            "fairness_gamma",
            "deficit_score_weight",
            "memory_score_weight",
        ),
    )
    parser.add_argument("--values", nargs="+", required=True)
    parser.add_argument("--output-root", default=None)
    return parser.parse_args()


def _parse_values(parameter: str, values: List[str]) -> List[Union[int, float]]:
    integer_parameters = {"hidden_dim", "epochs", "candidate_k"}
    converter = int if parameter in integer_parameters else float
    return [converter(value) for value in values]


def _set_value(config: PFGAConfig, parameter: str, value: Union[int, float]) -> None:
    if parameter == "learning_rate":
        config.training.learning_rate = float(value)
    elif parameter == "hidden_dim":
        config.model.hidden_dim = int(value)
    elif parameter == "lambda_p":
        config.loss.lambda_p = float(value)
    elif parameter == "epochs":
        config.training.epochs = int(value)
    elif parameter == "candidate_k":
        config.candidates.top_k = int(value)
    elif parameter == "fairness_weight":
        config.loss.fairness_weight = float(value)
    elif parameter in {"fairness_gamma", "deficit_score_weight", "memory_score_weight"}:
        setattr(config.model, parameter, float(value))
    else:
        raise ValueError(f"Unsupported parameter: {parameter}")


def main() -> None:
    args = parse_args()
    base = load_config(args.config)
    values = _parse_values(args.parameter, args.values)
    root = Path(
        args.output_root
        or f"{base.training.output_directory.rstrip('/')}_sensitivity_{args.parameter}"
    )
    summaries = {}
    for value in values:
        config = copy.deepcopy(base)
        _set_value(config, args.parameter, value)
        label = str(value).replace(".", "p")
        config.experiment.name = f"{base.experiment.name}_{args.parameter}_{label}"
        config.training.output_directory = str(root / label)
        config.validate()
        summaries[str(value)] = run_experiment(config)
    root.mkdir(parents=True, exist_ok=True)
    (root / "sensitivity_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
